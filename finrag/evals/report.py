"""
Renders out/REPORT.md from out/eval-results.json.

The report is deliberately conservative: it prints paired win/loss/tie counts
next to every headline delta, because at 17 answerable questions one question is
worth ~5.9 points and most of the gaps this report is used to argue about are
smaller than that.
"""
from __future__ import annotations

import json

from ..config import P
from .metrics import mean


def pct(x) -> str:
    return f"{(x or 0) * 100:.1f}%"


def n3(x) -> str:
    return f"{float(x or 0):.3f}"


def delta(a, b) -> str:
    d = (b or 0) - (a or 0)
    sign = "+" if d > 0 else ""
    return f"{sign}{d * 100:.1f} pts"


def table(headers: list, rows: list) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def main() -> None:
    src = P.out / "eval-results.json"
    if not src.exists():
        raise SystemExit("No out/eval-results.json. Run `python -m finrag.evals.run` first.")
    r = json.loads(src.read_text(encoding="utf-8"))

    ret = r["retrieval"]["summary"]
    gen = r["generation"]["summary"]

    def find(arm):
        return next((s for s in ret if s["arm"] == arm), None)

    fixed_dense = find("fixed/dense")
    sem_dense = find("semantic/dense")
    best_arm = max((s for s in ret if "rerank" not in s["arm"]), key=lambda s: s["ndcg"])
    base = best_arm["arm"].split("/")[0]
    best_hybrid = find(base + "/hybrid") or best_arm
    best_rerank = find(base + "/hybrid+rerank")
    fixed_rr = find("fixed/hybrid+rerank")
    sem_rr = find("semantic/hybrid+rerank")

    cs = r.get("chunkStats") or {}

    # The best arm overall, reranked ones included. An earlier version ranked only
    # the non-reranked arms and then reported that winner's rerank variant - so
    # when the two chunking strategies swapped places under reranking, the report
    # silently showed the runner-up and never mentioned the reversal.
    global_best = max(ret, key=lambda s: s["ndcg"])

    def rows_of(arm_id):
        run = next((x for x in r["retrieval"]["runs"] if x["id"] == arm_id), None)
        return run["rows"] if run else []

    def by_category(arm_id):
        rows = rows_of(arm_id)
        out = []
        for c in dict.fromkeys(x["category"] for x in rows):
            sub = [x for x in rows if x["category"] == c]
            out.append({"category": c, "n": len(sub),
                        "hitRate": mean([x["hitRate"] for x in sub]),
                        "ndcg": mean([x["ndcg"] for x in sub])})
        return out

    def paired(arm_a, arm_b, metric="hitRate"):
        """
        Paired per-question comparison between two arms.

        With 17 answerable questions a single question is worth ~5.9 points, so a
        difference of a few thousandths of nDCG is noise. Counting wins, losses and
        ties makes that visible where a bare delta hides it.
        """
        by_id = {x["qid"]: x for x in rows_of(arm_b)}
        a = rows_of(arm_a)
        wins = losses = ties = 0
        for ra in a:
            rb = by_id.get(ra["qid"])
            if not rb:
                continue
            if ra[metric] > rb[metric]:
                wins += 1
            elif ra[metric] < rb[metric]:
                losses += 1
            else:
                ties += 1
        return {"wins": wins, "losses": losses, "ties": ties, "n": len(a),
                "decisive": abs(wins - losses) > 1}

    def verdict(arm_a, arm_b, ndcg_a, ndcg_b):
        """Phrase a comparison honestly given the sample size."""
        p = paired(arm_a, arm_b)
        lead = arm_a if ndcg_a >= ndcg_b else arm_b
        if not p["decisive"]:
            per_q = 100 / (p["n"] or 1)
            return (f"**`{arm_a}` and `{arm_b}` are indistinguishable at this sample size** "
                    f"(nDCG@6 {n3(ndcg_a)} vs {n3(ndcg_b)}; paired over {p['n']} questions: "
                    f"{p['wins']} win / {p['losses']} loss / {p['ties']} tie). One question is "
                    f"worth ~{per_q:.1f} points here, so this gap is noise, not a result.")
        return (f"**`{lead}` wins** (nDCG@6 {n3(max(ndcg_a, ndcg_b))} vs "
                f"{n3(min(ndcg_a, ndcg_b))}; paired: {p['wins']} win / {p['losses']} loss / "
                f"{p['ties']} tie over {p['n']} questions).")

    def gen_stats(run):
        """
        Faithfulness is only ever measured on answers the pipeline already decided
        to emit: `verify` refuses anything it judges ungrounded, and the runner
        judges only non-refused rows. So the score is conditional on surviving an
        LLM grounding gate, and it can be raised simply by refusing more. Reporting
        it without its denominator - and without the answer rate beside it - makes
        a cautious, unhelpful arm look like the safest one.
        """
        rows = run.get("rows") or []
        judged = [x for x in rows if x.get("judged")]
        answered = [x for x in rows if not x["refused"] and not x["clarified"]]
        # The outcome this system exists to prevent, counted rather than averaged:
        # an answer that was delivered and was wrong, or cited a passage that does
        # not support it. An averaged 0-1 correctness score buries these.
        wrong = [x for x in judged
                 if x["judged"]["correctness"] < 1 or not x["judged"]["citationsValid"]]
        # A row whose model call threw is an infrastructure failure, not a decision.
        # Scoring it as a refusal silently credits/debits the model for a timeout.
        errored = [x for x in rows if x.get("error")]
        return {
            "judgedN": len(judged), "answeredN": len(answered),
            "faithfulness": mean([x["judged"]["faithfulness"] for x in judged]),
            "correctness": mean([x["judged"]["correctness"] for x in judged]),
            "wrongCount": len(wrong), "wrongIds": [x["qid"] for x in wrong],
            "erroredN": len(errored), "erroredIds": [x["qid"] for x in errored],
        }

    extra = {run["id"]: gen_stats(run) for run in r["generation"]["runs"]}

    # A row whose model call threw was scored as a refusal by the runner; say so
    # rather than letting a network timeout masquerade as a behavioural result.
    error_notes = "\n".join(
        f"- **{s['arm']}** had {extra[s['arm']]['erroredN']} row(s) fail with an infrastructure "
        f"error ({', '.join(extra[s['arm']]['erroredIds'])}); the runner scores those as "
        f"refusals, so its false-refusal figure is inflated by that amount."
        for s in gen if extra[s["arm"]]["erroredN"]
    )

    all_rows = r["generation"]["runs"][0]["rows"] if r["generation"]["runs"] else []
    n_questions = len({x["qid"] for x in all_rows})
    n_refuse = len({x["qid"] for x in all_rows if x["behaviour"]["want"] == "refuse"})
    n_clarify = len({x["qid"] for x in all_rows if x["behaviour"]["want"] == "clarify"})
    n_retrieval_q = len(rows_of("fixed/dense"))

    # --- failure analysis -------------------------------------------------
    # Includes wrong answers, not just wrong *behaviour*. The previous filter
    # checked only `behaviour.correct` and `faithfulness`, so an answer that was
    # faithful to its passages but reported the wrong figure never appeared here -
    # which is precisely the failure the project exists to catch.
    last_gen = r["generation"]["runs"][-1] if r["generation"]["runs"] else {"rows": []}
    failures = [
        x for x in (last_gen.get("rows") or [])
        if (not x["behaviour"]["correct"])
        or (x.get("judged") and (x["judged"]["faithfulness"] < 1
                                 or x["judged"]["correctness"] < 1
                                 or not x["judged"]["citationsValid"]))
        or x.get("error")
    ]

    ret_cols = ["Arm", "hit@6", "evid@6", "prec@6", "MRR", "nDCG@6", "p50 ms", "p95 ms"]

    def ret_row(s):
        return [f"`{s['arm']}`", n3(s["hitRate"]), n3(s["evidenceCoverage"]),
                n3(s["precisionAtK"]), n3(s["mrr"]), n3(s["ndcg"]),
                s["latencyP50"], s.get("latencyP95", "-")]

    chunk_verdict = verdict("fixed/dense", "semantic/dense",
                            fixed_dense["ndcg"], sem_dense["ndcg"])

    if fixed_rr and sem_rr:
        rr = paired("fixed/hybrid+rerank", "semantic/hybrid+rerank")
        if rr["decisive"]:
            rerank_verdict = ("Under reranking the picture changes: "
                              + verdict("fixed/hybrid+rerank", "semantic/hybrid+rerank",
                                        fixed_rr["ndcg"], sem_rr["ndcg"]))
        else:
            quoted_reversal = '"reversal"'
            rerank_verdict = (
                f"The same holds once reranking is applied (nDCG@6 {n3(fixed_rr['ndcg'])} vs "
                f"{n3(sem_rr['ndcg'])}; paired {rr['wins']}/{rr['losses']}/{rr['ties']}). "
                f"**This report makes no claim that either chunking strategy is better.** An "
                f"earlier draft argued a {quoted_reversal} between the two — that argument turned "
                f"on a single question, and does not survive a paired comparison.")
    else:
        rerank_verdict = "_Rerank arms not present in this run._"

    shipped = r["bestRetrievalArm"] + "+rerank"
    shipped_row = find(shipped)
    if global_best["arm"] != shipped:
        margin = abs(global_best["ndcg"] - (shipped_row or global_best)["ndcg"])
        selection_note = (
            f"Note the selection is nominal: `{global_best['arm']}` scores highest overall "
            f"(nDCG@6 {n3(global_best['ndcg'])}), and the margin over the shipped arm is "
            f"{n3(margin)} — inside the noise floor for {n_retrieval_q} questions. Do not read "
            f"the Stage 2 numbers as belonging to a demonstrated best configuration.")
    else:
        selection_note = ""

    if best_rerank:
        rerank_impact = (
            f"Reranking moves nDCG@6 by **{delta(best_hybrid['ndcg'], best_rerank['ndcg'])}** and "
            f"MRR by **{delta(best_hybrid['mrr'], best_rerank['mrr'])}**, at a latency cost of "
            f"**{best_rerank['latencyP50'] - best_hybrid['latencyP50']} ms at p50**.")
    else:
        rerank_impact = "_Rerank arm not present in this run._"

    spread_arm = next((s for s in gen if s.get("spread")), None)
    if spread_arm:
        reps = spread_arm["spread"]["repeats"]
        spread_table = table(
            ["Arm", "Answer rate", "Correct refusal", "Faithfulness", "Wrong-figure",
             "Unstable questions"],
            [[s["arm"],
              f"{pct(s['spread']['answerRate']['min'])} - {pct(s['spread']['answerRate']['max'])}",
              f"{pct(s['spread']['correctRefusal']['min'])} - "
              f"{pct(s['spread']['correctRefusal']['max'])}",
              f"{n3(s['spread']['faithfulness']['min'])} - "
              f"{n3(s['spread']['faithfulness']['max'])}",
              f"{s['spread']['wrong']['min']} - {s['spread']['wrong']['max']}",
              ", ".join(s["spread"]["flaky"]) if s["spread"]["flaky"] else "none"]
             for s in gen if s.get("spread")])
        spread_block = (
            f"**Run-to-run spread.** Stage 2 was repeated {reps} times on identical code. The "
            f"gates are model calls and do not settle on one answer, so each figure above is one "
            f"sample from these ranges:\n\n{spread_table}\n\n"
            f"A question listed as unstable passed in some repetitions and failed in others. "
            f"Quoting any single-run figure from the table above as *the* result overstates what "
            f"{reps} runs can support.")
    else:
        spread_block = ("**Single run.** Stage 2 ran once, so none of the figures above carry an "
                        "error bar. Set `EVAL_REPEATS=3` to measure run-to-run spread — on this "
                        "system it has been wide enough to change conclusions.")

    if failures:
        failure_rows = []
        for f in failures:
            j = f.get("judged")
            # Not truncated. A previous 160-character cut ended the only substantive
            # verdict mid-sentence, so the reader never learned what went wrong.
            why = str((j or {}).get("verdict") or f.get("error") or "behaviour mismatch")
            failure_rows.append([
                f["qid"], f["category"], f["behaviour"]["want"], f["behaviour"]["got"],
                n3(j["faithfulness"]) if j else "-",
                n3(j["correctness"]) if j else "-",
                ("yes" if j["citationsValid"] else "**no**") if j else "-",
                why.replace("|", "\\|"),
            ])
        failure_block = table(
            ["Question", "Category", "Expected", "Got", "Faith.", "Correct.", "Cites OK",
             "What went wrong"], failure_rows)
    else:
        failure_block = "_No failures in the final arm._"

    corpus_table = table(
        ["Document", "Form", "Filed", "Source"],
        [[d["docId"], d["form"], d["filingDate"], "[EDGAR](" + d["url"] + ")"]
         for d in r["corpus"]])

    chunk_table = table(
        ["Strategy", "Chunks", "Table chunks", "Mean chars", "Stdev", "Median", "p95", "Max"],
        [[k, v["chunks"], v["tableChunks"], v["meanChars"], v["stdevChars"],
          v["chars"]["median"], v["chars"]["p95"], v["chars"]["max"]] for k, v in cs.items()])

    sem_cat = {x["category"]: x["hitRate"] for x in by_category("semantic/dense")}
    category_table = table(
        ["Category", "n", "hit@6 (fixed/dense)", "hit@6 (semantic/dense)"],
        [[c["category"], c["n"], n3(c["hitRate"]), n3(sem_cat.get(c["category"], 0))]
         for c in by_category("fixed/dense")])

    chunking_table = table(ret_cols, [ret_row(fixed_dense), ret_row(sem_dense)])
    mode_table = table(ret_cols, [ret_row(s) for s in ret if "rerank" not in s["arm"]])
    rerank_table = table(ret_cols, [ret_row(s) for s in
                                    [find("fixed/hybrid"), fixed_rr,
                                     find("semantic/hybrid"), sem_rr] if s])

    gen_rows = []
    for s in gen:
        x = extra[s["arm"]]
        wrong_cell = f"{x['wrongCount']}/{x['judgedN']}"
        if x["wrongIds"]:
            wrong_cell += " (" + ", ".join(x["wrongIds"]) + ")"
        gen_rows.append([
            s["arm"], pct(s["answerRate"]),
            f"{n3(x['faithfulness'])} (n={x['judgedN']})", n3(x["correctness"]), wrong_cell,
            pct(s["citationValidity"]), pct(s["correctRefusal"]), pct(s["clarifyRate"]),
            pct(s["falseRefusal"]), s["latency"]["p50"],
        ])
    gen_table = table(
        ["Arm", "Answer rate", "Faithfulness (n judged)", "Correctness", "Wrong-figure",
         "Citations", "Correct refusal", "Clarified", "False refusal", "p50 ms"], gen_rows)

    baseline_miss = pct(1 - (gen[0]["answerRate"] if gen else 0))
    config_json = json.dumps(r["config"], indent=2)

    md = f"""# Financial Document Intelligence RAG — Evaluation Report

Generated {r['generatedAt']} · corpus of {len(r['corpus'])} SEC filings · {n_questions}-question golden set

---

## 1. What was measured

Three variables were isolated, one per stage, so each effect is attributable:

1. **Chunking strategy** — fixed-size vs semantic, holding retrieval constant (dense-only).
2. **Retrieval mode** — dense vs sparse (BM25) vs hybrid (RRF fusion), holding chunking at the winner.
3. **Reranking** — LLM cross-encoder on top of the best retrieval config.

Table handling is held **identical** across both chunking strategies (tables are atomic, split by row group with the header repeated when oversized). The only variable between the chunking arms is how *prose* is segmented — otherwise the comparison would silently be measuring table parsing instead.

**Corpus**

{corpus_table}

---

## 2. Chunking strategy comparison

### 2.1 What the two strategies produce

{chunk_table}

Fixed-size chunking produces **{cs.get('fixed', {}).get('chunks', '?')}** chunks with tight variance (stdev {cs.get('fixed', {}).get('stdevChars', '?')}); semantic chunking produces **{cs.get('semantic', {}).get('chunks', '?')}** (stdev {cs.get('semantic', {}).get('stdevChars', '?')}). The higher variance is the point: semantic chunks end where the subject changes, not at a character budget, so a short definition and a long risk-factor discussion are each kept whole.

### 2.2 Retrieval quality, dense-only (isolates chunking)

{chunking_table}

{chunk_verdict}

{rerank_verdict}

### 2.3 Per-category behaviour

{category_table}

---

## 3. Retrieval mode comparison

{mode_table}

Dense retrieval and BM25 fail differently, which is the whole argument for fusing them. Dense finds the passage that is *about* the right topic; BM25 finds the passage that literally contains the figure. In filings the figure is the answer, so lexical matching carries more weight than it would in a general-purpose corpus.

**The arm carried into Stage 2 was `{r['bestRetrievalArm']}`**, selected on nDCG among the non-reranked arms — so every end-to-end number in §5 is that configuration plus reranking, i.e. `{shipped}`.

{selection_note}

---

## 4. Reranking impact

{rerank_table}

{rerank_impact}

A cross-encoder reads query and passage *together*, so it can separate "NVIDIA FY2026 revenue" from "NVIDIA FY2025 revenue" — a distinction a bi-encoder cannot make, because it never sees the two texts at the same time. That is precisely the failure mode that matters in a corpus where every issuer files a structurally identical document every year.

---

## 5. End-to-end generation quality

{gen_table}

- **Answer rate is listed first on purpose.** Faithfulness is measured only on answers the pipeline chose to emit — `verify` refuses anything it judges ungrounded, and only non-refused rows reach the judge. It is therefore *conditional on answering*, and an arm can raise it by refusing more. Read the two columns together: the baseline arm scores the best faithfulness while failing {baseline_miss} of answerable questions.
- **Wrong-figure** counts delivered answers that were factually wrong or cited a passage that does not support them — the outcome this system exists to prevent, counted rather than averaged into a 0-1 score.
- **Correct refusal** is measured over just {n_refuse} questions and **Clarified** over {n_clarify}; one question moves them by 25 and 33 points respectively. Treat 100% as "no counter-example found", not as a rate.
- **Behaviour metrics read the graph's own control-flow flags** (`refused`, `clarified`), so they confirm which branch was taken, not that the text the user received was safe or useful.
{error_notes}

{spread_block}

**Latency.** p95 is omitted for generation: at n={n_questions} it is the second-largest observation, and single API retries against the 60 s timeout have produced values above 60,000 ms that describe the network, not the pipeline. p50 is reported; per-question timings are in `eval-results.json`. The framework's target was p95 < 6,000 ms and **this run does not demonstrate it is met**.

Note that a clarification counts as neither an answer nor a refusal: the system is waiting on the user, not declining.

---

## 6. Failure analysis

{failure_block}

Rows appear here when behaviour was wrong, **or** when a delivered answer was factually wrong or mis-cited. An earlier filter checked only behaviour and faithfulness, so an answer that was faithful to its passages while reporting the wrong figure could not appear — the exact failure this project exists to catch.

---

## 7. Configuration

```json
{config_json}
```

_Report generated by `python -m finrag.evals.report` from `out/eval-results.json`. Regenerate the underlying numbers with `python -m finrag.evals.run`._
"""

    P.out.mkdir(parents=True, exist_ok=True)
    (P.out / "REPORT.md").write_text(md, encoding="utf-8")
    print(f"-> out/REPORT.md  ({len(md) / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
