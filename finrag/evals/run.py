"""
Two-stage evaluation, so each variable is isolated and the bill stays small.

Stage 1 - retrieval sweep: which chunking strategy retrieves better, does hybrid
  beat dense, does reranking improve the ordering.
Stage 2 - end-to-end generation on three arms: does better retrieval actually
  produce more faithful answers and better refusals.

EVAL_REPEATS repeats stage 2. The gates are model calls and do not settle on one
answer: measured over three runs, correct refusal ranged 75%-100% and two
questions failed in two runs out of three - a spread wider than most of the
differences this report is used to argue about.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

from ..config import P, RETRIEVAL
from ..graph.rag import ask
from ..retrieval.hybrid import named_tickers, retrieve, retrieve_per_entity
from ..retrieval.rerank import rerank
from ..store.index import chunk_map
from .judge import judge_answer
from .metrics import (count_relevant, mean, percentiles, score_behaviour, score_ranking)

K = RETRIEVAL.top_k
REPEATS = int(os.getenv("EVAL_REPEATS", "1"))

RETRIEVAL_ARMS = [
    {"id": "fixed/dense", "strategy": "fixed", "mode": "dense"},
    {"id": "semantic/dense", "strategy": "semantic", "mode": "dense"},
    {"id": "fixed/sparse", "strategy": "fixed", "mode": "sparse"},
    {"id": "semantic/sparse", "strategy": "semantic", "mode": "sparse"},
    {"id": "fixed/hybrid", "strategy": "fixed", "mode": "hybrid"},
    {"id": "semantic/hybrid", "strategy": "semantic", "mode": "hybrid"},
]


def _bar(done: int, total: int, label: str) -> None:
    w = 24
    f = round(done / total * w)
    sys.stdout.write("\r  [" + "#" * f + "." * (w - f) + f"] {done}/{total} {label[:28]}   ")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def _fetch(question: str, strategy: str, mode: str) -> list:
    """
    Comparison questions take the per-entity path in production; measuring them
    under a single pass would describe a configuration that never runs.
    """
    tickers = named_tickers(question)
    if len(tickers) >= 2:
        per = max(RETRIEVAL.min_per_entity, -(-RETRIEVAL.rerank_window // len(tickers)))
        return retrieve_per_entity(question, strategy=strategy, mode=mode,
                                   tickers=tickers, per_entity=per)["hits"]
    return retrieve(question, strategy=strategy, mode=mode)["hits"]


def _is_wrong(row: dict) -> bool:
    j = row.get("judged")
    return bool(j) and (j["correctness"] < 1 or not j["citationsValid"])


def main() -> None:
    questions = json.loads((P.eval / "questions.json").read_text(encoding="utf-8"))["questions"]
    answerable = [q for q in questions if q.get("evidence")]

    print(f"=== Stage 1: retrieval sweep ({len(RETRIEVAL_ARMS) + 2} arms x {len(answerable)} questions) ===")
    relevant_totals = {
        s: {q["id"]: count_relevant(list(chunk_map(s).values()), q) for q in answerable}
        for s in ("fixed", "semantic")
    }

    retrieval_runs = []
    for arm in RETRIEVAL_ARMS:
        rows = []
        for n, q in enumerate(answerable, 1):
            t = time.time()
            hits = _fetch(q["question"], arm["strategy"], arm["mode"])
            rows.append({
                "qid": q["id"], "category": q["category"], "ms": int((time.time() - t) * 1000),
                **score_ranking(hits, q, k=K,
                                total_relevant=relevant_totals[arm["strategy"]][q["id"]]),
            })
            _bar(n, len(answerable), arm["id"])
        retrieval_runs.append({**arm, "rows": rows})

    for base in ("fixed", "semantic"):
        rows = []
        for n, q in enumerate(answerable, 1):
            t = time.time()
            tickers = named_tickers(q["question"])
            hits = _fetch(q["question"], base, "hybrid")
            # The floor is switched OFF: it gates ANSWERING, not ordering. With it
            # on, a question where every passage scores 3 returned an empty list
            # and read as nDCG 0 even with the right chunk ranked first.
            reranked = rerank(q["question"], hits[:RETRIEVAL.rerank_window],
                              top_n=RETRIEVAL.rerank_window,
                              groups=tickers if len(tickers) >= 2 else [],
                              apply_floor=False)["hits"]
            rows.append({
                "qid": q["id"], "category": q["category"], "ms": int((time.time() - t) * 1000),
                **score_ranking(reranked, q, k=K, total_relevant=relevant_totals[base][q["id"]]),
            })
            _bar(n, len(answerable), base + "/hybrid+rerank")
        retrieval_runs.append({"id": base + "/hybrid+rerank", "strategy": base,
                               "mode": "hybrid", "rows": rows})

    def summarise(run):
        r = run["rows"]
        return {
            "arm": run["id"],
            "hitRate": mean([x["hitRate"] for x in r]),
            "evidenceCoverage": mean([x["evidenceCoverage"] for x in r]),
            "precisionAtK": mean([x["precisionAtK"] for x in r]),
            "mrr": mean([x["mrr"] for x in r]),
            "ndcg": mean([x["ndcg"] for x in r]),
            "latencyP50": percentiles([x["ms"] for x in r])["p50"],
            "latencyP95": percentiles([x["ms"] for x in r])["p95"],
        }

    retrieval_summary = [summarise(r) for r in retrieval_runs]
    print("\n=== Retrieval results ===")
    header = "  {:<26}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}".format(
        "arm", "hit@6", "evid@6", "prec@6", "MRR", "nDCG@6", "p50ms")
    print(header)
    for s in retrieval_summary:
        print("  {:<26}{:>8.3f}{:>8.3f}{:>8.3f}{:>8.3f}{:>8.3f}{:>8}".format(
            s["arm"], s["hitRate"], s["evidenceCoverage"], s["precisionAtK"],
            s["mrr"], s["ndcg"], s["latencyP50"]))

    best = max((s for s in retrieval_summary if "rerank" not in s["arm"]), key=lambda s: s["ndcg"])
    print(f"\n  best retrieval arm without reranking: {best['arm']} (nDCG {best['ndcg']:.3f})")

    best_strategy, best_mode = best["arm"].split("/")
    gen_arms = [
        {"id": "baseline: fixed/dense, no rerank", "strategy": "fixed", "mode": "dense", "rerank": False},
        {"id": f"best retrieval: {best['arm']}, no rerank", "strategy": best_strategy,
         "mode": best_mode, "rerank": False},
        {"id": "best retrieval + rerank", "strategy": best_strategy, "mode": best_mode, "rerank": True},
    ]

    print(f"\n=== Stage 2: end-to-end generation ({len(gen_arms)} arms x {len(questions)} q x {REPEATS} run(s)) ===")
    gen_runs = []
    for arm in gen_arms:
        rows = []
        n = 0
        for rep in range(REPEATS):
            for q in questions:
                error = None
                judged = None
                try:
                    r = ask(q["question"], {"strategy": arm["strategy"], "mode": arm["mode"],
                                            "rerank": arm["rerank"]})
                except Exception as e:
                    error = str(e)
                    r = {"refused": True, "clarified": False, "answer": "",
                         "contexts": [], "totalMs": 0, "trace": []}

                behaviour = score_behaviour(r, q)
                # A refusal is scored by `behaviour`, not by the judge.
                if not r.get("refused") and not r.get("clarified") and not error:
                    try:
                        judged = judge_answer(question=q["question"], answer=r.get("answer", ""),
                                              gold=q.get("gold", ""),
                                              contexts=r.get("contexts") or []).model_dump()
                    except Exception as e:
                        error = f"judge: {e}"

                rows.append({
                    "qid": q["id"], "rep": rep, "category": q["category"], "error": error,
                    "refused": bool(r.get("refused")), "clarified": bool(r.get("clarified")),
                    "behaviour": behaviour, "ms": r.get("totalMs", 0),
                    "timings": r.get("timings", {}), "confidence": r.get("confidence", 0),
                    "citations": len(r.get("citations") or []), "judged": judged,
                    "answer": r.get("answer", ""),
                    "trace": [f"{t['node']}: {t['detail']}" for t in (r.get("trace") or [])],
                })
                n += 1
                _bar(n, len(questions) * REPEATS, arm["id"])
        gen_runs.append({**arm, "repeats": REPEATS, "rows": rows})

    def gen_summary(run):
        rows = run["rows"]
        ans = [r for r in rows if not r["refused"] and not r["clarified"]]
        judged = [r for r in ans if r["judged"]]
        must_answer = [r for r in rows if r["behaviour"]["want"] == "answer"]
        must_refuse = [r for r in rows if r["behaviour"]["want"] == "refuse"]
        must_clarify = [r for r in rows if r["behaviour"]["want"] == "clarify"]
        wrong = [r for r in judged if _is_wrong(r)]

        spread = None
        if run["repeats"] > 1:
            per = []
            for rep in range(run["repeats"]):
                sl = [r for r in rows if r["rep"] == rep]
                ma = [r for r in sl if r["behaviour"]["want"] == "answer"]
                mr = [r for r in sl if r["behaviour"]["want"] == "refuse"]
                jd = [r for r in sl if r["judged"]]
                per.append({
                    "answerRate": len([r for r in ma if not r["refused"] and not r["clarified"]]) / len(ma) if ma else 0,
                    "correctRefusal": len([r for r in mr if r["behaviour"]["correct"]]) / len(mr) if mr else 0,
                    "faithfulness": mean([r["judged"]["faithfulness"] for r in jd]),
                    "wrong": len([r for r in jd if _is_wrong(r)]),
                })

            def rng(k):
                return {"min": min(x[k] for x in per), "max": max(x[k] for x in per)}

            # Questions that pass in some repetitions and fail in others - the ones
            # a single sample would misclassify as a defect or a clean pass.
            flaky = []
            for qid in sorted({r["qid"] for r in rows}):
                outcomes = {r["behaviour"]["correct"] for r in rows if r["qid"] == qid}
                if len(outcomes) > 1:
                    flaky.append(qid)

            spread = {"repeats": run["repeats"], "answerRate": rng("answerRate"),
                      "correctRefusal": rng("correctRefusal"), "faithfulness": rng("faithfulness"),
                      "wrong": rng("wrong"), "flaky": flaky}

        return {
            "arm": run["id"],
            "judgedN": len(judged),
            "faithfulness": mean([r["judged"]["faithfulness"] for r in judged]),
            "correctness": mean([r["judged"]["correctness"] for r in judged]),
            "wrongCount": len(wrong),
            "wrongIds": [r["qid"] for r in wrong],
            "citationValidity": (len([r for r in judged if r["judged"]["citationsValid"]]) / len(judged)) if judged else 0,
            "answerRate": (len([r for r in must_answer if not r["refused"] and not r["clarified"]]) / len(must_answer)) if must_answer else 0,
            "correctRefusal": (len([r for r in must_refuse if r["behaviour"]["correct"]]) / len(must_refuse)) if must_refuse else 0,
            "clarifyRate": (len([r for r in must_clarify if r["clarified"]]) / len(must_clarify)) if must_clarify else 0,
            "falseRefusal": (len([r for r in must_answer if r["refused"]]) / len(must_answer)) if must_answer else 0,
            "latency": percentiles([r["ms"] for r in rows]),
            "errors": len([r for r in rows if r["error"]]),
            "erroredIds": [r["qid"] for r in rows if r["error"]],
            "spread": spread,
        }

    gen_sum = [gen_summary(r) for r in gen_runs]
    print("\n=== Generation results ===")
    print("  {:<42}{:>9}{:>8}{:>9}{:>10}{:>11}{:>8}".format(
        "arm", "answer%", "faith", "wrong", "refusal%", "falseRef%", "p50ms"))
    for s in gen_sum:
        print("  {:<42}{:>8.1f}%{:>8.3f}{:>6}/{:<3}{:>8.0f}%{:>10.1f}%{:>8}".format(
            s["arm"], s["answerRate"] * 100, s["faithfulness"], s["wrongCount"], s["judgedN"],
            s["correctRefusal"] * 100, s["falseRefusal"] * 100, s["latency"]["p50"]))

    manifest = json.loads((P.processed / "_manifest.json").read_text(encoding="utf-8"))
    P.out.mkdir(parents=True, exist_ok=True)
    (P.out / "eval-results.json").write_text(json.dumps({
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "config": {"topK": K, **RETRIEVAL.__dict__},
        "corpus": [{"docId": d["docId"], "form": d["form"], "filingDate": d["filingDate"],
                    "url": d["sourceUrl"]} for d in manifest["documents"]],
        "chunkStats": json.loads((P.index / "chunk-stats.json").read_text(encoding="utf-8")),
        "bestRetrievalArm": best["arm"],
        "retrieval": {"summary": retrieval_summary, "runs": retrieval_runs},
        "generation": {"summary": gen_sum, "runs": gen_runs},
    }, indent=2), encoding="utf-8")
    print("\n-> out/eval-results.json")
    print("Now run `python -m finrag.evals.report` to render out/REPORT.md")


if __name__ == "__main__":
    main()
