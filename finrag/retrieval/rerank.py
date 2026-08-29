"""
LLM cross-encoder reranker.

A bi-encoder scores query and document independently, so it can only measure
topical similarity. A cross-encoder reads the pair together and can tell
"NVIDIA's FY2026 revenue" from "NVIDIA's FY2025 revenue" - exactly the distinction
that matters in filings, and exactly the one dense retrieval gets wrong.

The response carries scores and NOTHING else. An earlier version also asked for a
one-line rationale per passage; structured output is emitted token by token, so
twenty rationales dominated the call (~8.9s median against a 6s budget for the
whole pipeline). Dropping the field was the single biggest latency win.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config import (OPENAI_API_KEY, OPENAI_BASE_URL, RERANK_MODEL, RETRIEVAL,
                      assert_openai)

# Enough to judge relevance; the deciding figures sit near the top of a chunk.
PASSAGE_CHARS = 700

# 6 rather than 10: with a 30-candidate window that is 5 concurrent calls instead
# of 3, and wall-clock is the slowest single call.
BATCH = 6


class _Score(BaseModel):
    id: int = Field(description="the [n] index of the passage")
    relevance: int = Field(ge=0, le=10, description="0 = irrelevant, 10 = directly answers")


class _Scores(BaseModel):
    scores: list[_Score]


PROMPT = """You are a relevance judge for a financial-filings search engine.
Score how well each numbered passage answers the QUESTION.

Scoring guide:
  9-10  contains the specific fact/figure asked for
  6-8   same topic and entity and period, but the exact figure is missing
  3-5   right topic, wrong entity or wrong fiscal period
  0-2   unrelated

Be strict about entity (which company) and period (which fiscal year/quarter).
A passage about the right metric for the WRONG company or year scores at most 4.
Return a score for every passage."""

_model = None


def _get_model():
    global _model
    assert_openai()
    if _model is None:
        kwargs = {"api_key": OPENAI_API_KEY, "model": RERANK_MODEL, "temperature": 0,
                  "max_retries": 2, "timeout": 60}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _model = ChatOpenAI(**kwargs).with_structured_output(_Scores)
    return _model


def rerank(question: str, hits: list[dict], *, top_n: int | None = None,
           groups: list[str] | None = None, apply_floor: bool = True) -> dict:
    """
    apply_floor: drop passages below `min_rerank_score`. That floor is an ANSWER
    quality gate - below it the generator should not see the passage at all. It is
    wrong when measuring RETRIEVAL, because a question where everything scores 3
    then returns an empty list and reads as nDCG 0 even with the right chunk first.
    """
    top_n = RETRIEVAL.rerank_to if top_n is None else top_n
    groups = groups or []
    if not hits:
        return {"hits": [], "ms": 0}
    t = time.time()

    def fmt(h: dict, gid: int) -> str:
        m = h["metadata"]
        return (f"[{gid}] ({m.get('ticker')} {m.get('form')} {m.get('filingDate')} | "
                f"{m.get('section')})\n{h['text'][:PASSAGE_CHARS]}")

    # Batch indices stay GLOBAL so returned ids map straight back to `hits`.
    batches = [[fmt(h, i + j) for j, h in enumerate(hits[i:i + BATCH])]
               for i in range(0, len(hits), BATCH)]

    def call(passages: list[str]):
        return _get_model().invoke([
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"QUESTION: {question}\n\nPASSAGES:\n" + "\n\n".join(passages)},
        ])

    results, failed = [], 0
    with ThreadPoolExecutor(max_workers=len(batches) or 1) as pool:
        futures = [pool.submit(call, b) for b in batches]
        for f in futures:
            try:
                results.append(f.result())
            except Exception:
                failed += 1

    scores = [s for r in results for s in r.scores]
    if not scores:
        # Reranking is an enhancement, never a hard dependency.
        return {"hits": hits[:top_n], "ms": int((time.time() - t) * 1000),
                "error": "no scores returned", "degraded": True}

    # Only ids the model actually returned count as scored; a model that
    # re-indexes per batch would otherwise have its collisions read as real scores.
    by_idx = {s.id: s for s in scores if 0 <= s.id < len(hits)}

    annotated = [
        {**h, "scored": i in by_idx,
         "rerankScore": by_idx[i].relevance if i in by_idx else None,
         "preRerankRank": h.get("rank")}
        for i, h in enumerate(hits)
    ]
    scored_hits = sorted([h for h in annotated if h["scored"]],
                         key=lambda h: h["rerankScore"], reverse=True)
    # Passages from a failed batch keep their fused position - they are NOT
    # dropped. The previous version scored them 0 and the floor deleted them, so a
    # single transient failure silently removed a third of the candidate window.
    unscored = [h for h in annotated if not h["scored"]]
    ranked = scored_hits + unscored

    def clears_floor(h: dict) -> bool:
        return (not apply_floor) or (not h["scored"]) or h["rerankScore"] >= RETRIEVAL.min_rerank_score

    if groups:
        # A global score floor cannot serve a comparison question: asked to compare
        # R&D intensity, the reranker scored every NVIDIA passage below the floor
        # and the kept set came back Apple-only. Each group keeps its best passage
        # regardless of the floor - half a comparison is not an answer.
        per = max(1, -(-top_n // len(groups)))
        out: list[dict] = []
        for g in groups:
            in_group = [h for h in ranked if h["metadata"].get("ticker") == g]
            kept = [h for i, h in enumerate(in_group) if i == 0 or clears_floor(h)]
            out.extend(kept[:per])
        selected = sorted(out, key=lambda h: (h["rerankScore"] if h["rerankScore"] is not None else -1),
                          reverse=True)
    else:
        selected = [h for h in ranked if clears_floor(h)]

    final = [{**h, "rank": r} for r, h in enumerate(selected[:top_n])]

    res = {"hits": final, "ms": int((time.time() - t) * 1000), "scoredCount": len(by_idx)}
    if failed or len(by_idx) < len(hits):
        res.update({
            "partial": failed,
            "unscored": len(hits) - len(by_idx),
            "error": f"{failed}/{len(batches)} rerank batches failed; "
                     f"{len(hits) - len(by_idx)} passage(s) unscored and kept at fused rank",
        })
    return res
