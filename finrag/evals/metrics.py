"""Retrieval and behaviour metrics against the hand-authored golden set."""
from __future__ import annotations

import math
import re
import statistics


def is_relevant(hit: dict, q: dict) -> bool:
    """
    A retrieved chunk counts as relevant when it comes from an expected issuer AND
    contains one of the question's `evidence` markers - the exact figure an analyst
    would need. Deliberately stricter than "same topic".
    """
    ev = q.get("evidence") or []
    if not ev:
        return False
    tickers = (q.get("expect") or {}).get("tickers") or []
    if tickers and hit.get("metadata", {}).get("ticker") not in tickers:
        return False
    text = hit.get("text") or ""
    return any(re.search(e, text) for e in ev)


def count_relevant(chunks: list, q: dict) -> int:
    return sum(1 for c in chunks if is_relevant(c, q))


def score_ranking(hits: list[dict], q: dict, *, k: int = 6, total_relevant: int | None = None) -> dict:
    top = hits[:k]
    rel = [1 if is_relevant(h, q) else 0 for h in top]
    hit_count = sum(rel)
    first = next((i for i, r in enumerate(rel) if r), -1)

    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = min(total_relevant if total_relevant is not None else hit_count, k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal))

    # Evidence coverage, NOT classical recall. Chunks overlap, so several contain
    # the same figure and an analyst needs one of them; dividing by "every chunk
    # containing the figure" punished the system for missing redundant duplicates.
    markers = q.get("evidence") or []
    tickers = (q.get("expect") or {}).get("tickers") or []
    covered = sum(
        1 for e in markers
        if any(re.search(e, h.get("text") or "")
               and (not tickers or h.get("metadata", {}).get("ticker") in tickers)
               for h in top)
    )

    return {
        "hitRate": 1 if hit_count else 0,
        "precisionAtK": hit_count / k,
        "evidenceCoverage": covered / len(markers) if markers else 0,
        "mrr": 0 if first < 0 else 1 / (first + 1),
        "ndcg": dcg / idcg if idcg > 0 else 0,
        "rankOfFirstRelevant": None if first < 0 else first + 1,
    }


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"p50": 0, "p95": 0, "max": 0}
    s = sorted(xs)
    at = lambda p: s[min(len(s) - 1, int(p / 100 * len(s)))]
    return {"p50": at(50), "p95": at(95), "max": s[-1]}


def score_behaviour(result: dict, q: dict) -> dict:
    """
    Behaviour scoring. Note this reads the graph's own control-flow flags, so it
    confirms which branch was taken - not that the text the user received was safe.
    """
    want = (q.get("expect") or {}).get("behaviour", "answer")
    clarified = bool(result.get("clarified"))
    refused = bool(result.get("refused"))
    got = "clarify" if clarified else "refuse" if refused else "answer"

    if want == "answer":
        return {"want": want, "got": got, "correct": got == "answer"}
    if want == "refuse":
        return {"want": want, "got": got, "correct": got == "refuse"}
    # clarify: asking back is right; declining outright is acceptable
    return {"want": want, "got": got, "correct": got in ("clarify", "refuse")}
