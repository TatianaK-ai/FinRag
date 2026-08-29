"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

from ..config import CORPUS, RETRIEVAL
from ..lib.embed import embed_query
from ..store.bm25 import load_bm25, search_bm25
from ..store.index import assert_index_fresh, chunk_map, query_vectors

VALID_MODES = ("dense", "sparse", "hybrid")


def rrf(lists: list[list[dict]], k: int | None = None, weights: list[float] | None = None) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    Chosen over score normalisation because cosine similarity and BM25 scores are
    not on comparable scales and their distributions shift per query - RRF needs
    only the ranks, so it is robust without per-query calibration.

        score(d) = sum_over_lists( weight_i / (K + rank_i(d)) )
    """
    k = RETRIEVAL.rrf_k if k is None else k
    acc: dict[str, dict] = {}
    for li, lst in enumerate(lists):
        w = weights[li] if weights else 1.0
        for rank, hit in enumerate(lst):
            cur = acc.setdefault(hit["id"], {"id": hit["id"], "score": 0.0, "sources": {}})
            cur["score"] += w / (k + rank + 1)
            cur["sources"][hit.get("source", f"list{li}")] = rank + 1
    return sorted(acc.values(), key=lambda x: x["score"], reverse=True)


def named_tickers(question: str) -> list[str]:
    """
    Which issuers a question explicitly refers to. Deterministic on purpose: it
    drives a retrieval budget, and a model call would add latency to decide
    something a regex settles.
    """
    q = question.lower()

    def word(needle: str) -> bool:
        # Word-bounded: a bare `in` check matched "apple" inside "pineapple" and
        # turned an unrelated question into a per-entity comparison.
        return re.search(rf"(^|[^a-z0-9]){re.escape(needle)}([^a-z0-9]|$)", q) is not None

    named = [
        c.ticker for c in CORPUS
        if word(c.ticker.lower())
        or word(re.sub(r"\b(inc|corporation|corp)\b\.?", "", c.name, flags=re.I).strip().lower())
    ]

    # "which of the three companies grew fastest" names none but means all.
    # Guarded on ZERO named issuers, not fewer than two: with `< 2` this fired for
    # questions naming exactly one company, which then forced the other two
    # issuers' passages into context below the relevance floor.
    if not named and re.search(
        r"\b(all three|the three|each (company|issuer)|every company|which company)\b", q
    ):
        return [c.ticker for c in CORPUS]
    return named


def retrieve(question: str, *, strategy: str = "fixed", mode: str = "hybrid",
             dense_k: int | None = None, sparse_k: int | None = None,
             flt: dict | None = None) -> dict:
    # A typo'd mode used to match no branch, leave both lists empty, and surface as
    # "Nothing in the indexed filings matched" - a confident refusal caused by a
    # spelling mistake.
    if mode not in VALID_MODES:
        raise ValueError(f'Unknown retrieval mode "{mode}". Expected one of {VALID_MODES}.')

    dense_k = dense_k or RETRIEVAL.dense_k
    sparse_k = sparse_k or RETRIEVAL.sparse_k

    assert_index_fresh(strategy)
    cmap = chunk_map(strategy)
    timings: dict[str, int] = {}
    dense: list[dict] = []
    sparse: list[dict] = []

    if mode in ("dense", "hybrid"):
        t = time.time()
        vec = embed_query(question)
        timings["embed"] = int((time.time() - t) * 1000)
        t2 = time.time()
        dense = [{**h, "source": "dense"} for h in query_vectors(strategy, vec, dense_k, flt)]
        timings["dense"] = int((time.time() - t2) * 1000)

    if mode in ("sparse", "hybrid"):
        t = time.time()
        # The dense store filters server-side; BM25 has no metadata index, but
        # chunk ids are ticker-prefixed so the same filter is a cheap prefix test.
        want = sparse_k * 6 if (flt or {}).get("ticker") else sparse_k
        raw = search_bm25(load_bm25(strategy), question, want)
        if (flt or {}).get("ticker"):
            raw = [h for h in raw if h["id"].startswith(f"{flt['ticker']}_")]
        sparse = [{**h, "source": "sparse"} for h in raw[:sparse_k]]
        timings["sparse"] = int((time.time() - t) * 1000)

    lists = [dense] if mode == "dense" else [sparse] if mode == "sparse" else [dense, sparse]
    weights = [RETRIEVAL.dense_weight, RETRIEVAL.sparse_weight] if mode == "hybrid" else None
    fused = rrf(lists, weights=weights)

    resolved = []
    for rank, f in enumerate(fused):
        chunk = cmap.get(f["id"])
        resolved.append({
            "id": f["id"], "rank": rank, "fusedScore": f["score"], "sources": f["sources"],
            "text": (chunk or {}).get("text", ""), "metadata": (chunk or {}).get("metadata", {}),
        })
    hits = [h for h in resolved if h["text"]]
    # Orphans are ids the index returned that no longer exist in the chunk store.
    # Counted rather than silently swallowed, which is how a de-synchronised index
    # used to look exactly like a normal, slightly worse result set.
    orphaned = len(resolved) - len(hits)

    return {"hits": hits, "timings": timings,
            "counts": {"dense": len(dense), "sparse": len(sparse),
                       "fused": len(hits), "orphaned": orphaned}}


def retrieve_per_entity(question: str, *, strategy: str, mode: str,
                        tickers: list[str], per_entity: int) -> dict:
    """
    Retrieval with a per-issuer budget, for comparison questions.

    A single pass over "Microsoft or NVIDIA revenue" does not split evenly: the
    measured 30-candidate window came back 27 NVDA / 2 MSFT / 1 AAPL, and the
    Microsoft figure was absent entirely. Reranking cannot recover what retrieval
    never surfaced.
    """
    with ThreadPoolExecutor(max_workers=len(tickers)) as pool:
        results = list(pool.map(
            lambda tk: retrieve(question, strategy=strategy, mode=mode, flt={"ticker": tk}),
            tickers,
        ))

    counts = {"dense": 0, "sparse": 0, "fused": 0, "orphaned": 0}
    # Passes run concurrently, so the cost is the slowest one, not their sum.
    timings = {"embed": 0, "dense": 0, "sparse": 0}
    by_ticker: list[list[dict]] = []
    for i, r in enumerate(results):
        counts["dense"] += r["counts"]["dense"]
        counts["sparse"] += r["counts"]["sparse"]
        counts["orphaned"] += r["counts"]["orphaned"]
        for k in timings:
            timings[k] = max(timings[k], r["timings"].get(k, 0))
        by_ticker.append([{**h, "entitySlot": tickers[i]} for h in r["hits"][:per_entity]])

    # Interleave so the head of the list alternates issuers; a downstream top-K cut
    # then keeps both sides rather than the strongest issuer's whole run.
    out: list[dict] = []
    for i in range(per_entity):
        for lst in by_ticker:
            if i < len(lst):
                out.append(lst[i])

    counts["fused"] = len(out)
    return {"hits": [{**h, "rank": r} for r, h in enumerate(out)],
            "timings": timings, "counts": counts}
