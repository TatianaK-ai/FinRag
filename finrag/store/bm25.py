"""
Sparse lexical index (Okapi BM25), written from scratch rather than imported so
the tokenizer can be tuned for filings.

This is the half of hybrid retrieval that dense vectors are bad at: exact matches
on "Item 1A", "10-Q", segment names, and reported figures. Embeddings happily
rank "revenue grew" next to "revenue declined"; BM25 does not confuse $416,161
with $391,035.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from ..config import P

K1 = 1.5
B = 0.75

STOP = set(
    "a an and are as at be but by for from has have had he in is it its of on or that the to was "
    "were will with we our us this these those which their they been also may can could would "
    "should such other than then there".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*%?")
_THOUSANDS = re.compile(r"(\d),(\d)")


def tokenize(text: str) -> list[str]:
    # 416,161 -> 416161 so a reported figure stays one token.
    lowered = _THOUSANDS.sub(r"\1\2", text.lower())
    while _THOUSANDS.search(lowered):
        lowered = _THOUSANDS.sub(r"\1\2", lowered)
    return [t for t in _TOKEN.findall(lowered) if len(t) > 1 and t not in STOP]


def build_bm25(docs: list[dict]) -> dict:
    """docs: [{'id': str, 'text': str}]"""
    df: dict[str, int] = defaultdict(int)
    postings: dict[str, list[list[int]]] = defaultdict(list)
    lengths: list[int] = []

    for i, d in enumerate(docs):
        toks = tokenize(d["text"])
        lengths.append(len(toks))
        tf: dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, n in tf.items():
            df[t] += 1
            postings[t].append([i, n])

    return {
        "ids": [d["id"] for d in docs],
        "lengths": lengths,
        "avgdl": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "N": len(docs),
        "df": dict(df),
        "postings": {k: v for k, v in postings.items()},
    }


def search_bm25(index: dict, query: str, k: int = 20) -> list[dict]:
    N, avgdl = index["N"], index["avgdl"]
    df, postings, lengths, ids = index["df"], index["postings"], index["lengths"], index["ids"]
    scores: dict[int, float] = defaultdict(float)

    for term in set(tokenize(query)):
        post = postings.get(term)
        if not post:
            continue
        n = df[term]
        idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
        for i, tf in post:
            norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * (lengths[i] / avgdl)))
            scores[i] += idf * norm

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"id": ids[i], "score": s, "rank": r} for r, (i, s) in enumerate(ranked)]


def _file(strategy: str) -> Path:
    return P.index / f"bm25.{strategy}.json"


def save_bm25(strategy: str, index: dict) -> Path:
    P.index.mkdir(parents=True, exist_ok=True)
    path = _file(strategy)
    path.write_text(json.dumps(index), encoding="utf-8")
    return path


# The serialised index is multi-megabyte; re-parsing it per query dominated eval
# wall-clock before this cache existed.
_cache: dict[str, dict] = {}


def load_bm25(strategy: str) -> dict:
    if strategy in _cache:
        return _cache[strategy]
    path = _file(strategy)
    if not path.exists():
        raise RuntimeError(f'Missing BM25 index for "{strategy}". Run `python -m finrag.store.run`.')
    idx = json.loads(path.read_text(encoding="utf-8"))
    _cache[strategy] = idx
    return idx
