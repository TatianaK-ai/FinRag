"""
Brute-force local vector store: vectors in a flat float32 binary, metadata in JSON.

Exists so the whole pipeline runs (and the evals reproduce) without a Pinecone
account, and so retrieval logic can be tested offline. At corpus scale an
exhaustive scan takes a few milliseconds.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from ..config import P


def _vec_file(s: str) -> Path:
    return P.index / f"vectors.{s}.f32"


def _meta_file(s: str) -> Path:
    return P.index / f"vectors.{s}.json"


def _l2(v) -> float:
    return math.sqrt(sum(x * x for x in v))


def upsert_local(strategy: str, chunks: list, vectors: list[list[float]]) -> int:
    P.index.mkdir(parents=True, exist_ok=True)
    dim = len(vectors[0])
    buf = bytearray()
    for v in vectors:
        norm = _l2(v) or 1.0            # store normalised -> dot == cosine
        buf.extend(struct.pack(f"<{dim}f", *[x / norm for x in v]))

    # Temp file + rename, so a crash cannot leave a .f32 disagreeing with its .json.
    tmp = _vec_file(strategy).with_suffix(".f32.tmp")
    tmp.write_bytes(bytes(buf))
    tmp.replace(_vec_file(strategy))

    _meta_file(strategy).write_text(json.dumps({
        "dim": dim,
        "records": [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks],
    }), encoding="utf-8")
    return len(chunks)


_cache: dict[str, dict] = {}


def _load(strategy: str) -> dict:
    if strategy in _cache:
        return _cache[strategy]
    mf = _meta_file(strategy)
    if not mf.exists():
        raise RuntimeError(f'Missing local vectors for "{strategy}". Run `python -m finrag.store.run`.')
    meta = json.loads(mf.read_text(encoding="utf-8"))
    raw = _vec_file(strategy).read_bytes()
    expected = len(meta["records"]) * meta["dim"] * 4
    if len(raw) != expected:
        raise RuntimeError(
            f'Local vector file for "{strategy}" is {len(raw)} bytes but its metadata '
            f'describes {len(meta["records"])} x {meta["dim"]} floats ({expected} bytes). Rebuild the index.'
        )
    meta["v"] = struct.unpack(f"<{len(raw) // 4}f", raw)
    _cache[strategy] = meta
    return meta


def query_local(strategy: str, vector: list[float], k: int = 20, flt: dict | None = None) -> list[dict]:
    meta = _load(strategy)
    dim, records, v = meta["dim"], meta["records"], meta["v"]
    norm = _l2(vector) or 1.0
    q = [x / norm for x in vector]

    scored: list[tuple[int, float]] = []
    for i, rec in enumerate(records):
        if flt and not all(rec["metadata"].get(kk) == vv for kk, vv in flt.items()):
            continue
        off = i * dim
        scored.append((i, sum(v[off + d] * q[d] for d in range(dim))))

    scored.sort(key=lambda t: t[1], reverse=True)
    return [
        {"id": records[i]["id"], "score": s, "rank": r,
         "text": records[i]["text"], "metadata": records[i]["metadata"]}
        for r, (i, s) in enumerate(scored[:k])
    ]
