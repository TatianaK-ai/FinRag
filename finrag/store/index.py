"""
Backend dispatch, chunk lookup, and the index/chunk binding.

Nothing used to tie a vector set to the chunks it was built from. An interrupted
index run (BM25 is written before embedding starts) left the two out of step, and
retrieval hid the damage by dropping ids that no longer resolve. In the worst case
an id survives but its TEXT has changed, so a vector selected on the old text
hands the generator different text under the same id - a silently wrong citation.

The fingerprint below makes that state loud instead of invisible.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import EMBED_DIM, EMBED_MODEL, P, VECTOR_BACKEND
from .local import query_local, upsert_local
from .pinecone_store import query_pinecone, upsert_pinecone

BACKEND = VECTOR_BACKEND


def upsert_vectors(strategy: str, chunks: list, vectors: list[list[float]]) -> int:
    if BACKEND == "local":
        return upsert_local(strategy, chunks, vectors)
    return upsert_pinecone(strategy, chunks, vectors)


def query_vectors(strategy: str, vector: list[float], k: int, flt: dict | None = None) -> list[dict]:
    if BACKEND == "local":
        return query_local(strategy, vector, k, flt)
    return query_pinecone(strategy, vector, k, flt)


_chunk_cache: dict[str, dict] = {}


def chunk_map(strategy: str) -> dict:
    """Full chunk text by id - a vector store's metadata is truncated, disk is not."""
    if strategy in _chunk_cache:
        return _chunk_cache[strategy]
    path = P.index / f"chunks.{strategy}.json"
    if not path.exists():
        raise RuntimeError(f'Missing chunks for "{strategy}". Run `python -m finrag.chunking.run`.')
    arr = json.loads(path.read_text(encoding="utf-8"))
    m = {c["id"]: c for c in arr}
    _chunk_cache[strategy] = m
    return m


def _manifest_path() -> Path:
    return P.index / "index-manifest.json"


def fingerprint_chunks(chunks) -> str:
    """Content fingerprint: ids and text, order-sensitive."""
    h = hashlib.sha256()
    for c in chunks:
        cid = c["id"] if isinstance(c, dict) else c.id
        text = c["text"] if isinstance(c, dict) else c.text
        h.update(cid.encode()); h.update(b"\x00")
        h.update(text.encode()); h.update(b"\x01")
    return f"{len(chunks)}:{h.hexdigest()[:16]}"


def record_index_build(strategy: str, chunks, backend: str) -> dict:
    m = {}
    if _manifest_path().exists():
        m = json.loads(_manifest_path().read_text(encoding="utf-8"))
    m[strategy] = {
        "backend": backend,
        "embedModel": EMBED_MODEL,
        "embedDim": EMBED_DIM,
        "chunkCount": len(chunks),
        "fingerprint": fingerprint_chunks(chunks),
        "builtAt": datetime.now(timezone.utc).isoformat(),
    }
    P.index.mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m[strategy]


_checked: set[str] = set()


def assert_index_fresh(strategy: str) -> None:
    """
    Verify the index on disk was built from the chunks now on disk. Throws rather
    than warns: a mismatch means retrieved ids may resolve to text the vectors were
    never derived from, and every answer built on that is untrustworthy.
    """
    if strategy in _checked:
        return
    _checked.add(strategy)

    if not _manifest_path().exists():
        raise RuntimeError(
            f'No index manifest for "{strategy}". Run `python -m finrag.store.run` to rebuild.'
        )
    entry = json.loads(_manifest_path().read_text(encoding="utf-8")).get(strategy)
    if not entry:
        raise RuntimeError(f'No index manifest entry for "{strategy}". Rebuild the index.')

    chunks = list(chunk_map(strategy).values())
    now = fingerprint_chunks(chunks)
    if entry["fingerprint"] != now:
        raise RuntimeError(
            f'Index for "{strategy}" is stale: built from {entry["chunkCount"]} chunks '
            f'({entry["fingerprint"]}) but {len(chunks)} chunks ({now}) are on disk now. '
            "Retrieved ids would resolve to text the vectors were not derived from. Rebuild."
        )
    if BACKEND != "local" and entry["embedDim"] != EMBED_DIM:
        raise RuntimeError(
            f'Index for "{strategy}" was built at dim {entry["embedDim"]}, but EMBED_DIM is {EMBED_DIM}.'
        )
