"""Pinecone serverless store: one namespace per chunking strategy."""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

from pinecone import Pinecone, ServerlessSpec

from ..config import EMBED_DIM, PINECONE_API_KEY, PINECONE_INDEX, assert_pinecone

_pc: Pinecone | None = None


def _client() -> Pinecone:
    global _pc
    assert_pinecone()
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def ensure_index():
    """Create the index on first run; each strategy gets its own namespace."""
    pc = _client()
    existing = {i["name"]: i for i in pc.list_indexes()}
    if PINECONE_INDEX in existing:
        dim = existing[PINECONE_INDEX]["dimension"]
        if dim != EMBED_DIM:
            raise RuntimeError(
                f'Pinecone index "{PINECONE_INDEX}" has dimension {dim}, but EMBED_MODEL '
                f"produces {EMBED_DIM}. Set EMBED_DIM={dim} or use a different index name."
            )
        return pc.Index(PINECONE_INDEX)

    print(f'  creating Pinecone index "{PINECONE_INDEX}" (dim {EMBED_DIM}, cosine, serverless)')
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    import time
    while not pc.describe_index(PINECONE_INDEX).status.get("ready"):
        time.sleep(2)
    return pc.Index(PINECONE_INDEX)


def upsert_pinecone(strategy: str, chunks: list, vectors: list[list[float]]) -> int:
    index = ensure_index()
    records = [
        {
            "id": c.id,
            "values": vectors[i],
            # Pinecone metadata has a 40 kB per-record ceiling; text is truncated
            # for storage but the full chunk is always re-read from disk before it
            # reaches the generator.
            "metadata": {**{k: v for k, v in c.metadata.items() if v is not None},
                         "text": c.text[:8000]},
        }
        for i, c in enumerate(chunks)
    ]

    batches = [records[i:i + 100] for i in range(0, len(records), 100)]
    done = 0

    def send(b):
        nonlocal done
        index.upsert(vectors=b, namespace=strategy)
        done += 1
        sys.stdout.write(f"\r  upsert {done}/{len(batches)}   ")
        sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(send, batches))
    sys.stdout.write("\n")
    return len(records)


def query_pinecone(strategy: str, vector: list[float], k: int = 20, flt: dict | None = None) -> list[dict]:
    index = ensure_index()
    res = index.query(vector=vector, top_k=k, include_metadata=True,
                      namespace=strategy, filter=flt or None)
    return [
        {"id": m["id"], "score": m.get("score"), "rank": r,
         "text": (m.get("metadata") or {}).get("text", ""),
         "metadata": m.get("metadata") or {}}
        for r, m in enumerate(res.get("matches", []))
    ]
