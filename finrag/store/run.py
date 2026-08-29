"""index: build BM25 + embed and upsert vectors for each strategy."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from ..chunking.common import embed_text
from ..config import P, STRATEGIES
from ..lib.embed import embed_all
from .bm25 import build_bm25, save_bm25
from .index import BACKEND, record_index_build, upsert_vectors


def main() -> None:
    only = [sys.argv[1]] if len(sys.argv) > 1 else list(STRATEGIES)
    for s in only:
        if s not in STRATEGIES:
            raise SystemExit(f'Unknown strategy "{s}". Expected one of: {", ".join(STRATEGIES)}')

    print(f"vector backend: {BACKEND}")
    for strategy in only:
        path = P.index / f"chunks.{strategy}.json"
        if not path.exists():
            print(f"  skip {strategy} (no chunks - run `python -m finrag.chunking.run`)")
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        chunks = [SimpleNamespace(id=c["id"], text=c["text"], metadata=c["metadata"]) for c in raw]

        print(f"\n=== Indexing: {strategy} ({len(chunks)} chunks) ===")

        # BM25 must index the SAME representation the embeddings see. A chunk
        # reading `| Revenue | $331,839 |` contains no company name, so a query
        # naming "Microsoft" cannot match it lexically at all.
        save_bm25(strategy, build_bm25([{"id": c.id, "text": embed_text(c)} for c in chunks]))
        print("  BM25 index built (provenance-prefixed)")

        vectors = embed_all([embed_text(c) for c in chunks], label=f"embed:{strategy}")
        n = upsert_vectors(strategy, chunks, vectors)
        # Recorded only after the upsert succeeds, so an interrupted run leaves a
        # manifest that does NOT match and the next retrieval fails loudly.
        rec = record_index_build(strategy, chunks, BACKEND)
        print(f"  {n} vectors upserted to {BACKEND} (fingerprint {rec['fingerprint']})")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
