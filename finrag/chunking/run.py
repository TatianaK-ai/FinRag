"""chunk: build both chunk sets and report their shape."""
from __future__ import annotations

import json
import statistics
import sys
from types import SimpleNamespace

from ..config import P, STRATEGIES
from .fixed import chunk_fixed
from .semantic import chunk_semantic


def _load_docs() -> list[dict]:
    files = sorted(f for f in P.processed.glob("*.json") if not f.name.startswith("_"))
    if not files:
        raise SystemExit("No processed docs. Run `python -m finrag.ingest.run` first.")
    docs = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        d["blocks"] = [SimpleNamespace(**b) for b in d["blocks"]]
        docs.append(d)
    return docs


def main() -> None:
    only = [sys.argv[1]] if len(sys.argv) > 1 else list(STRATEGIES)
    # Without this, `chunking.run semantik` fell through and wrote chunks.semantik.json.
    for s in only:
        if s not in STRATEGIES:
            raise SystemExit(f'Unknown strategy "{s}". Expected one of: {", ".join(STRATEGIES)}')

    docs = _load_docs()
    P.index.mkdir(parents=True, exist_ok=True)
    stats_path = P.index / "chunk-stats.json"
    all_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    for strategy in only:
        print(f"\n=== Chunking: {strategy} ===")
        chunks = []
        for doc in docs:
            c = chunk_fixed(doc) if strategy == "fixed" else chunk_semantic(doc)
            chunks.extend(c)
            print(f"  {doc['docId']:<22} {len(c):>4} chunks")

        sizes = sorted(c.metadata["chars"] for c in chunks)
        at = lambda p: sizes[int(p / 100 * (len(sizes) - 1))]
        all_stats[strategy] = {
            "chunks": len(chunks),
            "tableChunks": sum(1 for c in chunks if c.metadata["isTable"]),
            "meanChars": round(statistics.mean(sizes)),
            "stdevChars": round(statistics.pstdev(sizes)),
            "chars": {"min": sizes[0], "p25": at(25), "median": at(50),
                      "p75": at(75), "p95": at(95), "max": sizes[-1]},
        }
        (P.index / f"chunks.{strategy}.json").write_text(
            json.dumps([{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]),
            encoding="utf-8")
        s = all_stats[strategy]
        print(f"  total {s['chunks']} chunks ({s['tableChunks']} table) "
              f"median {s['chars']['median']} chars, p95 {s['chars']['p95']}")

    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print("\n=== Chunk statistics ===")
    print(f"  {'strategy':<10} {'chunks':>7} {'tables':>7} {'mean':>6} {'stdev':>6} {'median':>7} {'p95':>6}")
    for k, v in all_stats.items():
        print(f"  {k:<10} {v['chunks']:>7} {v['tableChunks']:>7} {v['meanChars']:>6} "
              f"{v['stdevChars']:>6} {v['chars']['median']:>7} {v['chars']['p95']:>6}")


if __name__ == "__main__":
    main()
