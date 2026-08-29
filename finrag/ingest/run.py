"""ingest: fetch (cached) -> clean -> structured blocks on disk."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import P
from .clean import clean_filing
from .edgar import fetch_corpus


def main() -> None:
    print("=== 1/3  Fetching filings from SEC EDGAR (cached on disk) ===")
    manifest = fetch_corpus()
    if not manifest:
        raise SystemExit("No filings available - check SEC_USER_AGENT and network.")

    print("\n=== 2/3  Cleaning + structuring ===")
    P.processed.mkdir(parents=True, exist_ok=True)
    docs = []
    for f in manifest:
        html = Path(f["rawPath"]).read_text(encoding="utf-8")
        blocks, stats = clean_filing(html)
        doc = {**{k: v for k, v in f.items() if k != "rawPath"},
               "blocks": [b.__dict__ for b in blocks], "stats": stats}
        (P.processed / f"{f['docId']}.json").write_text(json.dumps(doc), encoding="utf-8")
        docs.append(doc)
        print(f"  {f['docId']:<22} {stats['blocks']:>4} blocks "
              f"({stats['tableBlocks']:>3} tables) {stats['textChars'] // 1000}k chars, "
              f"{stats['sections']} sections, {stats['xbrlBytesRemoved'] // 1024}kB XBRL stripped")

    print("\n=== 3/3  Corpus summary ===")
    total = {
        "blocks": sum(d["stats"]["blocks"] for d in docs),
        "tables": sum(d["stats"]["tableBlocks"] for d in docs),
        "chars": sum(d["stats"]["textChars"] for d in docs),
        "xbrl": sum(d["stats"]["xbrlBytesRemoved"] for d in docs),
    }
    (P.processed / "_manifest.json").write_text(json.dumps({
        "documents": [{k: v for k, v in d.items() if k != "blocks"} for d in docs],
        "total": total,
    }, indent=2), encoding="utf-8")
    print(f"  {len(docs)} documents | {total['blocks']} blocks | {total['tables']} tables | "
          f"{total['chars'] // 1000}k chars")
    print(f"  Cleaning removed {total['xbrl'] // 1024}kB of inline XBRL.")


if __name__ == "__main__":
    main()
