"""One-shot CLI query: python -m finrag.cli "question" [--trace]"""
from __future__ import annotations

import argparse

from .config import STRATEGIES
from .graph.rag import ask


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask a question about the indexed SEC filings.")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--strategy", default="semantic", choices=STRATEGIES)
    ap.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()

    question = " ".join(a.question)
    r = ask(question, {"strategy": a.strategy, "mode": a.mode, "rerank": not a.no_rerank})

    print(f"\n{question}")
    print("-" * min(80, len(question)))
    print(r.get("answer") or "(no answer)")

    if r.get("citations"):
        print("\nSources:")
        for c in r["citations"]:
            print(f"  [{c['n']}] {c.get('company')} {c.get('form')} (filed {c.get('filingDate')}) - {c.get('section')}")
            print(f"      {c.get('sourceUrl')}")

    if a.trace:
        print("\nGraph trace:")
        for t in r.get("trace", []):
            print(f"  {t['ms']:>6}ms  {t['node']:<9} {t['detail']}")

    bits = [f"{r['totalMs']}ms"]
    if r.get("confidence"):
        bits.append(f"confidence {r['confidence']}")
    if r.get("verification"):
        bits.append(f"faithfulness {r['verification'].get('faithfulness')}")
    if r.get("refused"):
        bits.append("REFUSED")
    if r.get("clarified"):
        bits.append("CLARIFIED")
    print("\n" + " · ".join(bits))


if __name__ == "__main__":
    main()
