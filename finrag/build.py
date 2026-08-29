"""
One command for the whole build: ingest -> chunk -> index.

    python -m finrag.build

Runs the three stages in-process rather than chaining shell commands, so it
behaves identically on Windows and POSIX.
"""
from __future__ import annotations

import sys

STAGES = [
    ("ingest", "finrag.ingest.run"),
    ("chunk", "finrag.chunking.run"),
    ("index", "finrag.store.run"),
]


def main() -> None:
    from importlib import import_module
    for name, mod in STAGES:
        try:
            import_module(mod).main()
        except Exception as e:
            print(f"\nFAILED at \"{name}\": {e}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
