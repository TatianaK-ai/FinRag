"""
One-command offline verification: build the whole corpus and both indexes against
the bundled mock API, then run the test suite. No credentials, no spend.

    python tests/offline.py

The mock runs in this process on a daemon thread - unlike the Node original there
is no event-loop to starve, because the build stages are subprocesses and the
mock's server thread keeps answering while they run.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PORT = int(os.getenv("MOCK_PORT", "8099"))
ENV = {
    **os.environ,
    "OPENAI_API_KEY": "mock",
    "OPENAI_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
    "EMBED_DIM": "256",
    "VECTOR_BACKEND": "local",
    # Never write mock-derived chunks into the real index directory.
    "DATA_DIR": "data-test",
    "PYTHONPATH": str(ROOT),
}


def port_open(port: int, timeout: float = 0.5) -> bool:
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex(("127.0.0.1", port)) == 0


def run(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    code = subprocess.call([sys.executable, *args], cwd=ROOT, env=ENV)
    if code != 0:
        raise SystemExit(f"{label} failed (exit {code})")


def main() -> None:
    from tests.mock_openai import serve

    reuse = port_open(PORT)
    httpd = None if reuse else serve(PORT)
    deadline = time.time() + 10
    while not port_open(PORT):
        if time.time() > deadline:
            raise SystemExit(f"mock API did not come up on :{PORT}")
        time.sleep(0.2)
    print(f"{'reusing mock API already on' if reuse else 'mock API ready on'} :{PORT}")

    try:
        run("build: ingest + chunk + index", ["-m", "finrag.build"])
        run("tests", ["-m", "pytest", "-q", "tests"])
    finally:
        if httpd:
            httpd.shutdown()

    print("\nOffline verification passed.")


if __name__ == "__main__":
    main()
