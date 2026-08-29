"""
Test bootstrap.

`finrag.config` reads the environment once, at import time, so every variable the
suite depends on has to be set BEFORE any project module is imported. That is why
this happens at conftest module scope rather than in a fixture.
"""
from __future__ import annotations

import os
import socket

import pytest

os.environ.setdefault("OPENAI_API_KEY", "mock")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8099/v1")
os.environ.setdefault("EMBED_DIM", "256")
os.environ.setdefault("VECTOR_BACKEND", "local")
# Never let a test run write mock-derived chunks into the real index directory.
os.environ.setdefault("DATA_DIR", "data-test")

from .mock_openai import PORT, serve  # noqa: E402


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session", autouse=True)
def mock_api():
    """
    Start the mock, or quietly reuse an instance already listening on the port
    (one the developer started by hand, or the offline runner's own). The suite
    only needs *a* mock reachable there, not ownership of it.
    """
    if _port_open(PORT):
        yield None
        return
    httpd = serve(PORT)
    yield httpd
    httpd.shutdown()


@pytest.fixture(scope="session")
def built():
    """Skip integration tests when the indexes have not been built."""
    from finrag.config import P
    if not (P.index / "chunks.fixed.json").exists():
        pytest.skip("run `python tests/offline.py` (or `python -m finrag.build`) first")
    return True
