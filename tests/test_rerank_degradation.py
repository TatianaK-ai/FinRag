"""
Reranking is an enhancement, never a hard dependency.

Two failures are simulated against a real HTTP server rather than a stub: a
partially failing batch, and a total outage. Both used to be silent - unscored
passages were assigned 0 and then deleted by the score floor, removing a third of
the candidate window while a comment promised they would keep their fused rank.

`rerank.py` binds OPENAI_BASE_URL at import time (`from ..config import ...`), so
these tests rebind the module-level name and clear its cached client instead of
mutating the environment, which would have no effect.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from finrag.retrieval import rerank as rr


def hits(n: int) -> list[dict]:
    return [{"id": f"h{i}", "rank": i, "text": f"Passage {i} about revenue.",
             "metadata": {"ticker": "AAPL", "form": "10-K", "filingDate": "2025-10-31",
                          "section": "Item 7"}}
            for i in range(n)]


def _completion(payload: dict) -> bytes:
    return json.dumps({
        "id": "x", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": {},
    }).encode()


def _serve(handler_cls):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


@pytest.fixture
def point_at():
    """Point the reranker at a given base URL and restore afterwards."""
    saved_url, saved_key, saved_model = rr.OPENAI_BASE_URL, rr.OPENAI_API_KEY, rr._model

    def _apply(port: int):
        rr.OPENAI_BASE_URL = f"http://127.0.0.1:{port}/v1"
        rr.OPENAI_API_KEY = "test"
        rr._model = None

    yield _apply
    rr.OPENAI_BASE_URL, rr.OPENAI_API_KEY, rr._model = saved_url, saved_key, saved_model


def test_degraded_rerank_keeps_unscored_passages(point_at):
    """Serves scores for global ids 0..9 only, and fails the 2nd request outright."""
    state = {"calls": 0}

    class Flaky(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            with threading.Lock():
                state["calls"] += 1
                fail = state["calls"] == 2
            body = (json.dumps({"error": {"message": "induced failure"}}).encode() if fail
                    else _completion({"scores": [{"id": i, "relevance": 9} for i in range(10)]}))
            self.send_response(500 if fail else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = _serve(Flaky)
    point_at(srv.server_address[1])
    try:
        out = rr.rerank("what was revenue", hits(20), top_n=20)

        # Two things go wrong at once here, and both used to be silent:
        #   - the client retries the 500, and this server answers the retry with
        #     ids 0..9 again, colliding with the first batch;
        #   - so ten passages end up with no score of their own.
        # Previously they were assigned 0 and filtered out by min_rerank_score,
        # deleting a third of the candidate window while a comment promised they
        # would "fall back to their fused position".
        assert out["scoredCount"] == 10, "only ten distinct ids were genuinely scored"
        assert out["unscored"] == 10, "the other ten are unscored, not zero-scored"
        assert len(out["hits"]) == 20, "no passage is lost to a degraded rerank"
        assert any(h["id"] == "h15" for h in out["hits"]), \
            "an unscored passage survives at its fused rank"
        assert "unscored" in out["error"], "the degradation is reported, not swallowed"

        # Scored passages still outrank unscored ones.
        first_unscored = next(i for i, h in enumerate(out["hits"]) if not h["scored"])
        assert all(h["scored"] for h in out["hits"][:first_unscored])
    finally:
        srv.shutdown()


def test_total_rerank_failure_falls_back_to_fused_order(point_at):
    class Down(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.dumps({"error": {"message": "down"}}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = _serve(Down)
    point_at(srv.server_address[1])
    try:
        out = rr.rerank("what was revenue", hits(12), top_n=6)
        assert out["degraded"] is True, "reranking is an enhancement, never a hard dependency"
        assert len(out["hits"]) == 6, "falls back to the top of the fused order"
        assert out["error"], "the failure is surfaced"
    finally:
        srv.shutdown()
