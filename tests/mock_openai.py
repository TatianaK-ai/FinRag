"""
Offline stand-in for the OpenAI API.

Purpose is plumbing verification, not answer quality: it lets the full pipeline
(semantic chunking -> embedding -> indexing -> hybrid retrieval -> rerank ->
graph -> eval) run end to end with no credentials and no spend, so the wiring can
be tested independently of the models.

    python -m tests.mock_openai
    OPENAI_API_KEY=mock OPENAI_BASE_URL=http://127.0.0.1:8099/v1 python -m finrag.chunking.run

Embeddings are hashed bag-of-words, so cosine similarity still tracks lexical
overlap and retrieval smoke tests produce sensible orderings.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.getenv("MOCK_PORT", "8099"))
DIM = int(os.getenv("MOCK_DIM", "256"))

_WORD = re.compile(r"[a-z0-9]+")
_PASSAGE_SPLIT = re.compile(r"\n(?=\[\d+\]\s)")
_QUESTION = re.compile(r"QUESTION:\s*(.+)")


def embed(text: str) -> list[float]:
    v = [0.0] * DIM
    for t in _WORD.findall(text.lower())[:4000]:
        h = hashlib.md5(t.encode()).digest()
        i = struct.unpack("<I", h[:4])[0] % DIM
        sign = 1 if h[4] & 1 else -1
        v[i] += sign * (1 + math.log(1 + len(t)))
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def words(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def required_entities(question: str) -> list[str]:
    """
    Named entities the question demands. Capitalised mid-sentence words are a
    decent proxy for "the company being asked about" - enough for the mock to
    decide whether the corpus can plausibly answer at all, which is what makes
    the refusal path testable offline.
    """
    caps = re.findall(r"\b[A-Z][A-Za-z]{2,}\b", question)[1:]  # drop sentence-initial
    stop = {"what", "how", "much", "did", "the", "and"}
    return [w.lower() for w in caps if w.lower() not in stop]


def coverage(ctx: dict) -> float:
    """0..1 - how well the retrieved passages cover the question."""
    q = set(words(ctx["question"]))
    allw = set(words(" ".join(p["text"] for p in ctx["passages"])))
    entities = required_entities(ctx["question"])
    # A named entity absent from every passage means the corpus does not cover it.
    if entities and not any(e in allw for e in entities):
        return 0.0
    return (len(q & allw) / len(q)) if q else 0.0


def _generic(s):
    if not isinstance(s, dict):
        return None
    t = s.get("type")
    if t == "object":
        return {k: _generic(v) for k, v in (s.get("properties") or {}).items()}
    if t == "array":
        return []
    if t == "string":
        return (s.get("enum") or ["mock"])[0]
    if t in ("number", "integer"):
        return s.get("minimum", 0)
    if t == "boolean":
        return True
    return None


def synth(schema, ctx: dict):
    """Build a value that satisfies the requested JSON schema."""
    name = ctx["schemaName"]

    if "Scores" in name:
        # One score per passage, ranked by lexical overlap with the question. A
        # question naming an entity that is nowhere in the passages scores 0, so
        # the grade gate can refuse.
        q = set(words(ctx["question"]))
        entities = required_entities(ctx["question"])
        scores = []
        for p in ctx["passages"]:
            toks = set(words(p["text"]))
            if entities and not any(e in toks for e in entities):
                scores.append({"id": p["n"], "relevance": 0})
                continue
            hit = len(q & toks)
            scores.append({"id": p["n"],
                           "relevance": min(10, round(hit / max(1, len(q)) * 14))})
        return {"scores": scores}

    if name.startswith("Scope"):
        # Ambiguous only when the question names NO company at all and does not
        # deliberately span them. Naming a company outside the corpus (Tesla) is
        # not ambiguity - it is a question the corpus cannot answer, which the
        # refusal path handles downstream. Keying only off the three corpus names
        # would misroute those to clarification.
        named = bool(required_entities(ctx["question"]))
        spans_all = re.search(r"three companies|each company|all three|compare",
                              ctx["question"], re.I) is not None
        ambiguous = not named and not spans_all
        return {"namesSubject": named or spans_all, "isAmbiguous": ambiguous,
                "clarifyingQuestion": "Which company: Apple, Microsoft or NVIDIA?" if ambiguous else ""}

    if name.startswith("Grade"):
        c = coverage(ctx)
        if c > 0.3:
            return {"sufficient": True, "reason": "mock: passages look sufficient"}
        return {"sufficient": False,
                "reason": f"mock: coverage {c:.2f} - question names something absent from the corpus"}

    if name.startswith("Answer"):
        c = coverage(ctx)
        forecast = re.search(r"forecast|guidance|project|expect.*20\d\d", ctx["question"], re.I)
        if c <= 0.3 or forecast:
            return {"requestedFact": ctx["question"][:120], "premiseHolds": not forecast,
                    "answerable": False, "answer": "", "citations": [], "confidence": 0}
        return {"requestedFact": ctx["question"][:120], "premiseHolds": True, "answerable": True,
                "answer": f"Mock answer for: {ctx['question']} [1]", "citations": [1],
                "confidence": 0.75}

    if name.startswith("Verify"):
        return {"grounded": True, "unsupportedClaims": [], "faithfulness": 0.9,
                "answersTheQuestion": True, "mismatchReason": ""}

    if name.startswith("Judge"):
        # Must satisfy every required field of the judge schema. A partial object
        # fails validation client-side and LangChain silently retries, which looks
        # exactly like a hang.
        return {"faithfulness": 0.9, "unsupported": [], "correctness": 0.9, "relevance": 0.9,
                "citationsValid": True, "verdict": "mock judge - plumbing only, not a quality signal"}

    return _generic(schema)


def extract_context(body: dict) -> dict:
    msgs = body.get("messages") or []
    user = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "user")
    qm = _QUESTION.search(user)
    # Split on any line that OPENS a passage, not on a blank line preceding one.
    # A `\n\n(?=\[n\])` split can never fire before the FIRST passage, because the
    # prompt joins it to "PASSAGES:" with a single newline - so every mocked call
    # silently loses passage 1 and renumbers the rest, and the whole offline suite
    # runs green over that broken contract.
    passages = []
    for part in _PASSAGE_SPLIT.split(user):
        m = re.match(r"^\[(\d+)\]\s", part)
        if m:
            passages.append({"n": int(m.group(1)), "text": part})

    rf = (body.get("response_format") or {}).get("json_schema") or {}
    tools = body.get("tools") or []
    fn = (tools[0].get("function") if tools else None) or {}
    return {
        "question": qm.group(1).strip() if qm else user[:120],
        "passages": passages,
        "schemaName": rf.get("name") or fn.get("name") or "unknown",
        "schema": rf.get("schema") or fn.get("parameters"),
        "viaTool": bool(fn and not rf),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # quiet
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):  # noqa: N802
        if self.path.endswith("/models"):
            return self._json(200, {"object": "list", "data": []})
        self._json(404, {"error": {"message": f"mock: no route for {self.path}"}})

    def do_POST(self):  # noqa: N802
        try:
            if self.path.endswith("/embeddings"):
                body = self._read()
                inp = body.get("input")
                inp = inp if isinstance(inp, list) else [inp]
                return self._json(200, {
                    "object": "list", "model": body.get("model"),
                    "data": [{"object": "embedding", "index": i, "embedding": embed(str(t))}
                             for i, t in enumerate(inp)],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                })

            if self.path.endswith("/chat/completions"):
                body = self._read()
                ctx = extract_context(body)
                value = synth(ctx["schema"], ctx)
                if ctx["viaTool"]:
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_mock", "type": "function",
                        "function": {"name": ctx["schemaName"], "arguments": json.dumps(value)}}]}
                    finish = "tool_calls"
                else:
                    message = {"role": "assistant", "content": json.dumps(value)}
                    finish = "stop"
                return self._json(200, {
                    "id": "chatcmpl-mock", "object": "chat.completion", "model": body.get("model"),
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })

            self._json(404, {"error": {"message": f"mock: no route for {self.path}"}})
        except Exception as e:  # never let a mock failure look like a model failure
            self._json(500, {"error": {"message": str(e)}})


def serve(port: int = PORT) -> ThreadingHTTPServer:
    """
    Bind the port and serve on a daemon thread. Raises OSError if the port is
    taken - callers that only need *a* mock reachable there should probe first
    rather than assume ownership.
    """
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


if __name__ == "__main__":
    srv = serve()
    print(f"mock OpenAI listening on http://127.0.0.1:{PORT}/v1  (dim {DIM})", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
