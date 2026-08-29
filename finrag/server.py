"""FastAPI server: static UI plus a guarded JSON API."""
from __future__ import annotations

import os
import traceback

from fastapi import FastAPI, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import CORPUS, EMBED_MODEL, GEN_MODEL, P, ROOT
from .graph.rag import ask
from .middleware.guard import auth_enabled, bind_host, check_rate, guard_status, key_matches
from .store.index import BACKEND

MAX_QUESTION_CHARS = 500

app = FastAPI(title="FinRAG", docs_url=None, redoc_url=None)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    strategy: str = "semantic"
    mode: str = "hybrid"
    rerank: bool = True


def _presented(authorization: str | None, x_api_key: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return x_api_key


def _unauthorized() -> JSONResponse:
    # No detail about why: a caller learns only that it was rejected.
    return JSONResponse({"error": "unauthorized"}, status_code=401)


@app.get("/api/auth-status")
def auth_status():
    """Lets the page know whether to ask for a key, without revealing anything else."""
    return {"authRequired": auth_enabled()}


@app.get("/api/info")
def info(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    if auth_enabled() and not key_matches(_presented(authorization, x_api_key)):
        return _unauthorized()
    import json
    manifest = {"documents": []}
    mf = P.processed / "_manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    stats = {}
    sf = P.index / "chunk-stats.json"
    if sf.exists():
        stats = json.loads(sf.read_text(encoding="utf-8"))
    return {
        "companies": [c.__dict__ for c in CORPUS],
        "documents": manifest.get("documents", []),
        "chunkStats": stats,
        "models": {"generator": GEN_MODEL, "embeddings": EMBED_MODEL},
        "backend": BACKEND,
        "authRequired": auth_enabled(),
    }


@app.post("/api/ask")
def api_ask(body: AskBody, request: Request,
            authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    if auth_enabled() and not key_matches(_presented(authorization, x_api_key)):
        return _unauthorized()

    client = (request.client.host if request.client else "unknown")
    allowed, msg, retry = check_rate(client)
    if not allowed:
        return JSONResponse({"error": msg, "retryAfterSeconds": retry},
                            status_code=429, headers={"Retry-After": str(retry)})

    if body.mode not in ("dense", "sparse", "hybrid"):
        return JSONResponse({"error": "mode must be dense, sparse or hybrid"}, status_code=400)
    if body.strategy not in ("fixed", "semantic"):
        return JSONResponse({"error": "strategy must be fixed or semantic"}, status_code=400)

    try:
        r = ask(body.question, {"strategy": body.strategy, "mode": body.mode, "rerank": body.rerank})
    except Exception:
        # Logged in full, returned in outline: an internal message can carry index
        # paths or key-shaped strings a caller has no business seeing.
        traceback.print_exc()
        return JSONResponse({"error": "internal error"}, status_code=500)

    return {
        "answer": r.get("answer"),
        "refused": bool(r.get("refused")),
        "clarified": bool(r.get("clarified")),
        "citations": r.get("citations", []),
        "confidence": r.get("confidence", 0),
        "verification": r.get("verification"),
        "trace": r.get("trace", []),
        "timings": r.get("timings", {}),
        "totalMs": r.get("totalMs"),
        "contexts": [{
            "id": c["id"], "section": c["metadata"].get("section"),
            "ticker": c["metadata"].get("ticker"), "form": c["metadata"].get("form"),
            "filingDate": c["metadata"].get("filingDate"),
            "isTable": c["metadata"].get("isTable"),
            "rerankScore": c.get("rerankScore"),
            "preview": (c.get("text") or "")[:400],
        } for c in (r.get("contexts") or [])],
    }


_public = ROOT / "public"


@app.get("/")
def root():
    return FileResponse(_public / "index.html")


app.mount("/", StaticFiles(directory=str(_public)), name="static")


def main() -> None:
    import uvicorn
    port = int(os.getenv("PORT", "3000"))
    host = bind_host()
    print(f"FinRAG UI on http://{'localhost' if host == '0.0.0.0' else host}:{port}")
    for line in guard_status(port):
        print(line)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
