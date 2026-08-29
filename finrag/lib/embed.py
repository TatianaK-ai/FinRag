"""Batched embedding with a progress bar."""
from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import OpenAIEmbeddings

from ..config import EMBED_DIM, EMBED_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, assert_openai

_embedder: OpenAIEmbeddings | None = None


def embedder() -> OpenAIEmbeddings:
    global _embedder
    assert_openai()
    if _embedder is None:
        kwargs = {"api_key": OPENAI_API_KEY, "model": EMBED_MODEL, "dimensions": EMBED_DIM}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _embedder = OpenAIEmbeddings(**kwargs)
    return _embedder


def embed_query(q: str) -> list[float]:
    return embedder().embed_query(q)


def _bar(done: int, total: int, label: str = "") -> None:
    w = 24
    f = round(done / total * w) if total else w
    sys.stdout.write(f"\r  [{'#' * f}{'.' * (w - f)}] {done}/{total} {label}   ")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def embed_all(texts: list[str], *, batch_size: int = 128, concurrency: int = 4,
              label: str = "") -> list[list[float]]:
    e = embedder()
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    out: list[list[list[float]]] = [[] for _ in batches]
    done = 0

    def run(idx: int) -> None:
        nonlocal done
        out[idx] = e.embed_documents(batches[idx])
        done += 1
        _bar(done, len(batches), label)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(run, range(len(batches))))
    return [v for batch in out for v in batch]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-10)
