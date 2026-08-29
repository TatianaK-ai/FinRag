"""Configuration: paths, model choices, retrieval constants, corpus definition."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# DATA_DIR isolates derived artefacts (chunks, BM25, local vectors) so a test run
# cannot overwrite a real index. Raw filings stay shared: they are immutable
# downloads from EDGAR and re-fetching them in tests would be rude to the SEC.
_DATA = os.getenv("DATA_DIR", "data")


@dataclass(frozen=True)
class Paths:
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / _DATA / "processed"
    index: Path = ROOT / _DATA / "index"
    eval: Path = ROOT / "eval"
    out: Path = ROOT / "out"


P = Paths()


def _pick(*keys: str, default: str | None = None) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return default


# OPEN_API_KEY is accepted as a fallback: the original .env used that name.
OPENAI_API_KEY = _pick("OPENAI_API_KEY", "OPEN_API_KEY")
OPENAI_BASE_URL = _pick("OPENAI_BASE_URL")
PINECONE_API_KEY = _pick("PINECONE_API_KEY")
PINECONE_INDEX = _pick("PINECONE_INDEX_NAME", "PINECONE_INDEX", default="finrag-py")
SEC_USER_AGENT = _pick("SEC_USER_AGENT", default="FinRAG Research contact@example.com")

# --- Model choices -----------------------------------------------------------
# text-embedding-3-small @1536d: 8k context comfortably covers a ~1200-char chunk,
# and at $0.02/1M tokens the full two-strategy corpus embeds for under a cent.
EMBED_MODEL = _pick("EMBED_MODEL", default="text-embedding-3-small")
EMBED_DIM = int(_pick("EMBED_DIM", default="1536"))
GEN_MODEL = _pick("GEN_MODEL", default="gpt-4.1-mini")
RERANK_MODEL = _pick("RERANK_MODEL", default="gpt-4.1-mini")

# The scope gate looks like a one-bit classification, so it was briefly moved to a
# cheaper model to save latency. That was a bad trade and the eval caught it: the
# smaller model flags questions that clearly name a company as ambiguous, so
# correct refusal fell 100% -> 25%. The judgement is subtler than it looks.
SCOPE_MODEL = _pick("SCOPE_MODEL", default="gpt-4.1-mini")

# The judge is deliberately STRONGER than the generator, so the evaluation is not
# the generator grading its own homework.
JUDGE_MODEL = _pick("JUDGE_MODEL", default="gpt-4.1")

VECTOR_BACKEND = (_pick("VECTOR_BACKEND", default="pinecone") or "pinecone").lower()


@dataclass(frozen=True)
class RetrievalConfig:
    dense_k: int = 40
    sparse_k: int = 40
    # The reranker can only reorder what retrieval handed it. A dense-table chunk
    # competing against verbose prose on the same topic can land past rank 20, and
    # no amount of reranking recovers it - so the window must be wider than the
    # number of documents actually kept.
    rerank_window: int = 30
    # Floor on candidates per issuer for comparison questions. With three issuers a
    # flat 30-candidate window gives each only 10, and the table carrying
    # Microsoft's total revenue growth fused to rank 11 - one position outside the
    # budget - so the answer fell back to segment growth rates.
    min_per_entity: int = 14
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rerank_to: int = 6        # kept per named issuer
    max_rerank_to: int = 14   # ceiling once scaled by issuer count
    top_k: int = 6            # fed to the generator when reranking is off
    min_rerank_score: int = 4
    grade_threshold: int = 4


RETRIEVAL = RetrievalConfig()


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: str
    name: str


# Three large-cap issuers with genuinely different cost structures and misaligned
# fiscal calendars, which makes cross-document comparison meaningful.
CORPUS: list[Company] = [
    Company("AAPL", "0000320193", "Apple Inc."),
    Company("MSFT", "0000789019", "Microsoft Corporation"),
    Company("NVDA", "0001045810", "NVIDIA Corporation"),
]

FORMS = ["10-K", "10-Q"]
PER_FORM = {"10-K": 1, "10-Q": 1}
STRATEGIES = ["fixed", "semantic"]


def assert_openai() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Put it in .env (see .env.example). "
            "Get one at https://platform.openai.com/api-keys"
        )


def assert_pinecone() -> None:
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "Missing PINECONE_API_KEY. Put it in .env (see .env.example). "
            "Get one at https://app.pinecone.io -> your project -> API Keys"
        )
