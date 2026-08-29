# FinRAG — Financial Document Intelligence

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LangGraph-1C3C3C?style=flat-square)
![Pinecone](https://img.shields.io/badge/Pinecone-vector%20store-6f4cff?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-server-009688?style=flat-square&logo=fastapi&logoColor=white)
[![tests](https://github.com/TatianaK-ai/FinRag/actions/workflows/tests.yml/badge.svg)](https://github.com/TatianaK-ai/FinRag/actions/workflows/tests.yml)

A retrieval-augmented question-answering system over public SEC filings
(Apple, Microsoft, NVIDIA — 10-K and 10-Q). It answers questions with a figure
and a citation back to the exact passage in the exact filing, and **refuses**
when the filings do not support an answer.

Built with LangChain + LangGraph (Python), Pinecone, OpenAI.

Full write-up: [`docs/PROJECT.md`](docs/PROJECT.md) · framework:
[`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) · latest evaluation: `out/REPORT.md` ·
documentation PDF: [`docs/FinRAG-Documentation.pdf`](docs/FinRAG-Documentation.pdf).

## Demo

https://github.com/user-attachments/assets/eac1c2bd-ba70-4d93-9eb6-576f6b84a639

3:37 against the running server — a grounded answer with six EDGAR citations, two
refusals (out-of-corpus, then a false premise caught by the entity-grounding
gate), a clarification with the graph trace expanded, then the architecture and
the evaluation results with their limits.

If the player above does not load, it also plays on
[issue #1](https://github.com/TatianaK-ai/FinRag/issues/1). The file is committed
at [`demo/finrag-demo-live.mp4`](demo/finrag-demo-live.mp4), but GitHub serves
repository files as `application/octet-stream`, so that link downloads rather
than plays.

```
$ python -m finrag.cli "What was NVIDIA's gross margin in fiscal 2026?"

NVIDIA's gross margin in fiscal 2026 was 71.1% [1].

[1] NVDA 10-K filed 2026-02-25 — Item 7. MD&A
    https://www.sec.gov/Archives/edgar/data/1045810/...
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate  elsewhere
pip install -r requirements.txt
cp .env.example .env            # then fill in the keys
```

`.env` needs `OPENAI_API_KEY`, and `PINECONE_API_KEY` unless you set
`VECTOR_BACKEND=local`. `SEC_USER_AGENT` must be a real contact string — EDGAR
rejects anonymous automated traffic.

## Build the index

```bash
python -m finrag.build     # ingest -> chunk (both strategies) -> embed -> index
```

Raw filings are cached under `data/raw/`, so a rebuild does not re-hit EDGAR.

## Ask

```bash
python -m finrag.cli "What were Apple's total net sales in fiscal 2025?"
python -m finrag.server                    # chat UI on http://127.0.0.1:3000
```

**Set `APP_API_KEY` before exposing the server.** Without it the process binds to
`127.0.0.1` only and refuses off-box connections — every `/api/ask` call spends
real API credit, so an open endpoint is a billing liability before it is a data
one.

## Evaluate

```bash
EVAL_REPEATS=3 python -m finrag.evals.run    # experiment matrix, 3 repetitions
python -m finrag.evals.report                # renders out/REPORT.md
```

Stage 1 sweeps retrieval (2 chunking strategies × 3 modes, plus reranking);
stage 2 runs the full graph end to end and judges the answers with a *stronger*
model than the one that generated them. `EVAL_REPEATS` matters: the gates are
model calls and do not settle on one answer, and single-run figures have been
wide enough to change conclusions.

## Test

```bash
python tests/offline.py     # builds the corpus against a mock API, then runs pytest
pytest                      # tests only, assuming an index already exists
```

The offline runner needs **no credentials and spends nothing**: a bundled mock
OpenAI server (`tests/mock_openai.py`) serves hashed bag-of-words embeddings and
schema-shaped completions, so the whole pipeline runs end to end. It builds into
`DATA_DIR=data-test` so it can never overwrite a real index. Offline numbers
prove the wiring, never the quality.

The `tests` badge above runs `pytest` on GitHub Actions (Python 3.10 and 3.12).
CI covers the deterministic layers only — cleaning, tokenising, fusion, the
scope and guidance guards, auth and rate limiting. The graph and retrieval tests
skip there, because they need an index built from `data/raw/`, which is not in
the repository. `python tests/offline.py` is the full 32-test gate.

## Layout

```
finrag/
  config.py            models, retrieval constants, corpus, paths
  build.py             ingest -> chunk -> index in one command
  ingest/              EDGAR fetch, XBRL stripping, table-aware cleaning
  chunking/            fixed-size and semantic strategies (shared table handling)
  store/               BM25, local vectors, Pinecone, index freshness
  retrieval/           hybrid dense+sparse with RRF, LLM cross-encoder rerank
  graph/rag.py         the LangGraph state machine and its four refusal gates
  evals/               two-stage harness, LLM judge, report renderer
  middleware/guard.py  auth and rate limiting
  server.py, cli.py    FastAPI surface and command line
eval/questions.json    the golden set
tests/                 pytest suite + mock OpenAI server
docs/                  project write-up and framework
```
