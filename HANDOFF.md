# Handoff — what this project is and where it stands

This file exists because the Claude Code session that built this port lives under
the *other* project directory (`week_2`, the JavaScript original) and cannot be
resumed from here. Everything a fresh session needs to pick the work up is below.

Last updated: 2026-08-28.

---

## 1. What this is

`week_2_py` is a complete Python rewrite of `week_2` (LangChain.js + LangGraph.js).
Same corpus, same architecture, same evaluation methodology, same golden set.

A RAG system over six public SEC filings (Apple, Microsoft, NVIDIA — one 10-K and
one 10-Q each). It answers financial questions with a figure and a citation back
to the exact passage in the exact filing, and refuses when the filings do not
support an answer.

Read `docs/PROJECT.md` first — it is the full write-up, including the nineteen
iterations that produced the current design and the failures behind each one.
`docs/FRAMEWORK.md` is the one-page framework. `README.md` is the operating manual.

## 2. Opening it in PyCharm

- **File → Open** → `C:\Users\tatia\WebstormProjects\week_2_py`
- PyCharm should detect `.venv/`. If it does not:
  **Settings → Project → Python Interpreter → Add → Existing → `.venv\Scripts\python.exe`**
- Mark nothing as a sources root; the project is a plain package (`finrag/`) run
  with `python -m finrag.<module>`.
- Run configurations worth adding: `finrag.build`, `finrag.server`,
  `finrag.evals.run`, `finrag.evals.report`, and `tests/offline.py`.

Claude Code sessions are stored per directory, so a session started here begins
empty. Point it at this file.

## 3. State as of the last session

**Working and verified against the real APIs:**

| Stage | Result |
| --- | --- |
| Ingest | 6 documents, 1382 blocks, 348 tables, 846 kB of inline XBRL stripped |
| Chunking | fixed **1619** chunks (mean 871, stdev 393); semantic **1253** (mean 1060, stdev 725) |
| Indexing | BM25 + Pinecone index `finrag-py`; fingerprints `1619:49002b488d753121`, `1253:aeebb520aa1d61fc` |
| Graph | 7/7 behaviour checks correct (3 answer, 3 refuse, 1 clarify), answered p50 ≈ 6.0 s |
| Server | FastAPI, auth verified (401 no key / 401 wrong key / 200 correct key), UI answers with EDGAR citations |
| Tests | 32/32 pass offline against the bundled mock, no credentials, no spend |
| Eval | 3 repetitions, n=51 judged: answer rate 100%, wrong-figure 0/51, faithfulness 1.000, correct refusal 100%, false refusal 0%, p50 5,445 ms, hit@6 1.000 |

Both chunking counts match the JavaScript original exactly — that was the
acceptance criterion for the port, not a coincidence. The fixed chunker uses the
official `RecursiveCharacterTextSplitter` from `langchain-text-splitters`; a
hand-written recursive splitter produced 1719 chunks instead of 1619.

The full evaluation has been run twice. The first run scored 94.1% answer rate
and exposed a porting defect (below); after the fix the second run reached 100%
with no unstable questions and an empty failure table. `out/REPORT.pre-schema-fix.md`
keeps the first run for comparison.

For reference, the JavaScript original over the same golden set — 3 repetitions,
n=50 judged: answer rate 98.0%, wrong-figure 0/50, faithfulness 1.000, correct
refusal 100%, clarified 100%, false refusal 2.0%, p50 5,155 ms, hit@6 1.000.

**Nothing is committed.** Neither project has a git repository initialised here;
that was deliberate at the user's instruction.

## 4. Things that will bite you

These are all real failures already paid for once. They are documented in the
code at the point where they matter, but the expensive ones are worth knowing
before you touch anything.

**Never let a test run write into `data/index/`.** The offline build sets
`DATA_DIR=data-test`. Running it without that isolation once replaced
`chunks.semantic.json` with mock-derived chunks while the vector store still held
vectors keyed to the real ids — 466 fixed and 953 semantic ids whose stored text
differed from what the chunk store served. It read as a catastrophic retrieval
regression, not as corruption. `assert_index_fresh()` and the content fingerprint
now catch this, but the isolation is the actual fix.

**`finrag/config.py` reads the environment once, at import time.** Any test or
script that needs a different `OPENAI_BASE_URL`, `DATA_DIR` or `EMBED_DIM` has to
set it *before* the first `import finrag.*`. That is why `tests/conftest.py` does
it at module scope, and why `tests/test_rerank_degradation.py` rebinds
`rerank.OPENAI_BASE_URL` and clears `rerank._model` instead of touching `os.environ`.

**Pydantic `Field(description=...)` is not documentation.** Those strings are
serialised into the JSON schema sent to the model and steer its decision. Porting
`ScopeOut` as three bare fields cost a question: "which of the three companies
grew revenue fastest" read as ambiguous in 3 runs out of 3, because nothing in
the schema said that deliberately spanning all issuers still counts as
identifying the subject. Restoring the abridged *system prompt* did not fix it -
only restoring the field descriptions did. If you add a field to any schema in
`graph/rag.py`, describe it.

**`n=1` is not evidence.** Measured over three identical runs, correct refusal
ranged 75%–100% and two questions failed in two runs out of three. Twice during
development a conclusion was drawn from one or two runs and had to be retracted.
Run `EVAL_REPEATS=3` before believing any number, and read the spread table in
§5 of the report rather than the headline figures.

**Faithfulness is conditional on answering.** It is measured only on answers the
pipeline chose to emit — `verify` refuses anything it judges ungrounded, and only
non-refused rows reach the judge. An arm can raise its faithfulness simply by
refusing more. The report prints answer rate first for exactly this reason; do
not quote faithfulness without it.

**The scope gate must stay on `gpt-4.1-mini`.** Moving it to `gpt-4.1-nano` saved
260 ms and dropped correct refusal from 100% to 25%: the smaller model flags
questions that clearly name a company as ambiguous. The reason is written into
`config.py` so it is not re-attempted.

**Two of the four refusal gates are deterministic, not model calls.** The
guidance guard (`asks_for_guidance`) and the entity-grounding check
(`unsupported_entities`) exist because both LLM gates passed answers that were
confidently wrong — a 2027 "forecast" lifted from a lease-maturity table, and
NVIDIA's real acquisition price reported as the price of an imagined Intel deal.
If a test for either one calls `retrieve()` directly it is testing the wrong
layer; the guards live in the scope and verify nodes.

**A test that names no company routes to `clarify`, correctly.** This has caused
two false bug reports. If you write an end-to-end test about anything else, name
an issuer in the question.

## 5. Legality of the corpus

The filings are public-domain US government records published by the SEC for
exactly this purpose. `finrag/ingest/edgar.py` rate-limits requests and sends a
`SEC_USER_AGENT` identifying a real contact, which is what EDGAR's access policy
asks for. Keep both.

## 6. What is left

- Nothing outstanding in the port itself. The one caveat worth keeping in view:
  100% on the shipped arm is measured over 17 answerable questions × 3 repetitions,
  on the same golden set the system was iterated against. It means no
  counter-example was found, not that none exists.
- Demo video and the write-up document — the user's own submission tasks.
