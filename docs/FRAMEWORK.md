# Part 1 — The RAG Framework

## The one-liner

> My RAG app helps **equity and credit analysts** answer **specific quantitative and disclosure questions** from **the latest 10-K and 10-Q filings of AAPL, MSFT and NVDA (6 documents, ~1.3 M characters, 348 tables, sourced live from SEC EDGAR)** in **a web chat UI and a CLI** with **≥95% faithfulness** and **100% correct refusal on out-of-corpus questions**, at **p95 latency under 6 seconds**.

Against the three rules in the brief:

- **The corpus is named specifically.** Not "financial documents" — six identified filings, fetched by CIK from EDGAR, re-fetchable with one command, with their accession numbers recorded in the manifest. Every chunking and retrieval decision below follows from what those documents actually are: long, structurally repetitive, table-dense, and near-identical across issuers and years.
- **Faithfulness is the headline metric, not relevance.** The target is 95% of claims traceable to a retrieved passage, graded by a model stronger than the generator. Relevance is tracked too, but an answer that sounds right and cites a figure the filing never printed is the specific failure this system exists to prevent.
- **Latency has a ceiling.** p95 under 6 s end-to-end, including reranking and the grounding check. That budget is what stops retrieval from growing a third and fourth stage just because each one nudges nDCG.

---

## The framework

| Field | Decision |
| --- | --- |
| **Use case** | An analyst asks for a specific reported figure or disclosure ("What was NVIDIA's FY2026 gross margin?", "How did Apple's Greater China revenue move?") and needs the number with the filing and section it came from. Surfaces: a browser chat UI, a CLI, and a JSON HTTP API for embedding elsewhere. |
| **Corpus** | 6 filings — latest 10-K and 10-Q each for Apple, Microsoft, NVIDIA — pulled from the SEC EDGAR submissions API by CIK. English, inline-XBRL HTML, ~1.3 M cleaned characters, 348 tables, 1,382 structural blocks. The SEC is the source of truth; nothing is hand-edited after download. |
| **Ingestion + cleaning** | `data.sec.gov/submissions/CIK*.json` → primary document HTML → strip `<ix:header>`/`<ix:hidden>` (847 kB of XBRL tagging that would otherwise appear as text like `0000320193us-gaap:CommonStock…`) → drop `<script>`/`<style>` → convert every `<table>` to markdown with per-row compaction so `$`/`%`/parens re-attach to their figures → drop page-footer boilerplate (467 lines) → detect `Item N.` headings to label each block with its section, remembering each item's title so a bare running header (`Item 7` on its own) cannot strip it. Entities are decoded and NBSP/smart quotes normalised. Sections become chunk metadata and drive citations.

Every table chunk also carries a **synthesized caption** built deterministically from its own row labels and column headers — *"Financial table from Microsoft Corporation (MSFT) 10-K. Covers fiscal year 2026, fiscal year 2025. Reported line items: Revenue, Gross margin, …"*. Tables otherwise lose to prose in both retrievers: a grid of numbers embeds nowhere near a natural-language question, and BM25 rewards prose that repeats the query's words while a table repeats nothing. The concrete gap was vocabulary — the table says `2026`, the question says "fiscal year 2026". Captioning roughly doubled BM25 hit@6 (0.412 → 0.765). |
| **Ingestion + freshness** | `python -m finrag.ingest.run` re-fetches and re-cleans; raw HTML is cached on disk so re-runs are free. Cadence matches the filings themselves: quarterly for 10-Qs, annually for 10-Ks, so a weekly cron is more than sufficient. Freshness SLA: indexed within 24 h of a filing appearing on EDGAR. Every chunk carries `filingDate` and `periodOfReport`, so staleness is visible in the answer rather than hidden. |
| **Chunking + embedding** | Two strategies, built and measured against each other. **Fixed**: recursive character splitting at 1,200 chars / 200 overlap (~300 tokens) → 1,620 chunks, stdev 393. **Semantic**: sentence embeddings with a 1-sentence context buffer, cut where consecutive cosine distance exceeds the document's 90th percentile, bounded to 350–2,000 chars → 1,252 chunks, stdev 724. Tables are atomic in **both** (split by row group with the header repeated when oversized) so the comparison isolates prose segmentation. Embeddings: `text-embedding-3-small` @1536d — 8 k context comfortably covers a chunk, and the whole two-strategy corpus embeds for well under a cent. Chunk size and model were chosen together: 1,200 chars is small enough that a chunk carries one claim, large enough that a financial sentence keeps its qualifying clause. |
| **Retrieve** | Pinecone serverless (cosine, one namespace per chunking strategy) for dense, plus a from-scratch BM25 index for sparse. Both index the **same** provenance-prefixed text — indexing different representations silently crippled sparse retrieval on tables. Fused with Reciprocal Rank Fusion (k=60, equal weights) — chosen over score normalisation because cosine and BM25 scores are not comparably scaled and shift per query, whereas RRF needs only the ranks. Retrieve top-40 from each, fuse, rerank the top 30 in concurrent batches, keep 6 for the generator. A local brute-force vector backend (`VECTOR_BACKEND=local`) mirrors the same interface so the project runs and the evals reproduce without a Pinecone account. |

---

## Why hybrid, concretely

Dense retrieval and BM25 fail in opposite directions on this corpus, which is the entire argument for fusing them:

- **Dense misses exact tokens.** Embeddings place "revenue grew" and "revenue declined" close together, and cannot distinguish `$416,161` from `$391,035` — they are the same shape of thing. In filings the figure *is* the answer.
- **BM25 misses intent.** "How profitable is Apple's services business?" shares no useful term with the table headed `Gross margin` that answers it.
- **Both miss period.** Every issuer files a structurally identical document each year. The passage for FY2025 and the one for FY2024 differ by a handful of digits. Neither retriever alone reliably picks the right one — which is what the cross-encoder reranker is for, since it reads query and passage *together* and can act on "fiscal 2026" as a constraint rather than as a bag of tokens.

---

## The refusal path, designed first

The brief's point — that the "I don't know" path matters more than the happy path — is taken literally here: the graph was built refusal-first, and there are **four independent gates** before any answer is returned. Two of them (`premise` and the question-match half of `verify`) were added *because* evaluation caught them missing.

```
        ┌─→ scope ────┐
START ──┤             ├──→ gate ──ambiguous──→ clarify → END
        └─→ retrieve ─┘             │
     (fan out in parallel,      identified
      join at the gate)             ↓
                              [rerank] → grade ──insufficient──→ refuse → END
                                           │
                                       sufficient
                                           ↓
                                       generate ──false premise───→ refuse → END
                                           │    ──not answerable──→ refuse → END
                                           ↓
                                        verify ──ungrounded────────→ refuse → END
                                           │    ──answers the wrong
                                           │      question─────────→ refuse → END
                                           ↓
                                          END
```

1. **`scope`** — does the question even say which issuer it means? Judged on the question text alone: `scope` and `retrieve` fan out from `START` in parallel and join at `gate`, so scope never sees the passages and its verdict cannot be biased by whichever issuer retrieval happened to surface. Underspecified → `clarify`, which asks back. (Folding this into the generator did *not* work: shown six NVIDIA passages, it reasonably concluded the question was about NVIDIA.)
2. **`grade`** — is the retrieved evidence good enough to attempt an answer? When reranking has run, its calibrated 0–10 relevance is reused (no second round-trip); otherwise an explicit LLM grader decides. Best passage below 4/10 → refuse.
3. **`premise`** — does the question assume something the filings do not report? The generator must name the fact requested *before* answering, which is what stops it accepting a figure that merely shares a fiscal year. Asked for FY2027 revenue guidance, it previously answered from a 2027 lease-maturity figure — grounded, cited, and wrong.
4. **`verify`** — reads the answer back against its passages, checking two things: every claim supported, **and** that the answer addresses the metric, company and period actually asked about. Grounding alone is not sufficient.

A refusal is never a bare "I don't know": it states *why*, lists the closest material it did find, and names the corpus boundary. Seven of the 24 golden questions exercise this path (four that must be refused, three that must be clarified) — including one deliberately nasty case (Microsoft CEO compensation) where the section heading retrieves beautifully but contains no figures, because Item 11 is incorporated by reference to the proxy statement.

---

## Success criteria

| Metric | Target | Measured | Where it is measured |
| --- | --- | --- | --- |
| Faithfulness | ≥ 0.95 | **1.000** (n=51) | LLM judge, `out/REPORT.md` §5. **Conditional on answering** - measured only on answers the pipeline emitted, so it must be read beside answer rate and wrong-figure count |
| Correct refusal on out-of-corpus questions | 100% | **100%** | 4 golden questions, §5 - one question moves this 25 points |
| False refusal on answerable questions | ≤ 10% | **0.0%** | 17 golden questions, §5 |
| Retrieval hit@6 | ≥ 0.90 | **1.000** | Golden-set evidence matching, §2–4 |
| p95 end-to-end latency | < 6,000 ms | **not demonstrated** (p50 5,445 ms) | §5 |

Measured over three repetitions of the shipped arm (`fixed/hybrid+rerank`), plus
**0 wrong figures in 51 judged answers** - the outcome the four gates exist to
prevent. Every target except the latency tail is met, and the caveat under
§"Honest limitations" in `PROJECT.md` applies: these are training-set numbers on a
24-question set the gates were tuned against.

**On the latency ceiling.** The shipping pipeline makes four sequential model calls (scope → rerank → generate → verify). Two rounds of tuning brought the median under budget — dropping per-passage rationales from the reranker (8.9 s → 2.5 s), splitting the rerank window into concurrent batches (5.3 s → 2.1 s), A third attempt - running the scope gate on a cheaper model - was reverted after it broke behaviour classification. p95 is not demonstrated to be met, and at this sample size it is an extreme order statistic rather than a percentile, so §5 reports p50 and says so.
