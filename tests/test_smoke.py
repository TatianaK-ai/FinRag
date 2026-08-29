"""
End-to-end smoke tests, run entirely offline against the bundled mock API.

    pytest                    (assumes `python -m finrag.build` produced the indexes)
    python tests/offline.py   (builds against the mock first, then runs this)

These assert the pipeline's *behaviour* - cleaning removes XBRL, tables stay
intact, hybrid retrieval fuses both lists, the graph answers grounded questions
and refuses out-of-corpus ones. They deliberately assert nothing about answer
quality, which the mock cannot speak to.
"""
from __future__ import annotations

import importlib
import os
import re

import pytest

from finrag.chunking.common import table_caption
from finrag.graph.rag import ask, question_entities, unsupported_entities, asks_for_guidance
from finrag.ingest.clean import clean_filing
from finrag.retrieval.hybrid import named_tickers, retrieve, retrieve_per_entity, rrf
from finrag.store.bm25 import build_bm25, search_bm25, tokenize
from finrag.store.index import chunk_map, fingerprint_chunks

OPTS = {"strategy": "fixed", "mode": "hybrid", "rerank": True}


# --- unit-ish: cleaning ------------------------------------------------------

def test_cleaning_strips_xbrl_and_keeps_figures():
    sentence = "Total net sales increased during the period. " * 4
    html = f"""<html><body>
      <ix:header><ix:hidden>0000320193us-gaap:CommonStockMember2022-09-24</ix:hidden></ix:header>
      <p>Item 7. Management's Discussion</p>
      <p>{sentence}</p>
      <table><tr><td></td><td>2025</td><td>2024</td></tr>
        <tr><td>Total net sales</td><td>$</td><td>416,161</td><td>$</td><td>391,035</td></tr>
      </table></body></html>"""

    blocks, stats = clean_filing(html)
    allt = "\n".join(b.text for b in blocks)

    assert "us-gaap:" not in allt, "XBRL concept names must not survive cleaning"
    assert stats["xbrlBytesRemoved"] > 0
    assert "416,161" in allt, "reported figures must survive"
    assert len([b for b in blocks if b.type == "table"]) == 1
    assert any(b.section.startswith("Item 7") for b in blocks), "section label attached"


def test_table_compaction_folds_currency_glyphs():
    html = """<html><body><table>
      <tr><td></td><td>2025</td><td>2024</td></tr>
      <tr><td>Americas</td><td>$</td><td>178,353</td><td>$</td><td>167,045</td></tr>
      <tr><td>Europe</td><td>111,032</td><td></td><td>101,328</td><td></td></tr>
    </table></body></html>"""
    blocks, _ = clean_filing(html)
    table = next((b for b in blocks if b.type == "table"), None)
    assert table is not None
    assert "$178,353" in table.text, "currency glyph folded onto its figure"
    assert not re.search(r"\|\s*\$\s*\|", table.text), "no orphan $ column remains"


def test_lone_dash_is_a_value_not_a_prefix():
    # Regression: the em dash (nil marker) used to weld onto the next figure,
    # corrupting two cells at once.
    html = """<html><body><table>
      <tr><td></td><td>2025</td><td>Change</td><td>2024</td></tr>
      <tr><td>iPhone</td><td>$</td><td>209,586</td><td>&#8212;</td><td>%</td><td>$</td><td>201,183</td></tr>
    </table></body></html>"""
    blocks, _ = clean_filing(html)
    t = next(b.text for b in blocks if b.type == "table")
    assert "$209,586" in t
    assert "$201,183" in t, "the figure after the dash must stay intact"
    assert "209,586%" not in t and "201,183%" not in t, "no figure welded to a percent sign"


# --- unit-ish: BM25 ----------------------------------------------------------

def test_bm25_tokenizer_keeps_figures_and_item_refs_whole():
    toks = tokenize("Item 1A. Total net sales were $416,161 million, up 6% in 10-K.")
    assert "416161" in toks, "comma-separated figure kept as one token"
    assert "1a" in toks, "item reference preserved"
    assert "10-k" in toks, "form type preserved"
    assert "the" not in toks, "stopwords dropped"


def test_bm25_ranks_the_exact_figure_first():
    chunks = [
        {"id": "a", "text": "Total net sales were $416,161 million in fiscal 2025."},
        {"id": "b", "text": "Net sales increased across most segments during the year."},
        {"id": "c", "text": "Research and development expense was $34,550 million."},
    ]
    hits = search_bm25(build_bm25(chunks), "416,161 total net sales", 3)
    assert hits[0]["id"] == "a"


# --- unit-ish: fusion --------------------------------------------------------

def test_rrf_rewards_documents_in_both_lists():
    dense = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    sparse = [{"id": "q"}, {"id": "z"}, {"id": "w"}]
    fused = rrf([dense, sparse])
    assert fused[0]["id"] == "z", "the doc in both lists wins despite being rank 3 and 2"
    assert all(f["score"] <= fused[i - 1]["score"] for i, f in enumerate(fused) if i)


# --- unit-ish: captions ------------------------------------------------------

def test_table_caption_names_line_items_and_periods():
    md = "\n".join([
        "| (In millions) | 2026 | 2025 |",
        "| --- | --- | --- |",
        "| Revenue | $331,839 | $281,724 |",
        "| Gross margin | 225,465 | 193,893 |",
    ])
    cap = table_caption(md, {"company": "Microsoft Corporation", "ticker": "MSFT", "form": "10-K"})

    # The gap this closes: the table says "2026", a question says "fiscal year 2026".
    assert "fiscal year 2026" in cap, "periods spelled out, not left as bare years"
    assert "Microsoft Corporation" in cap, "company named in the caption"
    assert "Revenue" in cap, "line items listed"
    assert "331,839" not in cap, "caption describes structure, it does not restate figures"
    assert "In millions" not in cap, "units note is not a line item"


# --- unit-ish: entity detection ---------------------------------------------

def test_named_issuers_are_detected():
    assert sorted(named_tickers("Compare Apple and NVIDIA revenue")) == ["AAPL", "NVDA"]
    assert sorted(named_tickers("Microsoft or NVIDIA, which is bigger?")) == ["MSFT", "NVDA"]
    # Names nobody explicitly, but means everyone.
    assert len(named_tickers("Which of the three companies grew fastest?")) == 3
    # A single-issuer question must NOT trigger the per-entity path.
    assert named_tickers("What were Apple's total net sales in fiscal 2025?") == ["AAPL"]


def test_company_names_match_on_word_boundaries():
    # "pineapple" used to resolve to AAPL and push an unrelated question down the
    # per-entity comparison path.
    assert named_tickers("What is the pineapple revenue?") == []
    assert named_tickers("Tell me about apple products") == ["AAPL"]
    assert named_tickers("What is NVDA revenue?") == ["NVDA"]


def test_naming_one_issuer_never_widens_scope_to_all_three():
    # The "means all three" fallback fired on `< 2`, so a question naming exactly
    # one company resolved to three - and the group quota then forced the other
    # two issuers' passages into context below the relevance floor.
    assert named_tickers("Which company had higher revenue than Apple?") == ["AAPL"]
    assert named_tickers("Of the three companies, what was Apple total net sales?") == ["AAPL"]
    # Genuinely unspecified still means all three.
    assert len(named_tickers("Which company grew fastest?")) == 3
    assert len(named_tickers("Compare Apple and NVIDIA")) == 2


def test_guidance_guard_fires_only_on_requests_for_future_performance():
    # An earlier version of this test called retrieve() directly - a layer the
    # guard does not run in - so it passed green while covering none of the risk
    # it names. The guard lives in the scope node, so it is tested there.
    projections = [
        "What is Apple's revenue forecast for fiscal 2027?",
        "What is the revenue forecast for next year?",
        "What is NVIDIA earnings guidance for fiscal 2028?",
    ]
    legitimate = [
        "What are the projected future minimum lease payments?",    # a printed line item
        "What guidance does the 10-K give on revenue recognition policy?",
        "What does Apple say about its outlook for Greater China?",  # risk factors
        "What are projected lease payments in 2028?",                # future year, real schedule
        "What were Apple's total net sales in fiscal 2025?",
    ]
    for q in projections:
        assert asks_for_guidance(q) is True, f"should block: {q}"
    for q in legitimate:
        assert asks_for_guidance(q) is False, f"must NOT block: {q}"


def test_asked_about_entity_must_appear_in_the_cited_passages():
    # The q24 failure: a real NVIDIA acquisition passage, no mention of Intel, and
    # the model reported its price as the price of an imagined Intel deal.
    nvidia_acquisition = [{
        "text": "Total consideration consists of $13.0 billion paid at closing and "
                "$4 billion payable within one year.",
        "metadata": {"company": "NVIDIA Corporation", "ticker": "NVDA"},
    }]
    assert unsupported_entities(
        "What price did NVIDIA pay to acquire Intel's foundry business?",
        nvidia_acquisition) == ["intel"]

    # Issuer identity may live only in metadata - the prose often never says it.
    assert unsupported_entities(
        "What were Apple's total net sales in fiscal 2025?",
        [{"text": "Total net sales were $416,161 million.",
          "metadata": {"company": "Apple Inc.", "ticker": "AAPL"}}]) == []

    # Months, item numbers and question words are not entities.
    ents = question_entities(
        "What was Microsoft's total revenue in the three months ended March 31, 2026?")
    assert "microsoft" in ents
    assert "march" not in ents, "month names must not be treated as entities"

    # No contexts at all must not produce a spurious flag.
    assert unsupported_entities("What did Intel pay?", []) == []


# --- integration: retrieval and the graph ------------------------------------

def test_grounded_question_is_answered_with_citations(built):
    r = ask("What were Apple's total net sales in fiscal 2025?", OPTS)
    assert r["refused"] is False, "should not refuse a question the corpus answers"
    assert len(r["citations"]) >= 1, "must cite at least one passage"
    assert all(c["sourceUrl"].startswith("https://www.sec.gov/") for c in r["citations"])
    assert any(t["node"] == "verify" for t in r["trace"]), "grounding check must run"


def test_out_of_corpus_question_is_refused(built):
    r = ask("How much did Tesla spend on R&D in 2025?", OPTS)
    assert r["refused"] is True, "must refuse - Tesla is not in the corpus"
    assert r["citations"] == [], "a refusal must not cite anything"
    assert re.search(r"could not answer", r["answer"], re.I)
    assert any(t["node"] == "refuse" for t in r["trace"])


def test_hybrid_retrieval_draws_on_both_lists(built):
    res = retrieve("Apple net sales fiscal 2025", strategy="fixed", mode="hybrid")
    hits, counts = res["hits"], res["counts"]

    assert counts["dense"] > 0 and counts["sparse"] > 0
    assert hits
    # The invariant that holds regardless of embedding quality: RRF must carry
    # candidates through from *both* lists. How much the two lists agree is a
    # quality signal that depends on the embedding model, so it is measured in the
    # eval (nDCG per arm), not asserted here - under the mock's hashed embeddings
    # the two lists legitimately share nothing.
    assert any("dense" in h["sources"] for h in hits), "dense candidates survive fusion"
    assert any("sparse" in h["sources"] for h in hits), "sparse candidates survive fusion"
    assert all(h["fusedScore"] <= hits[i - 1]["fusedScore"] for i, h in enumerate(hits) if i)


def test_every_chunk_carries_provenance(built):
    for c in list(chunk_map("fixed").values())[:200]:
        m = c["metadata"]
        assert m["ticker"] and m["form"] and m["filingDate"] and m["section"]
        assert m["sourceUrl"].startswith("https://www.sec.gov/")


def test_underspecified_question_is_clarified(built):
    r = ask("What was revenue last year?", OPTS)
    assert r["clarified"] is True, "must ask which issuer rather than pick one"
    assert r["refused"] is False, "asking back is not a refusal"
    assert r["citations"] == [], "nothing to cite yet"
    assert any(t["node"] == "scope" for t in r["trace"])
    # Retrieval runs in parallel with scope and its result is thrown away; the
    # assertion that matters is that the question was clarified, not answered.
    assert any(t["node"] == "clarify" for t in r["trace"])


def test_cross_corpus_question_is_not_treated_as_ambiguous(built):
    r = ask("Compare Apple and NVIDIA revenue", OPTS)
    assert r["clarified"] is False, "naming both issuers is not ambiguous"


def test_fully_specified_question_is_never_clarified_over_its_period(built):
    r = ask("What was Microsoft's total revenue in the three months ended March 31, 2026?", OPTS)
    assert r["clarified"] is False, "company and period are both stated - nothing to ask"


def test_request_for_guidance_is_refused_deterministically(built):
    r = ask("What is Apple's revenue forecast for fiscal 2027?", OPTS)
    assert r["refused"] is True, "10-K/10-Q do not publish guidance"
    assert r["clarified"] is False, "the subject is clear - a refusal, not an ambiguity"
    # Must not depend on a model call: both LLM gates previously waved this
    # through, answering with a 2027 figure lifted from an obligations table.
    # `scope` and `retrieve` run concurrently for latency, so retrieval does happen
    # and its result is discarded. What must hold is that SCOPE made the decision.
    assert any(t["node"] == "scope" for t in r["trace"])
    assert r["citations"] == []
    assert re.search(r"do not contain management forecasts|guidance", r["answer"], re.I)


def test_allowed_forward_looking_question_reaches_retrieval(built):
    # Must name an issuer. Without one the scope gate correctly asks which company
    # and the question never reaches retrieval - which is right, but says nothing
    # about the guidance guard. The first version of this test omitted the company
    # and then blamed the guard for a clarification.
    r = ask("What are Apple's projected future minimum lease payments?", OPTS)
    assert r["clarified"] is False, "the issuer is named, so nothing to clarify"
    # It may or may not find the figure; what matters is the guard let it through.
    assert any(t["node"] == "retrieve" for t in r["trace"]), \
        "a real reported line item must not be blocked by the guidance guard"


def test_per_entity_retrieval_reports_real_timings(built):
    r = retrieve_per_entity("Compare Apple and NVIDIA revenue", strategy="fixed", mode="hybrid",
                            tickers=["AAPL", "NVDA"], per_entity=10)
    # An assertion that only checked `isinstance(timings["sparse"], int)` would
    # pass on the zero-initialised dict even if every sub-retrieval reported
    # nothing.
    assert r["timings"]["sparse"] > 0 or r["timings"]["dense"] > 0
    assert r["hits"]


def test_comparison_retrieves_both_sides(built):
    hits = retrieve_per_entity("Which had higher revenue, Microsoft or NVIDIA?",
                               strategy="fixed", mode="hybrid",
                               tickers=["MSFT", "NVDA"], per_entity=15)["hits"]
    seen = {h["metadata"]["ticker"] for h in hits}
    # A single pass measured 27 NVDA / 2 MSFT / 1 AAPL and missed the Microsoft
    # figure entirely; reranking cannot recover what retrieval never surfaced.
    assert {"MSFT", "NVDA"} <= seen, "both issuers represented"
    msft = sum(1 for h in hits if h["metadata"]["ticker"] == "MSFT")
    nvda = sum(1 for h in hits if h["metadata"]["ticker"] == "NVDA")
    assert min(msft, nvda) >= 5, f"neither side starved (MSFT {msft}, NVDA {nvda})"


def test_stale_index_is_refused_rather_than_mis_serving_text(built):
    chunks = list(chunk_map("fixed").values())
    # The fingerprint must be sensitive to TEXT, not just count - the dangerous
    # state is an id that still resolves but whose text changed, so a vector
    # selected on the old text serves the new one under the same id.
    a = fingerprint_chunks(chunks)
    mutated = [{**c, "text": c["text"] + " edited"} if i == 0 else c
               for i, c in enumerate(chunks)]
    assert fingerprint_chunks(mutated) != a, "changing one chunk must change the fingerprint"
    assert fingerprint_chunks(chunks[:-1]) != a, "dropping a chunk must change the fingerprint"
    assert a.startswith(f"{len(chunks)}:"), "fingerprint carries the chunk count"


def test_retrieval_reports_orphaned_ids(built):
    counts = retrieve("Apple net sales fiscal 2025", strategy="fixed", mode="hybrid")["counts"]
    # An index that returns ids absent from the chunk store used to look identical
    # to a slightly worse result set; now the count is always present.
    assert isinstance(counts["orphaned"], int), "orphan count is surfaced"
    assert counts["orphaned"] == 0, "a freshly built index has no orphans"


def test_unknown_retrieval_mode_raises_rather_than_refusing(built):
    # A typo'd mode used to match no branch, leave both lists empty, and surface as
    # "Nothing in the indexed filings matched" - a confident refusal caused by a
    # spelling mistake.
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        retrieve("Apple net sales", strategy="fixed", mode="hybird")


# --- guards ------------------------------------------------------------------

def _reload_guard(**env):
    """guard.py reads its limits at import time, so each case needs a fresh one."""
    from finrag.middleware import guard
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items()})
    try:
        return importlib.reload(guard)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_auth_rejects_a_wrong_key_and_accepts_the_right_one():
    g = _reload_guard(APP_API_KEY="unit-test-key")
    try:
        assert g.key_matches("unit-test-key") is True
        assert g.key_matches("wrong") is False
        # A wrong key of the SAME length must also fail - guards against a compare
        # that only checks length.
        assert g.key_matches("unit-test-kex") is False
        assert g.key_matches(None) is False, "no credentials at all is rejected"
        assert g.auth_enabled() is True
        assert g.bind_host() != "127.0.0.1" or os.getenv("BIND_HOST") == "127.0.0.1"
    finally:
        _reload_guard()


def test_no_key_configured_binds_to_loopback_only():
    g = _reload_guard(APP_API_KEY="")
    try:
        assert g.auth_enabled() is False
        # An unauthenticated endpoint that anyone can reach is a billing liability.
        assert g.bind_host() == "127.0.0.1"
        assert g.key_matches("anything") is False, "no key configured accepts nothing"
    finally:
        _reload_guard()


def test_rate_limiter_allows_exactly_the_budget_then_refuses():
    g = _reload_guard(RATE_LIMIT_PER_MINUTE="3", RATE_LIMIT_PER_DAY="100")
    try:
        assert g.check_rate("10.0.0.1")[0] is True
        assert g.check_rate("10.0.0.1")[0] is True
        assert g.check_rate("10.0.0.1")[0] is True
        ok, msg, retry = g.check_rate("10.0.0.1")
        assert ok is False, "the fourth request exceeds a budget of three"
        assert "requests/minute" in msg
        assert retry >= 0
        # The per-minute limit is per client, so a different caller is unaffected.
        assert g.check_rate("10.0.0.2")[0] is True
    finally:
        _reload_guard()


def test_daily_cap_bounds_total_spend_across_all_clients():
    g = _reload_guard(RATE_LIMIT_PER_MINUTE="1000", RATE_LIMIT_PER_DAY="2")
    try:
        assert g.check_rate("a")[0] is True
        assert g.check_rate("b")[0] is True
        ok, msg, _ = g.check_rate("c")
        assert ok is False, "the daily cap is global, not per client"
        assert "daily request cap" in msg
    finally:
        _reload_guard()
