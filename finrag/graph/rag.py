"""
The RAG graph.

The refusal path is the design centre, not an afterthought. A financial assistant
that invents a figure is worse than one that says it cannot find it, so several
independent gates stand before an answer is returned:

  scope    did the question say WHICH issuer it means?
  guidance does it ask for a projection a periodic report never contains?
  grade    is the retrieved evidence good enough to attempt this?
  premise  does the question assume something the filings do not report?
  entity   does the answer cite a passage mentioning what was asked about?
  verify   is the answer grounded AND about the fact that was asked?

    START ─┬─> scope ──────────┐
           └─> retrieve ───────┤   (run concurrently)
                              gate ──guidance/ambiguous──> refuse | clarify
                               │
                            rerank -> grade --insufficient--> refuse
                               │
                            generate --false premise--> refuse
                               │
                             verify --ungrounded / wrong entity--> refuse
                               v
                              END
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..config import (GEN_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, P, RETRIEVAL,
                      SCOPE_MODEL, assert_openai)
from ..retrieval.hybrid import named_tickers, retrieve, retrieve_per_entity
from ..retrieval.rerank import rerank


def _append(a: list | None, b: list | None) -> list:
    return (a or []) + (b or [])


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class State(TypedDict, total=False):
    question: str
    options: dict
    hits: list
    contexts: list
    answer: str
    citations: list
    confidence: float
    refused: bool
    refusalReason: str | None
    clarified: bool
    needsClarification: bool
    clarifyingQuestion: str
    requestedFact: str
    verification: dict
    trace: Annotated[list, _append]
    timings: Annotated[dict, _merge]


# ---------------------------------------------------------------- schemas ----
class AnswerOut(BaseModel):
    # Field descriptions are not decoration: they are serialised into the JSON
    # schema sent to the model and steer the decision. Abridging them during the
    # port measurably changed behaviour - see ScopeOut below.
    #
    # Naming the requested fact BEFORE answering is what stops the model from
    # settling for a figure that merely shares a year with the question. Asked
    # for "Apple's fiscal 2027 revenue forecast", it must notice that what it
    # found is a lease-maturity figure that happens to sit in a 2027 column.
    requestedFact: str = Field(
        max_length=160,
        description="the specific fact the question asks for: metric, company, period")
    premiseHolds: bool = Field(
        description="false if the question presupposes something the filings do not support "
                    "(an event that did not occur, or forward guidance a 10-K does not publish)")
    answerable: bool = Field(
        description="false if the passages do not contain the requested fact")
    answer: str = Field(
        description="markdown answer with inline [n] citations, or empty if not answerable")
    citations: list[int] = Field(description="the [n] passage numbers actually used")
    confidence: float = Field(ge=0, le=1)


class GradeOut(BaseModel):
    sufficient: bool
    reason: str = Field(max_length=200)


class VerifyOut(BaseModel):
    grounded: bool = Field(
        description="true only if every factual claim is supported by the passages")
    unsupportedClaims: list[str] = Field(
        description="claims not supported by any passage")
    faithfulness: float = Field(ge=0, le=1)
    # Grounding alone is not enough. An answer can quote a passage perfectly and
    # still answer a different question than the one asked - which is exactly how
    # a request for 2027 guidance got satisfied with a 2027 lease obligation.
    answersTheQuestion: bool = Field(
        description="true only if the answer addresses the metric, company and period "
                    "actually asked about")
    mismatchReason: str = Field(
        default="", max_length=200,
        description="if it does not, what was asked versus what was answered")


class ScopeOut(BaseModel):
    # Dropping these descriptions during the port cost a question. "Which of the
    # three companies grew revenue fastest" read as ambiguous in 3 runs out of 3,
    # because nothing in the bare schema told the model that deliberately spanning
    # all issuers still counts as identifying the subject. The system prompt says
    # so; the schema has to as well.
    namesSubject: bool = Field(
        description="true if the question identifies which company (or explicitly spans all of them)")
    isAmbiguous: bool = Field(
        description="true if answering would require guessing which issuer or period is meant")
    clarifyingQuestion: str = Field(
        default="", max_length=300, description="what to ask back, or empty")


GEN_PROMPT = """You answer questions about SEC filings for financial analysts.

Rules, in priority order:
1. Use ONLY the numbered passages. Never use outside knowledge, never estimate,
   never interpolate a figure that is not printed in a passage.
2. Cite every factual claim inline as [n], matching the passage number.
3. Always state the company and fiscal period a figure belongs to. Filings from
   different issuers and periods look alike; do not blur them.
4. Reproduce figures exactly as printed, including units and scale (the tables
   are usually "in millions"). Do not re-scale or round.
5. If the passages do not contain the answer, set answerable=false.

Before answering, state requestedFact: the exact metric, company and period being
asked for. Then check the premise:

PREMISE. Set premiseHolds=false when the question assumes something these filings
  do not support - a transaction that did not happen, or forward guidance, which
  a 10-K does not publish. A figure is NOT an answer merely because it shares a
  year with the question. If asked for fiscal 2027 revenue guidance, a 2027 lease
  or debt maturity is a different fact. Never substitute an adjacent number."""

SCOPE_PROMPT = """A corpus holds the latest 10-K and 10-Q for exactly three issuers: Apple,
Microsoft and NVIDIA. Their fiscal years do not align (Apple FY2025 ended
Sept 2025, Microsoft FY2026 ended June 2026, NVIDIA FY2026 ended Jan 2026).

Decide whether the question can be answered without guessing WHICH issuer is
meant. Judge the question text only - you are not shown any documents.

This gate is about WHICH COMPANY only. Never flag a question over its period.
Stating the right period is the job of the answering step. Flagging it here
turned fully specified questions - "Microsoft total revenue for the three
months ended March 31, 2026" - into needless clarifications.

isAmbiguous=true ONLY when the question refers to a company merely as "they",
  "the company", or not at all, e.g. "what was revenue last year?".
isAmbiguous=false when it names a company, OR deliberately spans all of them
  ("which of the three grew fastest", "compare X and Y").

A company named but ABSENT from the corpus (e.g. Tesla) is NOT ambiguous.
  The subject is perfectly clear - the corpus simply cannot answer it, and a
  later step declines for that reason. Asking "which of the three did you
  mean?" would be nonsense: the user told you exactly who they meant.

Picking an issuer the user never named produces a confidently wrong answer,
so prefer asking when genuinely torn."""

def _warm_openai_imports() -> None:
    """
    Force the openai SDK's lazy submodule imports at module load.

    `scope` and `retrieve` run concurrently, and both reach the SDK. The client
    imports `openai.resources.chat` / `.embeddings` lazily on first use, so two
    threads hitting it simultaneously raced on Python's import lock and raised
    `_DeadlockError: deadlock detected by _ModuleLock('openai.resources.embeddings')`.

    Importing them here - single-threaded, before the graph can fan out - removes
    the race. This is a Python-specific hazard the JavaScript original does not
    have, because ES module resolution is not lock-based.
    """
    try:
        import openai.resources.chat  # noqa: F401
        import openai.resources.embeddings  # noqa: F401
    except Exception:  # pragma: no cover - warming is best-effort
        pass


_warm_openai_imports()

_models: dict[str, Any] = {}


def _chat(schema, name: str, model_name: str | None = None):
    assert_openai()
    key = f"{name}:{model_name or GEN_MODEL}"
    if key not in _models:
        kwargs = {"api_key": OPENAI_API_KEY, "model": model_name or GEN_MODEL,
                  "temperature": 0, "max_retries": 2, "timeout": 60}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _models[key] = ChatOpenAI(**kwargs).with_structured_output(schema)
    return _models[key]


def _format_passages(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{i + 1}] {h['metadata'].get('company')} ({h['metadata'].get('ticker')}) "
        f"{h['metadata'].get('form')}, filed {h['metadata'].get('filingDate')}\n"
        f"Section: {h['metadata'].get('section')}\n{h['text']}"
        for i, h in enumerate(hits)
    )


# ------------------------------------------------------- deterministic gates --
GUIDANCE_WORD = re.compile(r"\b(forecasts?|guidance|projections?|projected|outlook|forward[- ]looking estimates?)\b", re.I)
# Filings are full of legitimate uses of these words - "projected future minimum
# lease payments", "guidance on revenue recognition". Matching the word alone
# refused all of them, trading a false-answer failure for a false-refusal one.
ACCOUNTING_CONTEXT = re.compile(
    r"\b(leases?|obligations?|maturit\w*|credit loss\w*|allowance|amorti[sz]\w*|depreciat\w*"
    r"|payments? due|contractual|commitments?|unrecognized|deferred|recognition|policy|accounting)\b", re.I)
NEXT_PERIOD = re.compile(r"\b(next|upcoming|coming)\s+(fiscal\s+)?(year|quarter)\b", re.I)

_horizon: int | None = None


def _corpus_horizon_year() -> int:
    """Latest period the corpus covers; anything past it is a projection."""
    global _horizon
    if _horizon is not None:
        return _horizon
    years: list[int] = []
    mf = P.processed / "_manifest.json"
    if mf.exists():
        for d in json.loads(mf.read_text(encoding="utf-8")).get("documents", []):
            raw = str(d.get("periodOfReport") or d.get("filingDate") or "")[:4]
            if raw.isdigit():
                years.append(int(raw))
    _horizon = max(years) if years else datetime.now().year
    return _horizon


def asks_for_guidance(question: str) -> bool:
    """
    A request for management's projection of FUTURE performance.

    The discriminating signal is not the word - it is a period beyond what the
    filings cover. "Revenue forecast for fiscal 2027" is unanswerable; "projected
    future minimum lease payments" is a printed line item.
    """
    if not GUIDANCE_WORD.search(question):
        return False
    if ACCOUNTING_CONTEXT.search(question):
        return False
    if NEXT_PERIOD.search(question):
        return True
    horizon = _corpus_horizon_year()
    return any(int(y) > horizon for y in re.findall(r"\b(?:19|20)\d{2}\b", question))


NOT_AN_ENTITY = {
    "what", "how", "much", "many", "did", "does", "the", "and", "which", "who", "when",
    "where", "was", "were", "are", "compare", "total", "net", "sales", "revenue",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "item", "part", "gaap", "usd", "inc", "corp",
}


def question_entities(question: str) -> list[str]:
    found = re.findall(r"\b[A-Z][A-Za-z&.'’]{2,}\b", question)
    out, seen = [], set()
    for w in found:
        e = re.sub(r"['’]s$", "", w).lower()
        if len(e) > 2 and e not in NOT_AN_ENTITY and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def unsupported_entities(question: str, contexts: list[dict] | None) -> list[str]:
    """
    Entities named in the question that appear in NO cited passage.

    Asked "What price did NVIDIA pay to acquire Intel's foundry business?" - an
    acquisition that never happened - retrieval surfaced a REAL NVIDIA acquisition
    disclosure mentioning neither Intel nor foundry, and the model reported its
    $17.0 billion as the price of the imagined deal. Every model gate passed it:
    the figures genuinely are in the passage, and the passage genuinely is about an
    acquisition price. Nothing asked whether it was about *Intel*.

    Metadata counts as presence: a chunk from Apple's 10-K is about Apple even
    where the prose never says so.
    """
    hay = " ".join(
        f"{c.get('text','')} {c.get('metadata',{}).get('company','')} {c.get('metadata',{}).get('ticker','')}"
        for c in (contexts or [])
    ).lower()
    if not hay.strip():
        return []
    return [e for e in question_entities(question) if e not in hay]


# ------------------------------------------------------------------ nodes ----
def node_scope(state: State) -> dict:
    t = time.time()
    q = state["question"]

    # Periodic reports state what HAPPENED; they do not publish guidance. That is a
    # property of the document type, not a judgement, and it is enforced in code
    # because the model gates could not hold it.
    if asks_for_guidance(q):
        return {
            "refusalReason": ("This corpus holds 10-K and 10-Q filings, which report results already "
                              "achieved. They do not contain management forecasts or guidance."),
            "trace": [{"node": "scope", "ms": int((time.time() - t) * 1000),
                       "detail": "asks for guidance - not present in periodic reports"}],
        }

    res = _chat(ScopeOut, "scope", SCOPE_MODEL).invoke([
        {"role": "system", "content": SCOPE_PROMPT},
        {"role": "user", "content": f"QUESTION: {q}"},
    ])
    ms = int((time.time() - t) * 1000)
    return {
        "needsClarification": res.isAmbiguous,
        "clarifyingQuestion": res.clarifyingQuestion,
        "timings": {"scope": ms},
        "trace": [{"node": "scope", "ms": ms,
                   "detail": f"ambiguous - {res.clarifyingQuestion}" if res.isAmbiguous else "subject identified"}],
    }


def node_retrieve(state: State) -> dict:
    t = time.time()
    opts = state.get("options") or {}
    strategy = opts.get("strategy", "semantic")
    mode = opts.get("mode", "hybrid")

    tickers = named_tickers(state["question"])
    multi = len(tickers) >= 2

    if multi:
        per = max(RETRIEVAL.min_per_entity, -(-RETRIEVAL.rerank_window // len(tickers)))
        r = retrieve_per_entity(state["question"], strategy=strategy, mode=mode,
                                tickers=tickers, per_entity=per)
    else:
        r = retrieve(state["question"], strategy=strategy, mode=mode)

    ms = int((time.time() - t) * 1000)
    spread = f" [per-entity: {'/'.join(tickers)}]" if multi else ""
    return {
        "hits": r["hits"],
        "timings": {"retrieve": ms},
        "trace": [{"node": "retrieve", "ms": ms,
                   "detail": f"{mode}/{strategy} -> {len(r['hits'])} candidates "
                             f"(dense {r['counts']['dense']}, sparse {r['counts']['sparse']}){spread}"}],
    }


def node_gate(state: State) -> dict:
    return {}


def node_rerank(state: State) -> dict:
    t = time.time()
    if not (state.get("options") or {}).get("rerank", True):
        hits = state["hits"][:RETRIEVAL.top_k]
        return {"hits": hits, "trace": [{"node": "rerank", "ms": 0,
                                         "detail": f"skipped - top {len(hits)} by fusion score"}]}

    tickers = named_tickers(state["question"])
    entities = max(1, len(tickers))
    # A comparison needs facts from several filings at once, so six slots shared
    # between three issuers leaves two each.
    keep = min(RETRIEVAL.rerank_to * entities, RETRIEVAL.max_rerank_to)
    window = max(RETRIEVAL.rerank_window, len(state["hits"]) if entities > 1 else 0)

    r = rerank(state["question"], state["hits"][:window],
               top_n=keep, groups=tickers if entities > 1 else [])
    ms = int((time.time() - t) * 1000)

    if r.get("degraded"):
        detail = f"DEGRADED ({r.get('error')}) - fused order kept"
    else:
        detail = f"{len(r['hits'])} kept, top score {r['hits'][0]['rerankScore'] if r['hits'] else 'n/a'}"
        if "partial" in r:
            detail += f" | PARTIAL: {r['error']}"
    return {"hits": r["hits"], "timings": {"rerank": ms},
            "trace": [{"node": "rerank", "ms": ms, "detail": detail}]}


def node_grade(state: State) -> dict:
    t = time.time()
    hits = state.get("hits") or []
    if not hits:
        return {"refusalReason": "Nothing in the indexed filings matched this question.",
                "trace": [{"node": "grade", "ms": 0, "detail": "no candidates"}]}

    # When reranking ran we already have calibrated relevance - reusing it avoids a
    # second round-trip. Without it, grade explicitly.
    if hits[0].get("rerankScore") is not None:
        best = hits[0]["rerankScore"]
        ok = best >= RETRIEVAL.grade_threshold
        return {
            "refusalReason": None if ok else
                f"Best passage scored {best}/10 for relevance, below the {RETRIEVAL.grade_threshold}/10 bar.",
            "trace": [{"node": "grade", "ms": 0,
                       "detail": f"rerank-derived: best={best} -> {'answer' if ok else 'refuse'}"}],
        }

    res = _chat(GradeOut, "grade").invoke([
        {"role": "system", "content": "Decide whether the passages contain enough information to "
                                      "answer the question. Be strict about company and fiscal period."},
        {"role": "user", "content": f"QUESTION: {state['question']}\n\nPASSAGES:\n"
                                    f"{_format_passages(hits[:RETRIEVAL.top_k])}"},
    ])
    ms = int((time.time() - t) * 1000)
    return {"refusalReason": None if res.sufficient else res.reason,
            "timings": {"grade": ms},
            "trace": [{"node": "grade", "ms": ms,
                       "detail": f"llm: {'answer' if res.sufficient else 'refuse'} - {res.reason}"}]}


def node_generate(state: State) -> dict:
    t = time.time()
    used = state["hits"]
    res = _chat(AnswerOut, "answer").invoke([
        {"role": "system", "content": GEN_PROMPT},
        {"role": "user", "content": f"QUESTION: {state['question']}\n\nPASSAGES:\n{_format_passages(used)}"},
    ])
    ms = int((time.time() - t) * 1000)

    if not res.premiseHolds:
        return {"refusalReason": f"The question assumes something these filings do not report "
                                 f"({res.requestedFact}). A figure that merely shares a fiscal year "
                                 f"is not the same fact.",
                "contexts": used, "timings": {"generate": ms},
                "trace": [{"node": "generate", "ms": ms, "detail": f"premise rejected - {res.requestedFact}"}]}
    if not res.answerable:
        return {"refusalReason": "The retrieved passages did not contain the answer.",
                "contexts": used, "timings": {"generate": ms},
                "trace": [{"node": "generate", "ms": ms, "detail": "model declined - not answerable"}]}

    citations = [{"n": n, **used[n - 1]["metadata"], "id": used[n - 1]["id"]}
                 for n in res.citations if 1 <= n <= len(used)]
    return {"answer": res.answer, "citations": citations, "confidence": res.confidence,
            "requestedFact": res.requestedFact, "contexts": used, "timings": {"generate": ms},
            "trace": [{"node": "generate", "ms": ms,
                       "detail": f"{len(res.answer)} chars, {len(citations)} citations, conf {res.confidence}"}]}


def node_verify(state: State) -> dict:
    t = time.time()
    res = _chat(VerifyOut, "verify").invoke([
        {"role": "system", "content":
            "Check the ANSWER against the PASSAGES. A claim is supported only if a passage states it. "
            "Numbers must match exactly, including scale.\n\n"
            "Then check it answers the QUESTION asked: same metric, same company, same fiscal period. "
            "An answer can quote a passage perfectly and still be about a different fact. "
            "Set answersTheQuestion=false in that case."},
        {"role": "user", "content": f"QUESTION:\n{state['question']}\n\nPASSAGES:\n"
                                    f"{_format_passages(state['contexts'])}\n\nANSWER:\n{state['answer']}"},
    ])
    ms = int((time.time() - t) * 1000)

    # Checked before the model's own verdicts, because it is the only one of the
    # three that cannot be talked round.
    missing = unsupported_entities(state["question"], state["contexts"])

    if missing:
        reason = (f"The answer cites no passage mentioning {', '.join(missing)}, which the question "
                  f"asks about. A figure from a different subject is not an answer about this one.")
    elif not res.grounded:
        reason = f"Answer failed the grounding check: {'; '.join(res.unsupportedClaims)}"
    elif not res.answersTheQuestion:
        reason = f"Answer was grounded but addressed a different fact: {res.mismatchReason}"
    else:
        reason = None

    detail = (f"grounded={res.grounded} answersQuestion={res.answersTheQuestion} "
              f"faithfulness={res.faithfulness}")
    if missing:
        detail += f" | ENTITY MISMATCH: {', '.join(missing)} absent from cited passages"

    return {"verification": res.model_dump(), "timings": {"verify": ms},
            "refusalReason": reason,
            "trace": [{"node": "verify", "ms": ms, "detail": detail}]}


def node_clarify(state: State) -> dict:
    issuers = sorted({c["metadata"].get("ticker") for c in (state.get("contexts") or state.get("hits") or [])
                      if c.get("metadata", {}).get("ticker")})
    extra = ""
    if len(issuers) > 1:
        extra = (f"\nThe passages I found span {', '.join(issuers)}, and their fiscal years do not "
                 "line up, so picking one for you would risk quoting the wrong period.\n")
    return {
        "clarified": True,
        "answer": ("I need one more detail before I can answer.\n\n"
                   f"**{state.get('clarifyingQuestion') or 'Which company and fiscal period do you mean?'}**\n"
                   f"{extra}\nThe corpus covers the latest 10-K and 10-Q for AAPL, MSFT and NVDA."),
        "citations": [], "confidence": 0,
        "trace": [{"node": "clarify", "ms": 0, "detail": state.get("clarifyingQuestion") or "underspecified"}],
    }


def node_refuse(state: State) -> dict:
    near = "\n".join(
        f"- {h['metadata'].get('ticker')} {h['metadata'].get('form')} "
        f"({h['metadata'].get('filingDate')}), {h['metadata'].get('section')}"
        for h in (state.get("hits") or [])[:3]
    )
    return {
        "refused": True,
        "answer": ("I could not answer this from the indexed filings.\n\n"
                   f"**Why:** {state.get('refusalReason') or 'insufficient supporting evidence.'}\n"
                   + (f"\n**Closest material I did find:**\n{near}\n" if near else "")
                   + "\nThe corpus covers the latest 10-K and 10-Q for AAPL, MSFT and NVDA only."),
        "citations": [], "confidence": 0,
        "trace": [{"node": "refuse", "ms": 0, "detail": state.get("refusalReason") or "insufficient evidence"}],
    }


# ------------------------------------------------------------------ graph ----
def _route_gate(s: State) -> str:
    if s.get("refusalReason"):
        return "refuse"
    return "clarify" if s.get("needsClarification") else "rerank"


_builder = StateGraph(State)
_builder.add_node("scope", node_scope)
_builder.add_node("retrieve", node_retrieve)
_builder.add_node("gate", node_gate)
_builder.add_node("rerank", node_rerank)
_builder.add_node("grade", node_grade)
_builder.add_node("generate", node_generate)
_builder.add_node("verify", node_verify)
_builder.add_node("clarify", node_clarify)
_builder.add_node("refuse", node_refuse)

# `scope` and `retrieve` fan out from START and run concurrently: scope needs only
# the question, so making retrieval wait for it put two independent calls in series
# for no reason.
_builder.add_edge(START, "scope")
_builder.add_edge(START, "retrieve")
_builder.add_edge("scope", "gate")
_builder.add_edge("retrieve", "gate")
_builder.add_conditional_edges("gate", _route_gate,
                               {"refuse": "refuse", "clarify": "clarify", "rerank": "rerank"})
_builder.add_edge("rerank", "grade")
_builder.add_conditional_edges("grade", lambda s: "refuse" if s.get("refusalReason") else "generate",
                               {"refuse": "refuse", "generate": "generate"})
_builder.add_conditional_edges("generate", lambda s: "refuse" if s.get("refusalReason") else "verify",
                               {"refuse": "refuse", "verify": "verify"})
_builder.add_conditional_edges("verify", lambda s: "refuse" if s.get("refusalReason") else END,
                               {"refuse": "refuse", END: END})
_builder.add_edge("clarify", END)
_builder.add_edge("refuse", END)

graph = _builder.compile()


def ask(question: str, options: dict | None = None) -> dict:
    t = time.time()
    out = graph.invoke({"question": question, "options": options or {},
                        "refused": False, "clarified": False})
    out["totalMs"] = int((time.time() - t) * 1000)
    return out
