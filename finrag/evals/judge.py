"""
LLM-as-judge for generation quality.

Runs on a STRONGER model than the generator so the evaluation is not the
generator grading its own homework. Faithfulness is scored against the retrieved
passages only; correctness against the hand-written gold answer. Keeping those
separate matters - an answer can be perfectly faithful to a passage that did not
contain what was asked for.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config import JUDGE_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, assert_openai


class JudgeOut(BaseModel):
    faithfulness: float = Field(ge=0, le=1, description="fraction of claims supported by the passages")
    unsupported: list[str]
    correctness: float = Field(ge=0, le=1, description="agreement with the gold answer")
    relevance: float = Field(ge=0, le=1)
    citationsValid: bool
    verdict: str = Field(max_length=300)


SYSTEM = """You grade a retrieval-augmented answer about SEC filings. Be exacting.

faithfulness: 1.0 only if EVERY figure and claim appears in the passages.
  A number that is close but not printed is unsupported. A figure attributed to
  the wrong company or the wrong fiscal period is unsupported.
correctness: compare against the gold answer. The key figures must match.
  Wording may differ freely. Extra correct detail is not penalised.
relevance: does it address the question asked, not an adjacent one.
citationsValid: false if a [n] marker cites a passage that does not support the
  sentence it is attached to."""

_judge = None


def _model():
    global _judge
    assert_openai()
    if _judge is None:
        kwargs = {"api_key": OPENAI_API_KEY, "model": JUDGE_MODEL, "temperature": 0,
                  "max_retries": 2, "timeout": 60}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _judge = ChatOpenAI(**kwargs).with_structured_output(JudgeOut)
    return _judge


def judge_answer(*, question: str, answer: str, gold: str, contexts: list[dict]) -> JudgeOut:
    passages = "\n\n".join(
        f"[{i + 1}] ({c.get('metadata', {}).get('ticker')} {c.get('metadata', {}).get('form')} "
        f"{c.get('metadata', {}).get('filingDate')})\n{c.get('text')}"
        for i, c in enumerate(contexts or [])
    )
    return _model().invoke([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"QUESTION:\n{question}\n\nGOLD ANSWER:\n{gold}\n\n"
            f"PASSAGES GIVEN TO THE SYSTEM:\n{passages or '(none)'}\n\n"
            f"SYSTEM ANSWER:\n{answer or '(refused)'}"},
    ])
