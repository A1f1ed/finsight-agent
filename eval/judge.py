"""LLM-as-judge: score a RAG answer on four dimensions with structured output.

The judge is a separate LLM call that reads the question, the *actual* context the
agent retrieved, the agent's answer, and a human-written reference answer + key
facts. It returns 1-5 scores with a one-line rationale per dimension.

Dimensions (the RAG triad + this product's citation requirement):
- faithfulness : is every claim supported by the retrieved context? (anti-hallucination / "answer grounding")
- relevance    : does the answer actually address the question?
- correctness  : does it match the reference answer / key facts?
- citation     : does it attribute claims to sources?

The judge model is configurable so it can differ from the generator (reduces
self-preference bias): pass judge_model=..., or set EVAL_JUDGE_MODEL in .env.
It defaults to the same CHAT_MODEL used by the agent.
"""

from __future__ import annotations

import json
import os
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from finsight.config import CHAT_MODEL

JUDGE_SYSTEM = """You are a strict, fair evaluator of a finance RAG assistant's answer.
You are given: the user QUESTION, the CONTEXT the assistant retrieved from its
knowledge base, the assistant's ANSWER, and a human-written REFERENCE answer with
KEY FACTS. Score the ANSWER on four dimensions using the 1-5 rubric below. Judge
only against the provided CONTEXT and REFERENCE — do not use outside knowledge to
rescue an ungrounded or wrong answer. Be concise; one sentence per rationale.

FAITHFULNESS (is every factual claim supported by the CONTEXT?)
5 = fully grounded, nothing invented | 3 = mostly grounded, minor unsupported embellishment
1 = major claims absent from or contradicting the CONTEXT (hallucination)

RELEVANCE (does it directly answer the QUESTION?)
5 = fully on-topic and complete | 3 = partially addresses it or drifts | 1 = off-topic/evasive

CORRECTNESS (vs the REFERENCE answer and KEY FACTS)
5 = matches reference, all key facts present and correct | 3 = partially correct, missing/minor-error key facts
1 = wrong or missing most key facts

CITATION (does it attribute claims to sources?)
5 = clear inline citations to source files | 3 = vague source mention | 1 = no attribution"""


class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = worst, 5 = best")
    reason: str = Field(description="one-sentence justification for the score")


class AnswerJudgement(BaseModel):
    faithfulness: DimensionScore
    relevance: DimensionScore
    correctness: DimensionScore
    citation: DimensionScore
    summary: str = Field(description="one-sentence overall verdict")


def normalize(score: int) -> float:
    """Map a 1-5 Likert score to [0, 1] (1 -> 0.0, 5 -> 1.0) for aggregation."""
    return (score - 1) / 4.0


def get_judge_model(judge_model: str | None = None):
    """Build the judge chat model: explicit arg > EVAL_JUDGE_MODEL env > CHAT_MODEL."""
    model_name = judge_model or os.getenv("EVAL_JUDGE_MODEL") or CHAT_MODEL
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )


def _build_user_prompt(
    question: str,
    context: str,
    answer: str,
    reference_answer: str,
    key_facts: list[str],
) -> str:
    facts = "\n".join(f"- {f}" for f in key_facts) or "(none provided)"
    ctx = context.strip() or "(the assistant retrieved NO context)"
    return f"""QUESTION:
{question}

CONTEXT (retrieved by the assistant):
{ctx}

ANSWER (to be scored):
{answer}

REFERENCE ANSWER (human-written, authoritative):
{reference_answer}

KEY FACTS the answer should contain:
{facts}

Return the four dimension scores (1-5) with a one-sentence rationale each, plus a
one-sentence overall summary."""


def _extract_json(text: str) -> dict:
    """Pull the first {...} block out of a raw model response (fallback path)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in judge response: {text[:200]!r}")
    return json.loads(match.group(0))


def judge_answer(
    question: str,
    context: str,
    answer: str,
    reference_answer: str,
    key_facts: list[str] | None = None,
    judge_model: str | None = None,
) -> AnswerJudgement:
    """Score one answer. Uses native structured output, falling back to JSON parsing.

    Structured-output support varies across OpenAI-compatible endpoints, so if the
    tool-calling path fails we re-ask for a plain JSON object and validate it — the
    LLM API is a real boundary where this fallback earns its keep.
    """
    model = get_judge_model(judge_model)
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=_build_user_prompt(
            question, context, answer, reference_answer, key_facts or []
        )),
    ]

    try:
        result = model.with_structured_output(AnswerJudgement).invoke(messages)
        if isinstance(result, AnswerJudgement):
            return result
    except Exception as exc:  # noqa: BLE001 - provider differences are numerous
        print(f"[judge] structured output failed ({exc.__class__.__name__}); falling back to JSON")

    schema_hint = json.dumps(
        {
            "faithfulness": {"score": 0, "reason": ""},
            "relevance": {"score": 0, "reason": ""},
            "correctness": {"score": 0, "reason": ""},
            "citation": {"score": 0, "reason": ""},
            "summary": "",
        }
    )
    raw = model.invoke(
        messages
        + [HumanMessage(content=f"Return ONLY a JSON object with this shape:\n{schema_hint}")]
    )
    return AnswerJudgement.model_validate(_extract_json(raw.text))
