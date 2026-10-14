"""Level 2 — End-to-end answer evaluation (LLM-as-judge).

Runs the FULL FinSight agent on each question (retrieve -> subagents -> synthesize),
captures the answer and the chunks the agent actually retrieved, then asks a judge
LLM to score faithfulness / relevance / correctness / citation.

Slow and token-costly by nature: each question is one full agent run (~1 min) plus
one judge call. Use --limit to sample. Retrieval quality is scored separately and
cheaply in retrieval_eval.py; this level measures the generation half.
"""

from __future__ import annotations

import shutil
import time

from langchain.messages import HumanMessage

from finsight.agent import WORKSPACE_ROOT, create_finsight_agent, new_thread_id

from . import metrics
from .common import read_retrieved_context
from .judge import judge_answer, normalize

# An answer "passes" when it is grounded, on-topic, factually sound and attributed.
# Transparent, deterministic thresholds (computed in code, not left to the judge).
PASS_THRESHOLDS = {"faithfulness": 4, "relevance": 4, "correctness": 3, "citation": 2}


def _passed(judgement) -> bool:
    return all(
        getattr(judgement, dim).score >= threshold
        for dim, threshold in PASS_THRESHOLDS.items()
    )


def evaluate_example(example: dict, judge_model: str | None = None, verbose: bool = True) -> dict:
    """Run the agent on one example, judge the answer, and clean up its workspace."""
    thread_id = new_thread_id()
    agent, backend = create_finsight_agent(thread_id)
    try:
        start = time.time()
        result = agent.invoke(
            {"messages": [HumanMessage(content=example["question"])]},
            {"configurable": {"thread_id": thread_id}},
        )
        latency = time.time() - start
        answer = result["messages"][-1].text
        context = read_retrieved_context(backend)

        judgement = judge_answer(
            question=example["question"],
            context=context,
            answer=answer,
            reference_answer=example["reference_answer"],
            key_facts=example.get("key_facts", []),
            judge_model=judge_model,
        )
    finally:
        # Eval threads are throwaway: drop the workspace so runs don't pile up.
        shutil.rmtree(WORKSPACE_ROOT / thread_id, ignore_errors=True)

    dims = {d: getattr(judgement, d).score for d in PASS_THRESHOLDS}
    return {
        "id": example["id"],
        "category": example["category"],
        "question": example["question"],
        "latency_sec": round(latency, 1),
        "faithfulness": dims["faithfulness"],
        "relevance": dims["relevance"],
        "correctness": dims["correctness"],
        "citation": dims["citation"],
        "passed": _passed(judgement),
        "reasons": {d: getattr(judgement, d).reason for d in PASS_THRESHOLDS},
        "summary": judgement.summary,
        "retrieved_context_chars": len(context),
        "answer_preview": answer[:400],
    }


def run(
    dataset: list[dict],
    judge_model: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Run answer eval over the dataset, printing progress per question."""
    rows: list[dict] = []
    for i, example in enumerate(dataset, start=1):
        if verbose:
            print(f"[{i:>2}/{len(dataset)}] {example['id']}: running agent + judge ...")
        row = evaluate_example(example, judge_model=judge_model, verbose=verbose)
        rows.append(row)
        if verbose:
            mark = "PASS" if row["passed"] else "FAIL"
            print(
                f"        {mark} faith={row['faithfulness']} rel={row['relevance']} "
                f"corr={row['correctness']} cite={row['citation']} "
                f"({row['latency_sec']}s)"
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    """Aggregate: pass rate, mean normalized score per dimension, mean latency."""
    return {
        "n": len(rows),
        "pass_rate": metrics.mean([1.0 if r["passed"] else 0.0 for r in rows]),
        "faithfulness": metrics.mean([normalize(r["faithfulness"]) for r in rows]),
        "relevance": metrics.mean([normalize(r["relevance"]) for r in rows]),
        "correctness": metrics.mean([normalize(r["correctness"]) for r in rows]),
        "citation": metrics.mean([normalize(r["citation"]) for r in rows]),
        "avg_latency_sec": metrics.mean([r["latency_sec"] for r in rows]),
    }


def summarize_by_category(rows: list[dict]) -> dict[str, dict]:
    categories = sorted({r["category"] for r in rows})
    return {cat: summarize([r for r in rows if r["category"] == cat]) for cat in categories}
