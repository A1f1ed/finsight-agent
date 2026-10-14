"""Level 1 — Retrieval evaluation.

Checks whether search_knowledge's underlying vector store returns the right
knowledge chunks for a question. Cheap and fast: one embedding call per question,
no agent run and no generation. Run this after every retrieval change (chunking,
hybrid search, reranker) to quantify the delta.

Metrics (see metrics.py): Hit Rate@k, Recall@k, Precision@k, MRR, key-term coverage.
"""

from __future__ import annotations

from finsight.rag import get_vector_store

from . import metrics
from .common import pct


def evaluate_example(vector_store, example: dict, k: int = 4) -> dict:
    """Retrieve for one dataset example and compute all retrieval metrics."""
    docs = vector_store.similarity_search(example["question"], k=k)
    retrieved_sources = [d.metadata.get("source", "unknown") for d in docs]
    retrieved_text = "\n\n".join(d.page_content for d in docs)
    relevant = set(example["relevant_sources"])
    return {
        "id": example["id"],
        "category": example["category"],
        "question": example["question"],
        "expected_sources": sorted(relevant),
        "retrieved_sources": retrieved_sources,
        "hit": metrics.hit_at_k(retrieved_sources, relevant),
        "recall": metrics.recall_at_k(retrieved_sources, relevant),
        "precision": metrics.precision_at_k(retrieved_sources, relevant),
        "mrr": metrics.reciprocal_rank(retrieved_sources, relevant),
        "keyterm": metrics.keyterm_coverage(retrieved_text, example.get("must_contain", [])),
    }


def run(dataset: list[dict], k: int = 4, verbose: bool = True) -> list[dict]:
    """Run retrieval eval over the dataset. Builds/loads the Chroma index once."""
    vector_store = get_vector_store()
    rows: list[dict] = []
    for i, example in enumerate(dataset, start=1):
        row = evaluate_example(vector_store, example, k=k)
        rows.append(row)
        if verbose:
            flag = "OK " if row["hit"] == 1.0 else "MISS"
            print(
                f"[{i:>2}/{len(dataset)}] {flag} {row['id']:<20} "
                f"recall={row['recall']:.2f} mrr={row['mrr']:.2f} "
                f"keyterm={pct(row['keyterm'])}"
            )
            if row["hit"] == 0.0:
                print(f"        expected {row['expected_sources']} got {row['retrieved_sources']}")
    return rows


def summarize(rows: list[dict]) -> dict:
    """Aggregate retrieval metrics over all rows (and mean latency-free quality)."""
    return {
        "n": len(rows),
        "hit_rate": metrics.mean([r["hit"] for r in rows]),
        "recall": metrics.mean([r["recall"] for r in rows]),
        "precision": metrics.mean([r["precision"] for r in rows]),
        "mrr": metrics.mean([r["mrr"] for r in rows]),
        "keyterm_coverage": metrics.mean([r["keyterm"] for r in rows]),
    }


def summarize_by_category(rows: list[dict]) -> dict[str, dict]:
    """Per-category breakdown, to spot which knowledge file retrieves worst."""
    categories = sorted({r["category"] for r in rows})
    return {
        cat: summarize([r for r in rows if r["category"] == cat])
        for cat in categories
    }
