"""Retrieval metrics — pure functions with no I/O, unit-testable without any API.

All functions take *lists of source labels* (e.g. "knowledge/risk_metrics.md") in
ranked order for the retrieved side, and a set/list of relevant sources for the
ground-truth side. They return floats in [0, 1] (0 = worst, 1 = perfect).

Run the built-in self-test (no network, no API key needed):
    python -m eval.metrics
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def hit_at_k(retrieved_sources: Sequence[str], relevant_sources: Iterable[str]) -> float:
    """1.0 if at least one relevant source appears in the top-k, else 0.0."""
    relevant = set(relevant_sources)
    return 1.0 if any(s in relevant for s in retrieved_sources) else 0.0


def recall_at_k(retrieved_sources: Sequence[str], relevant_sources: Iterable[str]) -> float:
    """Fraction of the relevant sources that were retrieved (source-level recall)."""
    relevant = set(relevant_sources)
    if not relevant:
        return 0.0
    hits = len(set(retrieved_sources) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved_sources: Sequence[str], relevant_sources: Iterable[str]) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if not retrieved_sources:
        return 0.0
    relevant = set(relevant_sources)
    hits = sum(1 for s in retrieved_sources if s in relevant)
    return hits / len(retrieved_sources)


def reciprocal_rank(retrieved_sources: Sequence[str], relevant_sources: Iterable[str]) -> float:
    """1 / rank of the first relevant source (0.0 if none found). Averaged => MRR."""
    relevant = set(relevant_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def keyterm_coverage(retrieved_text: str, must_contain: Sequence[str]) -> float | None:
    """Fraction of required key terms present in the concatenated retrieved text.

    Catches "right file, wrong chunk" — the source matched but the specific
    content needed to answer was not in the retrieved window. Case-insensitive.
    Returns None when there are no required terms (so it is skipped in the mean).
    """
    if not must_contain:
        return None
    haystack = retrieved_text.lower()
    hits = sum(1 for term in must_contain if term.lower() in haystack)
    return hits / len(must_contain)


def mean(values: Iterable[float | None]) -> float | None:
    """Mean over non-None values; None if there is nothing to average."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _self_test() -> None:
    """Deterministic checks of the metric math — runs offline, no API needed."""
    rel = {"b", "d"}
    assert hit_at_k(["a", "b", "c"], rel) == 1.0
    assert hit_at_k(["a", "c", "e"], rel) == 0.0
    assert recall_at_k(["a", "b", "c"], rel) == 0.5          # 1 of 2 relevant found
    assert recall_at_k(["b", "d", "x"], rel) == 1.0          # both found
    assert precision_at_k(["a", "b", "c", "d"], rel) == 0.5  # 2 of 4 relevant
    assert precision_at_k([], rel) == 0.0
    assert abs(reciprocal_rank(["x", "y", "b"], rel) - 1 / 3) < 1e-9
    assert reciprocal_rank(["x", "y", "z"], rel) == 0.0
    cov = keyterm_coverage("Sharpe ratio uses the risk-free rate", ["sharpe", "risk-free", "volatility"])
    assert abs(cov - 2 / 3) < 1e-9
    assert keyterm_coverage("anything", []) is None
    assert mean([1.0, 0.0, None, 0.5]) == 0.5
    assert mean([None, None]) is None
    print("[metrics] self-test passed: hit / recall / precision / MRR / keyterm / mean OK")


if __name__ == "__main__":
    _self_test()
