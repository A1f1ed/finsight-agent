"""Shared helpers for the eval harness: dataset loading, context capture, formatting."""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.jsonl"
REPORTS_DIR = EVAL_DIR / "reports"


def load_dataset(
    path: str | Path | None = None,
    limit: int | None = None,
    category: str | None = None,
) -> list[dict]:
    """Load the JSONL eval dataset, optionally filtered by category and truncated."""
    p = Path(path) if path else DATASET_PATH
    rows = [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if category:
        rows = [r for r in rows if r.get("category") == category]
    if limit:
        rows = rows[:limit]
    return rows


def read_retrieved_context(backend) -> str:
    """Read the chunks the agent actually retrieved, straight from its workspace.

    search_knowledge writes every retrieved chunk under the virtual path
    /retrieved/<batch>/chunk_N.md; with virtual_mode these land in
    <workspace>/retrieved/. Reading them back gives the judge the *real* grounding
    context (not an approximation), so faithfulness is scored against what the
    agent truly saw.
    """
    root = Path(backend.cwd) / "retrieved"
    if not root.exists():
        return ""
    chunks = [p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.md"))]
    return "\n\n---\n\n".join(chunks)


def pct(x: float | None) -> str:
    """Format a 0-1 ratio as a percentage, or 'n/a' when undefined."""
    return "n/a" if x is None else f"{x * 100:5.1f}%"
