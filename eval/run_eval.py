"""Unified eval entry point: run retrieval and/or answer evaluation, write a report.

Examples:
    python -m eval.run_eval --level retrieval            # fast + cheap (embeddings only)
    python -m eval.run_eval --level answer --limit 3     # sample the slow end-to-end path
    python -m eval.run_eval --level both                 # full harness
    python -m eval.run_eval --level retrieval --category risk_metrics

Reports (JSON + Markdown) are written to eval/reports/report_<timestamp>.*.
The dataset and score structure are portable: Langfuse / LangSmith can ingest them
directly (see eval/README.md), which is how task 2 closes the monitoring loop.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from . import answer_eval, retrieval_eval
from .common import REPORTS_DIR, load_dataset, pct

RETRIEVAL_METRICS = ["hit_rate", "recall", "precision", "mrr", "keyterm_coverage"]
ANSWER_METRICS = ["pass_rate", "faithfulness", "relevance", "correctness", "citation"]


def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    """Print and return a fixed-width text table (no external dependency)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    out = [f"\n{title}", line, sep]
    for row in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    text = "\n".join(out)
    print(text)
    return text


def _retrieval_report(rows: list[dict]) -> tuple[dict, str]:
    summary = retrieval_eval.summarize(rows)
    by_cat = retrieval_eval.summarize_by_category(rows)
    table = _print_table(
        "== Level 1: Retrieval ==",
        ["scope", "n", *RETRIEVAL_METRICS],
        [
            ["ALL", summary["n"], *[pct(summary[m]) for m in RETRIEVAL_METRICS]],
            *[
                [cat, s["n"], *[pct(s[m]) for m in RETRIEVAL_METRICS]]
                for cat, s in by_cat.items()
            ],
        ],
    )
    misses = [r for r in rows if r["hit"] == 0.0]
    if misses:
        print(f"\n  retrieval misses ({len(misses)}): " + ", ".join(r["id"] for r in misses))
    return {"summary": summary, "by_category": by_cat, "rows": rows}, table


def _answer_report(rows: list[dict], avg_latency: float | None) -> tuple[dict, str]:
    summary = answer_eval.summarize(rows)
    by_cat = answer_eval.summarize_by_category(rows)
    table = _print_table(
        "== Level 2: Answer quality (LLM-as-judge, 1-5 normalized to 0-1) ==",
        ["scope", "n", *ANSWER_METRICS],
        [
            ["ALL", summary["n"], *[pct(summary[m]) for m in ANSWER_METRICS]],
            *[
                [cat, s["n"], *[pct(s[m]) for m in ANSWER_METRICS]]
                for cat, s in by_cat.items()
            ],
        ],
    )
    fails = [r for r in rows if not r["passed"]]
    if fails:
        print(f"\n  failed answers ({len(fails)}):")
        for r in fails:
            print(f"    - {r['id']}: {r['summary']}")
    return {"summary": summary, "by_category": by_cat, "rows": rows}, table


def _write_reports(payload: dict, markdown: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"report_{stamp}.json"
    md_path = out_dir / f"report_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def _build_markdown(payload: dict, tables: list[str]) -> str:
    cfg = payload["config"]
    lines = [
        f"# FinSight Eval Report — {payload['timestamp']}",
        "",
        f"- level: **{cfg['level']}**  |  dataset: `{cfg['dataset']}` "
        f"({cfg['n_examples']} examples)  |  k={cfg['k']}  |  judge={cfg['judge_model'] or 'default (CHAT_MODEL)'}",
        "",
    ]
    lines += [t + "\n" for t in tables]
    lines.append(
        "\n_Produced by `eval/run_eval.py`. Dataset + scores are portable to "
        "Langfuse/LangSmith for continuous monitoring._"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FinSight evaluation harness.")
    parser.add_argument("--level", choices=["retrieval", "answer", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N examples")
    parser.add_argument("--k", type=int, default=4, help="retrieval top-k (matches search_knowledge)")
    parser.add_argument("--category", default=None, help="filter dataset by category")
    parser.add_argument("--judge-model", default=None, help="override the judge LLM (else EVAL_JUDGE_MODEL/CHAT_MODEL)")
    parser.add_argument("--dataset", default=None, help="path to a JSONL dataset (default eval/dataset.jsonl)")
    parser.add_argument("--out", default=None, help="report output dir (default eval/reports)")
    parser.add_argument("--no-report", action="store_true", help="print only, do not write report files")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, limit=args.limit, category=args.category)
    if not dataset:
        raise SystemExit("No examples matched — check --limit/--category/--dataset.")

    print(f"Loaded {len(dataset)} examples (level={args.level}, k={args.k}).")
    payload: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "level": args.level,
            "dataset": str(args.dataset or "eval/dataset.jsonl"),
            "n_examples": len(dataset),
            "k": args.k,
            "judge_model": args.judge_model,
        },
    }
    tables: list[str] = []

    if args.level in ("retrieval", "both"):
        rows = retrieval_eval.run(dataset, k=args.k)
        payload["retrieval"], table = _retrieval_report(rows)
        tables.append(table)

    if args.level in ("answer", "both"):
        rows = answer_eval.run(dataset, judge_model=args.judge_model)
        avg_latency = answer_eval.summarize(rows)["avg_latency_sec"]
        payload["answer"], table = _answer_report(rows, avg_latency)
        tables.append(table)

    if not args.no_report:
        out_dir = Path(args.out) if args.out else REPORTS_DIR
        markdown = _build_markdown(payload, tables)
        json_path, md_path = _write_reports(payload, markdown, out_dir)
        print(f"\nReport written:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
