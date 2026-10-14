# FinSight Evaluation Harness

Quantifies **how good the agent is**, so every retrieval/prompt change can be
measured instead of eyeballed. This is the `measure` half of the Roadmap's
`measure → optimize → monitor → re-iterate` loop.

## Why two levels

A RAG agent fails in two distinct places, and end-to-end "looks fine" can't tell
them apart:

```
question → [Level 1: retrieval]  did we fetch the right knowledge chunks?
         → [Level 2: generation] did we answer faithfully & correctly from them?
```

- **Level 1 — Retrieval** (`retrieval_eval.py`): cheap, fast, near-deterministic.
  One embedding call per question, no agent, no generation. Run it after *every*
  retrieval change (chunk size, hybrid search, reranker).
- **Level 2 — Answer quality** (`answer_eval.py`): runs the **full agent**
  (retrieve → subagents → synthesize), then an **LLM-as-judge** scores the answer.
  Slow (~1 min/question) and token-costly — sample it with `--limit`.

## Metrics

**Level 1 — retrieval** (ground truth = `relevant_sources` + `must_contain`):

| metric | meaning |
|---|---|
| Hit Rate@k | ≥1 relevant source in the top-k? (k=4, matches `search_knowledge`) |
| Recall@k | relevant sources retrieved ÷ all relevant sources |
| Precision@k | relevant items ÷ k |
| MRR | 1 ÷ rank of the first relevant source |
| Key-term coverage | required terms found in the retrieved text — catches "right file, wrong chunk" |

**Level 2 — answer** (LLM-as-judge, each 1–5, normalized to 0–1 via `(s-1)/4`):

| dimension | question it answers |
|---|---|
| Faithfulness | is every claim supported by the retrieved context? (**anti-hallucination / grounding**) |
| Relevance | does it actually address the question? |
| Correctness | does it match the reference answer + key facts? |
| Citation | does it attribute claims to sources? |

An answer **passes** when faithfulness≥4, relevance≥4, correctness≥3, citation≥2
(thresholds in `answer_eval.PASS_THRESHOLDS`, computed in code — not left to the judge).

> The judge scores faithfulness against the chunks the agent **actually retrieved**
> (read back from its `/retrieved/` workspace), not an approximation — so grounding
> is measured against reality.

## Run it

```bash
# from the finsight-agent/ project root, in the project's Python env
python -m eval.metrics                              # offline self-test of the metric math (no API)
python -m eval.run_eval --level retrieval           # Level 1 only — cheap
python -m eval.run_eval --level answer --limit 3    # Level 2 sample — slow
python -m eval.run_eval --level both                # full harness
python -m eval.run_eval --level retrieval --category risk_metrics
```

Requires `.env` with `DASHSCOPE_API_KEY` (embeddings + agent + judge). Optional
`EVAL_JUDGE_MODEL` overrides the judge LLM (defaults to `CHAT_MODEL`); using a
*different/stronger* judge reduces self-preference bias. Reports (JSON + Markdown)
are written to `eval/reports/report_<timestamp>.*`.

## Dataset

`dataset.jsonl` — one JSON object per line, ~18 questions over the 3 knowledge
files (6 each), incl. two Chinese cross-lingual retrieval cases. Schema:

```jsonc
{
  "id": "sharpe-formula",
  "category": "risk_metrics",                 // source file stem, for per-category breakdown
  "question": "How is the Sharpe ratio calculated?",
  "relevant_sources": ["knowledge/risk_metrics.md"], // Level 1 ground truth
  "must_contain": ["Sharpe", "risk-free", "volatility"], // key-term coverage
  "reference_answer": "Sharpe = (annualized return - rf) / annualized volatility ...",
  "key_facts": ["...", "..."]                 // Level 2 correctness ground truth
}
```

**To extend:** add a line. Keep `relevant_sources` at file granularity (robust to
re-chunking). When you fix a production bug or spot a regression, add the failing
question here — that is the `re-iterate` step feeding the dataset.

## Files

```
eval/
├── dataset.jsonl      # annotated ground-truth questions (portable to Langfuse/LangSmith)
├── metrics.py         # pure retrieval-metric functions + offline self-test
├── common.py          # dataset loading, workspace-context capture, formatting
├── retrieval_eval.py  # Level 1
├── judge.py           # LLM-as-judge (Pydantic structured output + JSON fallback)
├── answer_eval.py     # Level 2
├── run_eval.py        # CLI orchestrator + report writer
└── reports/           # generated JSON + Markdown reports (git-ignored)
```

## Notes & limitations

- **No new required deps.** Metrics are hand-rolled and transparent; the judge uses
  `langchain_openai` + `pydantic` (already present). Self-test needs no network.
- **Level 2 is stochastic** (agent + judge are LLMs): scores vary run to run. Use
  the same judge model and a fixed dataset for comparable deltas; treat small gaps
  as noise, act on consistent ones.
- **Cost control:** `--level retrieval` for frequent cheap checks; `--limit N` to
  sample Level 2. A full 18-question Level 2 run is ~18 agent runs + 18 judge calls.
- **Next (Roadmap task 2):** the JSONL dataset imports directly as a Langfuse
  dataset, and these scores map 1:1 onto Langfuse trace scores — so the same
  harness drives offline evals *and* online production monitoring.
