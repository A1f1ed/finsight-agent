"""FinSight evaluation harness.

Two levels:
- Level 1 retrieval_eval : cheap, deterministic vector-retrieval quality (recall/precision/MRR).
- Level 2 answer_eval    : full agent run + LLM-as-judge on faithfulness/relevance/correctness/citation.

Entry point: python -m eval.run_eval --level both
"""
