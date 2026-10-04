"""Quantify the leak that the masked corpus removes (supports the report's evaluation-design section).

    python eval/leak_check.py [--methods tfidf bm25 bge]

Runs the exact queries against the FULL abstracts (where each query text appears
verbatim) and against the masked abstracts, and writes MRR@10 for both to
results/leak_check.json. A big gap means the full-corpus number would measure
copy-matching, not retrieval.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import QUERY_FILES, RESULTS_DIR, load_corpus_for, read_jsonl  # noqa: E402
from run_eval import first_relevant_rank, query_metrics  # noqa: E402

from searchers.registry import METHOD_ORDER, build_searcher  # noqa: E402


def mrr_on(key: str, corpus: str, queries: list[dict]) -> float:
    searcher = build_searcher(key)
    searcher.fit(load_corpus_for(corpus))
    results = searcher.search_batch([q["query"] for q in queries], k=100)
    scores = []
    for q, hits in zip(queries, results):
        rank = first_relevant_rank(hits, set(q["relevant_docs"]))
        scores.append(query_metrics(rank, [d for d, _ in hits[:10]], q["relevant_docs"])[0])
    return statistics.mean(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["tfidf", "bm25", "bge"])
    methods = [m for m in METHOD_ORDER if m in ap.parse_args().methods]
    queries = read_jsonl(QUERY_FILES["exact"])
    out = {"n": len(queries), "mrr@10": {}}
    for key in methods:
        full, masked = mrr_on(key, "full", queries), mrr_on(key, "masked", queries)
        out["mrr@10"][key] = {"full_corpus": round(full, 4), "masked_corpus": round(masked, 4)}
        print(f"{key:<10} exact queries: full corpus MRR@10 {full:.3f}   masked corpus {masked:.3f}")
    (RESULTS_DIR / "leak_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
