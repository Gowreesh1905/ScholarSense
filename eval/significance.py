"""Paired bootstrap on per-query reciprocal rank for the comparisons the report makes.

    python eval/significance.py

Reads results/raw/, resamples queries with replacement (10,000 times, fixed seed) and
writes the 95% interval of the MRR@10 difference (A - B) to results/significance.json.
An interval that excludes 0 means the gap is unlikely to be query-sampling noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR, RESULTS_DIR  # noqa: E402

COMPARISONS = [("bge", "bm25"), ("bm25", "tfidf"), ("hybrid_rrf", "bge"), ("hybrid_rrf", "bm25"),
               ("hybrid", "hybrid_rrf"), ("hybrid", "bge"), ("bge", "specter"), ("glove", "word2vec")]
N_BOOT = 10_000


def rr_vector(method: str, query_set: str) -> tuple[list[str], np.ndarray] | None:
    path = RAW_DIR / f"{method}__{query_set}.json"
    if not path.exists():
        return None
    queries = json.loads(path.read_text(encoding="utf-8"))["queries"]
    qids = sorted(queries)
    rr = [1 / queries[q]["rank"] if queries[q]["rank"] is not None and queries[q]["rank"] <= 10 else 0.0
          for q in qids]
    return qids, np.array(rr)


def main() -> None:
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for query_set in ("exact", "paraphrased", "handwritten"):
        for a, b in COMPARISONS:
            va, vb = rr_vector(a, query_set), rr_vector(b, query_set)
            if va is None or vb is None:
                continue
            diff = va[1] - vb[1]
            idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
            boot = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out[f"{a} - {b} [{query_set}]"] = {
                "diff": round(float(diff.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "significant": bool(lo > 0 or hi < 0), "n": len(diff)}
            flag = "yes" if out[f"{a} - {b} [{query_set}]"]["significant"] else "no (interval includes 0)"
            print(f"{query_set:<12} {a:>10} - {b:<10} diff {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  significant: {flag}")
    (RESULTS_DIR / "significance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
