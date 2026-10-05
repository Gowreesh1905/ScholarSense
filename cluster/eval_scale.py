"""Evaluate at scale: P3's queries against 727 papers + the cluster-built distractors.

    python cluster/eval_scale.py            # after build_index.py has assembled the index

Same queries, same metrics (MRR@10, Recall@10) and same relevant papers as the 727-paper
evaluation (eval/run_eval.py); only the corpus grows. Exact and paraphrased queries search
the masked 727 + distractors; hand-written ones (if present) search the full 727 + distractors.
Writes results/scale/summary.json, summary.md, raw/ and mrr_727_vs_scale.png.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from cluster.common import RESULTS_SCALE, SCALE_METHODS, read_json, scale_texts, write_json  # noqa: E402
from cluster.index import ready_index  # noqa: E402
from common import QUERY_FILES, RESULTS_DIR, read_jsonl  # noqa: E402  (eval/common.py)
from run_eval import first_relevant_rank, query_metrics  # noqa: E402
from searchers.registry import METHOD_COLORS, available_methods, build_searcher  # noqa: E402

TOP_K = 100
BATCH = 128
LATENCY_QUERIES = 100
LATENCY_WARMUP = 5


def run_method(key: str, sets: dict[str, list[dict]], n: int) -> dict[str, dict]:
    searcher = build_searcher(key)
    out = {}
    for corpus in sorted({q["corpus"] for qs in sets.values() for q in qs}):
        t = time.time()
        texts = scale_texts(corpus, n)
        searcher.fit(texts)
        print(f"  {key:<11} fitted on {corpus} ({time.time() - t:.1f}s)", flush=True)
        for name, queries in sets.items():
            subset = [q for q in queries if q["corpus"] == corpus]
            if not subset:
                continue
            t = time.time()
            hits = []
            for i in range(0, len(subset), BATCH):
                hits += searcher.search_batch([q["query"] for q in subset[i:i + BATCH]], k=TOP_K)
            per_query, mrrs, recalls = {}, [], []
            for q, h in zip(subset, hits):
                rank = first_relevant_rank(h, set(q["relevant_docs"]))
                top10 = [d for d, _ in h[:10]]
                mrr, rec = query_metrics(rank, top10, q["relevant_docs"])
                mrrs.append(mrr)
                recalls.append(rec)
                per_query[q["qid"]] = {"rank": rank, "top10": top10}
            out[name] = {"method": key, "query_set": name, "n": len(subset), "corpus_size": len(texts),
                         "mrr@10": statistics.mean(mrrs), "recall@10": statistics.mean(recalls), "queries": per_query}
            print(f"  {key:<11} {name:<12} MRR@10 {out[name]['mrr@10']:.3f}  R@10 {out[name]['recall@10']:.3f}  "
                  f"({len(subset)} queries, {time.time() - t:.1f}s)", flush=True)
    return out


def latency(key: str, queries: list[str], n: int) -> dict:
    searcher = build_searcher(key)
    searcher.fit(scale_texts("full", n))
    for q in queries[:LATENCY_WARMUP]:
        searcher.search(q, k=10)
    times = []
    for q in queries:
        t = time.perf_counter()
        searcher.search(q, k=10)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"mean": round(statistics.mean(times), 2), "p95": round(times[int(0.95 * (len(times) - 1))], 2)}


def chart(rows: list[dict], sets: list[str], corpus_size: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(r["method"] for r in rows))
    by = {(r["method"], r["query_set"]): r for r in rows}
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.2), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], sets):
        for i, m in enumerate(methods):
            r = by.get((m, s))
            if not r:
                continue
            color = r["color"]
            ax.bar(i - 0.2, r["mrr@10_727"] or 0, 0.38, color=color, alpha=0.35, edgecolor=color)
            ax.bar(i + 0.2, r["mrr@10_scale"], 0.38, color=color)
            ax.text(i + 0.2, r["mrr@10_scale"] + 0.01, f"{r['mrr@10_scale']:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([by[(m, s)]["label"] if (m, s) in by else m for m in methods], rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{s} queries", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("MRR@10")
    fig.suptitle(f"MRR@10 at 727 papers (light) vs {corpus_size:,} papers (solid)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    manifest = ready_index(("full", "masked"))
    if not manifest:
        sys.exit("No assembled index. Run `python cluster/build_index.py` (and let it finish) first.")
    n = manifest["n_distractors"]
    corpus_size = manifest["corpus_size"]

    sets = {}
    for name in ("exact", "paraphrased", "handwritten"):
        if QUERY_FILES[name].exists():
            sets[name] = read_jsonl(QUERY_FILES[name])
    methods = [m for m in SCALE_METHODS if m in available_methods()]
    print(f"Corpus: {corpus_size:,} papers (727 + {n:,} distractors, built in {manifest['precision']}). "
          f"Query sets: { {k: len(v) for k, v in sets.items()} }. Methods: {methods}\n", flush=True)

    raw_dir = RESULTS_SCALE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in methods:
        print(f"[{key}]", flush=True)
        for name, raw in run_method(key, sets, n).items():
            (raw_dir / f"{key}__{name}.json").write_text(json.dumps(raw), encoding="utf-8")
            results[(key, name)] = raw

    pool = [q["query"] for q in sets["exact"]]
    sample = random.Random(0).sample(pool, min(LATENCY_QUERIES, len(pool)))
    print("\nLatency (single-query search on the full scaled corpus):", flush=True)
    lat = {}
    for key in methods:
        lat[key] = latency(key, sample, n)
        print(f"  {key:<11} mean {lat[key]['mean']:8.2f} ms   p95 {lat[key]['p95']:8.2f} ms", flush=True)

    small = {(r["method"], r["query_set"]): r for r in (read_json(RESULTS_DIR / "summary.json", {}) or {}).get("rows", [])}
    rows = []
    for (key, name), raw in results.items():
        s = small.get((key, name), {})
        searcher_label = build_searcher(key).label
        rows.append({
            "method": key, "label": searcher_label, "color": METHOD_COLORS.get(key, "#6b7280"), "query_set": name,
            "mrr@10_727": s.get("mrr@10"), "mrr@10_scale": round(raw["mrr@10"], 4),
            "recall@10_727": s.get("recall@10"), "recall@10_scale": round(raw["recall@10"], 4),
            "latency_ms_727": s.get("latency_ms_mean"), "latency_ms_scale": lat[key]["mean"],
            "latency_ms_p95_scale": lat[key]["p95"],
        })

    import torch
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_size": corpus_size,
        "n_distractors": n,
        "precision": manifest["precision"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "query_sets": {k: len(v) for k, v in sets.items()},
        "rows": rows,
        "charts": [{"file": "scale/mrr_727_vs_scale.png", "title": "MRR@10 at 727 vs. scale",
                    "caption": f"Each method's MRR@10 with the 727 papers alone (light) and with "
                               f"{n:,} arXiv distractors added (solid), on the same queries."}],
    }
    write_json(RESULTS_SCALE / "summary.json", summary)
    chart(rows, list(sets), corpus_size, RESULTS_SCALE / "mrr_727_vs_scale.png")

    lines = [f"| Method | Query set | MRR@10 (727) | MRR@10 ({corpus_size:,}) | R@10 (727) | R@10 ({corpus_size:,}) "
             f"| Latency ms ({corpus_size:,}) |", "|---|---|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "–" if v is None else f"{v:.3f}"  # noqa: E731
    for r in rows:
        lines.append(f"| {r['label']} | {r['query_set']} | {fmt(r['mrr@10_727'])} | {fmt(r['mrr@10_scale'])} | "
                     f"{fmt(r['recall@10_727'])} | {fmt(r['recall@10_scale'])} | {r['latency_ms_scale']:.1f} |")
    (RESULTS_SCALE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS_SCALE / 'summary.json'}, summary.md and mrr_727_vs_scale.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
