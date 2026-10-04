"""Run every method on every query set; write results/raw, latency.json, summary.json, summary.md.

    python eval/run_eval.py                                  # all available methods, all query sets
    python eval/run_eval.py --methods tfidf bm25 --sets exact --quick   # first 50 queries only

Per query: rank of the first relevant doc in the top 100 (None if absent),
MRR@10 = 1/rank if rank <= 10 else 0, Recall@10 = |relevant ∩ top 10| / |relevant|.
Latency is measured separately on the FULL corpus (single-query search()).

A --quick run writes to results/quick/ so it never overwrites the real results.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (QUERY_FILES, RESULTS_DIR, load_corpus_for, max_overlap,  # noqa: E402
                    read_jsonl)

from searchers.registry import METHOD_COLORS, METHOD_ORDER, available_methods, build_searcher  # noqa: E402

TOP_K = 100
QUICK_N = 50
LATENCY_QUERIES = 200
LATENCY_WARMUP = 5

SET_DESCRIPTIONS = {
    "exact": "Task/problem spans copied from each abstract, searched in the masked corpus "
             "(all task/problem spans removed). Keeps full word overlap with the paper's remaining text.",
    "paraphrased": "The same spans back-translated English-German-English, searched in the masked corpus. "
                   "Changes the wording, so vocabulary mismatch appears.",
    "handwritten": "Plain-language queries written by hand, avoiding the papers' technical terms, "
                   "searched in the full corpus.",
}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def first_relevant_rank(hits: list[tuple[int, float]], relevant: set[int]) -> int | None:
    for rank, (doc, _) in enumerate(hits, start=1):
        if doc in relevant:
            return rank
    return None


def query_metrics(rank: int | None, top10: list[int], relevant: list[int]) -> tuple[float, float]:
    mrr = 1.0 / rank if rank is not None and rank <= 10 else 0.0
    recall = len(set(top10) & set(relevant)) / len(relevant)
    return mrr, recall


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def load_query_sets(names: list[str], quick: bool) -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = {}
    for name in names:
        path = QUERY_FILES[name]
        if not path.exists():
            print(f"[skip] query set {name!r}: {path.name} does not exist yet")
            continue
        queries = read_jsonl(path)
        if quick:
            queries = queries[:QUICK_N]
        if queries:
            sets[name] = queries
    return sets


def fill_overlap(sets: dict[str, list[dict]]) -> None:
    """Compute `overlap` for queries that lack it (hand-written file; we don't edit it)."""
    corpora: dict[str, list[str]] = {}
    for queries in sets.values():
        for q in queries:
            if "overlap" in q:
                continue
            texts = corpora.setdefault(q["corpus"], load_corpus_for(q["corpus"]))
            q["overlap"] = round(max_overlap(q["query"], [texts[d] for d in q["relevant_docs"]]), 4)


def evaluate_method(key: str, sets: dict[str, list[dict]]) -> dict[str, dict]:
    """Fit `key` once per corpus it needs and run every query set. Returns {set: raw dict}."""
    searcher = build_searcher(key)
    per_set: dict[str, dict] = {
        name: {"method": key, "label": searcher.label, "family": searcher.family,
               "query_set": name, "n": len(queries),
               "corpus": "/".join(sorted({q["corpus"] for q in queries})), "queries": {}}
        for name, queries in sets.items()
    }

    corpora = sorted({q["corpus"] for queries in sets.values() for q in queries})
    for corpus in corpora:
        abstracts = load_corpus_for(corpus)
        t0 = time.perf_counter()
        searcher.fit(abstracts)
        fit_s = time.perf_counter() - t0
        for name, queries in sets.items():
            subset = [q for q in queries if q["corpus"] == corpus]
            if not subset:
                continue
            t0 = time.perf_counter()
            results = searcher.search_batch([q["query"] for q in subset], k=TOP_K)
            run_s = time.perf_counter() - t0
            for q, hits in zip(subset, results):
                relevant = q["relevant_docs"]
                rank = first_relevant_rank(hits, set(relevant))
                top10 = [d for d, _ in hits[:10]]
                per_set[name]["queries"][q["qid"]] = {"rank": rank, "top10": top10, "overlap": q["overlap"]}
            print(f"    {key:<11} {name:<12} corpus={corpus:<6} fit {fit_s:5.1f}s  "
                  f"{len(subset)} queries in {run_s:5.1f}s", flush=True)

    for name, raw in per_set.items():
        mrrs, recalls = [], []
        for q in sets[name]:
            r = raw["queries"][q["qid"]]
            mrr, rec = query_metrics(r["rank"], r["top10"], q["relevant_docs"])
            mrrs.append(mrr)
            recalls.append(rec)
        raw["mrr@10"] = statistics.mean(mrrs)
        raw["recall@10"] = statistics.mean(recalls)
        raw["found_in_top100"] = sum(v["rank"] is not None for v in raw["queries"].values()) / len(mrrs)
    return per_set


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------

def device_info() -> tuple[str, str | None]:
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", None


def measure_latency(key: str, queries: list[str]) -> dict:
    """Single-query search() on the full corpus: 5 warm-up queries, then timed queries."""
    searcher = build_searcher(key)
    searcher.fit(load_corpus_for("full"))
    for q in queries[:LATENCY_WARMUP]:
        searcher.search(q, k=10)
    times_ms = []
    for q in queries:
        t0 = time.perf_counter()
        searcher.search(q, k=10)
        times_ms.append((time.perf_counter() - t0) * 1000)
    times_ms.sort()
    p95 = times_ms[min(len(times_ms) - 1, int(0.95 * len(times_ms)))]
    return {"latency_ms_mean": statistics.mean(times_ms), "latency_ms_p95": p95, "n": len(times_ms)}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def read_json(path: Path, default):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def build_summary(out_dir: Path) -> dict:
    """Rebuild summary.json from raw/ and latency.json (keeps existing charts/examples)."""
    raw_dir = out_dir / "raw"
    latency = read_json(out_dir / "latency.json", {})
    old = read_json(out_dir / "summary.json", {})

    raws: dict[tuple[str, str], dict] = {}
    for path in raw_dir.glob("*__*.json"):
        raw = read_json(path, None)
        if raw:
            raws[(raw["method"], raw["query_set"])] = raw

    present = {m for m, _ in raws}
    methods = [m for m in METHOD_ORDER if m in present]
    set_names = [s for s in ("exact", "paraphrased", "handwritten") if any(s == q for _, q in raws)]

    rows = []
    for m in methods:
        for s in set_names:
            raw = raws.get((m, s))
            if raw is None:
                continue
            lat = latency.get("methods", {}).get(m, {})
            rows.append({
                "method": m, "label": raw["label"], "family": raw["family"],
                "color": METHOD_COLORS.get(m, "#6b7280"), "query_set": s,
                "mrr@10": round(raw["mrr@10"], 4), "recall@10": round(raw["recall@10"], 4),
                "latency_ms_mean": round(lat["latency_ms_mean"], 2) if lat else None,
                "latency_ms_p95": round(lat["latency_ms_p95"], 2) if lat else None,
            })

    query_sets = {}
    for s in set_names:
        raw = next(r for (_, q), r in raws.items() if q == s)
        query_sets[s] = {"n": raw["n"], "corpus": raw["corpus"], "description": SET_DESCRIPTIONS[s]}

    device, gpu = latency.get("device"), latency.get("gpu")
    if device is None:
        device, gpu = device_info()

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "gpu": gpu,
        "corpus_size": len(load_corpus_for("full")),
        "query_sets": query_sets,
        "rows": rows,
        "ablation": [m for m in ("bge", "hybrid_rrf", "hybrid") if m in methods],
        "charts": old.get("charts", []),
        "examples": old.get("examples", []),
    }


def write_markdown(summary: dict, path: Path) -> None:
    rows = summary["rows"]
    sets = list(summary["query_sets"])
    methods = list(dict.fromkeys(r["method"] for r in rows))
    by = {(r["method"], r["query_set"]): r for r in rows}

    columns: list[tuple[str, str, str, bool]] = []  # (header, set, field, higher_is_better)
    for s in sets:
        columns.append((f"{s} MRR@10", s, "mrr@10", True))
        columns.append((f"{s} R@10", s, "recall@10", True))
    columns.append(("Latency mean (ms)", sets[0], "latency_ms_mean", False))
    columns.append(("Latency p95 (ms)", sets[0], "latency_ms_p95", False))

    def value(m: str, s: str, field: str):
        row = by.get((m, s))
        return row[field] if row else None

    best = {}
    for header, s, field, higher in columns:
        vals = [v for m in methods if (v := value(m, s, field)) is not None]
        best[header] = (max(vals) if higher else min(vals)) if vals else None

    lines = ["| Method | " + " | ".join(h for h, *_ in columns) + " |",
             "|---|" + "---:|" * len(columns)]
    for m in methods:
        label = next(r["label"] for r in rows if r["method"] == m)
        cells = []
        for header, s, field, _ in columns:
            v = value(m, s, field)
            if v is None:
                cells.append("–")
                continue
            text = f"{v:.1f}" if field.startswith("latency") else f"{v:.3f}"
            cells.append(f"**{text}**" if v == best[header] else text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    hw = f"{summary['device']}" + (f" ({summary['gpu']})" if summary.get("gpu") else "")
    notes = [f"Corpus: {summary['corpus_size']} abstracts. Device for latency: {hw}.",
             "Query sets: " + "; ".join(f"**{s}** n={v['n']} ({v['corpus']} corpus)"
                                        for s, v in summary["query_sets"].items()) + ".",
             "Bold = best value in the column (highest accuracy, lowest latency). "
             "Latency: single-query `search()` on the full corpus, mean and p95 over "
             f"{LATENCY_QUERIES} queries after {LATENCY_WARMUP} warm-up queries."]
    path.write_text("\n\n".join(["\n".join(lines), *notes]) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", help="method keys (default: all available)")
    ap.add_argument("--sets", nargs="+", choices=list(QUERY_FILES), default=list(QUERY_FILES))
    ap.add_argument("--quick", action="store_true", help=f"first {QUICK_N} queries only; writes to results/quick/")
    ap.add_argument("--skip-latency", action="store_true", help="don't (re)measure latency")
    ap.add_argument("--latency-only", action="store_true", help="only measure latency")
    args = ap.parse_args()

    out_dir = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    available = available_methods()
    methods = args.methods or available
    unknown = [m for m in methods if m not in available]
    if unknown:
        sys.exit(f"Methods not available: {unknown}. Available: {available}")
    methods = [m for m in METHOD_ORDER if m in methods]
    print(f"Methods: {methods}")

    device, gpu = device_info()
    print(f"Device: {device}" + (f" ({gpu})" if gpu else ""))

    if not args.latency_only:
        sets = load_query_sets(args.sets, args.quick)
        fill_overlap(sets)
        print("Query sets:", {n: len(q) for n, q in sets.items()})
        for key in methods:
            print(f"\n[{key}]", flush=True)
            for name, raw in evaluate_method(key, sets).items():
                (raw_dir / f"{key}__{name}.json").write_text(json.dumps(raw), encoding="utf-8")
                print(f"    -> {name}: MRR@10 {raw['mrr@10']:.3f}  R@10 {raw['recall@10']:.3f}  "
                      f"found in top-{TOP_K}: {raw['found_in_top100']:.3f}")

    if not args.skip_latency:
        exact_path = QUERY_FILES["exact"]
        if not exact_path.exists():
            print("[skip] latency: queries_exact.jsonl does not exist yet")
        else:
            pool = [q["query"] for q in read_jsonl(exact_path)]
            sample = random.Random(0).sample(pool, min(LATENCY_QUERIES, len(pool)))
            latency = read_json(out_dir / "latency.json", {})
            latency.update({"device": device, "gpu": gpu, "methods": latency.get("methods", {})})
            print("\nLatency (full corpus, single-query search):")
            for key in methods:
                latency["methods"][key] = lat = measure_latency(key, sample)
                print(f"    {key:<11} mean {lat['latency_ms_mean']:8.2f} ms   p95 {lat['latency_ms_p95']:8.2f} ms")
            (out_dir / "latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")

    summary = build_summary(out_dir)
    if summary["rows"]:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_markdown(summary, out_dir / "summary.md")
        print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'summary.md'}\n")
        print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
