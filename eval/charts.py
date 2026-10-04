"""Make the four result charts from results/raw/ and results/summary.json.

    python eval/charts.py

Writes PNGs (150 dpi, white background) into results/ and adds each chart, with a
caption built from the actual numbers, to summary.json under "charts".
Method colors come from the registry (CONTRACT §4); each method also gets its own
marker shape so identity never depends on color alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OVERLAP_BINS, RESULTS_DIR, overlap_bin  # noqa: E402

from searchers.registry import METHOD_COLORS, METHOD_ORDER  # noqa: E402

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
MARKERS = {"tfidf": "o", "bm25": "s", "word2vec": "^", "glove": "v", "specter": "D",
           "bge": "P", "hybrid_rrf": "X", "hybrid": "*", "bm25_prf": "h"}
BIN_LABELS = ["[0, .2)", "[.2, .4)", "[.4, .6)", "[.6, .8)", "[.8, 1]"]
SET_ORDER = ["exact", "paraphrased", "handwritten"]


def style_axes(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def load_raw(raw_dir: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for path in raw_dir.glob("*__*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        out[(raw["method"], raw["query_set"])] = raw
    return out


def rr(rank) -> float:
    return 1.0 / rank if rank is not None and rank <= 10 else 0.0


def label_of(summary: dict, method: str) -> str:
    return next(r["label"] for r in summary["rows"] if r["method"] == method)


# ---------------------------------------------------------------------------
# 1. overlap vs MRR
# ---------------------------------------------------------------------------

def chart_overlap(raws, methods, summary, out: Path) -> dict:
    pooled_sets = [s for s in ("exact", "paraphrased") if any((m, s) in raws for m in methods)]
    n_bins = len(BIN_LABELS)
    counts = np.zeros(n_bins, dtype=int)
    mrr = {m: np.full(n_bins, np.nan) for m in methods}
    for m in methods:
        sums, ns = np.zeros(n_bins), np.zeros(n_bins)
        for s in pooled_sets:
            raw = raws.get((m, s))
            if raw is None:
                continue
            for q in raw["queries"].values():
                b = overlap_bin(q["overlap"])
                sums[b] += rr(q["rank"])
                ns[b] += 1
        with np.errstate(invalid="ignore"):
            mrr[m] = np.where(ns > 0, sums / np.maximum(ns, 1), np.nan)
        counts = ns.astype(int)  # identical for every method

    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(n_bins)
    for m in methods:
        ax.plot(x, mrr[m], color=METHOD_COLORS[m], marker=MARKERS[m], linewidth=1.8,
                markersize=8 if m != "hybrid" else 11, markeredgecolor="white", markeredgewidth=1,
                label=label_of(summary, m))
    ax.set_xticks(x, [f"{lab}\nn = {n}" for lab, n in zip(BIN_LABELS, counts)])
    ax.set_xlabel("Word overlap between query and the relevant abstract (fraction of query content words)",
                  color=INK, fontsize=10)
    ax.set_ylabel("MRR@10", color=INK, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_title("Retrieval quality vs. query/abstract word overlap (exact + paraphrased queries pooled)",
                 loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    save(fig, out / "overlap_vs_mrr.png")

    lo = next((i for i in range(n_bins) if counts[i] >= 20), 0)
    hi = next((i for i in reversed(range(n_bins)) if counts[i] >= 20), n_bins - 1)
    parts = []
    for m in ("tfidf", "bm25", "bge", "hybrid"):
        if m in mrr and not (np.isnan(mrr[m][lo]) or np.isnan(mrr[m][hi])):
            parts.append(f"{label_of(summary, m)} {mrr[m][hi]:.2f} → {mrr[m][lo]:.2f}")
    caption = (f"MRR@10 per word-overlap bin on {int(counts.sum())} pooled exact and paraphrased queries. "
               f"From the highest-overlap bin {BIN_LABELS[hi]} (n={counts[hi]}) to the lowest {BIN_LABELS[lo]} "
               f"(n={counts[lo]}): " + ", ".join(parts) + ".")
    return {"file": "overlap_vs_mrr.png",
            "title": "Retrieval quality vs. word overlap", "caption": caption}


# ---------------------------------------------------------------------------
# 2. speed vs accuracy
# ---------------------------------------------------------------------------

def chart_speed(summary, methods, out: Path) -> dict | None:
    rows = {r["method"]: r for r in summary["rows"] if r["query_set"] == "paraphrased"
            and r["latency_ms_mean"] is not None}
    ms = [m for m in methods if m in rows]
    if not ms:
        print("[skip] speed_vs_accuracy: need paraphrased results and latency")
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    pts = sorted(((rows[m]["latency_ms_mean"], rows[m]["mrr@10"], m) for m in ms))
    for i, (lat, acc, m) in enumerate(pts):
        ax.scatter(lat, acc, s=130 if m != "hybrid" else 220, color=METHOD_COLORS[m], marker=MARKERS[m],
                   edgecolor="white", linewidth=1.2, zorder=3)
        dy = 11 if i % 2 == 0 else -17  # alternate above/below so neighbouring labels don't collide
        ax.annotate(rows[m]["label"], (lat, acc), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per query (ms, log scale)", color=INK, fontsize=10)
    ax.set_ylabel("MRR@10 on the paraphrased queries", color=INK, fontsize=10)
    ax.set_ylim(0, min(1.0, max(p[1] for p in pts) * 1.25))
    ax.set_title("Speed vs. accuracy (paraphrased queries)", loc="left", color=INK, fontsize=11,
                 fontweight="bold")
    style_axes(ax, "both")
    save(fig, out / "speed_vs_accuracy.png")

    fastest = min(pts)
    best = max(pts, key=lambda p: p[1])
    caption = (f"Mean latency against MRR@10 on the paraphrased queries. The most accurate method is "
               f"{rows[best[2]]['label']} (MRR@10 {best[1]:.3f}, {best[0]:.1f} ms); the fastest is "
               f"{rows[fastest[2]]['label']} ({fastest[0]:.2f} ms, MRR@10 {fastest[1]:.3f}).")
    return {"file": "speed_vs_accuracy.png", "title": "Speed vs. accuracy", "caption": caption}


# ---------------------------------------------------------------------------
# 3 + 4. grouped bars
# ---------------------------------------------------------------------------

def grouped_bars(summary, methods, sets, out: Path, name: str, title: str, figsize, label_size: int) -> dict[str, float]:
    by = {(r["method"], r["query_set"]): r["mrr@10"] for r in summary["rows"]}
    n = len(methods)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sets))
    for i, m in enumerate(methods):
        vals = [by.get((m, s), np.nan) for s in sets]
        pos = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, vals, width * 0.88, color=METHOD_COLORS[m], label=label_of(summary, m))
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=label_size, color=INK)
    ax.set_xticks(x, [f"{s}\n(n = {summary['query_sets'][s]['n']})" for s in sets])
    ax.set_ylabel("MRR@10", color=INK, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    style_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    save(fig, out / name)
    return by


def describe(by, methods, labels, s) -> str:
    return ", ".join(f"{labels[m]} {by[(m, s)]:.3f}" for m in methods if (m, s) in by)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=RESULTS_DIR, help="results directory (default: results/)")
    out = ap.parse_args().dir
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raws = load_raw(out / "raw")
    labels = {r["method"]: r["label"] for r in summary["rows"]}
    methods = [m for m in METHOD_ORDER if m in labels]
    sets = [s for s in SET_ORDER if s in summary["query_sets"]]

    charts = [chart_overlap(raws, methods, summary, out)]
    speed = chart_speed(summary, methods, out)
    if speed:
        charts.append(speed)

    ablation = [m for m in summary.get("ablation", []) if m in labels]
    if len(ablation) >= 2:
        by = grouped_bars(summary, ablation, sets, out, "ablation.png",
                          "Ablation: dense only → + BM25 fusion (RRF) → + cross-encoder re-rank",
                          (8.5, 5), 9)
        caption = "MRR@10 of " + "; ".join(f"{s}: " + describe(by, ablation, labels, s) for s in sets) + "."
        charts.append({"file": "ablation.png", "title": "Ablation: fusion and re-ranking", "caption": caption})

    by = grouped_bars(summary, methods, sets, out, "results_by_set.png",
                      "MRR@10 for every method on each query set", (11.5, 5.4), 7)
    best = {s: max((m for m in methods if (m, s) in by), key=lambda m: by[(m, s)]) for s in sets}
    caption = "MRR@10 for all methods. Best per query set: " + "; ".join(
        f"{s} → {labels[best[s]]} ({by[(best[s], s)]:.3f})" for s in sets) + "."
    charts.append({"file": "results_by_set.png", "title": "MRR@10 by query set", "caption": caption})

    summary["charts"] = charts
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {summary_path} with {len(charts)} charts")


if __name__ == "__main__":
    main()
