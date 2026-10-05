"""Turn the cluster run reports into charts and results/cluster/summary.json (for the Results page).

    python cluster/charts.py
    python cluster/charts.py --main full --fault demo

Reads results/cluster/runs/*.json, scaling.json and precision_check.json. Produces:
  scaling.png               speedup on 1..n laptops vs. ideal (from bench_scaling.py)
  timeline_<run>.png        which worker processed which chunk, when (main build + fault run)
  chunks_per_laptop.png     share of the work each laptop did in the main build
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster.common import RESULTS_CLUSTER, laptop_of, read_json, write_json  # noqa: E402

LAPTOP_COLORS = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def load_runs() -> dict[str, dict]:
    runs = {}
    for path in sorted((RESULTS_CLUSTER / "runs").glob("*.json")):
        run = read_json(path)
        if run:
            runs[run["run"]] = run
    return runs


def laptop_colors(laptops: list[str]) -> dict[str, str]:
    return {lap: LAPTOP_COLORS[i % len(LAPTOP_COLORS)] for i, lap in enumerate(sorted(set(laptops)))}


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------

def scaling_table(runs: dict[str, dict], scaling: dict) -> list[dict]:
    base_lap = scaling["laptops"][0]
    by_set = {tuple(runs[n]["laptops"]): runs[n] for n in scaling["runs"] if n in runs}
    solo = {lap[0]: r["compute_s"] for lap, r in by_set.items() if len(lap) == 1}
    t1 = solo.get(base_lap)
    rows = []
    for laptops, r in sorted(by_set.items(), key=lambda kv: (len(kv[0]), kv[0])):
        n, t = len(laptops), r["compute_s"]
        row = {"laptops": list(laptops), "n": n, "seconds": t, "docs_per_s": r["docs_per_s"],
               "embed_stage_s": r["embed_stage_s"], "tokenize_stage_s": r["tokenize_stage_s"],
               "received_mb": r["received_mb"], "run": r["run"]}
        if t1:
            s = t1 / t
            row.update({"speedup": round(s, 2), "efficiency": round(s / n, 3)})
            if n > 1:  # Karp-Flatt: the serial fraction implied by the measured speedup
                row["serial_fraction"] = round((1 / s - 1 / n) / (1 - 1 / n), 3)
            if all(lap in solo for lap in laptops):
                # With unequal GPUs, the best possible time is the corpus divided by the
                # laptops' summed solo throughputs, not T1 / n.
                ideal_t = r["texts"] / sum(r["texts"] / solo[lap] for lap in laptops)
                row.update({"ideal_seconds": round(ideal_t, 2), "ideal_speedup": round(t1 / ideal_t, 2),
                            "cluster_efficiency": round(ideal_t / t, 3)})
        rows.append(row)
    return rows


def plot_scaling(rows: list[dict], scaling: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = {}
    for r in rows:   # the speedup curve: baseline alone, then the growing prefixes
        if r["laptops"] == scaling["laptops"][: r["n"]] and "speedup" in r:
            curve[r["n"]] = r
    ns = sorted(curve)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ns, ns, "--", color="#9ca3af", label="perfect (n×)")
    if all("ideal_speedup" in curve[n] for n in ns):
        ax.plot(ns, [curve[n]["ideal_speedup"] for n in ns], ":", color="#6b7280", marker="o", ms=4,
                label="ideal for these GPUs (sum of solo speeds)")
    ax.plot(ns, [curve[n]["speedup"] for n in ns], "-o", color="#2563eb", lw=2.2, label="measured")
    for n in ns:
        r = curve[n]
        ax.annotate(f"{r['speedup']:.2f}×\n{r['seconds']:.0f}s, eff {r['efficiency']:.0%}",
                    (n, r["speedup"]), textcoords="offset points", xytext=(-10, 10), ha="right", fontsize=8)
    ax.set_xticks(ns)
    ax.set_xlim(min(ns) - 0.3, max(ns) + 0.3)
    ax.set_ylim(0.8, max(ns) + 0.4)
    ax.set_xticklabels([" + ".join(curve[n]["laptops"]) for n in ns], fontsize=8)
    ax.set_xlabel("laptops")
    ax.set_ylabel(f"speedup vs. {scaling['laptops'][0]} alone")
    ax.set_title(f"Embedding + tokenizing {scaling['limit']:,} abstracts", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# timeline + per-laptop share
# ---------------------------------------------------------------------------

def plot_timeline(run: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def lane_key(worker):
        laptop, kind = laptop_of(worker)
        idx = worker.rsplit("-", 1)[-1]
        return laptop, kind != "gpu", int(idx) if idx.isdigit() else -1

    workers = sorted({t["worker"] for t in run["tasks"]}, key=lane_key)
    lane = {w: i for i, w in enumerate(workers)}
    colors = laptop_colors([laptop_of(w)[0] for w in workers])
    fig, ax = plt.subplots(figsize=(10, 1.4 + 0.22 * len(workers)))
    for t in run["tasks"]:
        lap, _ = laptop_of(t["worker"])
        ax.barh(lane[t["worker"]], t["end_s"] - t["start_s"], left=t["start_s"], height=0.8,
                color=colors[lap], alpha=1.0 if t["kind"] == "embed" else 0.45, edgecolor="white", linewidth=0.3)
    for e in run.get("events", []):
        if e["event"] == "left":
            ax.axvline(e["t"], color="#dc2626", ls="--", lw=1)
    lefts = sorted({(e["laptop"], e["t"]) for e in run.get("events", []) if e["event"] == "left"})
    if lefts:
        lap, t = lefts[0]
        ax.text(t, -0.9, f" {lap} disconnected", color="#dc2626", fontsize=8, va="top")
    ax.set_yticks(range(len(workers)))
    ax.set_yticklabels(workers, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("seconds since the job started")
    ax.set_title(f"Run '{run['run']}': {run['texts']:,} texts in {run['compute_s']:.0f}s "
                 f"(dark = GPU embedding, light = CPU tokenizing)", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_share(run: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    laps = sorted(run["per_laptop"])
    colors = laptop_colors(laps)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, kind, title in [(axes[0], "embed", "GPU: chunks embedded"), (axes[1], "tokenize", "CPU: chunks tokenized")]:
        vals = [run["per_laptop"][lap][f"{kind}_chunks"] for lap in laps]
        ax.bar(laps, vals, color=[colors[lap] for lap in laps])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v}\n({v / max(1, sum(vals)):.0%})", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.3 if vals else 1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Work done per laptop, run '{run['run']}'", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def run_summary(run: dict) -> dict:
    keep = ("run", "type", "texts", "laptops", "workers", "gpus", "precision", "chunk_size", "compute_s",
            "warmup_s", "embed_stage_s", "tokenize_stage_s", "received_mb", "docs_per_s", "per_laptop",
            "events", "resumed_chunks", "created")
    return {k: run[k] for k in keep if k in run}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="full", help="the main build run (default: full)")
    ap.add_argument("--fault", help="the fault-tolerance run (default: the latest run where a laptop left)")
    args = ap.parse_args()

    runs = load_runs()
    if not runs:
        sys.exit("No runs in results/cluster/runs/ yet.")
    summary = {"generated_at": datetime.now().isoformat(timespec="seconds"), "charts": []}

    scaling = read_json(RESULTS_CLUSTER / "scaling.json")
    if scaling:
        rows = scaling_table(runs, scaling)
        summary["scaling"] = {**{k: scaling[k] for k in ("laptops", "limit", "precision", "chunk_size", "warmup_s")},
                              "rows": rows}
        plot_scaling(rows, scaling, RESULTS_CLUSTER / "scaling.png")
        best = max((r for r in rows if "speedup" in r), key=lambda r: r["n"], default=None)
        caption = (f"{scaling['limit']:,} abstracts embedded and tokenized on growing sets of laptops. "
                   + (f"All {best['n']} laptops: {best['speedup']:.2f}× faster than {scaling['laptops'][0]} alone "
                      f"({best['efficiency']:.0%} efficiency)." if best else ""))
        summary["charts"].append({"file": "cluster/scaling.png", "title": "Speedup on 1 to n laptops", "caption": caption})
        print(f"scaling.png: {len(rows)} runs")

    main_run = runs.get(args.main)
    if main_run:
        summary["main_run"] = run_summary(main_run)
        plot_timeline(main_run, RESULTS_CLUSTER / f"timeline_{args.main}.png")
        plot_share(main_run, RESULTS_CLUSTER / "chunks_per_laptop.png")
        summary["charts"].append({"file": f"cluster/timeline_{args.main}.png", "title": "Who did what, when",
                                  "caption": f"Every chunk of the {main_run['texts']:,}-text build, by worker. "
                                             f"Finished in {main_run['compute_s']:.0f}s after {main_run.get('warmup_s', 0):.0f}s of model loading."})
        summary["charts"].append({"file": "cluster/chunks_per_laptop.png", "title": "Work per laptop",
                                  "caption": "Chunks are handed to whichever worker is free, so faster laptops take more."})
        print(f"timeline_{args.main}.png, chunks_per_laptop.png")

    fault_name = args.fault or next((n for n, r in sorted(runs.items(), key=lambda kv: kv[1].get("created", ""), reverse=True)
                                     if any(e["event"] == "left" for e in r.get("events", []))), None)
    if fault_name and fault_name in runs:
        fault = runs[fault_name]
        summary["fault_run"] = run_summary(fault)
        plot_timeline(fault, RESULTS_CLUSTER / f"timeline_{fault_name}.png")
        left = [e for e in fault["events"] if e["event"] == "left"]
        summary["charts"].append({
            "file": f"cluster/timeline_{fault_name}.png", "title": "Fault tolerance",
            "caption": (f"{left[0]['laptop']} was disconnected {left[0]['t']:.0f}s into the run; its unfinished chunks "
                        f"were re-run on the other laptops and all {fault['texts']:,} texts were still processed "
                        f"({fault['compute_s']:.0f}s in total).") if left else "No laptop left during this run."})
        print(f"timeline_{fault_name}.png")

    precision = read_json(RESULTS_CLUSTER / "precision_check.json")
    if precision:
        summary["precision_check"] = precision

    write_json(RESULTS_CLUSTER / "summary.json", summary)
    print("Wrote results/cluster/summary.json")


if __name__ == "__main__":
    main()
