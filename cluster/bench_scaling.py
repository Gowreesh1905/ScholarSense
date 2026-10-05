"""Scaling experiment: the same job on 1, 2 and 3 laptops (run on laptop A).

    python cluster/bench_scaling.py --laptops laptopA laptopB laptopC
    python cluster/bench_scaling.py --laptops laptopA laptopB laptopC --limit 30000 --no-solo

All laptops stay connected; each run is restricted to a subset of them. Runs:
  1. each laptop alone (its solo speed; needed for a fair "ideal" with unequal GPUs)
  2. the first two laptops, then all three (the speedup curve)
Models are loaded on every laptop before the first run, so the timings measure the
work itself; model loading is reported separately as a fixed (serial) cost.
Embeddings are received but not saved. Reports go to results/cluster/runs/ and the
experiment list to results/cluster/scaling.json; then run cluster/charts.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster.build_index import DEFAULT_CHUNK, connect, print_cluster  # noqa: E402
from cluster.common import RESULTS_CLUSTER, load_distractors, write_json  # noqa: E402
from cluster.job import check_environment, discover, run_job, warmup  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheduler", default="127.0.0.1")
    ap.add_argument("--laptops", nargs="+", required=True, help="in order; the first one is the 1-laptop baseline")
    ap.add_argument("--limit", type=int, default=30_000, help="distractors per run (default 30000)")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--no-solo", action="store_true", help="skip the solo runs of laptops 2..n")
    ap.add_argument("--pause", type=float, default=10, help="seconds between runs (lets GPUs cool a little)")
    args = ap.parse_args()

    client = connect(args.scheduler)
    view = discover(client)
    print_cluster(view, check_environment(client, view))
    missing = [lap for lap in args.laptops if lap not in view.laptops]
    if missing:
        sys.exit(f"Not connected: {missing}")

    texts = [d["abstract"] for d in load_distractors(args.limit)]
    print(f"Loading models on {', '.join(args.laptops)} ...", flush=True)
    t = time.time()
    warm = warmup(client, view, args.laptops, args.precision)
    warm_s = round(time.time() - t, 2)
    print(f"  ready in {warm_s}s", flush=True)

    plan = [[lap] for lap in (args.laptops[:1] if args.no_solo else args.laptops)]
    plan += [args.laptops[:k] for k in range(2, len(args.laptops) + 1)]
    names = []
    for i, laptops in enumerate(plan):
        name = f"bench-{args.limit // 1000}k-{'+'.join(laptops)}"
        print(f"\n=== run {i + 1}/{len(plan)}: {' + '.join(laptops)} ({len(texts):,} abstracts) ===", flush=True)
        report = run_job(client, texts, laptops, run_name=name, precision=args.precision,
                         chunk_size=args.chunk_size, restrict=True)
        report.update({"type": "bench", "warmup_s": warm_s, "warmup_per_worker": warm,
                       "created": datetime.now().isoformat(timespec="seconds")})
        write_json(RESULTS_CLUSTER / "runs" / f"{name}.json", report)
        names.append(name)
        print(f"  {report['compute_s']}s  ({report['docs_per_s']} abstracts/s)", flush=True)
        if i < len(plan) - 1:
            time.sleep(args.pause)

    write_json(RESULTS_CLUSTER / "scaling.json", {
        "laptops": args.laptops, "limit": args.limit, "precision": args.precision,
        "chunk_size": args.chunk_size, "runs": names, "warmup_s": warm_s,
        "warmup_per_worker": warm, "created": datetime.now().isoformat(timespec="seconds"),
    })
    print("\nSaved results/cluster/scaling.json. Now run: python cluster/charts.py")
    client.close()


if __name__ == "__main__":
    main()
