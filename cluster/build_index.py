"""Build the 100k-paper search index on the laptop cluster (run on laptop A).

    python cluster/build_index.py --check                      # who is connected, versions match?
    python cluster/build_index.py                              # full build, saves + assembles
    python cluster/build_index.py --limit 30000 --run-name demo --discard    # live demo / fault test

Every chunk of abstracts is embedded with BGE and SPECTER on some laptop's GPU and
tokenized for BM25 on some laptop's CPU cores. Finished chunks are saved under
cluster/output/runs/<run>/ as they arrive, so an interrupted build resumes where it
stopped. When all chunks are in, the index is assembled into .cache/ for the backend
and the evaluation. A timing report goes to results/cluster/runs/<run>.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster.common import RESULTS_CLUSTER, SCHEDULER_PORT, job_texts, load_distractors, write_json  # noqa: E402
from cluster.index import assemble, check_or_write_config, existing_chunks, save_chunk  # noqa: E402
from cluster.job import check_environment, discover, run_job, warmup  # noqa: E402

DEFAULT_CHUNK = 250


def connect(scheduler: str):
    from distributed import Client
    address = scheduler if "://" in scheduler else f"tcp://{scheduler}:{SCHEDULER_PORT}"
    print(f"Connecting to the scheduler at {address} ...", flush=True)
    return Client(address, timeout="20s")


def print_cluster(view, infos: dict) -> None:
    gpu_name = {i["worker"]: i.get("gpu") for i in infos.values()}
    print(f"\nConnected laptops ({len(view.laptops)}):")
    for lap in view.laptops:
        gpus = [view.workers[a]["name"] for a in view.gpu.get(lap, [])]
        names = ", ".join(f"{g} [{gpu_name.get(g) or 'no CUDA'}]" for g in gpus) or "no GPU worker"
        host = view.workers[(view.gpu.get(lap) or view.cpu.get(lap))[0]]["host"]
        print(f"  {lap:<12} {host:<16} GPU: {names};  CPU workers: {len(view.cpu.get(lap, []))}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheduler", default="127.0.0.1", help="scheduler IP (default: this laptop)")
    ap.add_argument("--limit", type=int, default=None, help="number of distractors (default: all prepared)")
    ap.add_argument("--run-name", default="full")
    ap.add_argument("--laptops", nargs="+", help="only use these laptops (default: all connected)")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--discard", action="store_true", help="don't save chunks or assemble (demo / timing runs)")
    ap.add_argument("--check", action="store_true", help="only show the connected laptops and check versions")
    ap.add_argument("--allow-commit-mismatch", action="store_true")
    args = ap.parse_args()

    client = connect(args.scheduler)
    view = discover(client)
    if not view.laptops:
        sys.exit("No workers connected. Start them with cluster/start_workers.ps1 on each laptop.")
    infos = check_environment(client, view, args.allow_commit_mismatch)
    print_cluster(view, infos)
    if args.check:
        print("Environment OK.")
        return

    laptops = args.laptops or view.laptops
    n_distractors = args.limit if args.limit is not None else len(load_distractors())
    texts = job_texts(n_distractors)
    print(f"Job: {len(texts):,} texts (727 original + 727 masked + {n_distractors:,} distractors), "
          f"chunks of {args.chunk_size}, {args.precision}, laptops: {', '.join(laptops)}")

    skip_embed, skip_tok = set(), set()
    on_result = None
    if not args.discard:
        check_or_write_config(args.run_name, {"n_distractors": n_distractors, "chunk_size": args.chunk_size,
                                              "precision": args.precision})
        skip_embed, skip_tok = existing_chunks(args.run_name)
        if skip_embed or skip_tok:
            print(f"Resuming: {len(skip_embed)} embed and {len(skip_tok)} tokenize chunks already saved.")
        on_result = lambda res: save_chunk(args.run_name, res)  # noqa: E731

    print("Loading models on every laptop (timed separately)...", flush=True)
    t = time.time()
    warm = warmup(client, view, laptops, args.precision)
    warm_s = round(time.time() - t, 2)
    print(f"  ready in {warm_s}s\n\nRunning (dashboard: http://{args.scheduler}:8787/status) ...", flush=True)

    report = run_job(client, texts, laptops, run_name=args.run_name, precision=args.precision,
                     chunk_size=args.chunk_size, on_result=on_result, skip_embed=skip_embed,
                     skip_tokenize=skip_tok, restrict=args.laptops is not None)
    report.update({
        "type": "build" if not args.discard else "demo",
        "n_distractors": n_distractors,
        "resumed_chunks": len(skip_embed) + len(skip_tok),
        "warmup_s": warm_s,
        "warmup_per_worker": warm,
        "gpus": {i["worker"]: i.get("gpu") for i in infos.values() if i.get("gpu")},
        "created": datetime.now().isoformat(timespec="seconds"),
    })

    out = RESULTS_CLUSTER / "runs" / f"{args.run_name}.json"
    write_json(out, report)
    print(f"\nDone in {report['compute_s']}s (+{warm_s}s model loading). "
          f"{report['docs_per_s']} texts/s, {report['received_mb']} MB received.")
    for lap, p in sorted(report["per_laptop"].items()):
        print(f"  {lap:<12} embedded {p['embed_chunks']:>4} chunks, tokenized {p['tokenize_chunks']:>4} chunks")
    for e in report["events"]:
        print(f"  t={e['t']}s  {e['worker']} {e['event']}")
    print(f"Timing report: {out}")

    if not args.discard:
        print("\nAssembling the index...", flush=True)
        manifest = assemble(args.run_name, n_distractors, args.chunk_size, args.precision)
        print(f"Index ready: {manifest['corpus_size']:,} papers. The backend and "
              f"`python cluster/eval_scale.py` will pick it up.")
    client.close()


if __name__ == "__main__":
    main()
