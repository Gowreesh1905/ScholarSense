"""The distributed job: send text chunks to the laptops, collect embeddings and tokens.

Used by build_index.py (the real 100k index) and bench_scaling.py (timing runs).

How work is shared: every chunk becomes one GPU task (embed with BGE + SPECTER) and
one CPU task (tokenize for BM25). Tasks carry a Dask resource tag, so GPU tasks only go
to GPU workers and CPU tasks only to CPU workers, on any laptop. The client keeps a
small window of tasks in flight per worker and submits the next chunk whenever one
finishes, so a faster laptop simply finishes more chunks. If a laptop disappears, Dask
re-runs its unfinished tasks on the others.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from cluster import tasks
from cluster.common import laptop_of

GPU_WINDOW_PER_WORKER = 3
CPU_WINDOW_PER_WORKER = 2

# Must match on every machine: the client pickles tasks for the workers.
STRICT_VERSIONS = ("dask", "distributed")
# Should match: different versions could compute slightly different embeddings/tokens.
SOFT_VERSIONS = ("torch", "sentence-transformers", "transformers", "nltk", "scikit-learn")


# ---------------------------------------------------------------------------
# cluster discovery
# ---------------------------------------------------------------------------

def _scheduler_workers(dask_scheduler=None) -> dict:
    return {
        addr: {"name": str(ws.name), "resources": dict(ws.resources), "host": ws.host}
        for addr, ws in dask_scheduler.workers.items()
    }


@dataclass
class ClusterView:
    workers: dict[str, dict]                                  # addr -> {name, resources, host}
    gpu: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))   # laptop -> addrs
    cpu: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    @property
    def laptops(self) -> list[str]:
        return sorted(set(self.gpu) | set(self.cpu))

    def name_to_addr(self) -> dict[str, str]:
        return {w["name"]: a for a, w in self.workers.items()}


def discover(client) -> ClusterView:
    workers = client.run_on_scheduler(_scheduler_workers)
    view = ClusterView(workers)
    for addr, w in workers.items():
        laptop, _ = laptop_of(w["name"])
        if w["resources"].get("GPU"):
            view.gpu[laptop].append(addr)
        elif w["resources"].get("CPU"):
            view.cpu[laptop].append(addr)
    return view


def check_environment(client, view: ClusterView, allow_commit_mismatch: bool = False) -> dict:
    """Compare every worker with this machine. Raises on mismatches that would break the job."""
    here = tasks.worker_info(probe_gpu=False)
    gpu_addrs = [a for addrs in view.gpu.values() for a in addrs]
    cpu_addrs = [a for addrs in view.cpu.values() for a in addrs]
    infos = {}
    if gpu_addrs:
        infos.update(client.run(tasks.worker_info, True, workers=gpu_addrs))
    if cpu_addrs:
        infos.update(client.run(tasks.worker_info, False, workers=cpu_addrs))
    errors, warnings = [], []
    for addr, info in infos.items():
        who = f"{info['worker']} ({info['host']})"
        if info["python"].rsplit(".", 1)[0] != here["python"].rsplit(".", 1)[0]:
            errors.append(f"{who}: Python {info['python']} vs {here['python']} here")
        for pkg in STRICT_VERSIONS:
            if info["packages"][pkg] != here["packages"][pkg]:
                errors.append(f"{who}: {pkg} {info['packages'][pkg]} vs {here['packages'][pkg]} here")
        for pkg in SOFT_VERSIONS:
            if info["packages"][pkg] != here["packages"][pkg]:
                warnings.append(f"{who}: {pkg} {info['packages'][pkg]} vs {here['packages'][pkg]} here")
        if info["commit"] != here["commit"]:
            (warnings if allow_commit_mismatch else errors).append(
                f"{who}: git commit {info['commit']} vs {here['commit']} here (run `git pull` on that laptop)")
        if addr in gpu_addrs and not info["cuda"]:
            errors.append(f"{who}: GPU worker but PyTorch has no CUDA (install the CUDA build of torch)")
    for w in warnings:
        print(f"  WARNING  {w}")
    if errors:
        raise RuntimeError("Cluster environment mismatch:\n  " + "\n  ".join(errors))
    return infos


def clock_offsets(client) -> dict[str, float]:
    """worker clock - client clock, per worker address (laptop clocks can differ by seconds)."""
    t0 = time.time()
    times = client.run(time.time)
    t1 = time.time()
    mid = (t0 + t1) / 2
    return {addr: t - mid for addr, t in times.items()}


def warmup(client, view: ClusterView, laptops: list[str], precision: str) -> dict[str, dict]:
    """Load models on the GPU workers and NLTK on the CPU workers; returns seconds per worker.

    Submitted as ordinary tasks (one pinned to each worker), not with client.run(): client.run
    executes on the worker's network loop, so a 20 s model load would stop its heartbeats and
    the scheduler would declare the worker dead.
    """
    from distributed import wait

    pending = {}
    for lap in laptops:
        for addr in view.gpu.get(lap, []):
            pending[client.submit(tasks.warmup_gpu, precision, workers=[addr], resources={"GPU": 1},
                                  allow_other_workers=False, pure=False, key=f"warmup-{addr}")] = addr
        for addr in view.cpu.get(lap, []):
            pending[client.submit(tasks.warmup_cpu, workers=[addr], resources={"CPU": 1},
                                  allow_other_workers=False, pure=False, key=f"warmup-{addr}")] = addr

    # A warm-up task is pinned to its worker, so if that worker dies it can never finish:
    # drop it instead of waiting forever.
    results = {}
    while pending:
        try:
            wait(list(pending), timeout=2, return_when="FIRST_COMPLETED")
        except TimeoutError:
            pass
        for fut in [f for f in pending if f.done()]:
            addr = pending.pop(fut)
            if fut.status == "finished":
                r = fut.result()
                results[r["worker"]] = {"seconds": round(r["seconds"], 2)}
            else:
                print(f"  WARNING  warm-up failed on {view.workers[addr]['name']}: {fut.exception()}")
        alive = set(client.run_on_scheduler(_scheduler_workers))
        for fut in [f for f, a in pending.items() if a not in alive]:
            print(f"  WARNING  {view.workers[pending.pop(fut)]['name']} left during warm-up; skipping it")
            fut.cancel()
    return results


# ---------------------------------------------------------------------------
# membership monitor (records laptops leaving/joining during a run)
# ---------------------------------------------------------------------------

class MembershipMonitor(threading.Thread):
    def __init__(self, client, t0: float, interval: float = 1.0):
        super().__init__(daemon=True)
        self.client, self.t0, self.interval = client, t0, interval
        self.events: list[dict] = []
        self._stop = threading.Event()
        self._known: dict[str, str] = {}

    def run(self):
        while not self._stop.is_set():
            try:
                now = {a: w["name"] for a, w in self.client.run_on_scheduler(_scheduler_workers).items()}
            except Exception:
                now = self._known
            t = round(time.time() - self.t0, 2)
            if self._known:
                for addr in self._known.keys() - now.keys():
                    self.events.append({"t": t, "event": "left", "worker": self._known[addr],
                                        "laptop": laptop_of(self._known[addr])[0]})
                    print(f"  [t={t:7.1f}s] worker LEFT:   {self._known[addr]}", flush=True)
                for addr in now.keys() - self._known.keys():
                    self.events.append({"t": t, "event": "joined", "worker": now[addr],
                                        "laptop": laptop_of(now[addr])[0]})
                    print(f"  [t={t:7.1f}s] worker JOINED: {now[addr]}", flush=True)
            self._known = now
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------

def _payload_bytes(res: dict) -> int:
    if res["kind"] == "embed":
        return int(res["bge"].nbytes + res["specter"].nbytes)
    return sum(len(tok) + 1 for doc in res["tokens"] for tok in doc)


def run_job(client, texts: list[str], laptops: list[str], *, run_name: str, precision: str,
            chunk_size: int, on_result=None, skip_embed: set[int] = frozenset(),
            skip_tokenize: set[int] = frozenset(), restrict: bool = True) -> dict:
    """Embed and tokenize `texts` on the given laptops. Returns a timing report.

    on_result(res) is called on this machine for every finished chunk (e.g. to save it).
    restrict=False lets workers that join mid-run take tasks too (used for the main build).
    """
    from distributed import as_completed

    view = discover(client)
    missing = [lap for lap in laptops if lap not in view.laptops]
    if missing:
        raise RuntimeError(f"Laptop(s) not connected: {missing}. Connected: {view.laptops}")
    gpu_addrs = [a for lap in laptops for a in view.gpu.get(lap, [])]
    cpu_addrs = [a for lap in laptops for a in view.cpu.get(lap, [])]
    if not gpu_addrs:
        raise RuntimeError(f"No GPU workers on {laptops}. Start them with cluster/start_workers.ps1.")

    offsets = clock_offsets(client)
    name_of = {a: w["name"] for a, w in view.workers.items()}
    offset_by_name = {name_of[a]: off for a, off in offsets.items() if a in name_of}

    starts = list(range(0, len(texts), chunk_size))
    pending = {
        "embed": [s for s in starts if s not in skip_embed],
        "tokenize": [s for s in starts if s not in skip_tokenize],
    }
    window = {
        "embed": GPU_WINDOW_PER_WORKER * len(gpu_addrs),
        "tokenize": CPU_WINDOW_PER_WORKER * max(1, len(cpu_addrs)),
    }
    in_flight = {"embed": 0, "tokenize": 0}

    def submit(kind: str):
        start = pending[kind].pop(0)
        chunk = texts[start:start + chunk_size]
        if kind == "embed":
            fn, args, res, addrs = tasks.embed_chunk, (start, chunk, precision), {"GPU": 1}, gpu_addrs
        elif cpu_addrs:
            fn, args, res, addrs = tasks.tokenize_chunk, (start, chunk), {"CPU": 1}, cpu_addrs
        else:  # no CPU workers started: the GPU workers tokenize too
            fn, args, res, addrs = tasks.tokenize_chunk, (start, chunk), {"GPU": 1}, gpu_addrs
        in_flight[kind] += 1
        return client.submit(fn, *args, key=f"{kind}-{run_name}-{start:07d}", resources=res,
                             workers=addrs if restrict else None, allow_other_workers=False,
                             pure=False, retries=2)

    t0 = time.time()
    monitor = MembershipMonitor(client, t0)
    monitor.start()
    ac = as_completed()
    for kind in ("embed", "tokenize"):
        while pending[kind] and in_flight[kind] < window[kind]:
            ac.add(submit(kind))

    total = len(pending["embed"]) + len(pending["tokenize"]) + sum(in_flight.values())
    records, done, received_bytes = [], 0, 0
    first_result = None
    try:
        for fut in ac:
            res = fut.result()
            received = time.time()
            fut.release()
            first_result = first_result or received
            kind = res["kind"]
            in_flight[kind] -= 1
            if pending[kind]:
                ac.add(submit(kind))

            off = offset_by_name.get(res["worker"], 0.0)
            laptop, _ = laptop_of(res["worker"])
            records.append({
                "kind": kind, "start": res["start"], "n": res["n"],
                "worker": res["worker"], "laptop": laptop,
                "start_s": round(res["t_start"] - off - t0, 3),
                "end_s": round(res["t_end"] - off - t0, 3),
                "received_s": round(received - t0, 3),
            })
            received_bytes += _payload_bytes(res)
            if on_result:
                on_result(res)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                print(f"  {done:>5}/{total} chunks  ({time.time() - t0:6.1f}s)", flush=True)
    finally:
        monitor.stop()

    t_end = time.time()

    def stage_end(kind):
        ends = [r["received_s"] for r in records if r["kind"] == kind]
        return round(max(ends), 2) if ends else 0.0

    per_laptop = defaultdict(lambda: {"embed_chunks": 0, "tokenize_chunks": 0, "embed_busy_s": 0.0,
                                      "tokenize_busy_s": 0.0, "docs_embedded": 0})
    for r in records:
        p = per_laptop[r["laptop"]]
        p[f"{r['kind']}_chunks"] += 1
        p[f"{r['kind']}_busy_s"] = round(p[f"{r['kind']}_busy_s"] + (r["end_s"] - r["start_s"]), 2)
        if r["kind"] == "embed":
            p["docs_embedded"] += r["n"]

    return {
        "run": run_name,
        "texts": len(texts),
        "chunk_size": chunk_size,
        "precision": precision,
        "laptops": laptops,
        "workers": {lap: {"gpu": [name_of[a] for a in view.gpu.get(lap, [])],
                          "cpu": len(view.cpu.get(lap, []))} for lap in laptops},
        "compute_s": round(t_end - t0, 2),
        "first_result_s": round((first_result or t_end) - t0, 2),
        "embed_stage_s": stage_end("embed"),
        "tokenize_stage_s": stage_end("tokenize"),
        "received_mb": round(received_bytes / 1e6, 1),
        "docs_per_s": round(len(texts) / max(1e-9, t_end - t0), 1),
        "per_laptop": dict(per_laptop),
        "events": monitor.events,
        "tasks": sorted(records, key=lambda r: (r["kind"], r["start"])),
    }
