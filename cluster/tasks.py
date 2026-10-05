"""Functions that run on the Dask workers (every laptop imports this from its own checkout).

GPU workers run `embed_chunk` (BGE + SPECTER); CPU workers run `tokenize_chunk` (the BM25
tokenizer). Models are loaded once per worker process and reused by every chunk.
Each task returns its own timings so the client can draw who did what, and when.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENCODE_BATCH = 64


def _worker_name(dask_worker=None) -> str:
    if dask_worker is not None:  # functions run with client.run() get the worker passed in
        return str(dask_worker.name)
    try:
        from distributed import get_worker
        return get_worker().name
    except Exception:  # running outside a worker (tests)
        return f"{socket.gethostname()}-local"


@lru_cache(maxsize=None)
def _models(precision: str):
    """BGE and SPECTER on this worker's GPU, loaded once per process and precision."""
    import torch
    from sentence_transformers import SentenceTransformer

    from searchers.models import BGE_MODEL, SPECTER_MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bge = SentenceTransformer(BGE_MODEL, device=device)
    specter = SentenceTransformer(SPECTER_MODEL, device=device)
    if precision == "fp16" and device == "cuda":
        bge.half()
        specter.half()
    return bge, specter


def _specter_clean(text: str) -> str:
    # The same cleaning BERTSearcher applies before SPECTER (bert_specter_contextual_search.py).
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def warmup_gpu(precision: str, dask_worker=None) -> dict:
    """Load the models (and initialise CUDA) before timing starts."""
    t = time.time()
    bge, specter = _models(precision)
    bge.encode(["warm-up"], batch_size=1)
    specter.encode(["warm-up"], batch_size=1)
    return {"worker": _worker_name(dask_worker), "seconds": time.time() - t}


def warmup_cpu(dask_worker=None) -> dict:
    """Load NLTK and the TF-IDF preprocessor before timing starts."""
    t = time.time()
    from data import content_words
    content_words("warm up the tokenizer")
    return {"worker": _worker_name(dask_worker), "seconds": time.time() - t}


def embed_chunk(start: int, texts: list[str], precision: str) -> dict:
    """BGE (normalised, as BGESearcher stores it) and SPECTER (raw, as SpecterAdapter caches it)."""
    t0 = time.time()
    bge, specter = _models(precision)
    b = bge.encode(texts, batch_size=ENCODE_BATCH, normalize_embeddings=True, convert_to_numpy=True)
    s = specter.encode([_specter_clean(t) for t in texts], batch_size=ENCODE_BATCH, convert_to_numpy=True)
    return {
        "kind": "embed", "start": start, "n": len(texts),
        "bge": b.astype("float16"), "specter": s.astype("float16"),   # half the bytes on the network
        "worker": _worker_name(), "t_start": t0, "t_end": time.time(),
    }


def tokenize_chunk(start: int, texts: list[str]) -> dict:
    """The exact tokens BM25Searcher would produce (data.content_words)."""
    t0 = time.time()
    from data import content_words
    tokens = [content_words(t) for t in texts]
    return {
        "kind": "tokenize", "start": start, "n": len(texts), "tokens": tokens,
        "worker": _worker_name(), "t_start": t0, "t_end": time.time(),
    }


def worker_info(probe_gpu: bool = True, dask_worker=None) -> dict:
    """What this worker runs on, so the client can refuse mismatched laptops.
    probe_gpu=False skips importing torch (CPU workers never need it)."""
    import importlib.metadata as md

    def version(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None

    gpu = None
    cuda = False
    if probe_gpu:
        try:
            import torch
            cuda = torch.cuda.is_available()
            gpu = torch.cuda.get_device_name(0) if cuda else None
        except Exception:
            pass

    return {
        "worker": _worker_name(dask_worker),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("dask", "distributed", "torch", "sentence-transformers",
                                              "transformers", "nltk", "scikit-learn", "numpy")},
        "commit": commit,
        "cuda": cuda,
        "gpu": gpu,
        "pid": os.getpid(),
        "time": time.time(),
    }
