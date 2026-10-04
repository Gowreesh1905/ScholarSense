"""On-disk cache for expensive artifacts, under <repo>/.cache/ (gitignored)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def text_hash(texts: list[str], *extra: object) -> str:
    """sha1 over `extra` (model id, tag, parameters) and every text, in order."""
    h = hashlib.sha1()
    for part in extra:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    for t in texts:
        h.update(t.encode("utf-8", errors="surrogatepass"))
        h.update(b"\x1e")
    return h.hexdigest()


def cache_path(name: str) -> Path:
    """Path of a file inside the cache dir (the dir is created if needed)."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / name


def cached_encode(model_id: str, texts: list[str],
                  encode_fn: Callable[[list[str]], np.ndarray], tag: str = "") -> np.ndarray:
    """Return encode_fn(texts), cached at .cache/emb_<sha1(model_id + tag + all texts)>.npy."""
    path = cache_path(f"emb_{text_hash(texts, model_id, tag)}.npy")
    if path.exists():
        try:
            return np.load(path)
        except Exception as exc:  # corrupt/partial file -> re-encode
            print(f"[cache] Could not read {path.name} ({exc}); re-encoding.")

    embeddings = np.asarray(encode_fn(texts))
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, embeddings)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written cache file
    return embeddings
