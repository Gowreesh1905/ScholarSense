"""Saving chunks during a build, resuming, and assembling the finished index.

Assembling writes the embeddings and tokens into the same .cache/ files the searchers
already read (searchers/cache.py), keyed on the exact list of texts they are fitted on.
So BGESearcher, SpecterAdapter, BM25Searcher and HybridSearcher use the cluster-built
index with no code changes: their fit() finds a cache hit instead of re-embedding.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from cluster.common import (INDEX_MANIFEST, OUTPUT_DIR, job_texts, read_json, scale_texts,
                            sha1_texts, write_json)
from searchers.cache import cache_path, text_hash
from searchers.models import BGE_MODEL, SPECTER_MODEL

N_ORIGINAL = 727


def run_dir(run_name: str) -> Path:
    return OUTPUT_DIR / "runs" / run_name


def _chunk_path(run_name: str, kind: str, start: int) -> Path:
    ext = "npz" if kind == "embed" else "pkl"
    return run_dir(run_name) / "chunks" / f"{kind}_{start:07d}.{ext}"


def save_chunk(run_name: str, res: dict) -> None:
    path = _chunk_path(run_name, res["kind"], res["start"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        if res["kind"] == "embed":
            np.savez(f, bge=res["bge"], specter=res["specter"])
        else:
            pickle.dump(res["tokens"], f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def existing_chunks(run_name: str) -> tuple[set[int], set[int]]:
    folder = run_dir(run_name) / "chunks"
    if not folder.exists():
        return set(), set()
    embed, tok = set(), set()
    for p in folder.iterdir():
        kind, _, rest = p.stem.partition("_")
        if p.suffix in (".npz", ".pkl") and rest.isdigit():
            (embed if kind == "embed" else tok).add(int(rest))
    return embed, tok


def check_or_write_config(run_name: str, config: dict) -> None:
    """A resumed run must embed the same texts with the same settings."""
    path = run_dir(run_name) / "config.json"
    old = read_json(path)
    if old is None:
        write_json(path, config)
        return
    diff = {k: (old.get(k), v) for k, v in config.items() if old.get(k) != v}
    if diff:
        raise RuntimeError(
            f"Run {run_name!r} already exists with different settings {diff}. "
            f"Use another --run-name or delete {run_dir(run_name)}."
        )


def _write_cache(name: str, write) -> None:
    path = cache_path(name)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def cache_files(n_distractors: int, kinds=("full", "masked")) -> dict[str, Path]:
    """The .cache/ files the searchers will look for, per (kind, artifact)."""
    out = {}
    for kind in kinds:
        texts = scale_texts(kind, n_distractors)
        out[f"{kind}/bge"] = cache_path(f"emb_{text_hash(texts, BGE_MODEL, '')}.npy")
        out[f"{kind}/specter"] = cache_path(f"emb_{text_hash(texts, SPECTER_MODEL, '')}.npy")
        out[f"{kind}/bm25"] = cache_path(f"bm25_tok_{text_hash(texts, 'bm25')}.pkl")
    return out


def assemble(run_name: str, n_distractors: int, chunk_size: int, precision: str) -> dict:
    """Join the chunks of a finished build and seed the searchers' caches."""
    texts = job_texts(n_distractors)
    starts = list(range(0, len(texts), chunk_size))
    embed_done, tok_done = existing_chunks(run_name)
    missing = [s for s in starts if s not in embed_done or s not in tok_done]
    if missing:
        raise RuntimeError(f"Run {run_name!r} is missing {len(missing)} chunks; run build_index.py again to resume.")

    bge, specter, tokens = [], [], []
    for s in starts:
        with np.load(_chunk_path(run_name, "embed", s)) as z:
            bge.append(z["bge"].astype(np.float32))       # float32: numpy has no fast float16 matmul
            specter.append(z["specter"].astype(np.float32))
        with open(_chunk_path(run_name, "tokenize", s), "rb") as f:
            tokens.extend(pickle.load(f))
    bge, specter = np.vstack(bge), np.vstack(specter)
    assert len(bge) == len(specter) == len(tokens) == len(texts)

    # job order: original 727 | masked 727 | distractors
    parts = {
        "full": np.r_[0:N_ORIGINAL, 2 * N_ORIGINAL:len(texts)],
        "masked": np.r_[N_ORIGINAL:2 * N_ORIGINAL, 2 * N_ORIGINAL:len(texts)],
    }
    files = cache_files(n_distractors)
    for kind, rows in parts.items():
        _write_cache(files[f"{kind}/bge"].name, lambda f, a=bge[rows]: np.save(f, a))
        _write_cache(files[f"{kind}/specter"].name, lambda f, a=specter[rows]: np.save(f, a))
        toks = [tokens[i] for i in rows]
        _write_cache(files[f"{kind}/bm25"].name, lambda f, t=toks: pickle.dump(t, f, protocol=pickle.HIGHEST_PROTOCOL))

    manifest = {
        "run": run_name,
        "n_distractors": n_distractors,
        "corpus_size": N_ORIGINAL + n_distractors,
        "precision": precision,
        "chunk_size": chunk_size,
        "texts_sha1": sha1_texts(texts),
        "cache_files": {k: p.name for k, p in files.items()},
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(INDEX_MANIFEST, manifest)
    return manifest


def ready_index(kinds=("full",)) -> dict | None:
    """The index manifest if every cache file it needs exists, else None."""
    manifest = read_json(INDEX_MANIFEST)
    if not manifest:
        return None
    try:
        files = cache_files(manifest["n_distractors"], kinds)
    except FileNotFoundError:
        return None
    return manifest if all(p.exists() for p in files.values()) else None
