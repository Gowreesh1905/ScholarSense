"""Paths and helpers shared by the cluster scripts, the backend and the scale evaluation.

The scaled corpus = the 727 original papers (doc_index 0-726, unchanged) followed by
arXiv computer-science abstracts used as distractors. Because the 727 keep their
doc_index, every query's `relevant_docs` stays valid at any corpus size.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLUSTER_DIR = ROOT / "cluster"
DATA_DIR = CLUSTER_DIR / "data"                       # gitignored: downloaded corpus
OUTPUT_DIR = CLUSTER_DIR / "output"                   # gitignored: embedding chunks, index manifest
RESULTS_CLUSTER = ROOT / "results" / "cluster"        # committed: run reports, charts
RESULTS_SCALE = ROOT / "results" / "scale"            # committed: evaluation at scale

DISTRACTORS_FILE = DATA_DIR / "distractors.jsonl"
DISTRACTORS_MANIFEST = DATA_DIR / "distractors_manifest.json"
INDEX_MANIFEST = OUTPUT_DIR / "index_manifest.json"
MASKED_CORPUS = ROOT / "eval" / "data" / "corpus_masked.json"

SCHEDULER_PORT = 8786
DASHBOARD_PORT = 8787

# Methods that run on the scaled corpus. TF-IDF is left out (BM25 represents the lexical
# family and scored within 0.005 of it at 727); Word2Vec/GloVe are left out because the
# static-vector family was far behind at 727 and retraining Word2Vec on 100k adds nothing.
SCALE_METHODS = ["bm25", "specter", "bge", "hybrid_rrf", "hybrid"]

_LAPTOP_NAME = re.compile(r"^(?P<laptop>.+)-(?P<kind>gpu|cpu)(?:-\d+)?$")


@dataclass(frozen=True)
class ScalePaper:
    doc_index: int
    arxiv_id: str | None
    abstract: str
    title: str | None


def laptop_of(worker_name: str) -> tuple[str, str]:
    """'laptopB-cpu-3' -> ('laptopB', 'cpu'); 'laptopA-gpu' -> ('laptopA', 'gpu')."""
    m = _LAPTOP_NAME.match(str(worker_name))
    if not m:
        return str(worker_name), "unknown"
    return m["laptop"], m["kind"]


def sha1_texts(texts: list[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8", errors="surrogatepass"))
        h.update(b"\x1e")
    return h.hexdigest()


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


@lru_cache(maxsize=1)
def _distractors() -> tuple[dict, ...]:
    if not DISTRACTORS_FILE.exists():
        raise FileNotFoundError(
            f"{DISTRACTORS_FILE} not found. Run `python cluster/prepare_corpus.py` on this laptop first."
        )
    with open(DISTRACTORS_FILE, encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f if line.strip())


def load_distractors(n: int | None = None) -> list[dict]:
    """The first n distractors ({arxiv_id, title, abstract, categories, year}), in file order."""
    rows = _distractors()
    if n is not None and n > len(rows):
        raise ValueError(f"Asked for {n} distractors but {DISTRACTORS_FILE.name} has {len(rows)}.")
    return list(rows[:n] if n is not None else rows)


def original_texts(kind: str) -> list[str]:
    """The 727 abstracts: 'full' = original text, 'masked' = task/problem spans removed (P3)."""
    if kind == "full":
        from data import load_corpus
        return [p.abstract for p in load_corpus()]
    if kind == "masked":
        return json.loads(MASKED_CORPUS.read_text(encoding="utf-8"))
    raise ValueError(f"kind must be 'full' or 'masked', not {kind!r}")


def scale_texts(kind: str, n_distractors: int) -> list[str]:
    """The exact list of texts a searcher is fitted on at scale. Everything that seeds or
    reads the embedding caches must build its list with this function, so the cache keys match."""
    return original_texts(kind) + [d["abstract"] for d in load_distractors(n_distractors)]


def job_texts(n_distractors: int) -> list[str]:
    """What the cluster embeds: original 727 + masked 727 + distractors (each text once)."""
    return original_texts("full") + original_texts("masked") + [d["abstract"] for d in load_distractors(n_distractors)]


def scale_papers(n_distractors: int) -> list[ScalePaper]:
    from data import load_corpus
    papers = [ScalePaper(p.doc_index, p.arxiv_id, p.abstract, None) for p in load_corpus()]
    base = len(papers)
    papers += [
        ScalePaper(base + i, d.get("arxiv_id"), d["abstract"], d.get("title"))
        for i, d in enumerate(load_distractors(n_distractors))
    ]
    return papers
