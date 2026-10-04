"""Shared paths and helpers for the evaluation scripts (team/CONTRACT.md §7).

Run every script from the repo root, e.g. `python eval/build_queries.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # root-level data.py and the searchers/ package

EVAL_DIR = ROOT / "eval"
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"

CORPUS_MASKED = DATA_DIR / "corpus_masked.json"
QUERY_FILES = {
    "exact": DATA_DIR / "queries_exact.jsonl",
    "paraphrased": DATA_DIR / "queries_paraphrased.jsonl",
    "handwritten": DATA_DIR / "queries_handwritten.jsonl",
}

OVERLAP_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def overlap(query: str, doc_text: str) -> float:
    """|content words of query ∩ content words of doc| ÷ |content words of query| (CONTRACT §7)."""
    from data import content_words

    q = set(content_words(query))
    if not q:
        return 0.0
    return len(q & set(content_words(doc_text))) / len(q)


def max_overlap(query: str, doc_texts: list[str]) -> float:
    """Overlap against several relevant docs: the maximum."""
    from data import content_words

    q = set(content_words(query))
    if not q:
        return 0.0
    return max(len(q & set(content_words(t))) / len(q) for t in doc_texts)


def overlap_bin(value: float) -> int:
    """Index 0-4 of the bin [0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1]."""
    return min(int(value * 5), 4)


def load_corpus_for(corpus: str) -> list[str]:
    """Abstracts to search: 'masked' -> corpus_masked.json, 'full' -> original abstracts."""
    from data import load_corpus

    if corpus == "masked":
        with open(CORPUS_MASKED, encoding="utf-8") as f:
            return json.load(f)
    return [p.abstract for p in load_corpus()]
