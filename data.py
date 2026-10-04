"""
ScholarSense — shared dataset loader (team/CONTRACT.md §1).

Every module identifies papers by `doc_index` (position in `load_corpus()`),
never by arXiv ID. `arxiv_id` is always read as a string: reading it as a
float turns "2101.06860" into "2101.0686".

    python data.py      # prints a short summary of the corpus and spans
"""

from __future__ import annotations

import contextlib
import io
import string
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

CSV_PATH: Path = Path(__file__).resolve().parent / "abstract_sentences.csv"

ASPECT_OF_LABEL = {"task": "task", "problem": "problem", "idea": "method", "result": "result"}


@dataclass(frozen=True)
class Paper:
    doc_index: int            # position in load_corpus()
    arxiv_id: str | None      # read as str; None for the 18 papers without one
    abstract: str


@dataclass(frozen=True)
class Span:
    doc_index: int
    aspect: str               # "task" | "problem" | "method" | "result"
    start: int                # cleaned char offsets into the abstract, end exclusive
    end: int
    text: str                 # == abstract[start:end]
    raw_start: int            # original offsets from the CSV (clamped to the abstract length)
    raw_end: int


@lru_cache(maxsize=1)
def _read_csv() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, dtype={"arxiv_id": str})


@lru_cache(maxsize=1)
def _corpus() -> tuple[Paper, ...]:
    df = _read_csv()
    order: dict[str, int] = {}
    ids: list[str | None] = []
    for abstract, arxiv_id in zip(df["abstract"], df["arxiv_id"]):
        if not isinstance(abstract, str):
            continue
        if abstract not in order:
            order[abstract] = len(order)
            ids.append(None)
        idx = order[abstract]
        if ids[idx] is None and isinstance(arxiv_id, str) and arxiv_id.strip():
            ids[idx] = arxiv_id.strip()
    return tuple(Paper(i, ids[i], abstract) for abstract, i in order.items())


def load_corpus() -> list[Paper]:
    """727 papers. Order = first appearance of each unique abstract in the CSV.
    arxiv_id = first non-null arxiv_id among that abstract's rows (read with dtype=str)."""
    return list(_corpus())


def arxiv_url(arxiv_id: str | None) -> str | None:
    """'https://arxiv.org/abs/<id>' or None."""
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None


def clean_span(abstract: str, start: int, end: int, min_words: int = 4) -> tuple[int, int] | None:
    """Clamp end to len(abstract); return None if start >= len(abstract).
    If the span starts mid-word (abstract[start-1] and abstract[start] are both alphanumeric),
    move start forward past the partial word. If it ends mid-word (abstract[end-1] and
    abstract[end] both alphanumeric), move end back to before the partial word.
    Then strip surrounding whitespace and leading punctuation.
    Return None if fewer than min_words words remain."""
    n = len(abstract)
    start = max(0, int(start))
    end = min(int(end), n)
    if start >= n or end <= start:
        return None

    # Partial word at the start: "...th|e model" -> skip "e"
    if start > 0 and abstract[start - 1].isalnum() and abstract[start].isalnum():
        while start < end and abstract[start].isalnum():
            start += 1
    # Partial word at the end: "To address th|e" -> drop "th"
    if end < n and abstract[end - 1].isalnum() and abstract[end].isalnum():
        while end > start and abstract[end - 1].isalnum():
            end -= 1

    while start < end and (abstract[start].isspace() or abstract[start] in string.punctuation):
        start += 1
    while end > start and abstract[end - 1].isspace():
        end -= 1

    if len(abstract[start:end].split()) < min_words:
        return None
    return start, end


def load_spans(min_words: int = 4) -> list[Span]:
    """All CSV rows cleaned with clean_span; rows that return None are dropped;
    exact duplicates (same doc_index, aspect, text) are dropped. Order = CSV order."""
    df = _read_csv()
    doc_of = {p.abstract: p.doc_index for p in _corpus()}
    spans: list[Span] = []
    seen: set[tuple[int, str, str]] = set()
    for abstract, start, end, label in zip(df["abstract"], df["start_idx"], df["end_idx"], df["label"]):
        aspect = ASPECT_OF_LABEL.get(label)
        if aspect is None or not isinstance(abstract, str):
            continue
        cleaned = clean_span(abstract, start, end, min_words)
        if cleaned is None:
            continue
        s, e = cleaned
        doc_index = doc_of[abstract]
        text = abstract[s:e]
        if (doc_index, aspect, text) in seen:
            continue
        seen.add((doc_index, aspect, text))
        n = len(abstract)
        spans.append(Span(doc_index, aspect, s, e, text,
                          raw_start=min(int(start), n), raw_end=min(int(end), n)))
    return spans


_tfidf_preprocessor = None
_tfidf_lock = threading.Lock()


def content_words(text: str) -> list[str]:
    """The TF-IDF preprocessing (lowercase, tokenize, drop punctuation/stopwords/<3 chars,
    POS-aware lemmatize). Implemented by reusing TFIDFSearcher.preprocess from
    tfidf_lexical_search.py (one shared instance, created lazily).
    Used by BM25 (P2) and the word-overlap measure (P3), so both use identical tokens."""
    global _tfidf_preprocessor
    if _tfidf_preprocessor is None:
        with _tfidf_lock:
            if _tfidf_preprocessor is None:
                # The teammate's module prints on import and in the constructor.
                with contextlib.redirect_stdout(io.StringIO()):
                    from tfidf_lexical_search import TFIDFSearcher
                    _tfidf_preprocessor = TFIDFSearcher([])
    return _tfidf_preprocessor.preprocess(text)


def _mid_word(abstract: str, start: int, end: int) -> bool:
    starts_mid = start > 0 and abstract[start - 1].isalnum() and abstract[start].isalnum()
    ends_mid = end < len(abstract) and abstract[end - 1].isalnum() and abstract[end].isalnum()
    return starts_mid or ends_mid


if __name__ == "__main__":
    import random
    import sys
    from collections import Counter

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    papers = load_corpus()
    spans = load_spans()
    abstracts = [p.abstract for p in papers]

    print(f"Papers:                 {len(papers)}")
    print(f"Papers without arXiv ID: {sum(p.arxiv_id is None for p in papers)}")
    short_ids = [p.arxiv_id for p in papers if p.arxiv_id and len(p.arxiv_id.split('.')[-1]) < 5
                 and p.arxiv_id[:2] >= "15"]
    print(f"Truncated-looking IDs:   {len(short_ids)} {short_ids[:5]}")

    print("\nSpans per aspect (after cleaning):")
    for aspect, n in Counter(s.aspect for s in spans).most_common():
        print(f"  {aspect:<8} {n}")
    print(f"  {'total':<8} {len(spans)}")

    bad = [s for s in spans if _mid_word(abstracts[s.doc_index], s.start, s.end)]
    mismatched = [s for s in spans if abstracts[s.doc_index][s.start:s.end] != s.text]
    print(f"\nSpans starting/ending mid-word: {len(bad)}")
    print(f"Spans whose text != abstract[start:end]: {len(mismatched)}")

    print("\n5 random cleaned spans:")
    for s in random.Random(0).sample(spans, 5):
        print(f"  [{s.doc_index:>3} {s.aspect:<7}] {s.text!r}")

    print(f"\ncontent_words('Neural networks were learning faster'): "
          f"{content_words('Neural networks were learning faster')}")
