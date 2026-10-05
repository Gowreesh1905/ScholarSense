"""Aspect search (team/CONTRACT.md §5, team/P5_aspect_report.md part 2).

Search only the *task*, *problem*, *method* or *result* sentences of the
abstracts, using the Scholar Inbox span labels (loaded by data.load_spans()).
Each span is embedded with BGE; a query is scored against the spans of one
aspect and each paper keeps its best-matching span.

Not a BaseSearcher: its search() also takes an aspect and returns the span.

    python -m searchers.aspect      # coverage + top hits for 3 example queries
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data import Paper, Span
from searchers.cache import cached_encode
from searchers.models import BGE_MODEL, BGE_QUERY_PREFIX, get_bge

ASPECTS = ["task", "problem", "method", "result"]


@dataclass
class AspectHit:
    doc_index: int
    score: float          # cosine similarity of the best span
    span_start: int       # char offsets into the abstract
    span_end: int
    span_text: str


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


class AspectSearcher:
    key = "aspect"

    def __init__(self) -> None:
        self._spans: dict[str, list[Span]] = {a: [] for a in ASPECTS}
        self._embeddings: dict[str, np.ndarray] = {}

    def fit(self, papers: list[Paper], spans: list[Span]) -> None:
        """Embed every span (no query prefix) and keep them grouped by aspect.

        Fully replaces any earlier index. Spans whose doc_index is not in
        `papers` are ignored.
        """
        n_papers = len(papers)
        usable = [s for s in spans if s.aspect in ASPECTS and 0 <= s.doc_index < n_papers]

        self._spans = {a: [] for a in ASPECTS}
        self._embeddings = {}
        if not usable:
            return

        model = get_bge()

        def encode(texts: list[str]) -> np.ndarray:
            return model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

        all_embeddings = _normalize(cached_encode(BGE_MODEL, [s.text for s in usable], encode, tag="aspect-spans"))

        for aspect in ASPECTS:
            rows = [i for i, s in enumerate(usable) if s.aspect == aspect]
            self._spans[aspect] = [usable[i] for i in rows]
            self._embeddings[aspect] = all_embeddings[rows]

    def search(self, query: str, aspect: str, k: int = 10) -> list[AspectHit]:
        """Top-k papers by their best span of `aspect`, best first."""
        if aspect not in ASPECTS:
            raise ValueError(f"Unknown aspect {aspect!r}. Choose one of: {', '.join(ASPECTS)}")
        spans = self._spans[aspect]
        if not spans or k <= 0:
            return []

        model = get_bge()
        q = _normalize(model.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True, convert_to_numpy=True))
        scores = self._embeddings[aspect] @ q

        # Best score first; ties go to the lower doc_index, then the earlier span.
        order = sorted(range(len(spans)), key=lambda i: (-scores[i], spans[i].doc_index, spans[i].start))
        hits: list[AspectHit] = []
        seen: set[int] = set()
        for i in order:
            span = spans[i]
            if span.doc_index in seen:
                continue
            seen.add(span.doc_index)
            hits.append(AspectHit(span.doc_index, float(scores[i]), span.start, span.end, span.text))
            if len(hits) == k:
                break
        return hits

    def coverage(self) -> dict[str, int]:
        """Number of papers that have at least one span of each aspect."""
        return {a: len({s.doc_index for s in self._spans[a]}) for a in ASPECTS}


if __name__ == "__main__":
    import sys

    from data import load_corpus, load_spans

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    papers = load_corpus()
    searcher = AspectSearcher()
    searcher.fit(papers, load_spans())

    print(f"Papers: {len(papers)}")
    print("Coverage (papers with at least one span):")
    for aspect, n in searcher.coverage().items():
        print(f"  {aspect:<8} {n}")

    queries = [
        "methods break down when the camera moves a lot",
        "making blurry photos sharp again",
        "teaching robots to grasp unfamiliar objects",
    ]
    for query in queries:
        print(f"\n{'=' * 100}\nQuery: {query}")
        for aspect in ASPECTS:
            print(f"  -- {aspect}")
            for hit in searcher.search(query, aspect, k=3):
                print(f"     [{hit.doc_index:>3}] {hit.score:.3f}  {hit.span_text[:150]!r}")
