"""Searcher interface every retrieval method implements (team/CONTRACT.md §2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSearcher(ABC):
    key: str          # e.g. "bm25"  (see the registry table in CONTRACT §4)
    label: str        # e.g. "BM25"
    family: str       # "lexical" | "static" | "contextual" | "hybrid"
    description: str  # one sentence shown in the UI
    score_type: str   # "cosine" | "bm25" | "rrf" | "cross-encoder"

    @abstractmethod
    def fit(self, abstracts: list[str]) -> None:
        """Build the index over these abstracts. doc_index = position in this list.
        May be called again with a different list (evaluation uses a masked corpus);
        it must then fully replace the old index."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to k (doc_index, score) pairs, best first."""

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[tuple[int, float]]]:
        """Default: loop over search(). Embedding models should override this
        and encode all queries in one batch."""
        return [self.search(q, k) for q in queries]


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """The k highest scores as (doc_index, score) pairs, best first.

    Ties are broken by the lower doc_index, so results are deterministic.
    Returns plain Python ints and floats.
    """
    scores = np.asarray(scores).ravel()
    k = min(int(k), scores.shape[0])
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.lexsort((idx, -scores[idx]))]
    return [(int(i), float(scores[i])) for i in idx]
