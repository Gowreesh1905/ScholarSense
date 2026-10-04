"""BM25 searcher implementation (team/CONTRACT.md §4, team/P2_models.md §1)."""

from __future__ import annotations

import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

from data import content_words
from searchers.base import BaseSearcher, top_k
from searchers.cache import cache_path, text_hash


class BM25Searcher(BaseSearcher):
    key = "bm25"
    label = "BM25"
    family = "lexical"
    description = "Keyword matching with term-frequency saturation and length normalisation."
    score_type = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._n_docs: int = 0

    def fit(self, abstracts: list[str]) -> None:
        """Build the BM25 index over the provided abstracts using content_words tokenization.

        Tokenized abstracts are cached on disk so subsequent fits on identical corpus
        are instantaneous. Fully rebuilds the index if called again with a different corpus.
        """
        abstracts_list = list(abstracts)
        self._n_docs = len(abstracts_list)
        if self._n_docs == 0:
            self._bm25 = None
            return

        path = cache_path(f"bm25_tok_{text_hash(abstracts_list, 'bm25')}.pkl")
        tokenized = None
        if path.exists():
            try:
                with open(path, "rb") as f:
                    tokenized = pickle.load(f)
            except Exception:
                tokenized = None

        if tokenized is None:
            tokenized = [content_words(text) for text in abstracts_list]
            try:
                tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(tokenized, f)
                os.replace(tmp, path)
            except Exception:
                pass

        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def scores(self, query: str) -> np.ndarray:
        """Return raw BM25 scores across all corpus documents for the given query.

        Reused by downstream methods such as BM25 query expansion (PRF).
        """
        if self._bm25 is None or self._n_docs == 0:
            return np.zeros(0, dtype=float)

        tokens = content_words(query)
        if not tokens:
            return np.zeros(self._n_docs, dtype=float)

        return np.asarray(self._bm25.get_scores(tokens), dtype=float)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to k (doc_index, score) pairs, best first."""
        if self._bm25 is None or self._n_docs == 0:
            return []

        tokens = content_words(query)
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)
        return [(i, s) for i, s in top_k(raw_scores, k) if s > 0]
