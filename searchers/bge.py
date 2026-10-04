"""BGE-base dense retrieval searcher (team/CONTRACT.md §4, team/P2_models.md §3)."""

from __future__ import annotations

import numpy as np

from searchers.base import BaseSearcher, top_k
from searchers.cache import cached_encode
from searchers.models import BGE_MODEL, BGE_QUERY_PREFIX, get_bge


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


class BGESearcher(BaseSearcher):
    key = "bge"
    label = "BGE-base"
    family = "contextual"
    description = "State-of-the-art dense retrieval model (BAAI/bge-base-en-v1.5), 768d contextual vectors."
    score_type = "cosine"

    def __init__(self) -> None:
        self._corpus: np.ndarray | None = None
        self._n_docs: int = 0

    def fit(self, abstracts: list[str]) -> None:
        """Embed and cache corpus abstracts using BGE-base.

        Documents are encoded WITHOUT prefix. Fully rebuilds index when called again.
        """
        abstracts_list = list(abstracts)
        self._n_docs = len(abstracts_list)
        if self._n_docs == 0:
            self._corpus = None
            return

        model = get_bge()

        def encode_corpus(texts: list[str]) -> np.ndarray:
            return model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

        embeddings = cached_encode(BGE_MODEL, abstracts_list, encode_corpus)
        self._corpus = _normalize(embeddings)

    def scores(self, query: str) -> np.ndarray:
        """Return cosine similarity against all corpus documents for the prefixed query."""
        if self._corpus is None or self._n_docs == 0:
            return np.zeros(0, dtype=float)

        model = get_bge()
        prefixed = BGE_QUERY_PREFIX + query
        q = _normalize(model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True))
        return self._corpus @ q

    def rank_all(self, query: str) -> np.ndarray:
        """Return all document indices sorted best-first (descending similarity).

        Exposed for reuse by downstream methods such as HybridSearcher.
        """
        sims = self.scores(query)
        if len(sims) == 0:
            return np.empty(0, dtype=int)
        idx = np.arange(len(sims))
        return idx[np.lexsort((idx, -sims))]

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to k (doc_index, score) pairs, best first."""
        if self._corpus is None or self._n_docs == 0:
            return []

        sims = self.scores(query)
        return top_k(sims, k)

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[tuple[int, float]]]:
        """Encode all prefixed queries in a batch and compute cosine similarities."""
        if self._corpus is None or self._n_docs == 0:
            return [[] for _ in queries]

        if not queries:
            return []

        model = get_bge()
        prefixed_queries = [BGE_QUERY_PREFIX + q for q in queries]
        q_batch = model.encode(prefixed_queries, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        q_batch = _normalize(q_batch)

        all_sims = q_batch @ self._corpus.T
        return [top_k(all_sims[i], k) for i in range(len(queries))]
