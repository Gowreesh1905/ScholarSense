"""GloVe 300d pretrained static embedding searcher (team/CONTRACT.md §4, team/P2_models.md §2)."""

from __future__ import annotations

import numpy as np

from searchers.base import BaseSearcher, top_k
from searchers.cache import cached_encode
from searchers.models import GLOVE_MODEL, get_glove


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


class GloVeSearcher(BaseSearcher):
    key = "glove"
    label = "GloVe 300d (pretrained)"
    family = "static"
    description = "Average of pretrained GloVe 300d word vectors (Wikipedia + Gigaword, 6B tokens)."
    score_type = "cosine"

    def __init__(self) -> None:
        self._corpus: np.ndarray | None = None
        self._n_docs: int = 0

    def fit(self, abstracts: list[str]) -> None:
        """Embed and cache corpus abstracts using the pretrained GloVe 300d model.

        Fully rebuilds index when called again with a different corpus.
        """
        abstracts_list = list(abstracts)
        self._n_docs = len(abstracts_list)
        if self._n_docs == 0:
            self._corpus = None
            return

        model = get_glove()

        def encode_corpus(texts: list[str]) -> np.ndarray:
            return model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

        embeddings = cached_encode(GLOVE_MODEL, abstracts_list, encode_corpus)
        self._corpus = _normalize(embeddings)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to k (doc_index, score) pairs, best first."""
        if self._corpus is None or self._n_docs == 0:
            return []

        model = get_glove()
        q = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        # A query with no known words produces an all-zero vector
        if not np.any(q):
            return []

        q = _normalize(q)
        sims = self._corpus @ q
        return top_k(sims, k)

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[tuple[int, float]]]:
        """Encode all queries at once in a batch and compute cosine similarities."""
        if self._corpus is None or self._n_docs == 0:
            return [[] for _ in queries]

        if not queries:
            return []

        model = get_glove()
        q_batch = model.encode(queries, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        q_batch = _normalize(q_batch)

        all_sims = q_batch @ self._corpus.T

        results: list[list[tuple[int, float]]] = []
        for i, q in enumerate(q_batch):
            if not np.any(q):
                results.append([])
            else:
                results.append(top_k(all_sims[i], k))
        return results
