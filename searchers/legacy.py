"""Adapters that put the original three searchers behind BaseSearcher.

The teammates' modules (tfidf_lexical_search.py, word2vec_pytorch_search.py,
bert_specter_contextual_search.py) are used as-is, unmodified.
"""

from __future__ import annotations

import contextlib
import io
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from searchers.base import BaseSearcher, top_k
from searchers.cache import cache_path, cached_encode, text_hash
from searchers.models import SPECTER_MODEL, get_specter


def _quiet():
    """Swallow the progress prints the original modules make."""
    return contextlib.redirect_stdout(io.StringIO())


def _silent(*args, **kwargs):
    """Picklable no-op logger for the cached Word2VecSearcher."""


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


class TFIDFAdapter(BaseSearcher):
    key = "tfidf"
    label = "TF-IDF"
    family = "lexical"
    description = "Keyword matching on lemmatised words, weighted by how rare each word is in the corpus."
    score_type = "cosine"

    def __init__(self):
        self._searcher = None
        self._matrix = None

    def fit(self, abstracts: list[str]) -> None:
        with _quiet():
            from tfidf_lexical_search import TFIDFSearcher
            self._searcher = TFIDFSearcher(list(abstracts))
            self._matrix = self._searcher.get_corpus_embeddings()

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = self._searcher.get_query_embedding(query)
        if q.nnz == 0:  # no query word is in the vocabulary
            return []
        sims = cosine_similarity(q, self._matrix).ravel()
        # Papers sharing no word with the query aren't matches; don't rank them by chance.
        return [(i, s) for i, s in top_k(sims, k) if s > 0]


class Word2VecAdapter(BaseSearcher):
    key = "word2vec"
    label = "Word2Vec (corpus-trained)"
    family = "static"
    description = "Average of word vectors learned from only these 727 abstracts (skip-gram, PyTorch)."
    score_type = "cosine"

    PARAMS = {"vector_size": 100, "epochs": 5}

    def __init__(self):
        self._searcher = None
        self._corpus = None

    def fit(self, abstracts: list[str]) -> None:
        abstracts = list(abstracts)
        path = cache_path(f"word2vec_{text_hash(abstracts, sorted(self.PARAMS.items()))}.pkl")
        searcher = None
        if path.exists():
            try:
                with open(path, "rb") as f:
                    searcher = pickle.load(f)
            except Exception as exc:  # corrupt/incompatible cache -> retrain
                print(f"[word2vec] Cache load failed ({exc}); retraining.")

        if searcher is None:
            from word2vec_pytorch_search import Word2VecSearcher
            searcher = Word2VecSearcher(abstracts, verbose=False, **self.PARAMS)
            searcher.get_corpus_embeddings()  # populate before caching
            searcher._log = _silent           # its default no-op lambda can't be pickled
            with open(path, "wb") as f:
                pickle.dump(searcher, f)

        self._searcher = searcher
        self._corpus = searcher.get_corpus_embeddings()

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = self._searcher.get_query_embedding(query)
        if not np.any(q):  # no known words
            return []
        sims = cosine_similarity(q.reshape(1, -1), self._corpus).ravel()
        return top_k(sims, k)


class SpecterAdapter(BaseSearcher):
    key = "specter"
    label = "SPECTER"
    family = "contextual"
    description = "Transformer trained on citation links between scientific papers (allenai/specter)."
    score_type = "cosine"

    def __init__(self):
        self._searcher = None
        self._corpus = None

    def fit(self, abstracts: list[str]) -> None:
        with _quiet():
            from bert_specter_contextual_search import BERTSearcher
            self._searcher = BERTSearcher(list(abstracts), get_specter())

        def encode(_texts):
            with _quiet():
                return self._searcher.get_corpus_embeddings()

        self._corpus = _normalize(cached_encode(SPECTER_MODEL, list(abstracts), encode))

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = _normalize(self._searcher.get_query_embedding(query))
        return top_k(self._corpus @ q, k)

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[tuple[int, float]]]:
        cleaned = [self._searcher._clean_text(q) for q in queries]
        q = _normalize(self._searcher.model.encode(cleaned, batch_size=32, convert_to_numpy=True))
        sims = q @ self._corpus.T
        return [top_k(row, k) for row in sims]
