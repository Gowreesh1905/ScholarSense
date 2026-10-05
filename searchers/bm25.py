"""BM25 searcher implementation (team/CONTRACT.md §4, team/P2_models.md §1)."""

from __future__ import annotations

import os
import pickle

import numpy as np
from scipy import sparse

from data import content_words
from searchers.base import BaseSearcher, top_k
from searchers.cache import cache_path, text_hash

# rank_bm25's BM25Okapi floors negative IDFs (words in more than half the documents)
# at EPSILON * the average IDF. Kept identical so scores match rank_bm25 exactly.
EPSILON = 0.25


class BM25Searcher(BaseSearcher):
    key = "bm25"
    label = "BM25"
    family = "lexical"
    description = "Keyword matching with term-frequency saturation and length normalisation."
    score_type = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._weights: sparse.csc_matrix | None = None   # docs x vocab, BM25 term weight per (doc, word)
        self._idf: np.ndarray | None = None
        self._vocab: dict[str, int] = {}
        self._n_docs: int = 0

    def fit(self, abstracts: list[str]) -> None:
        """Build the BM25 index over the provided abstracts using content_words tokenization.

        Tokenized abstracts are cached on disk so subsequent fits on identical corpus
        are instantaneous. Fully rebuilds the index if called again with a different corpus.
        """
        abstracts_list = list(abstracts)
        self._n_docs = len(abstracts_list)
        if self._n_docs == 0:
            self._weights = None
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

        self._build(tokenized)

    def _build(self, tokenized: list[list[str]]) -> None:
        """Precompute every document's BM25 weight for every word it contains (the Okapi formula
        of rank_bm25), so a query is one sparse matrix-vector product instead of a Python loop
        over all documents per query word. That matters at 100k documents."""
        vocab: dict[str, int] = {}
        indptr, indices, data = [0], [], []
        for doc in tokenized:
            counts: dict[int, int] = {}
            for tok in doc:
                j = vocab.setdefault(tok, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            indices.extend(counts)
            data.extend(counts.values())
            indptr.append(len(indices))
        n = len(tokenized)
        tf = sparse.csr_matrix((np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64),
                                np.asarray(indptr, dtype=np.int64)), shape=(n, max(1, len(vocab))))

        doc_len = np.asarray([len(doc) for doc in tokenized], dtype=np.float64)
        avgdl = doc_len.sum() / n
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(n - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = EPSILON * idf.mean()

        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        f = tf.data
        tf.data = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * doc_len[rows] / avgdl))

        self._weights = tf.tocsc()
        self._idf = idf
        self._vocab = vocab

    @property
    def idf(self) -> dict[str, float]:
        """IDF of every word in the corpus vocabulary."""
        return {w: float(self._idf[j]) for w, j in self._vocab.items()} if self._idf is not None else {}

    def _scores_for_tokens(self, tokens: list[str]) -> np.ndarray:
        q: dict[int, int] = {}
        for tok in tokens:                     # repeated query words count again, as in rank_bm25
            j = self._vocab.get(tok)
            if j is not None:
                q[j] = q.get(j, 0) + 1
        if not q:
            return np.zeros(self._n_docs, dtype=float)
        cols = np.fromiter(q, dtype=np.int64)
        weights = np.fromiter(q.values(), dtype=np.float64) * self._idf[cols]
        return np.asarray(self._weights[:, cols] @ weights).ravel()

    def scores(self, query: str) -> np.ndarray:
        """Return raw BM25 scores across all corpus documents for the given query.

        Reused by downstream methods such as BM25 query expansion (PRF).
        """
        if self._weights is None or self._n_docs == 0:
            return np.zeros(0, dtype=float)
        return self._scores_for_tokens(content_words(query))

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to k (doc_index, score) pairs, best first."""
        if self._weights is None or self._n_docs == 0:
            return []

        tokens = content_words(query)
        if not tokens:
            return []

        raw_scores = self._scores_for_tokens(tokens)
        return [(i, s) for i, s in top_k(raw_scores, k) if s > 0]
