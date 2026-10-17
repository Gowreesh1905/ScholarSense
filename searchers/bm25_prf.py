"""BM25 with pseudo-relevance feedback / query expansion (team/CONTRACT.md §4, team/P5_aspect_report.md optional part).

Answers "couldn't keyword search just be patched?": run BM25, assume the top 5
papers are relevant, add their most characteristic words to the query, and search
again. It can only add words that appear in the corpus, so it helps a little with
vocabulary mismatch but cannot bridge a gap the corpus itself does not contain.
"""

from __future__ import annotations

import math
import os
import pickle
from collections import Counter

from rank_bm25 import BM25Okapi

from data import content_words
from searchers.base import BaseSearcher, top_k
from searchers.bm25 import BM25Searcher
from searchers.cache import cache_path, text_hash

FEEDBACK_DOCS = 5      # top papers of the first search that are treated as relevant
EXPANSION_TERMS = 10   # new words added to the query
ORIGINAL_WEIGHT = 2    # original query words are repeated, expansion words are used once


class BM25PRFSearcher(BaseSearcher):
    key = "bm25_prf"
    label = "BM25 + query expansion"
    family = "lexical"
    description = "BM25, then adds the 10 most characteristic words of the top 5 results to the query and searches again."
    score_type = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._first_pass = BM25Searcher(k1=k1, b=b)
        self._bm25: BM25Okapi | None = None
        self._tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._n_docs = 0

    def fit(self, abstracts: list[str]) -> None:
        """Index the abstracts twice: BM25Searcher for the first pass, and BM25Okapi over the same tokens for the second.

        Fully rebuilds the index when called again with a different corpus.
        """
        abstracts_list = list(abstracts)
        self._n_docs = len(abstracts_list)
        self._tokens, self._idf, self._bm25 = [], {}, None
        if self._n_docs == 0:
            return

        self._first_pass.fit(abstracts_list)
        self._tokens = self._tokenize(abstracts_list)
        self._bm25 = BM25Okapi(self._tokens, k1=self.k1, b=self.b)

        doc_freq = Counter(term for doc in self._tokens for term in set(doc))
        self._idf = {t: math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    @staticmethod
    def _tokenize(abstracts: list[str]) -> list[list[str]]:
        """content_words() of every abstract, cached on disk (the POS tagging takes a few seconds)."""
        path = cache_path(f"bm25prf_tok_{text_hash(abstracts, 'bm25prf')}.pkl")
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        tokens = [content_words(text) for text in abstracts]
        try:
            tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(tokens, f)
            os.replace(tmp, path)
        except Exception:
            pass
        return tokens

    def expansion_terms(self, query_tokens: list[str], feedback_docs: list[int]) -> list[str]:
        """Words of the feedback papers that are not in the query, best first: total frequency x IDF."""
        in_query = set(query_tokens)
        weight: Counter[str] = Counter()
        for doc in feedback_docs:
            for term, count in Counter(self._tokens[doc]).items():
                if term not in in_query:
                    weight[term] += count * self._idf[term]
        ranked = sorted(weight.items(), key=lambda item: (-item[1], item[0]))  # ties: alphabetical, so results are stable
        return [term for term, _ in ranked[:EXPANSION_TERMS]]

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to k (doc_index, score) pairs, best first."""
        if self._bm25 is None or self._n_docs == 0:
            return []
        query_tokens = content_words(query)
        if not query_tokens:
            return []

        first = [(d, s) for d, s in top_k(self._first_pass.scores(query), FEEDBACK_DOCS) if s > 0]
        if not first:  # no query word occurs in the corpus: nothing to learn from
            return []

        expanded = query_tokens * ORIGINAL_WEIGHT + self.expansion_terms(query_tokens, [d for d, _ in first])
        scores = self._bm25.get_scores(expanded)
        return [(i, s) for i, s in top_k(scores, k) if s > 0]
