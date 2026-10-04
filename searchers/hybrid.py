"""Hybrid searcher combining BM25 keyword search and BGE dense retrieval (team/CONTRACT.md §4, team/P2_models.md §4)."""

from __future__ import annotations

import numpy as np

from searchers.base import BaseSearcher
from searchers.bge import BGESearcher
from searchers.bm25 import BM25Searcher
from searchers.models import get_cross_encoder


class HybridSearcher(BaseSearcher):
    # Class defaults (overridden per instance based on rerank flag)
    key = "hybrid"
    label = "Hybrid + re-rank"
    family = "hybrid"
    description = "BM25 + BGE retrieval fused with RRF, re-ranked with MiniLM cross-encoder."
    score_type = "cross-encoder"

    def __init__(self, rerank: bool = True) -> None:
        self.rerank = bool(rerank)
        self.bm25 = BM25Searcher()
        self.bge = BGESearcher()
        self._abstracts: list[str] = []

        if not self.rerank:
            self.key = "hybrid_rrf"
            self.label = "BM25 + BGE (RRF)"
            self.family = "hybrid"
            self.description = "Reciprocal Rank Fusion over BM25 keyword search and BGE dense retrieval (top-100 each)."
            self.score_type = "rrf"
        else:
            self.key = "hybrid"
            self.label = "Hybrid + re-rank"
            self.family = "hybrid"
            self.description = "BM25 + BGE retrieval fused with RRF, re-ranked with MiniLM cross-encoder."
            self.score_type = "cross-encoder"

    def fit(self, abstracts: list[str]) -> None:
        """Fit both BM25 and BGE sub-searchers and retain corpus abstracts for cross-encoder re-ranking.

        Fully rebuilds the index when called again with a different corpus.
        """
        self._abstracts = list(abstracts)
        self.bm25.fit(self._abstracts)
        self.bge.fit(self._abstracts)

    def _fuse_and_rerank(
        self,
        query: str,
        bm25_hits: list[tuple[int, float]],
        bge_hits: list[tuple[int, float]],
        k: int,
    ) -> list[tuple[int, float]]:
        """Combine top-100 lists using RRF (k_rrf=60) and optionally re-rank top-30 with cross-encoder."""
        if not self._abstracts:
            return []

        rrf_scores: dict[int, float] = {}
        for rank, (doc_idx, _) in enumerate(bm25_hits, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (60.0 + rank))

        for rank, (doc_idx, _) in enumerate(bge_hits, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (60.0 + rank))

        if not rrf_scores:
            return []

        # Sort candidate documents descending by RRF score; break ties by lower doc_index
        rrf_sorted = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))

        if not self.rerank:
            return [(int(d), float(s)) for d, s in rrf_sorted[:k]]

        # Re-rank top-30 candidates with cross-encoder
        top30 = rrf_sorted[:30]
        remaining = rrf_sorted[30:]
        if not top30:
            return []

        pairs = [(query, self._abstracts[doc_idx]) for doc_idx, _ in top30]
        ce_model = get_cross_encoder()
        ce_scores = ce_model.predict(pairs, batch_size=32)
        ce_scores = np.asarray(ce_scores).ravel()

        reranked_top30 = [(int(top30[i][0]), float(ce_scores[i])) for i in range(len(top30))]
        reranked_top30.sort(key=lambda item: (-item[1], item[0]))

        combined = reranked_top30 + [(int(d), float(s)) for d, s in remaining]
        return combined[:k]

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Retrieve top-100 from BM25 and BGE, fuse with RRF, and optionally re-rank top-30."""
        if not self._abstracts:
            return []

        bm25_hits = self.bm25.search(query, k=100)
        bge_hits = self.bge.search(query, k=100)
        return self._fuse_and_rerank(query, bm25_hits, bge_hits, k)

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[tuple[int, float]]]:
        """Batch the BGE dense retrieval step, then perform RRF and re-ranking for each query."""
        if not self._abstracts:
            return [[] for _ in queries]

        if not queries:
            return []

        bge_batch = self.bge.search_batch(queries, k=100)
        results: list[list[tuple[int, float]]] = []
        for i, query in enumerate(queries):
            bm25_hits = self.bm25.search(query, k=100)
            results.append(self._fuse_and_rerank(query, bm25_hits, bge_batch[i], k))
        return results
