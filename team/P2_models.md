# P2 — New models

Read `team/README.md` and `team/CONTRACT.md` first (especially §2 interface, §3 shared helpers, §4 registry table).

**Your job:** implement 4 new search methods (BM25, GloVe, BGE, Hybrid) as `BaseSearcher` subclasses. The Hybrid class provides two registry entries: `hybrid_rrf` (ablation, no re-rank) and `hybrid` (with re-rank).

**You own:** `searchers/bm25.py`, `searchers/glove.py`, `searchers/bge.py`, `searchers/hybrid.py`.
**Don't edit:** anything else. The registry entries for your classes are already written by P1; you don't register anything yourself.

**Before merge #1** (P1's skeleton isn't on `main` yet): code against the CONTRACT signatures. If you need to run something, make temporary local copies of `base.py`/`models.py`/`cache.py` that follow the CONTRACT, but **don't commit them**; delete them when you merge P1's real ones.

---

## Shared rules for all four classes

- Subclass `BaseSearcher`; set the class attributes `key`, `label`, `family`, `description`, `score_type` exactly as in the CONTRACT registry table.
- `fit(abstracts)` may be called again with a different list (P3 evaluates on a masked corpus); it must fully rebuild the index.
- `search` returns `list[tuple[int, float]]`, best first, at most `k` items. Use `np.argpartition` + sort for top-k; plain Python ints and floats in the output.
- Load models only through `searchers.models` getters. Cache corpus embeddings with `searchers.cache.cached_encode`.
- Write short, useful descriptions (shown in the UI), e.g. BM25: "Keyword matching with term-frequency saturation and length normalisation."

## 1. `searchers/bm25.py` → `BM25Searcher`

- `from rank_bm25 import BM25Okapi` (default parameters, k1 = 1.5, b = 0.75).
- Tokenize corpus and queries with `data.content_words` (the same preprocessing as TF-IDF, so the two lexical methods are directly comparable).
- `fit`: tokenize all abstracts, build `BM25Okapi(tokenized)`. Preprocessing 727 abstracts takes a few seconds (POS tagging); that's fine.
- `search`: `scores = bm25.get_scores(content_words(query))`; if the query has no tokens, return `[]`.
- Also expose `scores(query) -> np.ndarray` (all document scores); P5's optional query-expansion method reuses it.

## 2. `searchers/glove.py` → `GloVeSearcher`

- Model: `get_glove()` (average of pretrained GloVe 300d word vectors, trained on 6 billion tokens). It contrasts with our Word2Vec, which is trained on only 727 abstracts.
- Feed raw text: the model does its own lowercasing and tokenizing.
- `encode(..., normalize_embeddings=True, convert_to_numpy=True)` → cosine = dot product.
- Override `search_batch` to encode all queries at once.
- A query with no known words gives a zero vector → return `[]`.

## 3. `searchers/bge.py` → `BGESearcher`

- Model: `get_bge()`.
- **Queries get the prefix** `BGE_QUERY_PREFIX` (from `searchers.models`); documents do not. This matters for short queries.
- `encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)`; cosine = dot product.
- Corpus embeddings via `cached_encode("BAAI/bge-base-en-v1.5", abstracts, fn)`.
- Override `search_batch`.
- Expose `rank_all(query) -> np.ndarray` (doc indices sorted best-first) or similar, for the hybrid to reuse.

## 4. `searchers/hybrid.py` → `HybridSearcher(rerank: bool = True)`

```
Query ─┬─► BM25 top-100 ─┐
       │                  ├─► Reciprocal Rank Fusion ─► top-30 ─► cross-encoder re-rank ─► top-k
       └─► BGE  top-100 ─┘
```
- `__init__(self, rerank=True)`: create its own `BM25Searcher()` and `BGESearcher()`. They share the loaded BGE model through `get_bge()` and the cached embeddings, so nothing loads twice.
- Set class attributes per instance: `rerank=False` → key `hybrid_rrf`, label "BM25 + BGE (RRF)", score_type `rrf`. `rerank=True` → key `hybrid`, label "Hybrid + re-rank", score_type `cross-encoder`.
- `fit`: fit both sub-searchers; keep the abstracts (the cross-encoder needs the text).
- **RRF:** `score(d) = Σ 1 / (60 + rank_in_list(d))` over the BM25 and BGE top-100 lists, with 1-based ranks; a doc missing from a list contributes 0.
- `rerank=False`: return the top-k by RRF score.
- `rerank=True`: take the top-30 by RRF; `get_cross_encoder().predict([(query, abstracts[d]) for d in top30], batch_size=32)`; sort by that score; return the top-k. If k > 30, append the remaining RRF-ordered docs after the 30 (keep their RRF scores; P3 needs ranks up to 100).
- `search_batch`: batch the BGE part; loop for the rest.

## Testing

After merge #1:
```bash
python -m searchers.smoke bm25 glove bge hybrid_rrf hybrid
```
Check that:
- [ ] Every method returns sensible top-3 results for the default queries (robots/grasping papers for the grasp query, etc.).
- [ ] BGE is clearly better than GloVe on the paraphrase-style queries; BM25 is similar to TF-IDF.
- [ ] Running smoke twice: the second run's BGE fit is near-instant (cache hit).
- [ ] `fit` twice with different lists works (quick test: fit on the first 100 abstracts, search, then fit on all 727, search).
- [ ] Hybrid with re-rank on GPU answers in well under 1 second per query.

Push to `p2-models` and tell P1 before **H4 (merge #2)**.

## After merge #2

- Help P3 if evaluation shows problems in your methods (e.g. a crash on an empty query, or a slow `search_batch`).
- Don't tune hyperparameters on the evaluation queries themselves. If you try something (e.g. top-50 instead of top-30 for re-ranking), report it as a separate ablation row, not as a silent change.
