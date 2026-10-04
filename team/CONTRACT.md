# Shared contract

Everyone codes against these interfaces. P1 implements sections 1–4 first and pushes them to `main` by H1.5. **Don't change anything here without P1 agreeing**; if you must, update this file in the same commit and tell everyone.

---

## 1. `data.py` (repo root, owner P1)

```python
from dataclasses import dataclass

CSV_PATH: Path  # <repo>/abstract_sentences.csv

@dataclass(frozen=True)
class Paper:
    doc_index: int            # position in load_corpus()
    arxiv_id: str | None      # read as str; None for the 18 papers without one
    abstract: str

def load_corpus() -> list[Paper]:
    """727 papers. Order = first appearance of each unique abstract in the CSV.
    arxiv_id = first non-null arxiv_id among that abstract's rows (read with dtype=str)."""

def arxiv_url(arxiv_id: str | None) -> str | None:
    """'https://arxiv.org/abs/<id>' or None."""

ASPECT_OF_LABEL = {"task": "task", "problem": "problem", "idea": "method", "result": "result"}

@dataclass(frozen=True)
class Span:
    doc_index: int
    aspect: str               # "task" | "problem" | "method" | "result"
    start: int                # cleaned char offsets into the abstract, end exclusive
    end: int
    text: str                 # == abstract[start:end]
    raw_start: int            # original offsets from the CSV (clamped to the abstract length)
    raw_end: int

def clean_span(abstract: str, start: int, end: int, min_words: int = 4) -> tuple[int, int] | None:
    """Clamp end to len(abstract); return None if start >= len(abstract).
    If the span starts mid-word (abstract[start-1] and abstract[start] are both alphanumeric),
    move start forward past the partial word. If it ends mid-word (abstract[end-1] and
    abstract[end] both alphanumeric), move end back to before the partial word.
    Then strip surrounding whitespace and leading punctuation.
    Return None if fewer than min_words words remain."""

def load_spans(min_words: int = 4) -> list[Span]:
    """All CSV rows cleaned with clean_span; rows that return None are dropped;
    exact duplicates (same doc_index, aspect, text) are dropped. Order = CSV order."""

def content_words(text: str) -> list[str]:
    """The TF-IDF preprocessing (lowercase, tokenize, drop punctuation/stopwords/<3 chars,
    POS-aware lemmatize). Implemented by reusing TFIDFSearcher.preprocess from
    tfidf_lexical_search.py (one shared instance, created lazily).
    Used by BM25 (P2) and the word-overlap measure (P3), so both use identical tokens."""
```

## 2. Searcher interface: `searchers/base.py` (owner P1)

`searchers/` is a package (`searchers/__init__.py` exists, can be empty).

```python
from abc import ABC, abstractmethod

class BaseSearcher(ABC):
    key: str          # e.g. "bm25"  (see the registry table below)
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
```

Rules:
- Don't load models in `__init__` directly; use the shared getters in `searchers/models.py`, so one model is loaded once even if several searchers use it.
- Embedding searchers cache corpus embeddings with `searchers/cache.py`.
- Constructors take no required arguments (the registry calls them with none, except the hybrid variants below).

## 3. Shared helpers (owner P1)

`searchers/models.py`: every getter is `functools.lru_cache`d, so a model loads once per process.
```python
def get_device() -> str                     # "cuda" if available else "cpu"
def get_specter() -> SentenceTransformer    # "allenai/specter"
def get_bge() -> SentenceTransformer        # "BAAI/bge-base-en-v1.5"
def get_glove() -> SentenceTransformer      # "sentence-transformers/average_word_embeddings_glove.6B.300d"
def get_cross_encoder() -> CrossEncoder     # "cross-encoder/ms-marco-MiniLM-L6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```

`searchers/cache.py`
```python
def cached_encode(model_id: str, texts: list[str],
                  encode_fn: Callable[[list[str]], np.ndarray], tag: str = "") -> np.ndarray:
    """Return encode_fn(texts), cached at .cache/emb_<sha1(model_id + tag + all texts)>.npy."""
```

## 4. Registry: `searchers/registry.py` (owner P1)

```python
METHOD_ORDER: list[str]          # display order, see table
DEFAULT_UI_METHODS = ["tfidf", "bm25", "bge", "hybrid"]
def build_searcher(key: str) -> BaseSearcher     # new, unfitted instance
def available_methods() -> list[str]             # keys in METHOD_ORDER whose module imports OK
```
Entries import their module lazily inside the factory. If a module doesn't exist yet (not merged), `available_methods()` skips it with a printed warning instead of crashing.

| key | label | family | score_type | module : class | owner |
|---|---|---|---|---|---|
| `tfidf` | TF-IDF | lexical | cosine | `searchers/legacy.py : TFIDFAdapter` | P1 |
| `bm25` | BM25 | lexical | bm25 | `searchers/bm25.py : BM25Searcher` | P2 |
| `word2vec` | Word2Vec (corpus-trained) | static | cosine | `searchers/legacy.py : Word2VecAdapter` | P1 |
| `glove` | GloVe 300d (pretrained) | static | cosine | `searchers/glove.py : GloVeSearcher` | P2 |
| `specter` | SPECTER | contextual | cosine | `searchers/legacy.py : SpecterAdapter` | P1 |
| `bge` | BGE-base | contextual | cosine | `searchers/bge.py : BGESearcher` | P2 |
| `hybrid_rrf` | BM25 + BGE (RRF) | hybrid | rrf | `searchers/hybrid.py : HybridSearcher(rerank=False)` | P2 |
| `hybrid` | Hybrid + re-rank | hybrid | cross-encoder | `searchers/hybrid.py : HybridSearcher(rerank=True)` | P2 |
| `bm25_prf` *(optional)* | BM25 + query expansion | lexical | bm25 | `searchers/bm25_prf.py : BM25PRFSearcher` | P5 |

Note: the old frontend key `bert` is renamed to **`specter`**.

**Method colors** (used by both the UI and the charts so they match):
`tfidf #d97706`, `bm25 #dc2626`, `word2vec #0891b2`, `glove #059669`, `specter #9333ea`, `bge #2563eb`, `hybrid_rrf #64748b`, `hybrid #db2777`, `bm25_prf #ea580c`, `aspect #0f766e`.

## 5. Aspect search: `searchers/aspect.py` (owner P5)

Not a `BaseSearcher` (different signature).
```python
ASPECTS = ["task", "problem", "method", "result"]

@dataclass
class AspectHit:
    doc_index: int
    score: float          # cosine similarity of the best span
    span_start: int       # char offsets into the abstract
    span_end: int
    span_text: str

class AspectSearcher:
    key = "aspect"
    def fit(self, papers: list[Paper], spans: list[Span]) -> None
    def search(self, query: str, aspect: str, k: int = 10) -> list[AspectHit]   # best span per paper, best first
    def coverage(self) -> dict[str, int]    # number of papers that have >= 1 span of each aspect
```

## 6. HTTP API (owner P1; consumer P4)

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`.

**`GET /api/health`**
```json
{
  "status": "ready",
  "corpus_size": 727,
  "device": "cuda",
  "methods": [
    {"key": "bm25", "label": "BM25", "family": "lexical",
     "description": "…", "score_type": "bm25", "color": "#dc2626"}
  ],
  "default_methods": ["tfidf", "bm25", "bge", "hybrid"],
  "aspects": ["all", "task", "problem", "method", "result"],
  "aspect_coverage": {"task": 600, "problem": 480, "method": 650, "result": 620}
}
```
(Coverage numbers are examples.) `methods` lists only available methods, in `METHOD_ORDER`.

**`POST /api/search`**
Request:
```json
{"query": "text", "k": 5, "methods": ["tfidf", "bm25", "bge", "hybrid"], "aspect": "all"}
```
- `query`: 1–500 chars. `k`: 1–20, default 5.
- `methods`: 1–4 keys, default `default_methods`. An unknown key → 400.
- `aspect`: one of `aspects`, default `"all"`.

Response:
```json
{
  "query": "text",
  "aspect": "problem",
  "took_ms": 182.4,
  "methods": [
    {
      "key": "bm25", "label": "BM25", "family": "lexical", "description": "…",
      "score_type": "bm25", "color": "#dc2626", "took_ms": 2.1,
      "results": [
        {"rank": 1, "doc_index": 42, "score": 12.31,
         "arxiv_id": "2111.12503", "url": "https://arxiv.org/abs/2111.12503",
         "snippet": "first 300 chars…", "abstract": "full abstract",
         "matched_span": null}
      ]
    }
  ]
}
```
- `arxiv_id` and `url` can be `null`.
- When `aspect != "all"`, one extra entry is **appended** to `methods` with `key: "aspect"`, `label: "Aspect: Problem"`, `family: "contextual"`, `score_type: "cosine"`, `color: "#0f766e"`. Its results have `"matched_span": {"start": 120, "end": 210, "text": "…"}` (char offsets into `abstract`).
- Scores are raw. Only `score_type == "cosine"` is on a 0–1 scale.

**`GET /api/results`**: contents of `results/summary.json`; 404 `{"detail": "No results yet"}` if it doesn't exist.
**`GET /api/results/files/<name>`**: static files from `results/` (the chart PNGs).

## 7. Evaluation files (owner P3, except the hand-written file)

All in `eval/data/`, JSON Lines (one object per line).

`corpus_masked.json`: list of 727 strings, same order as `load_corpus()`. Each abstract with **all its task and problem spans** removed (use the `raw_start`/`raw_end` offsets, so truncated fragments are removed too; replace each removed range with one space, then collapse whitespace).

`queries_exact.jsonl`
```json
{"qid": "exact-0001", "query": "cleaned task/problem span text", "aspect": "task",
 "relevant_docs": [42], "corpus": "masked", "overlap": 0.62}
```
`queries_paraphrased.jsonl`: same fields, plus `"source_qid": "exact-0001", "original": "…"`; qids `para-0001…`.

`queries_handwritten.jsonl` (**owner P5**)
```json
{"qid": "hw-01", "query": "natural question in plain words", "relevant_docs": [42, 77],
 "corpus": "full", "notes": "why these papers are relevant"}
```
- `corpus: "masked"` → evaluate against `corpus_masked.json`; `"full"` → against the original abstracts.
- `overlap` = |content words of query ∩ content words of the relevant doc in that corpus| ÷ |content words of query|, using `data.content_words`. For several relevant docs, use the maximum. P3 computes it for all three sets (P5 may leave it out).

## 8. `results/summary.json` (owner P3; consumers P4, P5)

```json
{
  "generated_at": "2026-10-04T18:00:00",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "corpus_size": 727,
  "query_sets": {
    "exact":       {"n": 940, "corpus": "masked", "description": "…"},
    "paraphrased": {"n": 940, "corpus": "masked", "description": "…"},
    "handwritten": {"n": 20,  "corpus": "full",   "description": "…"}
  },
  "rows": [
    {"method": "bm25", "label": "BM25", "family": "lexical", "color": "#dc2626",
     "query_set": "paraphrased", "mrr@10": 0.41, "recall@10": 0.63,
     "latency_ms_mean": 1.8, "latency_ms_p95": 3.0}
  ],
  "ablation": ["bge", "hybrid_rrf", "hybrid"],
  "charts": [
    {"file": "overlap_vs_mrr.png", "title": "…", "caption": "…"}
  ],
  "examples": [
    {"qid": "para-0123", "query": "…", "query_set": "paraphrased", "relevant_docs": [42],
     "ranks": {"tfidf": 57, "bm25": 31, "bge": 1, "hybrid": 1},
     "explanation": "…"}
  ]
}
```
- `ranks`: rank of the first relevant doc within the top 100, or `null` if not found.
- Latency is the same for every query set (measured once per method); repeat it in each row.
- Numbers above are placeholders, not expected values.
