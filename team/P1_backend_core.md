# P1 — Backend core + integrator

Read `team/README.md` and `team/CONTRACT.md` first. CONTRACT sections 1–4 and 6 are yours to implement.

**Your job:** build the shared foundation everyone else plugs into, fix the existing bugs, rebuild the backend around a method registry, and merge everyone's branches.

**You own:** `data.py`, `searchers/__init__.py`, `searchers/base.py`, `searchers/models.py`, `searchers/cache.py`, `searchers/registry.py`, `searchers/legacy.py`, `searchers/smoke.py`, `backend/*`, `main_search_engine.py`, and the one fix in `bert_specter_contextual_search.py`.
**Don't edit:** `tfidf_lexical_search.py`, `word2vec_pytorch_search.py`, `word2vec_static_search.py`, `notebooks/`, `frontend/`, other people's files.

---

## Phase 1 (H0.5 → H1.5): contract skeleton — highest priority, everyone waits on this

Push to `main` as soon as this works. Keep it small and correct; polish later.

1. **`data.py`**: implement CONTRACT §1 exactly.
   - `pd.read_csv(CSV_PATH, dtype={"arxiv_id": str})`.
   - `load_corpus()`: group rows by abstract in first-appearance order; `arxiv_id` = first non-null id for that abstract, else `None`. Must return 727 papers, 18 with `arxiv_id is None`.
   - `clean_span()` / `load_spans()` as specified. Print nothing.
   - `content_words()`: create one `TFIDFSearcher([])` lazily (its constructor only sets up the vectorizer) and return `instance.preprocess(text)`.
   - Add a `if __name__ == "__main__":` block that prints: number of papers, papers without an id, number of spans per aspect, and 5 random cleaned spans. Check that no cleaned span starts or ends mid-word.
2. **SPECTER double-load fix** in `bert_specter_contextual_search.py`: the model currently loads at import time (the "Cell 3" GPU check and "Cell 4" model load, roughly lines 60–96), so importing the module loads SPECTER, then the engine loads it again. Move Cells 3–4 into an `if __name__ == "__main__":` block (or a `load_specter()` function called from there). Keep the teammate's comment style. `BERTSearcher` must stay unchanged and importable without loading anything.
3. **`searchers/` package**: `__init__.py` (empty), `base.py`, `models.py`, `cache.py`, `registry.py` per CONTRACT §2–4.
   - `cache.py` writes to `<repo>/.cache/` (already gitignored).
   - The registry lists all keys from the CONTRACT table, including P2's and P5's, with lazy imports. Missing modules are skipped with a warning.
4. **`searchers/legacy.py`**: adapters that wrap the existing classes without modifying them:
   - `TFIDFAdapter`: `fit` creates `TFIDFSearcher(abstracts)` and calls `get_corpus_embeddings()`; `search` = cosine similarity of `get_query_embedding(query)` against the corpus matrix.
   - `Word2VecAdapter`: `fit` trains `Word2VecSearcher(abstracts, vector_size=100, epochs=5, verbose=False)`. Cache the trained searcher with pickle in `.cache/` keyed by sha1 of the abstracts + the parameters (the old engine did this; reuse that idea). If the query has no known words (all-zero vector), return `[]`.
   - `SpecterAdapter`: uses `get_specter()`; corpus embeddings via `cached_encode("allenai/specter", ...)`; cosine similarity.
   - Fill in `key`, `label`, `family`, `description`, `score_type` from the CONTRACT table.
5. **`searchers/smoke.py`**: `python -m searchers.smoke [keys...] [--query "..."]` fits each available method (or the given keys) on the full corpus and prints, per method: fit time, query time, and the top 3 (doc_index, score, first 80 chars). Default queries: `"making blurry photos sharp again"`, `"self-driving cars sensing surroundings with laser scanners"`, `"teaching robots to grasp unfamiliar objects"`. P2 and P5 use this to test their work.
6. Update `backend/requirements.txt`: add `rank-bm25`, `transformers`, `sentencepiece`, `sacremoses`, `matplotlib`.
7. Run `python data.py` and `python -m searchers.smoke tfidf word2vec specter`. Commit and push to `main`. Tell everyone: **merge #1 is ready.**

## Phase 2 (H1.5 → H4): backend on the registry

1. **`backend/engine.py`** rewrite:
   - At startup: `load_corpus()`, then for each key in `available_methods()`: `build_searcher(key)` then `fit([p.abstract for p in papers])`. Log fit time per method.
   - If `searchers/aspect.py` is importable, build `AspectSearcher` and fit it with `load_corpus()` and `load_spans()`. Otherwise aspects are unavailable (health returns `"aspects": ["all"]`).
   - `search(query, k, methods, aspect)` times **each method separately** (`time.perf_counter`) and builds the response exactly as in CONTRACT §6 (`doc_index`, nullable `arxiv_id`/`url`, `snippet` = first 300 chars + "…" if longer, `matched_span`).
   - Delete the old `load_data` (the float-ID bug) and the old W2V/BERT cache code (now handled in `legacy.py` / `cache.py`).
2. **`backend/app.py`**:
   - `SearchRequest`: add `methods: list[str] | None` (1–4 items) and `aspect: str = "all"`. Validate both and return 400 with a clear message on bad values.
   - `GET /api/health` per CONTRACT (include `color` per method from the CONTRACT color list; keep the table in one place, e.g. `registry.METHOD_COLORS`).
   - `GET /api/results` → read `results/summary.json`, 404 if missing.
   - Mount `results/` as static files at `/api/results/files` (`fastapi.staticfiles.StaticFiles`, `check_dir=False` so it starts even before P3 creates the folder, or create the folder at startup).
3. **`main_search_engine.py`**: replace its `load_data` with `data.load_corpus()` (fixes the ID bug in the CLI). Ideally switch the CLI to the registry too: show the top 3 for each of the original 3 methods by default, plus an optional `--methods bm25,bge,hybrid` argument. Keep the interactive loop as it is.
4. Test with curl:
   ```bash
   python backend/app.py
   curl http://127.0.0.1:8000/api/health
   curl -X POST http://127.0.0.1:8000/api/search -H "Content-Type: application/json" -d "{\"query\": \"teaching robots to grasp unfamiliar objects\", \"methods\": [\"tfidf\", \"bge\"], \"aspect\": \"problem\"}"
   ```

## Merge duty

- **H1.5 (merge #1):** your skeleton.
- **H4 (merge #2):** merge `p2-models` and `p5-aspect`. Run `python -m searchers.smoke` (all methods) and the curl tests. Fix only *integration* problems yourself; send bugs inside someone's module back to them.
- **H7 (merge #3):** merge everything (`p3-eval` with results, `p4-frontend`, `p5-aspect` report). Do a full run: backend, then frontend, every method, every aspect, Results page.
- After each merge, tell everyone to `git merge origin/main`.

## Done when

- [ ] `python data.py`: 727 papers, 18 without an ID, no ID like `2101.0686` (the trailing zero must be kept: `2101.06860`).
- [ ] Importing `bert_specter_contextual_search` doesn't load a model.
- [ ] `/api/health` lists all available methods with colors and aspect coverage.
- [ ] `/api/search` returns per-method `took_ms`, handles `methods` and `aspect`, returns `null` url for papers without an ID.
- [ ] `/api/results` and `/api/results/files/*.png` work once P3's results exist.
- [ ] The CLI still runs.
