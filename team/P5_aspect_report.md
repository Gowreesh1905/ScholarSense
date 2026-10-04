# P5 — Aspect search + hand-written queries + report

Read `team/README.md` and `team/CONTRACT.md` first (especially §1 `Span`/`load_spans`, §5 aspect interface, §7 hand-written query format).

**Your job has three parts:**
1. **Hand-written queries** (first, P3 needs them): ~20 natural queries with the correct papers marked.
2. **Aspect-based search:** search only the task, problem, method or result parts of abstracts, using the dataset's labels.
3. **The report, slides and demo script**, filled with the real numbers at the end.

**You own:** `eval/data/queries_handwritten.jsonl`, `searchers/aspect.py`, `searchers/bm25_prf.py` (optional), `report/`.
**Don't edit:** anything else. P1 wires your `AspectSearcher` into the API; P4 builds its UI.

---

## Part 1 (H0.5 → H2): hand-written queries → `eval/data/queries_handwritten.jsonl`

These are the most convincing test of vocabulary mismatch, because they're written the way a person actually searches.

- Write **~20 queries** in plain language that **avoids the papers' own technical terms**. E.g. "making blurry photos sharp again" rather than "image deblurring"; "self-driving cars sensing surroundings with laser scanners" rather than "LiDAR 3D object detection".
- The corpus is mostly computer vision (images, 3D, video, point clouds, autonomous driving, robotics; also some reinforcement learning and language models). Pick topics that exist. Avoid NLP topics like translation: only ~1 paper.
- For each query, find the relevant papers (1–5 per query, by `doc_index`):
  - Collect candidates from **several** methods so you don't favor one: before merge #1, search the CSV text by keyword; after merge #1, `python -m searchers.smoke tfidf specter --query "..."`; after merge #2, add `bm25 bge`.
  - **Read the abstracts** and mark a paper relevant only if its main topic answers the query. Don't mark a paper relevant just because a model ranked it first.
- Format (CONTRACT §7): `{"qid": "hw-01", "query": "...", "relevant_docs": [42, 77], "corpus": "full", "notes": "..."}`. `doc_index` = position in `data.load_corpus()`.
- Commit and tell P3 by **H2**. 20 good queries beat 40 rushed ones.

## Part 2 (H2 → H4): `searchers/aspect.py`

Implement CONTRACT §5 exactly.

- `fit(papers, spans)`:
  - Keep spans grouped by aspect (`task`, `problem`, `method`, `result`; the CSV label `idea` is already mapped to `method` by `data.py`).
  - Embed each span's `text` with `get_bge()` (**no** query prefix for spans), `normalize_embeddings=True`, cached with `cached_encode("BAAI/bge-base-en-v1.5", texts, fn, tag="aspect-spans")`.
- `search(query, aspect, k)`:
  - Embed the query **with** `BGE_QUERY_PREFIX`.
  - Score every span of that aspect (dot product); for each paper keep its **best** span; return the top-k papers as `AspectHit` with that span's `start`, `end`, `text`.
  - Unknown aspect → `ValueError`.
- `coverage()`: number of distinct papers having ≥1 span per aspect.
- A `if __name__ == "__main__":` block: fit on `load_corpus()` and `load_spans()`, print coverage, then for 3 queries print the top 3 hits per aspect with the matched span text.
- Test: `python -m searchers.aspect`. The matched spans should read as whole words (`data.load_spans` cleans them; if one doesn't, tell P1).
- Find **3 demo queries** where aspect search clearly helps. E.g. a problem-style query ("methods break down when the camera moves a lot") on **Problem** returns papers that *tackle* that problem, while whole-abstract BGE also returns papers that merely mention it. Write them down for the demo and report.
- Push to `p5-aspect` before **H4 (merge #2)**.

## Optional (only if Parts 1–2 are done by H4.5): `searchers/bm25_prf.py`

BM25 with pseudo-relevance feedback (query expansion). It answers the examiner's likely question: "couldn't keyword search just be patched?"

- `BM25PRFSearcher(BaseSearcher)`, key `bm25_prf`, family `lexical`, score_type `bm25`, label "BM25 + query expansion". The registry entry already exists (P1).
- `fit`: build an internal `BM25Searcher` (from P2's `searchers/bm25.py`) and keep the tokenized docs.
- `search`: run BM25; take the top 5 docs; score their terms (frequency in those docs × IDF from the BM25 object); add the top 10 terms not already in the query; search again with the original query tokens **twice** plus the expansion terms once.
- Tell P3 so it appears in the results table.

## Part 3 (H4 → end): report, slides, demo script, all in `report/`

Write in plain, short English. Every number comes from `results/summary.md` / `summary.json` (P3 sends the final version at H7). Leave `TBD` placeholders until then; never invent numbers.

**`report/REPORT.md`**
1. Introduction and research gap: vocabulary mismatch, with an example.
2. Dataset: Scholar Inbox abstracts, 727 papers, the labeled spans, the data problems we found and fixed (float IDs, truncated spans).
3. Methods: a table of all methods by family (lexical / static / contextual / hybrid) with one line each on how it works; the hybrid pipeline diagram; aspect search.
4. Evaluation design: the three query sets; **why we mask the corpus** (the leak); back-translation; MRR@10, Recall@10, latency; hardware (from `summary.json`). P3 sends a paragraph for this.
5. Results: the results table, the overlap chart, speed vs. accuracy, the ablation (what BM25 fusion adds, what re-ranking adds).
6. Aspect search: what it is, coverage, the 3 demo queries.
7. Example queries: P3's 3 examples with explanations.
8. Honest findings and limitations: where lexical methods win and why; one relevant paper per span-query; short spans are ambiguous; back-translation keeps technical terms; small CV-heavy corpus; 20 hand-written queries is a small set.
9. Conclusion: does semantic search close the vocabulary gap? Answer from the data.

**`report/SLIDES.md`**: an 8–10 slide outline (title + 3–4 bullets + which chart/screenshot each), matching the demo order.

**`report/DEMO.md`**: the ≈6-minute script from `team/README.md`, with the **exact** queries to type (tested against the final build), which methods to select, what to point at, and a fallback if the backend is slow (screenshots in `report/img/`).

## Done when
- [ ] `queries_handwritten.jsonl` committed with ~20 queries (by H2).
- [ ] `python -m searchers.aspect` works; the API returns highlighted spans after merge #2.
- [ ] `REPORT.md`, `SLIDES.md`, `DEMO.md` complete with the real numbers; no `TBD` left.
- [ ] Demo queries tested on the final merged build.
