# P3 — Evaluation

Read `team/README.md` and `team/CONTRACT.md` first (especially §1 `data.py`, §2 interface, §7 query files, §8 `summary.json`).

**Your job:** build the query sets, run every method on them, and produce the numbers and charts that prove (or disprove) the project's claim. **Report the results as they come out.** If BM25 beats the BERT models somewhere, that's a finding to explain, not a bug to hide.

**You own:** `eval/` (all of it except `eval/data/queries_handwritten.jsonl`, which P5 writes) and `results/`.
**Don't edit:** anything else.

**Before merge #1**, you can already write `build_queries.py` against the CONTRACT `data.py` signatures. If you need to run it before P1 pushes, make a temporary local copy of the loader and **don't commit it**.

---

## Why the evaluation is designed this way (put this reasoning in the report too)

- **Queries:** the CSV's task and problem spans are natural short descriptions of what a paper is about, so they serve as queries; the correct answer is the paper they came from.
- **The leak:** each span is copied word for word from its own abstract. Searching the original abstracts would let keyword methods win just by matching the copied text. So exact and paraphrased queries are evaluated against a **masked corpus**: every abstract with all its task and problem spans removed. The paper is still findable from its method and result text.
- **Paraphrasing:** back-translation (English → German → English) changes the wording but keeps the meaning, which recreates vocabulary mismatch.
- **Hand-written queries** (P5) are evaluated on the full corpus, because they aren't taken from the abstracts.

## Scripts (all run from the repo root)

### 1. `eval/build_queries.py`
- `corpus_masked.json`: for each paper, take every task/problem span from `load_spans()`, remove `abstract[raw_start:raw_end]` (replace with one space), collapse whitespace. Save the list of 727 strings in `load_corpus()` order.
- `queries_exact.jsonl`: one query per cleaned task/problem `Span` (≥4 words), with `relevant_docs = [span.doc_index]`, `corpus = "masked"`, `aspect`, and `overlap` (CONTRACT §7, computed against the masked text of that doc). Drop exact duplicate query texts that point to different docs (they're ambiguous).
- Print: number of queries by aspect, overlap distribution (mean and a 5-bin histogram), and the shortest masked abstract (check that none are empty).

### 2. `eval/paraphrase.py`
- `Helsinki-NLP/opus-mt-en-de` then `Helsinki-NLP/opus-mt-de-en` with `transformers` (`MarianTokenizer`, `MarianMTModel`). On GPU, batch size 32, `num_beams=4`, `max_new_tokens=256`. Run it on the GPU laptop: ~1,000 spans should take a few minutes there, much longer on CPU.
- Write `queries_paraphrased.jsonl` (same fields + `source_qid`, `original`; recompute `overlap` for the new text).
- Print how many paraphrases are identical to the original (expected: some) and the new overlap histogram. **If overlap barely drops, say so in the report**; it's a known limit of back-translation (technical terms survive translation).
- Commit the generated `.jsonl` files so nobody has to rerun this.

### 3. `eval/run_eval.py`
```bash
python eval/run_eval.py                                  # all available methods, all query sets
python eval/run_eval.py --methods tfidf bm25 --sets exact --quick   # first 50 queries only
```
- For each query set: choose the corpus (`masked` → `corpus_masked.json`, `full` → original abstracts). For each method from `available_methods()`: `build_searcher(key)`, `fit(corpus)`, then `search_batch(queries, k=100)`.
- Per query: rank of the first relevant doc in the top 100 (`None` if absent). **MRR@10** = 1/rank if rank ≤ 10 else 0. **Recall@10** = |relevant ∩ top 10| ÷ |relevant|. Average over queries.
- **Latency:** measured separately on the full corpus, per method: 5 warm-up queries, then single-query `search()` on 200 queries from the exact set, timed with `time.perf_counter`; report mean and p95 in ms. Run it on the GPU laptop and record the device and GPU name.
- Save per-query results to `results/raw/<method>__<set>.json` (qid → rank, top-10 doc indices), so charts and examples need no re-run.
- Write `results/summary.json` (CONTRACT §8; take `label`, `family`, `color` from the searcher/registry) and `results/summary.md` (a markdown results table for the report: rows = methods in `METHOD_ORDER`, columns = MRR@10 and R@10 per query set, plus latency; bold the best value per column).
- Skip a query set if its file doesn't exist yet (e.g. hand-written before P5 delivers) and say so in the output.
- Develop it on `tfidf` alone before merge #2, then run all methods after.

### 4. `eval/charts.py` (matplotlib, PNG, 150 dpi, white background, method colors from CONTRACT)
1. **`overlap_vs_mrr.png`, the key chart.** Pool the exact + paraphrased queries; bin by overlap `[0, .2), [.2, .4), [.4, .6), [.6, .8), [.8, 1]`; one line per method (MRR@10 per bin). Put the number of queries per bin under the x labels. Expected story: lexical methods fall as overlap drops, semantic ones hold up better. Plot what actually happens.
2. **`speed_vs_accuracy.png`:** x = mean latency (ms, log scale), y = MRR@10 on the paraphrased set, one labeled point per method.
3. **`ablation.png`:** grouped bars of MRR@10 for `bge`, `hybrid_rrf`, `hybrid` on each query set.
4. **`results_by_set.png`:** grouped bars of MRR@10 for every method on each query set.
- Add each chart to `summary.json` `charts` with a one-sentence `caption` stating what it shows **using the actual numbers**.

### 5. `eval/examples.py`
- Find candidates from `results/raw/`: (a) a query where lexical methods rank the paper > 20 and BGE/hybrid rank it 1–3; (b) the opposite, where BM25 wins and BGE fails; (c) one where re-ranking fixes BGE's mistake (or makes it worse).
- Pick 3, read the query and the abstract, and write a 2–3 sentence `explanation` of *why* each method succeeded or failed (shared or missing words, synonyms, a specific technical term). Store them in `summary.json` `examples`.

## Timeline
- H0.5–H1.5: `build_queries.py`.
- H1.5–H4: `paraphrase.py` (on the GPU laptop), `run_eval.py` working on `tfidf`, `word2vec`, `specter`.
- H4–H7: full run with all methods on the GPU laptop, `charts.py`, `examples.py`, commit `results/`.
- H7: send P5 the final `results/summary.md` and the charts for the report.

## Done when
- [ ] `eval/data/` holds `corpus_masked.json`, `queries_exact.jsonl`, `queries_paraphrased.jsonl` (and P5's hand-written file).
- [ ] `results/summary.json` validates against CONTRACT §8; P4's Results page renders it.
- [ ] 4 charts + `summary.md` committed; the numbers come from the GPU laptop.
- [ ] 3 examples with honest explanations.
- [ ] A short "evaluation design" paragraph sent to P5: the leak and the masking, paraphrasing, the metrics, and limitations (single relevant paper per span-query; short spans can be ambiguous; back-translation keeps technical terms).
