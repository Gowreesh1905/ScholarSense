# ScholarSense: slide outline (10 slides, about 6 minutes plus demo)

The order matches `report/DEMO.md`. Numbers are copied from `results/summary.json`; if P3 re-runs the evaluation, update
slides 7 and 8 and the TBD-HW item on slide 10. Charts are in `results/`.

---

## Slide 1. Title
**ScholarSense: does semantic search close the vocabulary gap?**
- Comparing 8 retrieval methods on 727 research-paper abstracts (Scholar Inbox data)
- What is new in phase 2: more methods, a real evaluation, aspect search, a web app with a results page
- Team names and roles
- *Visual:* none, or the app's logo.

## Slide 2. The problem: vocabulary mismatch
- Keyword search needs the query and the paper to share words; people rarely write in a paper's own words
- Example from our corpus: *"figuring out how far away everything is in a picture taken by a camera with a 360-degree view"*
  → the right papers are about "omnidirectional depth estimation"
- Question: do semantic methods fix this, and what does a hybrid add?
- *Visual:* the query on the left, the three target abstracts' key phrases on the right, few shared words highlighted.

## Slide 3. Data
- 727 abstracts, 2,538 labelled spans (task, problem, method, result), mostly computer vision
- Bugs found and fixed: 60 arXiv IDs corrupted by float parsing; 22.4% of spans cut mid-word; 13 rows past the abstract end
- After cleaning: 2,339 spans; papers per aspect: task 501, problem 535, method 668, result 635
- *Visual:* a before/after of one dirty span ("To address th" → whole words).

## Slide 4. Eight methods in four families
- Lexical: TF-IDF, BM25. Static: Word2Vec (corpus-trained), GloVe (pretrained). Contextual: SPECTER, BGE. Hybrid: BM25 + BGE (RRF), then + cross-encoder re-rank
- All behind one interface, so the web app and the evaluation treat them the same
- *Visual:* the hybrid pipeline diagram (BM25 top-100 + BGE top-100 → RRF → top-30 → cross-encoder → top-k).

## Slide 5. How we test fairly
- Queries = the task and problem spans of each paper (1,030); the correct answer is the paper they came from
- **The leak:** a span is copied from its own abstract, so keyword search scores 0.96–0.97 MRR@10 on the original text but 0.52 once spans are masked out
- So we search a **masked corpus**; we also back-translate queries (English → German → English) to create mismatch; 20 hand-written plain-language queries on the full corpus
- Metrics: MRR@10, Recall@10, latency
- *Visual:* the leak table (TF-IDF 0.962 → 0.519, BM25 0.972 → 0.523, BGE 0.766 → 0.498).

## Slide 6. LIVE DEMO 1: side by side
- Type the 360° depth query with TF-IDF, BM25, BGE and Hybrid: lexical methods miss it, BGE finds two of three papers in its top 2
- Second query (robot grasp from simple shapes): lexical miss, BGE and Hybrid rank the paper first
- *Visual:* screenshot `report/img/01-mismatch.png` (fallback if the demo fails).

## Slide 7. Results page: the key chart
- Keyword search is hard to beat on exact queries (BM25 0.523 vs BGE 0.498); BGE's lead on paraphrased queries (+0.023) is not significant
- Overlap chart: as word overlap falls from the top bin to the bottom bin, BM25 drops 0.83 → 0.14 and BGE only 0.63 → 0.30
- Older semantic methods (Word2Vec, GloVe, SPECTER) score 0.20–0.27
- *Visual:* `results/overlap_vs_mrr.png` and the results table. (Hand-written column: **TBD-HW**.)

## Slide 8. The ablation and the cost
- BGE 0.498 → + BM25 by rank fusion 0.560 (significant) → + cross-encoder 0.548 (no gain, within noise)
- Same pattern on paraphrased queries: 0.462 → 0.507 → 0.506
- Latency: fusion 17 ms, re-ranking 150 ms (about 9× slower for no measured gain)
- *Visual:* `results/ablation.png` beside `results/speed_vs_accuracy.png`.

## Slide 9. LIVE DEMO 2: aspect search
- Same query on the **Problem** aspect versus the whole abstract, with the matched sentence highlighted
- Query: *"results look unrealistic or contain visible artifacts"* (Problem): five papers that state this as their problem, versus an off-target mix from whole-abstract BGE
- Also: *"runs in real time at tens of frames per second"* (Result)
- Be upfront: chosen by looking at the output; it helps when the query reads like a complaint or a result
- *Visual:* screenshot `report/img/04-aspect.png`.

## Slide 10. Honest findings and conclusion
- Where keywords win: when the query shares words with the paper (example para-0158: BM25 and TF-IDF rank it 1st, BGE 53rd)
- Where re-ranking hurts: example para-0132, correct paper 1st by BGE, 23rd after re-ranking
- Limits: one correct paper per query; back-translation keeps technical terms (mean overlap 0.422 → 0.374); 727 papers, mostly computer vision; 20 hand-written queries
- **Answer:** semantic search closes the gap partly. It helps most at low word overlap; fusing it with BM25 is better than either; "semantic" does not always beat "keyword"
- *Visual:* a three-line summary; hand-written result: **TBD-HW**.
