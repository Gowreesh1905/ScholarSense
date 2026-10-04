# Evaluation design (for the report, section 4) and key findings

All numbers are from `results/summary.json`, `results/leak_check.json` and `results/significance.json`.
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (the brief expected an RTX 4060; the numbers below are from the 4050).
The hand-written query set was **not available** when this was run (P5's `queries_handwritten.jsonl`);
`python eval/run_eval.py` picks it up automatically once the file exists, and adds a third column to every table.

## Evaluation design paragraph

We test how well each method finds a paper from a short description of what it is about. The queries are the
task and problem spans that the Scholar Inbox annotators marked in each abstract (1,030 queries with at least four
words); the correct answer is the paper the span came from. There is one catch: every span is copied word for word
from its own abstract, so searching the original abstracts would reward any method that matches the copied text.
We measured this: on the original abstracts TF-IDF and BM25 score 0.96 and 0.97 MRR@10, but 0.52 once the spans
are removed (BGE: 0.77 vs 0.50). So the exact and paraphrased queries are searched in a **masked corpus**: every
abstract with all of its task and problem spans cut out. The paper can still be found from its method and result
sentences. To recreate the vocabulary mismatch problem, we also **back-translate** each query English → German →
English (Helsinki-NLP Marian models, beam search of 4), which changes the wording but keeps the meaning. Hand-written
queries (P5) are searched in the full corpus, because they are not taken from the abstracts. We report **MRR@10**
(1 / rank of the first correct paper if it is in the top 10, else 0) and **Recall@10**, averaged over queries, and
**latency** (mean and p95 of a single-query search on the full corpus after 5 warm-up queries, 200 queries).

## Limitations (state these in the report)

- **One correct paper per span query.** Other papers in the corpus may be just as relevant (the corpus is mostly
  computer vision, with many near-duplicate GAN/NeRF/image-editing papers), so MRR is a lower bound on real quality.
- **Short spans are ambiguous.** The shortest queries are four words and generic ("high resolution semantic image
  synthesis"); many papers fit them.
- **Back-translation barely reduces word overlap.** Mean overlap between query and the relevant masked abstract
  only fell from 0.422 (exact) to 0.374 (paraphrased); 5.0% of paraphrases are identical to the original
  (8.0% ignoring case and punctuation), because technical terms survive translation. Back-translation also
  sometimes introduces errors ("editing operations" → "machining processes").
- **Masking is not perfect.** Spans shorter than four words are not masked, and a few papers lose most of their
  text (one paper with a 28-word abstract was reduced to one character, so its 2 queries were dropped; the other
  1,030 queries are kept). The masked text can also start or end mid-word, because the CSV offsets are dirty.
- **Overlap is a rough proxy.** It counts shared lemmatised content words, and low-overlap queries also tend to be
  shorter or more generic, so the overlap chart mixes "vocabulary mismatch" with "harder query".
- **Hand-written queries are a small set** (about 20), so the third column has wide error bars.
- The corpus is small (727 papers) and one machine produced all the latency numbers.

## Key findings (report them as they are)

MRR@10 (masked corpus, 1,030 queries per set):

| Method | exact | paraphrased |
|---|---:|---:|
| TF-IDF | 0.519 | 0.438 |
| BM25 | 0.523 | 0.439 |
| Word2Vec (corpus-trained) | 0.272 | 0.215 |
| GloVe 300d | 0.237 | 0.208 |
| SPECTER | 0.233 | 0.197 |
| BGE-base | 0.498 | 0.462 |
| BM25 + BGE (RRF) | 0.560 | 0.507 |
| Hybrid + re-rank | 0.548 | 0.506 |

1. **Keyword search is hard to beat on these queries.** BM25 and TF-IDF are as good as or better than BGE on exact
   queries (BM25 − BGE = +0.025, 95% CI [+0.001, +0.049], just significant). On paraphrased queries BGE is ahead by
   +0.023 MRR@10, but the 95% interval [−0.003, +0.048] includes zero, so **we cannot claim that BGE beats BM25 on
   the paraphrased set as a whole**.
2. **Where semantics do help is at low word overlap** (`overlap_vs_mrr.png`, 2,060 pooled queries). From the
   highest-overlap bin [0.8, 1] (n=112) to the lowest [0, 0.2) (n=326): TF-IDF 0.78 → 0.16, BM25 0.83 → 0.14,
   BGE 0.63 → 0.30, Hybrid + re-rank 0.82 → 0.21. BGE is the best single method in the lowest bin and loses to
   BM25 in the high-overlap bins. This is the vocabulary-mismatch result, but with the overlap caveat above.
3. **Older semantic methods are much worse than keyword search** here: corpus-trained Word2Vec, pretrained GloVe
   and SPECTER all score 0.20–0.27. BGE beats SPECTER by 0.26 MRR@10 (significant). SPECTER is trained to compare
   whole papers (title + abstract), not to answer short queries, which is a likely reason; we did not test it.
4. **Fusion is the best idea in the project.** BM25 + BGE (RRF) beats BGE (+0.062 exact, +0.046 paraphrased) and
   BM25 (+0.037 exact, +0.068 paraphrased); all four gaps are significant.
5. **Cross-encoder re-ranking did not help.** Hybrid + re-rank vs RRF alone: −0.012 exact, −0.001 paraphrased,
   both within noise, and it makes a query slower by about 9× (150 ms vs 17 ms mean). It does raise Recall@10
   slightly (0.766 vs 0.757 exact; 0.717 vs 0.708 paraphrased).
6. **Speed:** TF-IDF/BM25 take about 2–3 ms per query, BGE about 11 ms, RRF fusion 17 ms, re-ranking 150 ms
   (mean, GPU).
