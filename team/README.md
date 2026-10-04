# ScholarSense — one-day team plan

**How to use this folder:** every teammate tells their Claude:

> Read `team/README.md`, `team/CONTRACT.md` and `team/P<N>_<role>.md`, then do the work in `team/P<N>_<role>.md`. Only edit the files that brief says I own.

| Person | Brief | Owns |
|---|---|---|
| P1 — Backend core + integrator | [P1_backend_core.md](P1_backend_core.md) | `data.py`, `searchers/base.py`, `searchers/models.py`, `searchers/cache.py`, `searchers/registry.py`, `searchers/legacy.py`, `searchers/smoke.py`, `backend/`, `main_search_engine.py`, the SPECTER fix in `bert_specter_contextual_search.py` |
| P2 — New models | [P2_models.md](P2_models.md) | `searchers/bm25.py`, `searchers/glove.py`, `searchers/bge.py`, `searchers/hybrid.py` |
| P3 — Evaluation | [P3_evaluation.md](P3_evaluation.md) | `eval/` (except `eval/data/queries_handwritten.jsonl`), `results/` |
| P4 — Frontend | [P4_frontend.md](P4_frontend.md) | `frontend/` |
| P5 — Aspect search + report | [P5_aspect_report.md](P5_aspect_report.md) | `searchers/aspect.py`, `searchers/bm25_prf.py` (optional), `eval/data/queries_handwritten.jsonl`, `report/` |

Names: P1 = ______  P2 = ______  P3 = ______  P4 = ______  P5 = ______

---

## What the project is (context for every Claude)

ScholarSense compares retrieval methods on 727 research-paper abstracts (the Scholar Inbox dataset, `abstract_sentences.csv`). The research question: **do semantic methods solve the vocabulary-mismatch problem that keyword search has?** (e.g. the query says "spoken language", the paper says "speech").

**Today the repo has:**
- 3 methods: TF-IDF (`tfidf_lexical_search.py`), Word2Vec trained on the corpus in PyTorch (`word2vec_pytorch_search.py`), SPECTER (`bert_specter_contextual_search.py`). `word2vec_static_search.py` (gensim) is unused reference code; leave it alone.
- A CLI (`main_search_engine.py`), a FastAPI backend (`backend/app.py`, `backend/engine.py`), and a React + Vite + TypeScript + Tailwind v4 frontend (`frontend/`).
- Nothing is measured: there is no evaluation.

**What we are adding today:**
1. Bug fixes + a method registry, so methods plug in without hardcoding (P1).
2. 4 new methods: BM25, GloVe, BGE, Hybrid (BM25 + BGE → RRF → cross-encoder re-rank), plus a `hybrid_rrf` ablation (P2).
3. A real evaluation: 3 query sets, MRR@10 / Recall@10 / latency, charts (P3).
4. **Aspect-based search**: search only the *task*, *problem*, *method* or *result* parts of abstracts, using the dataset's labels (P5).
5. UI: model picker, aspect selector with highlighted matches, Results page (P4).

**What we are NOT doing:** a multi-laptop cluster, the 50k arXiv corpus, or fine-tuning. Don't add them.

## Dataset facts (verified; every Claude should rely on these)

- `abstract_sentences.csv`: 2,538 rows, columns `arxiv_id, abstract, start_idx, end_idx, label`.
- Each row is one labeled span: `abstract[start_idx:end_idx]` (character offsets, end exclusive).
- Labels: `idea` 678, `task` 664, `result` 652, `problem` 544. **We display `idea` as "method".**
- 727 unique abstracts. Corpus order = order of first appearance of each unique abstract in the CSV. **`doc_index` = position in that order. Everything identifies papers by `doc_index`, never by arXiv ID.**
- `arxiv_id` **must be read as a string** (`dtype={"arxiv_id": str}`). Reading it as a float turns `2101.06860` into `2101.0686` (60 papers) — that is the current bug.
- 18 abstracts have no arXiv ID in any row → `arxiv_id = None`, `url = None`.
- Spans are dirty: ~20% start or end mid-word (e.g. `"...To address th"`), 13 rows have `start_idx` past the end of the abstract, some are 0–3 words. `data.load_spans()` cleans them (see CONTRACT).
- Task/problem spans: 1,208 raw → about 830–1,050 usable after cleaning (≥4 words).
- The corpus is mostly computer vision (381/727 abstracts mention "image"). NLP topics are rare (1 abstract mentions machine translation).
- Abstracts: median ~168 words, max ~297 (fits in 512 tokens).

## Environment

- **Python 3.14** on Windows. Everyone uses the same versions.
- One laptop has an RTX 4060 (CUDA). **Final evaluation numbers and latency must be produced on that laptop.** Other laptops can develop on CPU.
- Install (everyone, at the start):
  ```bash
  pip install -r backend/requirements.txt
  pip install rank-bm25 sentencepiece sacremoses matplotlib transformers
  ```
  GPU laptop only: `pip install torch --index-url https://download.pytorch.org/whl/cu126`
- Pre-download models (everyone, at the start, ~1.5 GB):
  ```bash
  python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('allenai/specter'); SentenceTransformer('BAAI/bge-base-en-v1.5'); SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"
  ```
  P3 also: `python -c "from transformers import MarianMTModel, MarianTokenizer; [(MarianTokenizer.from_pretrained(m), MarianMTModel.from_pretrained(m)) for m in ['Helsinki-NLP/opus-mt-en-de','Helsinki-NLP/opus-mt-de-en']]"`
- Frontend: Node 24, `cd frontend && npm install`.

## Git rules

- Each person works on their own branch: `p1-core`, `p2-models`, `p3-eval`, `p4-frontend`, `p5-aspect`.
- **Only edit files you own.** If you need a change in someone else's file, ask that person; don't edit it yourself.
- **Don't change anything in `team/CONTRACT.md` without P1 agreeing.** Other people's code depends on it.
- P1 merges branches into `main` at each merge point. After each merge, everyone runs `git merge origin/main` on their branch.
- Never commit `.cache/`, model weights, or `node_modules/` (already in `.gitignore`). **Do** commit `results/*.json`, `results/*.md`, `results/*.png`, `eval/data/*.jsonl`.

## Schedule (H = hours from start)

| Time | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| H0–H0.5 | **All together:** read CONTRACT, assign names, create branches, start installs/downloads | | | | |
| H0.5–H1.5 | **Contract skeleton → push to main** | BM25, BGE (against CONTRACT) | `build_queries.py` | Mock API + registry-driven UI | Hand-written queries |
| **H1.5 merge #1** | P1's skeleton on main; everyone merges it | | | | |
| H1.5–H4 | Backend engine + API | GloVe, Hybrid, smoke tests | `paraphrase.py`, `run_eval.py` on TF-IDF | Model picker, aspect selector | `aspect.py` |
| **H4 merge #2** | P2 models + P5 aspect into main | | | P4 switches from mock to real API | |
| H4–H7 | Aspect + results endpoints, fix integration bugs | Help P3, tune, fix bugs | Full eval on GPU laptop, charts, examples | Results page | Report draft, optional `bm25_prf` |
| **H7 merge #3** | Everything into main | | | | |
| H7–H8 | **All:** full run-through, fill report numbers, rehearse demo | | | | |

**P1 is the bottleneck in the first 1.5 hours.** Until merge #1, everyone else codes against the interfaces in CONTRACT.md (they are exact), using stubs/mocks where needed.

## Demo (≈6 minutes, for the review)
1. Search a vocabulary-mismatch query with TF-IDF, BM25, BGE and Hybrid side by side: lexical methods miss, semantic methods find it.
2. Results page: the overlap chart (scores fall for lexical methods as word overlap drops) and the results table.
3. The ablation: BGE → +BM25 (RRF) → +re-rank.
4. Aspect search: same query on "Problem" vs whole abstract, with the matched span highlighted.
5. Honest findings: where lexical methods win, and why.

## Stretch goals (only if everything above works)
- Topic map: KMeans + t-SNE/PCA of BGE embeddings, plotted with the query as a point (whoever is free).
- Optional `bm25_prf` query-expansion method (P5 brief).
