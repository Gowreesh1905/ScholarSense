# ScholarSense 🔍

A comparative semantic search engine for research paper abstracts, built as an NLP case study inspired by the **Scholar Inbox** paper.

The system lets a user type a free-text query and retrieves the most relevant papers from a corpus of 727 research-paper abstracts, using three different NLP retrieval methods side-by-side.

---

## Research Gap We Address

Traditional **keyword-based search** (TF-IDF) fails when the user's query and the paper use _different words_ to express the same idea — the **vocabulary mismatch problem**.

> **Example:** Searching for *"automatic translation of spoken language"* should match papers about *Neural Machine Translation*, even though none of those exact words appear in the query.

This project investigates whether **semantic representations** (Word2Vec, BERT) solve this problem better than pure lexical matching.

---

## The Three Methods

| Method | Type | Key Idea |
|:---|:---|:---|
| **TF-IDF** | Lexical / Keyword | Represents text as a sparse bag-of-words weighted by rarity. Only exact word matches score. |
| **Word2Vec** | Static Semantic | Learns a dense vector per word from co-occurrence. Words used in similar contexts get similar vectors. Documents are the _average_ of their word vectors. |
| **BERT / SPECTER** | Contextual Semantic | Transformer model that encodes the full sentence context. The same word can have different vectors in different sentences. SPECTER is fine-tuned specifically on scientific papers. |

---

## Project Structure

```
ScholarSense/
│
├── abstract_sentences.csv          ← Dataset (Scholar Inbox authors)
│
├── tfidf_lexical_search.py         ← TF-IDF module (Member 1)
├── word2vec_static_search.py       ← Word2Vec module (Member 2)  [reference, needs gensim]
├── word2vec_pytorch_search.py      ← Word2Vec, inline PyTorch version (no gensim needed)
├── bert_specter_contextual_search.py ← BERT module (Member 3)
│
├── main_search_engine.py           ← Orchestrator + interactive CLI
│
├── backend/                        ← FastAPI web API (wraps the same 3 searchers)
│   ├── engine.py                   ← Loads corpus, builds/caches all 3 searchers
│   ├── app.py                      ← REST API: /api/health, /api/search
│   └── requirements.txt
│
└── frontend/                       ← React + Vite + TypeScript + Tailwind web UI
```

> **Note:** `word2vec_static_search.py` uses `gensim`, which requires Microsoft C++ Build Tools to install on Python 3.14. `word2vec_pytorch_search.py` is a drop-in replacement that works without any extra installation — both the CLI and the web backend use it.

---

## Web UI (recommended)

A browser-based search UI that queries all three methods at once and shows
ranked results side by side, with badges highlighting papers found by
multiple methods.

**1. Backend (FastAPI) — one-time setup:**

```bash
pip install -r backend/requirements.txt
```

**2. Frontend — one-time setup:**

```bash
cd frontend
npm install
```

**3. Run (two terminals):**

```bash
# Terminal 1 — API server (http://localhost:8000)
python backend/app.py

# Terminal 2 — web UI (http://localhost:5173)
cd frontend
npm run dev
```

Open **http://localhost:5173** in a browser. The first search after a
fresh `python backend/app.py` start will be slow (training Word2Vec +
embedding all abstracts with BERT/SPECTER, same one-time cost as the CLI);
after that, the trained Word2Vec vectors and BERT corpus embeddings are
cached to disk under `.cache/`, so subsequent server restarts skip
straight to "ready."

To sanity-check the API on its own, without the UI:

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/search -H "Content-Type: application/json" -d "{\"query\": \"automatic translation of spoken language\", \"k\": 5}"
```

---

## CLI (alternative)

The original interactive terminal search still works standalone, with no
backend/frontend setup required — useful for quick experiments.

### Prerequisites
- Python 3.10 – 3.14
- An NVIDIA GPU is **strongly recommended** for BERT (CPU works but is slow)

### Step 1 — Install dependencies

```bash
pip install scikit-learn nltk pandas numpy sentence-transformers
```

For **GPU acceleration** (recommended — e.g. RTX 4060):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

For **CPU only**:
```bash
pip install torch
```

### Step 2 — Place the dataset

Make sure `abstract_sentences.csv` (from the Scholar Inbox authors) is in the same folder as the scripts.

### Step 3 — Run

```bash
python main_search_engine.py
```

On first run it will:
1. Download NLTK tokenizer data (~5 MB, one-time)
2. Download the SPECTER model weights from HuggingFace (~400 MB, one-time)
3. Train the Word2Vec model on the corpus (~15 seconds)
4. Generate BERT embeddings for all 727 papers (~7 seconds on RTX 4060 GPU, ~2 min on CPU)

After setup, the interactive prompt appears immediately on every subsequent run.

---

## How to Search

Once running, type any natural-language query and press **Enter**:

```
================================================================================
  SCHOLARSENSE — INTERACTIVE SEMANTIC SEARCH ENGINE
================================================================================
  Corpus  : 727 research papers (Scholar Inbox dataset)
  Methods : TF-IDF  |  Word2Vec  |  BERT/SPECTER
  Device  : CUDA
================================================================================

🔍  Query: methods for automatic translation of spoken language
```

The system returns the **Top 3 most similar papers** from each method, showing:
- **Similarity score** (0 = no match, 1 = perfect match)
- **arXiv ID** and a direct link to the paper on `arxiv.org`
- A **300-character abstract snippet**

Type `exit` to quit.

---

## Pipeline Overview

```
User Query (free text)
        │
        ▼
┌──────────────────────────────────────────────────┐
│              Preprocessing                        │
│   Lowercase → Tokenize → Remove stopwords →      │
│   Lemmatize  (TF-IDF & Word2Vec only)            │
│   Raw text passed directly for BERT              │
└──────────────────────────────────────────────────┘
        │
        ▼
┌───────────────┐   ┌───────────────┐   ┌──────────────────┐
│    TF-IDF     │   │   Word2Vec    │   │  BERT / SPECTER  │
│ Sparse vector │   │ Avg word vec  │   │  768-dim dense   │
│  (vocab-dim)  │   │  (100-dim)    │   │   context vec    │
└───────┬───────┘   └───────┬───────┘   └────────┬─────────┘
        │                   │                     │
        └───────────────────┼─────────────────────┘
                            │
                            ▼
                   Cosine Similarity
              (query vector vs all 727 docs)
                            │
                            ▼
                    Ranked Results (Top-K)
```

---

## Dataset

`abstract_sentences.csv` is released by the Scholar Inbox authors alongside their ACL Systems Demo paper:

> *"Scholar Inbox: Personalized Paper Recommendations for Scientists"*

The file contains **2,538 sentence-level annotations** across **727 unique paper abstracts**, with columns: `arxiv_id`, `abstract`, `start_idx`, `end_idx`, `label`.

We use only the `arxiv_id` and `abstract` columns. The sentence-level labels are ignored.

---

## Important Clarification

This case study is a **simplified comparative study** inspired by Scholar Inbox's semantic search functionality. Scholar Inbox itself uses GTE-Large embeddings and a content-based recommendation model with user ratings and active learning — it does not use TF-IDF, Word2Vec, or raw BERT as its final retrieval methods.
