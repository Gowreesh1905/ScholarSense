"""
ScholarSense — Main Search Engine
===================================
Compares three retrieval methods side-by-side:
  1. TF-IDF       (lexical / keyword matching)
  2. Word2Vec     (static semantic — inline PyTorch implementation, no gensim needed)
  3. BERT/SPECTER (contextual semantic)

Usage:
    python main_search_engine.py
"""

import sys, io
# Force UTF-8 output so box-drawing and other Unicode chars work on all
# Windows terminals regardless of the system code-page (cp1252 etc.)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Teammates' modules ────────────────────────────────────────────────────────
from tfidf_lexical_search import TFIDFSearcher
from bert_specter_contextual_search import BERTSearcher

# ── Inline PyTorch Word2Vec (no gensim required — see word2vec_pytorch_search.py) ──
from word2vec_pytorch_search import Word2VecSearcher

# ── BERT / PyTorch ────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import torch


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath="abstract_sentences.csv"):
    """
    Loads the Scholar Inbox abstract_sentences.csv dataset.

    Returns
    -------
    abstracts : list of str   — unique abstract texts (index-aligned)
    arxiv_ids : list of str   — corresponding arXiv paper IDs
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # Keep first occurrence of each abstract (preserves order for index alignment)
    df_unique = df.drop_duplicates(subset="abstract").dropna(subset=["abstract"])
    abstracts = df_unique["abstract"].tolist()
    arxiv_ids = df_unique["arxiv_id"].astype(str).tolist()

    print(f"Loaded {len(abstracts)} unique research papers!\n")
    return abstracts, arxiv_ids


# =============================================================================
# DISPLAY
# =============================================================================

def display_top_k(query_embedding, corpus_embeddings,
                  abstracts, arxiv_ids, k=5, model_name=""):
    """
    Ranks all corpus documents by cosine similarity to the query and
    prints the top-k results with arXiv ID and abstract snippet.
    """
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]

    header = f"[ {model_name} | Top {k} Results ]"
    print("=" * 70)
    print(header)
    print("=" * 70)
    for rank, idx in enumerate(top_k_indices, 1):
        score       = similarities[idx]
        arxiv_id    = arxiv_ids[idx]
        snippet     = abstracts[idx][:300].replace("\n", " ") + "..."
        arxiv_url   = f"https://arxiv.org/abs/{arxiv_id}"

        print(f"  #{rank}  Score: {score:.4f}   arXiv: {arxiv_id}")
        print(f"       Link   : {arxiv_url}")
        print(f"       Snippet: {snippet}")
        print()
    print("-" * 70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    abstracts, arxiv_ids = load_data("abstract_sentences.csv")

    print("Initializing models (first run may take ~30 seconds)...\n")

    # ── 2. TF-IDF ─────────────────────────────────────────────────────────────
    print("> Setting up TF-IDF...")
    tfidf_searcher = TFIDFSearcher(abstracts)
    tfidf_corpus   = tfidf_searcher.get_corpus_embeddings()
    print()

    # ── 3. Word2Vec (inline PyTorch — no gensim needed) ───────────────────────
    print("> Setting up Word2Vec (training on corpus)...")
    w2v_searcher = Word2VecSearcher(abstracts, vector_size=100, epochs=5)
    w2v_corpus   = w2v_searcher.get_corpus_embeddings()
    print()

    # ── 4. BERT / SPECTER ─────────────────────────────────────────────────────
    print("> Setting up BERT (SPECTER)...")
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    specter       = SentenceTransformer("allenai/specter", device=device)
    bert_searcher = BERTSearcher(abstracts, specter)
    bert_corpus   = bert_searcher.get_corpus_embeddings()
    print()

    # ── 5. Interactive loop ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SCHOLARSENSE — INTERACTIVE SEMANTIC SEARCH ENGINE")
    print("=" * 70)
    print(f"  Corpus  : {len(abstracts)} research papers (Scholar Inbox dataset)")
    print(f"  Methods : TF-IDF  |  Word2Vec  |  BERT/SPECTER")
    print(f"  Device  : {device.upper()}")
    print("=" * 70)
    print("  Type your query and press Enter.  Type 'exit' to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input(">> Search Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            print("  (empty — please type a query)\n")
            continue
        if query.lower() == "exit":
            print("Goodbye!")
            break

        print()
        display_top_k(tfidf_searcher.get_query_embedding(query),
                      tfidf_corpus, abstracts, arxiv_ids,
                      k=3, model_name="TF-IDF (Lexical)")

        display_top_k(w2v_searcher.get_query_embedding(query),
                      w2v_corpus, abstracts, arxiv_ids,
                      k=3, model_name="Word2Vec (Static Semantic)")

        display_top_k(bert_searcher.get_query_embedding(query),
                      bert_corpus, abstracts, arxiv_ids,
                      k=3, model_name="BERT / SPECTER (Contextual)")


if __name__ == "__main__":
    main()
