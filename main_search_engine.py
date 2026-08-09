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

import re
import string

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ── Teammates' modules ────────────────────────────────────────────────────────
from tfidf_lexical_search import TFIDFSearcher
from bert_specter_contextual_search import BERTSearcher

# ── BERT / PyTorch ────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import torch


# =============================================================================
# INLINE WORD2VEC  (pure PyTorch — no gensim required)
# =============================================================================
# gensim has no pre-built wheel for Python 3.14 and requires C++ Build Tools
# to compile.  This lightweight replacement trains a skip-gram Word2Vec model
# directly on the corpus using PyTorch, then uses average-word-embedding to
# produce document vectors — identical in concept to the teammate's module.
# =============================================================================

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]

def _ensure_nltk():
    for path, pkg in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


class _SkipGram(torch.nn.Module):
    """Minimal skip-gram model with negative sampling."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center  = torch.nn.Embedding(vocab_size, embed_dim)
        self.context = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.xavier_uniform_(self.center.weight)
        torch.nn.init.xavier_uniform_(self.context.weight)

    def forward(self, center_ids, context_ids):
        c = self.center(center_ids)           # (B, D)
        ctx = self.context(context_ids)       # (B, D)
        return (c * ctx).sum(dim=1)           # dot product


class Word2VecSearcher:
    """
    Static Semantic Search using Word2Vec (pure PyTorch skip-gram).

    Trains a Word2Vec model from scratch on the provided corpus and
    represents each document as the average of its word vectors.
    No external NLP libraries beyond PyTorch and NLTK are required.
    """

    def __init__(self, abstracts, vector_size=100, window=5,
                 min_count=2, epochs=5, neg_samples=5, lr=0.025,
                 batch_size=512, random_seed=42):
        _ensure_nltk()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)
        self.vector_size = vector_size
        self._corpus_embeddings = None
        self.abstracts = abstracts

        torch.manual_seed(random_seed)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # ── 1. Preprocess ─────────────────────────────────────────────────────
        print(f"[Word2Vec] Preprocessing {len(abstracts)} abstracts...")
        self.tokenized_corpus = [self.preprocess(a) for a in abstracts]

        # ── 2. Build vocabulary ───────────────────────────────────────────────
        freq = defaultdict(int)
        for tokens in self.tokenized_corpus:
            for t in tokens:
                freq[t] += 1
        vocab = [w for w, c in freq.items() if c >= min_count]
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = vocab
        vocab_size = len(vocab)
        print(f"[Word2Vec] Vocabulary size: {vocab_size}")

        # ── 3. Build training pairs (skip-gram) ───────────────────────────────
        pairs = []
        for tokens in self.tokenized_corpus:
            ids = [self.word2id[t] for t in tokens if t in self.word2id]
            for i, center in enumerate(ids):
                for j in range(max(0, i-window), min(len(ids), i+window+1)):
                    if j != i:
                        pairs.append((center, ids[j]))
        if not pairs:
            raise ValueError("No training pairs generated — corpus may be too small.")
        pairs = torch.tensor(pairs, dtype=torch.long)        # (N, 2)
        print(f"[Word2Vec] Training pairs: {len(pairs):,}")

        # ── 4. Train ──────────────────────────────────────────────────────────
        model = _SkipGram(vocab_size, vector_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(pairs[:, 0], pairs[:, 1])
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True)
        print(f"[Word2Vec] Training {epochs} epoch(s) on {device_str.upper()}...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for center_ids, ctx_ids in loader:
                center_ids = center_ids.to(self.device)
                ctx_ids    = ctx_ids.to(self.device)

                # Positive samples
                pos_score = model(center_ids, ctx_ids)
                pos_label = torch.ones_like(pos_score)

                # Negative samples (random)
                neg_ids   = torch.randint(0, vocab_size, ctx_ids.shape,
                                          device=self.device)
                neg_score = model(center_ids, neg_ids)
                neg_label = torch.zeros_like(neg_score)

                loss = criterion(torch.cat([pos_score, neg_score]),
                                 torch.cat([pos_label, neg_label]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

        # Extract the center-word embedding matrix as NumPy
        self.word_vectors = model.center.weight.detach().cpu().numpy()
        print("[Word2Vec] Training complete!")

    # ── Preprocessing (same pipeline as teammates) ────────────────────────────

    @staticmethod
    def _wordnet_pos(tag):
        if tag.startswith("J"): return wordnet.ADJ
        if tag.startswith("V"): return wordnet.VERB
        if tag.startswith("R"): return wordnet.ADV
        return wordnet.NOUN

    def preprocess(self, text):
        if not isinstance(text, str):
            return []
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        tokens = word_tokenize(text)
        cleaned = []
        for tok in tokens:
            tok = "".join(c for c in tok if c not in self.punctuation)
            if not tok.isalpha() or len(tok) < 3 or tok in self.stop_words:
                continue
            cleaned.append(tok)
        if not cleaned:
            return []
        tagged = nltk.pos_tag(cleaned)
        return [self.lemmatizer.lemmatize(w, self._wordnet_pos(t))
                for w, t in tagged]

    def _avg_vector(self, tokens):
        vecs = [self.word_vectors[self.word2id[t]]
                for t in tokens if t in self.word2id]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_corpus_embeddings(self):
        if self._corpus_embeddings is None:
            print(f"[Word2Vec] Computing document embeddings...")
            self._corpus_embeddings = np.vstack(
                [self._avg_vector(toks) for toks in self.tokenized_corpus]
            )
            print(f"[Word2Vec] Corpus embeddings shape: {self._corpus_embeddings.shape}")
        return self._corpus_embeddings

    def get_query_embedding(self, query_string):
        tokens = self.preprocess(query_string)
        return self._avg_vector(tokens)


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
