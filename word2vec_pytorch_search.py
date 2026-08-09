# ================================================================
# Static Semantic Search using Word2Vec — inline PyTorch implementation
#
# gensim has no pre-built wheel for Python 3.13/3.14 and requires the
# Microsoft C++ Build Tools to compile from source. This module trains
# a skip-gram Word2Vec model directly on the corpus using PyTorch, then
# represents each document as the average of its word vectors —
# identical in concept to the gensim-based module, no extra install.
#
# Pipeline:
#   Abstracts / Query
#          |
#   Preprocessing (lowercase -> tokenize -> remove punctuation
#                  & stop words -> lemmatize)
#          |
#   Skip-gram Word2Vec (trained on the corpus)
#          |
#   Average of word vectors  ->  Embedding Vectors (NumPy arrays)
# ================================================================

import re
import string
from collections import defaultdict

import numpy as np
import torch

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


def ensure_nltk_resources():
    for path, pkg in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


class _SkipGram(torch.nn.Module):
    """Minimal skip-gram model with negative sampling."""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center = torch.nn.Embedding(vocab_size, embed_dim)
        self.context = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.xavier_uniform_(self.center.weight)
        torch.nn.init.xavier_uniform_(self.context.weight)

    def forward(self, center_ids, context_ids):
        c = self.center(center_ids)       # (B, D)
        ctx = self.context(context_ids)   # (B, D)
        return (c * ctx).sum(dim=1)       # dot product


class Word2VecSearcher:
    """
    Static Semantic Search using Word2Vec (pure PyTorch skip-gram).

    Trains a Word2Vec model from scratch on the provided corpus and
    represents each document as the average of its word vectors.
    No external NLP libraries beyond PyTorch and NLTK are required.

    Methods
    -------
    preprocess(text)
        Lowercase -> tokenize -> strip punctuation/stop words -> lemmatize.
    get_corpus_embeddings()
        Returns a 2D NumPy array of shape (n_papers, vector_size).
    get_query_embedding(query_string)
        Returns a 1D NumPy array of shape (vector_size,).
    """

    def __init__(self, abstracts, vector_size=100, window=5,
                 min_count=2, epochs=5, neg_samples=5, lr=0.025,
                 batch_size=512, random_seed=42, verbose=True):
        ensure_nltk_resources()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)
        self.vector_size = vector_size
        self._corpus_embeddings = None
        self.abstracts = abstracts
        self._log = print if verbose else (lambda *a, **k: None)

        torch.manual_seed(random_seed)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # -- 1. Preprocess --------------------------------------------------
        self._log(f"[Word2Vec] Preprocessing {len(abstracts)} abstracts...")
        self.tokenized_corpus = [self.preprocess(a) for a in abstracts]

        # -- 2. Build vocabulary ---------------------------------------------
        freq = defaultdict(int)
        for tokens in self.tokenized_corpus:
            for t in tokens:
                freq[t] += 1
        vocab = [w for w, c in freq.items() if c >= min_count]
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = vocab
        vocab_size = len(vocab)
        self._log(f"[Word2Vec] Vocabulary size: {vocab_size}")

        # -- 3. Build training pairs (skip-gram) ------------------------------
        pairs = []
        for tokens in self.tokenized_corpus:
            ids = [self.word2id[t] for t in tokens if t in self.word2id]
            for i, center in enumerate(ids):
                for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                    if j != i:
                        pairs.append((center, ids[j]))
        if not pairs:
            raise ValueError("No training pairs generated — corpus may be too small.")
        pairs = torch.tensor(pairs, dtype=torch.long)  # (N, 2)
        self._log(f"[Word2Vec] Training pairs: {len(pairs):,}")

        # -- 4. Train ----------------------------------------------------------
        model = _SkipGram(vocab_size, vector_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(pairs[:, 0], pairs[:, 1])
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self._log(f"[Word2Vec] Training {epochs} epoch(s) on {device_str.upper()}...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for center_ids, ctx_ids in loader:
                center_ids = center_ids.to(self.device)
                ctx_ids = ctx_ids.to(self.device)

                pos_score = model(center_ids, ctx_ids)
                pos_label = torch.ones_like(pos_score)

                neg_ids = torch.randint(0, vocab_size, ctx_ids.shape, device=self.device)
                neg_score = model(center_ids, neg_ids)
                neg_label = torch.zeros_like(neg_score)

                loss = criterion(torch.cat([pos_score, neg_score]),
                                  torch.cat([pos_label, neg_label]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self._log(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

        self.word_vectors = model.center.weight.detach().cpu().numpy()
        self._log("[Word2Vec] Training complete!")

    @staticmethod
    def _wordnet_pos(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("R"):
            return wordnet.ADV
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
        return [self.lemmatizer.lemmatize(w, self._wordnet_pos(t)) for w, t in tagged]

    def _avg_vector(self, tokens):
        vecs = [self.word_vectors[self.word2id[t]] for t in tokens if t in self.word2id]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def get_corpus_embeddings(self):
        if self._corpus_embeddings is None:
            self._log("[Word2Vec] Computing document embeddings...")
            self._corpus_embeddings = np.vstack(
                [self._avg_vector(toks) for toks in self.tokenized_corpus]
            )
            self._log(f"[Word2Vec] Corpus embeddings shape: {self._corpus_embeddings.shape}")
        return self._corpus_embeddings

    def get_query_embedding(self, query_string):
        tokens = self.preprocess(query_string)
        return self._avg_vector(tokens)
