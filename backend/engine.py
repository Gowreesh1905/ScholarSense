"""
ScholarSense search engine wrapper for the web API.

Loads the corpus once and builds all three searchers (TF-IDF, Word2Vec,
BERT/SPECTER) exactly the way the CLI (main_search_engine.py) does, reusing
the teammates' modules untouched. The two expensive artifacts — the trained
Word2Vec vectors and the BERT corpus embeddings — are cached to disk so
restarting the API server during development doesn't re-train / re-embed
the whole corpus every time.
"""

import hashlib
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
sys.path.insert(0, str(ROOT))  # so we can import the teammates' root-level modules

from tfidf_lexical_search import TFIDFSearcher            # noqa: E402
from bert_specter_contextual_search import BERTSearcher   # noqa: E402
from word2vec_pytorch_search import Word2VecSearcher       # noqa: E402


def load_data(filepath: Path):
    """Loads abstract_sentences.csv -> (abstracts, arxiv_ids), index-aligned."""
    df = pd.read_csv(filepath)
    df_unique = df.drop_duplicates(subset="abstract").dropna(subset=["abstract"])
    abstracts = df_unique["abstract"].tolist()
    arxiv_ids = df_unique["arxiv_id"].astype(str).tolist()
    return abstracts, arxiv_ids


def _corpus_hash(abstracts: list[str]) -> str:
    h = hashlib.sha1()
    for a in abstracts:
        h.update(a.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


@dataclass
class SearchHit:
    rank: int
    score: float
    arxiv_id: str
    url: str
    snippet: str
    abstract: str


@dataclass
class MethodInfo:
    key: str
    label: str
    description: str


METHODS: list[MethodInfo] = [
    MethodInfo("tfidf", "TF-IDF", "Lexical / keyword matching — only exact word matches score."),
    MethodInfo("word2vec", "Word2Vec", "Static semantic embeddings — one fixed vector per word."),
    MethodInfo("bert", "BERT / SPECTER", "Contextual embeddings from a transformer fine-tuned on scientific papers."),
]


class SearchEngine:
    """Builds/holds all three searchers and answers ranked top-k queries."""

    def __init__(self, csv_path: Path = ROOT / "abstract_sentences.csv", verbose: bool = True):
        self._log = print if verbose else (lambda *a, **k: None)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._log("[engine] Loading dataset...")
        self.abstracts, self.arxiv_ids = load_data(csv_path)
        self._log(f"[engine] Loaded {len(self.abstracts)} unique abstracts.")

        self._hash = _corpus_hash(self.abstracts)
        CACHE_DIR.mkdir(exist_ok=True)

        self._log("[engine] Fitting TF-IDF...")
        self.tfidf = TFIDFSearcher(self.abstracts)
        self.tfidf_corpus = self.tfidf.get_corpus_embeddings()

        self.word2vec = self._load_or_train_word2vec()
        self.word2vec_corpus = self.word2vec.get_corpus_embeddings()

        self.specter_model = SentenceTransformer("allenai/specter", device=self.device)
        self.bert = BERTSearcher(self.abstracts, self.specter_model)
        self.bert_corpus = self._load_or_embed_bert()

        self._log("[engine] Ready.")

    # ------------------------------------------------------------------
    # Disk-cached heavy artifacts
    # ------------------------------------------------------------------

    def _word2vec_cache_path(self) -> Path:
        return CACHE_DIR / f"word2vec_{self._hash}.pkl"

    def _load_or_train_word2vec(self) -> Word2VecSearcher:
        path = self._word2vec_cache_path()
        if path.exists():
            self._log(f"[engine] Loading cached Word2Vec model from {path.name}...")
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:  # corrupt/incompatible cache -> retrain
                self._log(f"[engine] Cache load failed ({exc}); retraining.")

        self._log("[engine] Training Word2Vec (no cache found)...")
        model = Word2VecSearcher(self.abstracts, vector_size=100, epochs=5, verbose=self._log is print)
        model.get_corpus_embeddings()  # populate before caching
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return model

    def _bert_cache_path(self) -> Path:
        return CACHE_DIR / f"bert_{self._hash}.npy"

    def _load_or_embed_bert(self) -> np.ndarray:
        path = self._bert_cache_path()
        if path.exists():
            self._log(f"[engine] Loading cached BERT/SPECTER embeddings from {path.name}...")
            try:
                return np.load(path)
            except Exception as exc:
                self._log(f"[engine] Cache load failed ({exc}); re-embedding.")

        self._log("[engine] Embedding corpus with BERT/SPECTER (no cache found, this can take a while on CPU)...")
        embeddings = self.bert.get_corpus_embeddings()
        np.save(path, embeddings)
        return embeddings

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _rank(self, query_embedding, corpus_embeddings, k: int) -> list[SearchHit]:
        if hasattr(query_embedding, "reshape") and len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten()
        top_k = similarities.argsort()[-k:][::-1]
        hits = []
        for rank, idx in enumerate(top_k, 1):
            abstract = self.abstracts[idx]
            snippet = abstract[:300].replace("\n", " ").strip()
            if len(abstract) > 300:
                snippet += "..."
            hits.append(SearchHit(
                rank=rank,
                score=float(similarities[idx]),
                arxiv_id=self.arxiv_ids[idx],
                url=f"https://arxiv.org/abs/{self.arxiv_ids[idx]}",
                snippet=snippet,
                abstract=abstract,
            ))
        return hits

    def search(self, query: str, k: int = 5) -> dict:
        query = query.strip()
        start = time.perf_counter()

        tfidf_hits = self._rank(self.tfidf.get_query_embedding(query), self.tfidf_corpus, k)
        w2v_hits = self._rank(self.word2vec.get_query_embedding(query), self.word2vec_corpus, k)
        bert_hits = self._rank(self.bert.get_query_embedding(query), self.bert_corpus, k)

        took_ms = (time.perf_counter() - start) * 1000

        return {
            "query": query,
            "took_ms": round(took_ms, 1),
            "methods": [
                {"key": "tfidf", "label": METHODS[0].label, "description": METHODS[0].description, "results": tfidf_hits},
                {"key": "word2vec", "label": METHODS[1].label, "description": METHODS[1].description, "results": w2v_hits},
                {"key": "bert", "label": METHODS[2].label, "description": METHODS[2].description, "results": bert_hits},
            ],
        }

    def status(self) -> dict:
        return {
            "status": "ready",
            "corpus_size": len(self.abstracts),
            "device": self.device,
            "methods": [{"key": m.key, "label": m.label, "description": m.description} for m in METHODS],
        }
