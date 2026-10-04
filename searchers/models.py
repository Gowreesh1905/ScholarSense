"""Shared model getters (team/CONTRACT.md §3).

Every getter is lru_cached, so each model loads once per process no matter
how many searchers use it. Imports are lazy so importing this module is cheap.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

SPECTER_MODEL = "allenai/specter"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
GLOVE_MODEL = "sentence-transformers/average_word_embeddings_glove.6B.300d"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=None)
def get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=None)
def _sentence_transformer(model_id: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, device=get_device())


def get_specter() -> SentenceTransformer:
    return _sentence_transformer(SPECTER_MODEL)


def get_bge() -> SentenceTransformer:
    return _sentence_transformer(BGE_MODEL)


def get_glove() -> SentenceTransformer:
    return _sentence_transformer(GLOVE_MODEL)


@lru_cache(maxsize=None)
def get_cross_encoder() -> CrossEncoder:
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER_MODEL, device=get_device())
