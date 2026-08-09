# ================================================================
# Lexical Search using TF-IDF
# Sparse Term-Frequency Document Vectors for Research-Paper Abstracts
#
# Developer Role: TF-IDF Developer (Lexical Search)
#
# This module implements lexical (exact-word) representation for
# research-paper search using TF-IDF (Term Frequency - Inverse
# Document Frequency).
#
# Unlike Word2Vec or BERT, TF-IDF has NO notion of meaning: it only
# knows whether a word literally occurs in a document, weighted by
# how rare that word is across the whole corpus. Two abstracts that
# describe the same idea with different vocabulary ("car" vs.
# "automobile") will NOT be considered similar. This is the exact
# trade-off this arm of the case study is meant to measure, against
# Word2Vec's static semantic embeddings and BERT/SPECTER's
# contextual embeddings.
#
# Because TF-IDF relies on exact lexical matches, preprocessing
# matters much more here than for BERT: every inflected/derived
# form of a word must be reduced to the same token, or the vectorizer
# will treat "network", "networks" and "networking" as three
# unrelated, independently-weighted dimensions.
#
# Pipeline:
#   Abstracts / Query
#          ↓
#   Preprocessing (lowercase → tokenize → remove punctuation
#                  & stop words → lemmatize)
#          ↓
#   TfidfVectorizer (fit on the corpus, reused to transform queries)
#          ↓
#   Sparse Embedding Vectors (scipy sparse matrices)
#
# NOTE: This module deliberately contains NO cosine-similarity or
#       ranking logic. It only produces embeddings; comparison and
#       ranking are handled by a separate component of the case study.
# ================================================================


# ============================================================
# Cell 1 — Install Dependencies (run once in terminal)
# ============================================================
# pip install scikit-learn nltk numpy pandas
#
# scikit-learn : TfidfVectorizer (fit on the corpus, transform queries)
# nltk         : tokenization, English stop-word list, WordNet lemmatizer
# numpy        : used to preview/inspect the sparse embeddings
# pandas       : loads the dataset from CSV (used by the caller/notebook)
# ============================================================


# ============================================================
# Cell 2 — Imports
# ============================================================

import re                                        # Built-in: basic text cleaning (HTML tags, URLs, whitespace)
import string                                    # Built-in: the punctuation character set

import numpy as np                               # Used only to preview/inspect the sparse embeddings
import pandas as pd                              # Loads the dataset from CSV

import nltk                                      # Classical NLP toolkit (tokenizer, stop words, lemmatizer)
from nltk.corpus import stopwords, wordnet       # English stop-word list + WordNet POS constants
from nltk.stem import WordNetLemmatizer          # Rule/lexicon-based lemmatizer
from nltk.tokenize import word_tokenize          # Splits a sentence into a list of word tokens

from sklearn.feature_extraction.text import TfidfVectorizer  # Builds the TF-IDF vocabulary/matrix

print("All libraries imported successfully!")


# ============================================================
# Cell 3 — NLTK Resource Bootstrap
# ============================================================
# NLTK ships code, but the data files (tokenizer models, stop-word
# list, WordNet) must be downloaded once. This helper downloads any
# missing resource quietly so the class works on a fresh machine.
#
# (Kept identical to the Word2Vec developer's bootstrap so both
# classical-NLP arms of the case study share one preprocessing
# convention.)
# ============================================================

# (resource path used by nltk.data.find, package name used by nltk.download)
_NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),                                  # sentence/word tokenizer tables
    ("tokenizers/punkt", "punkt"),                                          # tokenizer (older NLTK versions)
    ("corpora/stopwords", "stopwords"),                                     # English stop-word list
    ("corpora/wordnet", "wordnet"),                                         # lexical database used for lemmatization
    ("corpora/omw-1.4", "omw-1.4"),                                         # WordNet multilingual data (WordNet dependency)
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),  # POS tagger for POS-aware lemmatization
]


def ensure_nltk_resources():
    """
    Download the NLTK data files required for preprocessing, if missing.

    Safe to call repeatedly: a resource that is already present on disk
    is detected by `nltk.data.find` and is not downloaded again.
    """
    for resource_path, package_name in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)
