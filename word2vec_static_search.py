# ================================================================
# Static Semantic Search using Word2Vec
# Average Word Embeddings for Research-Paper Abstracts
#
# Developer Role: Word2Vec Developer (Static Semantic Search)
#
# This module implements static semantic representation for
# research-paper search using Word2Vec (gensim).
#
# Word2Vec learns ONE fixed ("static") vector per word, independent
# of the sentence the word appears in. Because the model works at
# the WORD level, a document-level strategy is required: here we use
# the Average Word Embedding approach — the document vector is the
# mean of the vectors of all its (preprocessed) words.
#
# Pipeline:
#   Abstracts / Query
#          ↓
#   Preprocessing (lowercase → tokenize → remove punctuation
#                  & stop words → lemmatize)
#          ↓
#   Word2Vec (trained on the corpus, or a pre-trained model)
#          ↓
#   Average of word vectors  →  Embedding Vectors (NumPy arrays)
#
# NOTE: This module deliberately contains NO cosine-similarity or
#       ranking logic. It only produces embeddings; comparison and
#       ranking are handled by a separate component of the case study.
# ================================================================


# ============================================================
# Cell 1 — Install Dependencies (run once in terminal)
# ============================================================
# pip install gensim nltk numpy pandas
#
# gensim : Word2Vec implementation (training + pre-trained models)
# nltk   : tokenization, English stop-word list, WordNet lemmatizer
# numpy  : stores the embeddings as numerical arrays
# pandas : loads the dataset from CSV (used by the caller/notebook)
#
# NOTE: gensim currently ships wheels for Python <= 3.12. On newer
#       interpreters create an environment first, e.g.:
#           uv venv --python 3.12 venv
# ============================================================


# ============================================================
# Cell 2 — Imports
# ============================================================

import re                                        # Built-in: basic text cleaning (HTML tags, URLs, whitespace)
import string                                    # Built-in: the punctuation character set

import numpy as np                               # Numerical arrays for storing embeddings

import nltk                                      # Classical NLP toolkit (tokenizer, stop words, lemmatizer)
from nltk.corpus import stopwords, wordnet       # English stop-word list + WordNet POS constants
from nltk.stem import WordNetLemmatizer          # Rule/lexicon-based lemmatizer
from nltk.tokenize import word_tokenize          # Splits a sentence into a list of word tokens

from gensim.models import Word2Vec               # Trains a Word2Vec model from scratch on our corpus
import gensim.downloader as api                  # Downloads pre-trained vectors (e.g. word2vec-google-news-300)


# ============================================================
# Cell 3 — NLTK Resource Bootstrap
# ============================================================
# NLTK ships code, but the data files (tokenizer models, stop-word
# list, WordNet) must be downloaded once. This helper downloads any
# missing resource quietly so the class works on a fresh machine.
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


# ============================================================
# Cell 4 — Define the Word2VecSearcher Class
# ============================================================

class Word2VecSearcher:
    """
    Static Semantic Search using Word2Vec + Average Word Embeddings.

    Unlike TF-IDF (which matches literal words) this class represents
    text by MEANING: words that appear in similar contexts receive
    similar vectors, so a query can match an abstract even when they
    share no vocabulary ("car" vs "automobile").

    Unlike BERT/SPECTER, the vectors are STATIC: each word has exactly
    one vector regardless of context, so "bank" (river) and "bank"
    (finance) collapse into a single representation. That trade-off is
    precisely what this arm of the case study is meant to measure.

    Document strategy
    -----------------
    Word2Vec produces WORD vectors, not document vectors. This class
    uses the Average Word Embedding approach: the document vector is
    the element-wise mean of the vectors of all its in-vocabulary
    preprocessed tokens.

    Parameters
    ----------
    abstracts : list of str
        The research-paper abstracts to embed.
    use_pretrained : bool, default False
        False -> train a Word2Vec model from scratch on `abstracts`
                 (fast, offline, and the vocabulary matches the
                 scientific domain of the corpus).
        True  -> load pre-trained vectors via gensim-data
                 (broader general-English vocabulary, but a large
                 one-off download: ~1.6 GB for Google News).
    pretrained_model_name : str, default "word2vec-google-news-300"
        gensim-data model name, used only when `use_pretrained=True`.
    vector_size : int, default 300
        Embedding dimensionality when training from scratch.
        (Ignored for pre-trained models, which fix their own size.)
    window : int, default 5
        Context window size used during training.
    min_count : int, default 2
        Words occurring fewer than `min_count` times are ignored.
    epochs : int, default 30
        Training passes over the corpus. Our corpus is small
        (a few hundred abstracts), so more epochs help.
    sg : int, default 1
        1 = skip-gram (better for small corpora and rare technical
        terms), 0 = CBOW (faster).
    workers : int, default 4
        Worker threads used for training.
    random_seed : int, default 42
        Seed for reproducible training runs.

    Attributes
    ----------
    tokenized_corpus : list of list of str
        The preprocessed token lists, one per abstract.
    word_vectors : gensim.models.KeyedVectors
        The lookup table mapping a word to its static vector.
    vector_size : int
        Dimensionality of every embedding produced by this class.

    Methods
    -------
    preprocess(text)
        Lowercase → tokenize → strip punctuation/stop words → lemmatize.
    get_corpus_embeddings()
        Returns a 2D NumPy array of shape (n_papers, vector_size).
    get_query_embedding(query_string)
        Returns a 1D NumPy array of shape (vector_size,).
    """

    def __init__(
        self,
        abstracts,
        use_pretrained=False,
        pretrained_model_name="word2vec-google-news-300",
        vector_size=300,
        window=5,
        min_count=2,
        epochs=30,
        sg=1,
        workers=4,
        random_seed=42,
    ):
        """Preprocess the corpus and build (train or load) the Word2Vec model."""
        # ---- Store the raw corpus -------------------------------------
        self.abstracts = abstracts

        # ---- Set up the preprocessing tools ---------------------------
        # Downloaded once, then reused for every abstract and every query.
        ensure_nltk_resources()
        self.stop_words = set(stopwords.words("english"))   # e.g. "the", "of", "and"
        self.lemmatizer = WordNetLemmatizer()               # "networks" -> "network"
        self.punctuation = set(string.punctuation)          # ! " # $ % & ' ( ) * + , - . / ...

        # Cache for get_corpus_embeddings(): the mean-vector computation
        # is done once and reused on later calls.
        self._corpus_embeddings = None

        # ---- Step 1: preprocess every abstract ------------------------
        # Word2Vec consumes a list of token lists, so preprocessing must
        # happen before training. The SAME function is later applied to
        # the query, guaranteeing corpus and query are treated identically.
        print(f"Preprocessing {len(self.abstracts)} abstracts...")
        self.tokenized_corpus = [self.preprocess(abstract) for abstract in self.abstracts]

        # ---- Step 2: obtain the word vectors --------------------------
        if use_pretrained:
            # Option A — pre-trained vectors (downloaded on first use).
            # These are trained on billions of general-English words, so
            # they cover everyday vocabulary far better than our corpus,
            # but they miss corpus-specific jargon and are a big download.
            print(f"Loading pre-trained vectors '{pretrained_model_name}' (large one-off download)...")
            self.model = None
            self.word_vectors = api.load(pretrained_model_name)
        else:
            # Option B — train from scratch on the abstracts themselves.
            # Skip-gram + a low min_count works well on a small, dense,
            # domain-specific corpus like research-paper abstracts.
            print(f"Training Word2Vec on the corpus (vector_size={vector_size}, epochs={epochs})...")
            self.model = Word2Vec(
                sentences=self.tokenized_corpus,  # the preprocessed token lists
                vector_size=vector_size,          # dimensionality of each word vector
                window=window,                    # context words considered left/right
                min_count=min_count,              # ignore very rare words
                sg=sg,                            # 1 = skip-gram, 0 = CBOW
                epochs=epochs,                    # training passes over the corpus
                workers=workers,                  # parallel worker threads
                seed=random_seed,                 # reproducibility
            )
            # KeyedVectors: the trained word -> vector lookup table.
            self.word_vectors = self.model.wv

        # Every embedding this class returns has this dimensionality,
        # whether it came from a trained or a pre-trained model.
        self.vector_size = self.word_vectors.vector_size

        print(f"Word2VecSearcher initialized with {len(self.abstracts)} abstracts.")
        print(f"Vocabulary size:     {len(self.word_vectors.index_to_key)}")
        print(f"Embedding dimension: {self.vector_size}")

    # --------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------

    @staticmethod
    def _wordnet_pos(treebank_tag):
        """
        Map a Penn-Treebank POS tag to the POS constant WordNet expects.

        The lemmatizer needs to know the part of speech to be accurate:
        without it, "learning" (verb) stays "learning" and "was" stays
        "wa". Defaults to NOUN, which is WordNet's own default.
        """
        if treebank_tag.startswith("J"):
            return wordnet.ADJ      # adjective
        if treebank_tag.startswith("V"):
            return wordnet.VERB     # verb
        if treebank_tag.startswith("R"):
            return wordnet.ADV      # adverb
        return wordnet.NOUN         # noun (default)

    def preprocess(self, text):
        """
        Turn a raw string into a clean list of lemmatized content words.

        Steps
        -----
        0. Basic cleaning : strip HTML tags, URLs and LaTeX-ish markers.
        1. Lowercasing    : "Neural" and "neural" become the same token.
        2. Tokenization   : split the sentence into a list of words.
        3. Filtering      : drop punctuation, digits, single characters
                            and standard English stop words.
        4. Lemmatization  : reduce inflected forms to their base form
                            ("networks" -> "network", "learned" -> "learn")
                            so all variants share one vector.

        Parameters
        ----------
        text : str
            Raw abstract or query text.

        Returns
        -------
        list of str
            The cleaned, lemmatized tokens.
        """
        # Guard against NaN / non-string input coming from a CSV column
        if not isinstance(text, str):
            return []

        # --- Step 0: basic cleaning ---------------------------------
        text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags: <br>, <p>, </div>
        text = re.sub(r"http\S+|www\.\S+", " ", text) # remove URLs
        text = re.sub(r"\s+", " ", text).strip()      # collapse repeated whitespace

        # --- Step 1: lowercase --------------------------------------
        text = text.lower()

        # --- Step 2: tokenize ---------------------------------------
        tokens = word_tokenize(text)

        # --- Step 3: remove punctuation, numbers and stop words -----
        cleaned_tokens = []
        for token in tokens:
            # Strip punctuation attached to a word ("state-of-the-art" -> "stateoftheart",
            # "model." -> "model"). Tokens that were pure punctuation become "".
            token = "".join(char for char in token if char not in self.punctuation)

            if not token.isalpha():        # drops "", "2021", "3d", stray symbols
                continue
            if len(token) < 3:             # drops noise like "et", "al", "e"
                continue
            if token in self.stop_words:   # drops "the", "we", "of", "which", ...
                continue

            cleaned_tokens.append(token)

        if not cleaned_tokens:
            return []

        # --- Step 4: POS-aware lemmatization ------------------------
        # POS-tag first so the lemmatizer knows whether "training" is a
        # noun or a verb, then reduce each token to its dictionary form.
        tagged_tokens = nltk.pos_tag(cleaned_tokens)
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token, self._wordnet_pos(tag))
            for token, tag in tagged_tokens
        ]

        return lemmatized_tokens

    # --------------------------------------------------------------
    # Document representation: Average Word Embedding
    # --------------------------------------------------------------

    def _average_word_vectors(self, tokens):
        """
        Average the Word2Vec vectors of `tokens` into a single vector.

        This is the Average Word Embedding strategy that turns word-level
        Word2Vec output into a document-level representation.

        Out-of-vocabulary tokens (words the model never saw, or words
        removed by `min_count`) have no vector and are simply skipped.
        A document with no in-vocabulary token at all falls back to a
        zero vector, which keeps the output array rectangular.

        Parameters
        ----------
        tokens : list of str
            Preprocessed tokens of one document or query.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (vector_size,), dtype float32.
        """
        # Collect a vector for every token the model actually knows
        vectors = [self.word_vectors[token] for token in tokens if token in self.word_vectors]

        # No known word -> return a zero vector rather than crashing
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)

        # The document vector is the element-wise mean of its word vectors
        return np.mean(vectors, axis=0).astype(np.float32)

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def get_corpus_embeddings(self):
        """
        Generate document embeddings for ALL abstracts.

        Pipeline:
            Preprocessed tokens → Word2Vec lookup → mean vector → NumPy array

        The result is cached, so repeated calls are free.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (n_papers, vector_size).
            Example: 727 abstracts with 300-dim vectors -> (727, 300).
        """
        # Return the cached matrix if it has already been computed
        if self._corpus_embeddings is not None:
            return self._corpus_embeddings

        print(f"Generating average-word-embedding vectors for {len(self.tokenized_corpus)} abstracts...")

        # One mean vector per abstract, stacked into a 2D matrix.
        # Row order matches `self.abstracts` exactly, so row i always
        # corresponds to abstract i across TF-IDF, Word2Vec and BERT.
        self._corpus_embeddings = np.vstack(
            [self._average_word_vectors(tokens) for tokens in self.tokenized_corpus]
        )

        print(f"Corpus embeddings generated!")
        print(f"Shape: {self._corpus_embeddings.shape}")

        return self._corpus_embeddings

    def get_query_embedding(self, query_string):
        """
        Generate an embedding for a single user query.

        The query goes through the EXACT SAME preprocessing and the SAME
        word-vector table as the abstracts, so the query vector and the
        document vectors live in the same vector space and are directly
        comparable.

        Pipeline:
            Query → preprocessing → Word2Vec lookup → mean vector → NumPy array

        Parameters
        ----------
        query_string : str
            The user's natural-language search query.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (vector_size,).
            A query whose words are all out-of-vocabulary returns a
            zero vector.
        """
        # Step 1: identical preprocessing to the corpus
        query_tokens = self.preprocess(query_string)

        # Step 2: average the vectors of the query's known words
        return self._average_word_vectors(query_tokens)
