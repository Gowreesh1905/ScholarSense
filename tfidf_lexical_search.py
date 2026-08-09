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


# ============================================================
# Cell 4 — Define the TFIDFSearcher Class
# ============================================================

class TFIDFSearcher:
    """
    Lexical Search using TF-IDF (Term Frequency - Inverse Document Frequency).

    Unlike Word2Vec or BERT, this class has no concept of meaning: it
    represents each document as a sparse vector over the corpus
    vocabulary, weighted by how frequent a word is in that document
    versus how rare it is across the whole corpus. Only exact,
    lemma-matching words contribute to similarity, which is precisely
    the limitation this arm of the case study is meant to illustrate
    against the two semantic (Word2Vec / BERT) arms.

    Because TF-IDF only ever "sees" the tokens it was fit on, both the
    preprocessing AND the fitted vocabulary must be shared between the
    corpus and any later query - this class guarantees that by fitting
    ONE `TfidfVectorizer` on the corpus and reusing it (via `.transform`,
    never `.fit` again) for every subsequent query.

    Parameters
    ----------
    abstracts : list of str
        The research-paper abstracts to embed.
    max_features : int or None, default None
        If set, keeps only the top `max_features` terms by corpus
        frequency (caps the vocabulary/embedding dimension).
    ngram_range : tuple of (int, int), default (1, 1)
        Range of n-gram sizes to extract (e.g. (1, 2) keeps unigrams
        AND bigrams like "neural_network").
    sublinear_tf : bool, default True
        Replaces raw term frequency `tf` with `1 + log(tf)`, which
        stops very high-frequency words from dominating the vector -
        a standard TF-IDF refinement (Manning et al., IR textbook).

    Attributes
    ----------
    preprocessed_corpus : list of str
        The raw abstracts, kept for reference (tokens are produced
        on demand by `preprocess`, which the vectorizer calls itself).
    vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        The fitted vectorizer. Its vocabulary is frozen after the
        first call to `get_corpus_embeddings()` / `get_query_embedding()`.
    vector_size : int
        Dimensionality of every embedding this class produces
        (i.e. the fitted vocabulary size), available once fitted.

    Methods
    -------
    preprocess(text)
        Lowercase → tokenize → strip punctuation/stop words → lemmatize.
        Returns a list of tokens (the vectorizer's custom tokenizer).
    get_corpus_embeddings()
        Returns a 2D SciPy sparse matrix of shape (n_papers, vector_size).
    get_query_embedding(query_string)
        Returns a 1x(vector_size) SciPy sparse row vector for the query.
    """

    def __init__(self, abstracts, max_features=None, ngram_range=(1, 1), sublinear_tf=True):
        """Store the corpus and configure (but do not yet fit) the vectorizer."""
        # ---- Store the raw corpus -------------------------------------
        self.abstracts = abstracts

        # ---- Set up the preprocessing tools ---------------------------
        # Downloaded once, then reused for every abstract and every query.
        ensure_nltk_resources()
        self.stop_words = set(stopwords.words("english"))   # e.g. "the", "of", "and"
        self.lemmatizer = WordNetLemmatizer()               # "networks" -> "network"
        self.punctuation = set(string.punctuation)          # ! " # $ % & ' ( ) * + , - . / ...

        # ---- Configure the TF-IDF vectorizer ---------------------------
        # `tokenizer=self.preprocess` means scikit-learn calls OUR
        # preprocessing function (lowercase/tokenize/stopword-strip/
        # lemmatize) on every document AND on every later query, so
        # both are guaranteed to be treated identically.
        #
        # `preprocessor=lambda text: text` and `lowercase=False` disable
        # scikit-learn's OWN default cleaning step, since `preprocess`
        # already does all of that itself. `token_pattern=None` silences
        # scikit-learn's warning that `token_pattern` is unused when a
        # custom `tokenizer` is supplied.
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocess,
            preprocessor=lambda text: text,
            lowercase=False,
            token_pattern=None,
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
        )

        # Cache for get_corpus_embeddings(): fit_transform() is only
        # ever run once, on first use, by either public method below.
        self._corpus_embeddings = None
        self.vector_size = None

        print(f"TFIDFSearcher created for {len(self.abstracts)} abstracts "
              f"(vectorizer not yet fit - happens lazily on first use).")

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

        This is passed directly to `TfidfVectorizer` as its `tokenizer`,
        so it is called on every abstract during fitting AND on every
        query during `.transform()` - the exact same function, every time.

        Steps
        -----
        0. Basic cleaning : strip HTML tags, URLs and stray whitespace.
        1. Lowercasing    : "Neural" and "neural" become the same token.
        2. Tokenization   : split the sentence into a list of words.
        3. Filtering      : drop punctuation, digits, single/double
                            characters and standard English stop words.
        4. Lemmatization  : reduce inflected forms to their base form
                            ("networks" -> "network", "learned" -> "learn")
                            so all variants share one TF-IDF column.

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
    # Internal: fit the vectorizer exactly once
    # --------------------------------------------------------------

    def _ensure_fitted(self):
        """
        Fit the TfidfVectorizer on the corpus, if it has not been fit yet.

        This is called automatically by BOTH `get_corpus_embeddings()`
        and `get_query_embedding()`, so either method can be called
        first and the vectorizer is still only ever fit ONCE (fitting
        twice would silently change the vocabulary/IDF weights and
        break query/corpus comparability).
        """
        if self._corpus_embeddings is not None:
            return  # already fit - nothing to do

        print(f"Fitting TF-IDF vectorizer on {len(self.abstracts)} abstracts "
              f"(preprocessing happens inside the vectorizer)...")

        # fit_transform() runs `preprocess` on every abstract, builds the
        # vocabulary/IDF weights from the result, and returns the corpus
        # matrix in one pass - a second, separate transform pass would be
        # wasted work.
        self._corpus_embeddings = self.vectorizer.fit_transform(self.abstracts)
        self.vector_size = len(self.vectorizer.vocabulary_)

        print(f"TFIDFSearcher fitted with {len(self.abstracts)} abstracts.")
        print(f"Vocabulary size:     {self.vector_size}")
        print(f"Embedding dimension: {self.vector_size}  (one dimension per vocabulary term)")

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def get_corpus_embeddings(self):
        """
        Generate TF-IDF document vectors for ALL abstracts.

        Pipeline:
            Raw abstracts → preprocess (tokenizer) → TfidfVectorizer.fit_transform → sparse matrix

        The vectorizer is fit only once; repeated calls return the
        cached matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            A 2D SPARSE matrix of shape (n_papers, vector_size).
            Example: 727 abstracts over a vocabulary of ~4000 terms ->
            (727, 4000), with the vast majority of entries being 0.0.

        Note
        ----
        Unlike Word2Vec/BERT (which return dense NumPy arrays), TF-IDF
        vectors are naturally sparse - most terms in the vocabulary do
        not appear in any single abstract. `scipy.sparse` matrices work
        directly with `sklearn.metrics.pairwise.cosine_similarity`, so
        no conversion is required downstream. Call `.toarray()` on the
        result if a dense NumPy array is ever needed instead.
        """
        self._ensure_fitted()
        return self._corpus_embeddings

    def get_query_embedding(self, query_string):
        """
        Generate a TF-IDF vector for a single user query.

        The query is tokenized with the EXACT SAME `preprocess` function
        and projected onto the EXACT SAME (already-fitted) vocabulary as
        the corpus - `.transform()` is used, never `.fit()`, so a query
        can never introduce new vocabulary/IDF weights that the corpus
        vectors don't also have.

        Pipeline:
            Query → preprocess (tokenizer) → TfidfVectorizer.transform → sparse row vector

        Parameters
        ----------
        query_string : str
            The user's natural-language search query.

        Returns
        -------
        scipy.sparse.csr_matrix
            A SPARSE row vector of shape (1, vector_size). A query whose
            words are all out-of-vocabulary returns an all-zero row.
        """
        # Fitting must have happened already (or happens now, on first
        # use) - a query can only be projected onto a vocabulary that
        # already exists.
        self._ensure_fitted()

        # `.transform` (not `.fit_transform`) reuses the corpus vocabulary/
        # IDF weights untouched, keeping corpus and query comparable.
        return self.vectorizer.transform([query_string])


print("TFIDFSearcher class defined successfully!")


# ============================================================
# Cell 5 — Load the Dataset
# ============================================================
# The dataset "abstract_sentences.csv" is provided by the
# Scholar Inbox authors. Each row contains an abstract with
# sentence-level annotations. The same abstract appears
# multiple times, so we use drop_duplicates() to get one
# entry per unique paper.
#
# drop_duplicates() preserves the original order, ensuring
# consistency across TF-IDF, Word2Vec, and BERT implementations.
# ============================================================

# Load the dataset provided by the Scholar Inbox authors
# Make sure "abstract_sentences.csv" is in the same folder as your script
df = pd.read_csv("abstract_sentences.csv")

# Extract the 'abstract' column and drop duplicates
# drop_duplicates() guarantees that the order of papers stays EXACTLY the same
# for TF-IDF, Word2Vec, and BERT.
unique_abstracts = df['abstract'].drop_duplicates().dropna().tolist()

print(f"Successfully loaded {len(unique_abstracts)} unique research papers!")

# ---------------------------------------------------------
# 'unique_abstracts' is now a standard Python list of strings.
# You can now pass this list into your TF-IDF, Word2Vec, or BERT code!
# ---------------------------------------------------------

# Preview the first abstract (truncated for display)
print(f"\nExample abstract (first 300 characters):")
print(f"{unique_abstracts[0][:300]}...")


# ============================================================
# Cell 6 — Initialize the TFIDFSearcher
# ============================================================
# Pass the unique abstracts to create the searcher instance.
# Fitting the vectorizer happens lazily on first use (see Cell 7).
# ============================================================

searcher = TFIDFSearcher(unique_abstracts)


# ============================================================
# Cell 7 — Generate Corpus Embeddings
# ============================================================
# This fits the TF-IDF vocabulary on ALL abstracts and returns
# their sparse document vectors.
#
# Expected output: a 2D SciPy sparse matrix
#   - Rows    = number of unique abstracts (727)
#   - Columns = TF-IDF vocabulary size (corpus-dependent)
# ============================================================

corpus_embeddings = searcher.get_corpus_embeddings()

print(f"\nCorpus embeddings shape: {corpus_embeddings.shape}")


# ============================================================
# Cell 8 — Generate Query Embedding
# ============================================================
# The query is processed through the SAME preprocessing and the
# SAME fitted vocabulary as the corpus, so both embeddings live
# in the same TF-IDF vector space (i.e. share the same columns).
# ============================================================

# Example query — a natural-language search for research papers
query = "AI methods for identifying malicious network activity"

query_embedding = searcher.get_query_embedding(query)

print(f"Query: '{query}'")
print(f"Query embedding shape: {query_embedding.shape}")


# ============================================================
# Cell 9 — Inspect Embeddings
# ============================================================
# Verify the generated embeddings are correct.
# No similarity calculation or ranking is performed here.
# ============================================================

print("=" * 60)
print("CORPUS EMBEDDINGS")
print("=" * 60)
print(f"Type:            {type(corpus_embeddings)}")
print(f"Data type:       {corpus_embeddings.dtype}")
print(f"Shape:           {corpus_embeddings.shape}")
print(f"  -> {corpus_embeddings.shape[0]} documents, each represented as a {corpus_embeddings.shape[1]}-dimensional sparse vector")
print(f"Non-zero entries: {corpus_embeddings.nnz} / {corpus_embeddings.shape[0] * corpus_embeddings.shape[1]} "
      f"({100 * corpus_embeddings.nnz / (corpus_embeddings.shape[0] * corpus_embeddings.shape[1]):.3f}% dense)")
print(f"\nFirst document embedding (first 10 non-zero terms):")
first_row = corpus_embeddings[0].tocoo()
terms = searcher.vectorizer.get_feature_names_out()
top_terms = sorted(zip(first_row.col, first_row.data), key=lambda x: -x[1])[:10]
for col, weight in top_terms:
    print(f"  {terms[col]:<20s} {weight:.4f}")

print()
print("=" * 60)
print("QUERY EMBEDDING")
print("=" * 60)
print(f"Type:            {type(query_embedding)}")
print(f"Data type:       {query_embedding.dtype}")
print(f"Shape:           {query_embedding.shape}")
print(f"  -> 1 query, represented as a {query_embedding.shape[1]}-dimensional sparse vector")
print(f"Non-zero entries: {query_embedding.nnz}")
print(f"\nQuery embedding (non-zero terms):")
query_row = query_embedding.tocoo()
for col, weight in sorted(zip(query_row.col, query_row.data), key=lambda x: -x[1]):
    print(f"  {terms[col]:<20s} {weight:.4f}")

print()
print("=" * 60)
print("VERIFICATION")
print("=" * 60)
print(f"Corpus and query embeddings have the same dimension: "
      f"{corpus_embeddings.shape[1] == query_embedding.shape[1]} "
      f"({corpus_embeddings.shape[1]} == {query_embedding.shape[1]})")
print(f"\n-> Both embeddings exist in the same {corpus_embeddings.shape[1]}-dimensional vector space.")
print(f"-> They can be compared using cosine similarity or other metrics")
print(f"   (handled by a separate component).")
