# ================================================================
# Contextual Semantic Search using SPECTER
# Transformer-Based Scientific Document Embeddings
#
# Developer Role: BERT Developer (Contextual Semantic Search)
#
# This script implements contextual semantic representation for
# research-paper search using SPECTER (allenai/specter), a
# transformer-based scientific document embedding model.
#
# SPECTER is specifically designed for generating embeddings of
# scientific documents (titles, abstracts) and is therefore more
# appropriate for this project than a generic BERT model.
#
# Pipeline:
#   Abstracts / Query
#          ↓
#   Basic Cleaning (HTML tags, whitespace)
#          ↓
#   SPECTER (allenai/specter)
#          ↓
#   Embedding Vectors (NumPy arrays)
# ================================================================


# ============================================================
# Cell 1 — Install Dependencies (run once in terminal)
# ============================================================
# pip install sentence-transformers torch numpy
#
# sentence-transformers : loads pre-trained transformer models
#                         and generates embeddings
# torch                 : PyTorch deep-learning framework,
#                         required for GPU support
# numpy                 : stores embeddings as numerical arrays
# ============================================================


# ============================================================
# Cell 2 — Imports
# ============================================================

import re                                       # Built-in: basic text cleaning (HTML tags, whitespace)
import numpy as np                              # Numerical arrays for storing embeddings
import torch                                    # PyTorch: GPU support and deep-learning backend
from sentence_transformers import SentenceTransformer  # Loads & runs the SPECTER model
import pandas as pd                             # Loads the dataset from CSV

print("All libraries imported successfully!")


# ============================================================
# Cell 3 — Check GPU Availability
# ============================================================
# Transformer models are computationally intensive.
# A GPU significantly speeds up embedding generation.
# If no GPU is available, the model still works on CPU.
# ============================================================

print("PyTorch version:", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:            ", torch.cuda.get_device_name(0))
else:
    print("Running on CPU  (GPU not detected — this is fine, it will just be slower)")


# ============================================================
# Cell 4 — Load the SPECTER Model
# ============================================================
# SPECTER = Scientific Paper Embeddings using
#           Citation-informed TransformERs
#
# - Developed by Allen Institute for AI (allenai)
# - Specifically trained on scientific documents
# - Uses citation relationships to learn document similarity
# - More appropriate for research-paper search than generic BERT
# ============================================================

# Select the best available device: GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the SPECTER model using sentence-transformers
# This downloads the model weights on first run (~400 MB)
model = SentenceTransformer(
    "allenai/specter",
    device=device
)

# Verify the model loaded correctly and check the embedding dimension
embedding_dimension = model.get_sentence_embedding_dimension()

print(f"SPECTER model loaded successfully!")
print(f"Device:              {device}")
print(f"Embedding dimension: {embedding_dimension}")


# ============================================================
# Cell 5 — Define the BERTSearcher Class
# ============================================================

class BERTSearcher:
    """
    Contextual Semantic Search using SPECTER (allenai/specter).

    This class generates transformer-based contextual embeddings
    for research-paper abstracts and user queries using SPECTER,
    a model specifically designed for scientific documents.

    The class performs ONLY basic text cleaning and does NOT apply
    traditional NLP preprocessing (no lowercasing, no stop-word
    removal, no lemmatization, no stemming, no manual tokenization)
    because transformer models rely on the original sentence
    structure and context.

    Parameters
    ----------
    abstracts : list of str
        A list of research-paper abstracts to embed.
    specter_model : SentenceTransformer
        A pre-loaded SPECTER model instance.

    Methods
    -------
    get_corpus_embeddings()
        Returns a 2D NumPy array of shape (n_papers, embedding_dim).
    get_query_embedding(query_string)
        Returns a 1D NumPy array of shape (embedding_dim,).
    """

    def __init__(self, abstracts, specter_model):
        """
        Initialize the BERTSearcher.

        Parameters
        ----------
        abstracts : list of str
            Research-paper abstracts to generate embeddings for.
        specter_model : SentenceTransformer
            The loaded SPECTER model (allenai/specter).
        """
        # Store the raw abstracts — cleaning happens at embedding time
        self.abstracts = abstracts

        # Store the SPECTER model reference
        # Both corpus and query embeddings use the SAME model
        # so they exist in the SAME vector space
        self.model = specter_model

        print(f"BERTSearcher initialized with {len(self.abstracts)} abstracts.")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def _clean_text(self, text):
        """
        Apply ONLY basic text cleaning.

        What this does:
        - Removes HTML tags  (e.g., <br>, <p>, </div>)
        - Collapses multiple whitespace characters into a single space
        - Strips leading and trailing whitespace

        What this does NOT do (intentionally):
        - Does NOT lowercase the text
        - Does NOT remove stop words
        - Does NOT lemmatize or stem
        - Does NOT remove punctuation
        - Does NOT manually tokenize

        The transformer's built-in tokenizer handles all tokenization.

        Parameters
        ----------
        text : str
            The raw text to clean.

        Returns
        -------
        str
            The cleaned text with linguistic content preserved.
        """
        # Remove HTML tags (e.g., <br>, <p class="...">, </div>)
        text = re.sub(r"<[^>]+>", " ", text)

        # Collapse multiple whitespace characters into a single space
        text = re.sub(r"\s+", " ", text)

        # Strip leading and trailing whitespace
        text = text.strip()

        return text

    def get_corpus_embeddings(self):
        """
        Generate embeddings for ALL research-paper abstracts.

        Pipeline:
            Raw abstracts → Basic cleaning → SPECTER → NumPy array

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (n_papers, embedding_dim).
            For SPECTER, embedding_dim is typically 768.

        Example
        -------
        If there are 727 unique abstracts:
            Output shape: (727, 768)
        """
        # Step 1: Apply basic cleaning to each abstract
        # Preserves sentence structure, word order, and context
        cleaned_abstracts = [self._clean_text(abstract) for abstract in self.abstracts]

        print(f"Generating embeddings for {len(cleaned_abstracts)} abstracts...")
        print("(This may take a few minutes depending on corpus size and hardware)")

        # Step 2: Pass all cleaned abstracts through SPECTER
        # - The model's internal tokenizer handles sub-word tokenization
        # - show_progress_bar=True displays a progress bar during encoding
        # - batch_size=16 processes 16 abstracts at a time (memory-efficient)
        # - convert_to_numpy=True returns a NumPy array directly
        corpus_embeddings = self.model.encode(
            cleaned_abstracts,
            show_progress_bar=True,
            batch_size=16,
            convert_to_numpy=True
        )

        print(f"Corpus embeddings generated!")
        print(f"Shape: {corpus_embeddings.shape}")

        return corpus_embeddings

    def get_query_embedding(self, query_string):
        """
        Generate an embedding for a single user query.

        The query is processed through the EXACT SAME model
        and cleaning pipeline as the corpus abstracts, ensuring
        both exist in the same vector space.

        Pipeline:
            User query → Basic cleaning → SPECTER → NumPy array

        Parameters
        ----------
        query_string : str
            The user's search query.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (embedding_dim,).
            For SPECTER, this is typically (768,).
        """
        # Step 1: Apply the SAME basic cleaning as corpus abstracts
        cleaned_query = self._clean_text(query_string)

        # Step 2: Encode the single query through SPECTER
        # The model's internal tokenizer handles tokenization
        query_embedding = self.model.encode(
            cleaned_query,
            convert_to_numpy=True
        )

        return query_embedding

print("BERTSearcher class defined successfully!")


# ============================================================
# Cell 6 — Load the Dataset
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
# Cell 7 — Initialize the BERTSearcher
# ============================================================
# Pass the unique abstracts and the loaded SPECTER model
# to create the searcher instance.
# ============================================================

searcher = BERTSearcher(unique_abstracts, model)


# ============================================================
# Cell 8 — Generate Corpus Embeddings
# ============================================================
# This encodes ALL abstracts through SPECTER.
# Each abstract is converted into a dense vector.
#
# Expected output: a 2D NumPy array
#   - Rows    = number of unique abstracts (727)
#   - Columns = SPECTER embedding dimension (768)
# ============================================================

corpus_embeddings = searcher.get_corpus_embeddings()

print(f"\nCorpus embeddings shape: {corpus_embeddings.shape}")


# ============================================================
# Cell 9 — Generate Query Embedding
# ============================================================
# The query is processed through the SAME model and pipeline
# as the corpus abstracts, so both embeddings exist in the
# same 768-dimensional vector space.
# ============================================================

# Example query — a natural-language search for research papers
query = "AI methods for identifying malicious network activity"

query_embedding = searcher.get_query_embedding(query)

print(f"Query: '{query}'")
print(f"Query embedding shape: {query_embedding.shape}")


# ============================================================
# Cell 10 — Inspect Embeddings
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
print(f"  → {corpus_embeddings.shape[0]} documents, each represented as a {corpus_embeddings.shape[1]}-dimensional vector")
print(f"\nFirst document embedding (first 10 values):")
print(f"  {corpus_embeddings[0][:10]}")
print(f"\nEmbedding statistics:")
print(f"  Min:  {corpus_embeddings.min():.6f}")
print(f"  Max:  {corpus_embeddings.max():.6f}")
print(f"  Mean: {corpus_embeddings.mean():.6f}")
print(f"  Std:  {corpus_embeddings.std():.6f}")

print()
print("=" * 60)
print("QUERY EMBEDDING")
print("=" * 60)
print(f"Type:            {type(query_embedding)}")
print(f"Data type:       {query_embedding.dtype}")
print(f"Shape:           {query_embedding.shape}")
print(f"  → 1 query, represented as a {query_embedding.shape[0]}-dimensional vector")
print(f"\nQuery embedding (first 10 values):")
print(f"  {query_embedding[:10]}")

print()
print("=" * 60)
print("VERIFICATION")
print("=" * 60)
print(f"Corpus and query embeddings have the same dimension: "
      f"{corpus_embeddings.shape[1] == query_embedding.shape[0]} "
      f"({corpus_embeddings.shape[1]} == {query_embedding.shape[0]})")
print(f"\n→ Both embeddings exist in the same {corpus_embeddings.shape[1]}-dimensional vector space.")
print(f"→ They can be compared using cosine similarity or other metrics")
print(f"   (handled by a separate component).")
