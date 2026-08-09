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


