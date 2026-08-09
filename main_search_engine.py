import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Import your teammates' modules
from tfidf_lexical_search import TFIDFSearcher
# from word2vec_static_search import Word2VecSearcher
from bert_specter_contextual_search import BERTSearcher

def load_data(filepath="abstract_sentences.csv"):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    # Extract the 'abstract' column and drop duplicates
    unique_abstracts = df['abstract'].drop_duplicates().dropna().tolist()
    print(f"Loaded {len(unique_abstracts)} unique research papers!\n")
    return unique_abstracts

def display_top_k(query_embedding, corpus_embeddings, abstracts, k=5, model_name=""):
    """
    Computes cosine similarity between the query and all documents,
    and prints the top K most similar abstracts.
    """
    # Reshape query embedding to (1, dim) if it is 1D
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Calculate cosine similarity using scikit-learn
    # This automatically handles both Sparse Matrices (TF-IDF) and Dense Arrays (Word2Vec/BERT)
    similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten()
    
    # Sort to get the highest similarity indices
    top_k_indices = similarities.argsort()[-k:][::-1]
    
    print(f"--- {model_name} Top {k} Results ---")
    for rank, idx in enumerate(top_k_indices, 1):
        score = similarities[idx]
        abstract_text = abstracts[idx]
        
        # Truncate abstract for clean display in the terminal
        short_abstract = abstract_text[:250].replace('\n', ' ') + "..."
        print(f"{rank}. [Similarity: {score:.4f}] {short_abstract}")
    print("\n")

def main():
    # 1. Load Data
    abstracts = load_data("abstract_sentences.csv")
    
    print("Initializing Models (This might take a minute)...\n")
    
    # 2. Initialize TF-IDF
    print("-> Setting up TF-IDF...")
    tfidf_searcher = TFIDFSearcher(abstracts)
    tfidf_corpus = tfidf_searcher.get_corpus_embeddings()
    
    # 3. Initialize Word2Vec
    # print("-> Setting up Word2Vec (Training on Corpus)...")
    # w2v_searcher = Word2VecSearcher(abstracts, use_pretrained=False, epochs=50) 
    # w2v_corpus = w2v_searcher.get_corpus_embeddings()
    
    # 4. Initialize BERT (Specter)
    print("-> Setting up BERT (Specter)...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    specter_model = SentenceTransformer("allenai/specter", device=device)
    bert_searcher = BERTSearcher(abstracts, specter_model)
    bert_corpus = bert_searcher.get_corpus_embeddings()
    
    print("\n" + "="*80)
    print("ALL MODELS READY. STARTING COMPARISON ENGINE!")
    print("="*80 + "\n")
    
    # 5. Define our tricky test query designed to show the vocabulary mismatch
    # It avoids exact words like "Neural Machine Translation" or "NLP"
    test_query = "methods for automatic translation of spoken language"
    print(f"TEST SEARCH QUERY: '{test_query}'\n")
    
    # --- Execute TF-IDF ---
    tfidf_q_emb = tfidf_searcher.get_query_embedding(test_query)
    display_top_k(tfidf_q_emb, tfidf_corpus, abstracts, k=3, model_name="TF-IDF (Lexical)")
    
    # --- Execute Word2Vec ---
    # w2v_q_emb = w2v_searcher.get_query_embedding(test_query)
    # display_top_k(w2v_q_emb, w2v_corpus, abstracts, k=3, model_name="Word2Vec (Static Semantic)")
    
    # --- Execute BERT ---
    bert_q_emb = bert_searcher.get_query_embedding(test_query)
    display_top_k(bert_q_emb, bert_corpus, abstracts, k=3, model_name="BERT Specter (Contextual)")
    
    print("Experiment Complete! Observe how TF-IDF relies on exact words, while BERT captures the overarching meaning.")

if __name__ == "__main__":
    main()
