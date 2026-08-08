import pandas as pd

# 1. Load the dataset provided by the Scholar Inbox authors
# Make sure "abstract_sentences.csv" is in the same folder as your script
df = pd.read_csv("abstract_sentences.csv")

# 2. Extract the 'abstract' column and drop duplicates
# drop_duplicates() guarantees that the order of papers stays EXACTLY the same
# for TF-IDF, Word2Vec, and BERT.
unique_abstracts = df['abstract'].drop_duplicates().dropna().tolist()

print(f"Successfully loaded {len(unique_abstracts)} unique research papers!")

# ---------------------------------------------------------
# 'unique_abstracts' is now a standard Python list of strings.
# You can now pass this list into your TF-IDF, Word2Vec, or BERT code!
# ---------------------------------------------------------