# === STEP 2: Retriever ===

import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Updated ChromaDB client (no deprecated Settings)
client = chromadb.PersistentClient(path="chroma_db")

# Load the existing collection
collection = client.get_collection("esdm_products")

# Define retrieval function
def retrieve_similar_products(user_query, top_k=5):
    query_embedding = model.encode([user_query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["metadatas"][0]  # returns list of dicts
