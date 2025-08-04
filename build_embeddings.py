# === build_embeddings.py ===
import pandas as pd
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import uuid
import numpy as np
import json  # Added for JSON serialization

# === STEP 1: Load Excel File ===
file_path = "ESDM_Dataset.xlsx"
df = pd.read_excel(file_path)

# === STEP 2: Data Preparation ===
# Ensure Product_id is always valid
df["Product_id"] = df["Product_id"].fillna("").astype(str)
df["Product_id"] = df["Product_id"].apply(lambda x: x.strip() or str(uuid.uuid4()))

# List of all columns to embed (all columns except Product_id)
embedding_columns = [col for col in df.columns if col != "Product_id"]

# === STEP 3: Initialize ChromaDB Client ===
client = PersistentClient(path="chroma_db")
collection_name = "esdm_products"

# 🚨 Overwrite existing collection if it exists
existing_collections = [c.name for c in client.list_collections()]
if collection_name in existing_collections:
    print(f"⚠️ Collection '{collection_name}' already exists. Deleting it...")
    client.delete_collection(name=collection_name)
    print(f"✅ Deleted existing collection '{collection_name}'.")

collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}  # Using cosine similarity
)

# === STEP 4: Embedding Process (Modified) ===
model = SentenceTransformer("all-MiniLM-L6-v2")
batch_size = 32  # Optimal batch size for most GPUs
chunks = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]

for batch in tqdm(chunks, desc="Processing batches"):
    # Prepare IDs - ensure they're unique and non-empty
    ids = batch["Product_id"].tolist()
    
    # Initialize lists for embeddings and metadata
    combined_embeddings = []
    metadatas = []
    documents = []  # This will now store structured JSON documents
    
    for _, row in batch.iterrows():
        # Create individual column embeddings
        column_embeddings = []
        doc_parts = {}  # Using dictionary instead of list for structured storage
        
        for col in embedding_columns:
            value = row[col]
            
            # Skip null/empty values
            if pd.isna(value) or value == "":
                continue
                
            # Convert value to string and embed
            text_value = str(value)
            embedding = model.encode(text_value, show_progress_bar=False)
            column_embeddings.append(embedding)
            doc_parts[col] = text_value  # Store as key-value pair
            
        # Combine embeddings (average) and create document text
        if column_embeddings:
            avg_embedding = np.mean(column_embeddings, axis=0).tolist()
            document_text = json.dumps(doc_parts)  # Convert dict to JSON string
        else:
            avg_embedding = model.encode("").tolist()
            document_text = "{}"  # Empty JSON object
            
        combined_embeddings.append(avg_embedding)
        documents.append(document_text)
        
        # Prepare metadata (all original fields)
        metadata = {}
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                metadata[col] = None
            elif isinstance(value, (int, float)):
                metadata[col] = float(value)
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                metadata[col] = str(value)
            else:
                metadata[col] = str(value)
        metadatas.append(metadata)
    
    # Add batch to ChromaDB
    collection.add(
        ids=ids,
        embeddings=combined_embeddings,
        metadatas=metadatas,
        documents=documents  # Now contains JSON strings
    )

print("✅ Embedding build complete. Collection stored in 'chroma_db'.")