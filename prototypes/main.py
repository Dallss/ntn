# perfume_recommender_from_npy.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# CONFIG
# -------------------------------
EMBEDDINGS_NPY = "embeddings.npy"
METADATA_CSV = "metadata.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_N = 5

# -------------------------------
# LOAD EMBEDDINGS AND METADATA
# -------------------------------
print(f"Loading embeddings from {EMBEDDINGS_NPY} ...")
embeddings = np.load(EMBEDDINGS_NPY)  # shape = (num_perfumes, 384)

print(f"Loading metadata from {METADATA_CSV} ...")
metadata = pd.read_csv(METADATA_CSV, sep=";")

print(f"Loaded {len(metadata)} perfumes and embeddings of shape {embeddings.shape}")

# -------------------------------
# LOAD EMBEDDING MODEL (for query)
# -------------------------------
model = SentenceTransformer(MODEL_NAME)

# -------------------------------
# USER QUERY
# -------------------------------
user_query = "I like warm, spicy perfumes with vanilla and jasmine"
query_embedding = model.encode(user_query)

# -------------------------------
# CALCULATE SIMILARITY
# -------------------------------
similarities = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
metadata['similarity'] = similarities

# -------------------------------
# GET TOP RECOMMENDATIONS
# -------------------------------
recommendations = metadata.sort_values('similarity', ascending=False).head(TOP_N)

# -------------------------------
# PRINT RESULTS
# -------------------------------
for i, row in recommendations.iterrows():
    print(f"{i+1}. {row['Perfume']} by {row['Brand']} ({row['Gender']})")
    print(f"   Rating: {row['Rating Value']} ({row['Rating Count']} reviews)")
    print(f"   Top notes: {row['Top']}")
    print(f"   Middle notes: {row['Middle']}")
    print(f"   Base notes: {row['Base']}")
    
    main_accords = [str(row[f'mainaccord{i}']) for i in range(1,6)
                    if f'mainaccord{i}' in row and pd.notna(row[f'mainaccord{i}']) and row[f'mainaccord{i}'] != '']
    print(f"   Main accords: {', '.join(main_accords)}")
    print("   ---")
