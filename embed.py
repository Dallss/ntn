# embed_perfumes_scaled.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "fra_cleaned.csv"         # input CSV
EMBEDDINGS_NPY = "embeddings.npy"    # embeddings file
METADATA_CSV = "metadata.csv"        # metadata file
ENCODING = "windows-1252"
SEPARATOR = ";"                       # semicolon CSV
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

# -------------------------------
# LOAD CSV
# -------------------------------
print(f"Loading CSV from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, encoding=ENCODING, sep=SEPARATOR, on_bad_lines="skip")

# Fill missing text columns with empty string
text_columns = ['Top', 'Middle', 'Base',
                'mainaccord1','mainaccord2','mainaccord3','mainaccord4','mainaccord5']
for col in text_columns:
    if col not in df.columns:
        print(f"Warning: Column {col} missing, filling with empty strings")
        df[col] = ""
df[text_columns] = df[text_columns].fillna('')

# -------------------------------
# CREATE SEMANTIC TEXT
# -------------------------------
def create_text(row):
    notes = f"Top notes: {row['Top']}. Middle notes: {row['Middle']}. Base notes: {row['Base']}."
    accords = f"Main accords: {', '.join([row[f'mainaccord{i}'] for i in range(1,6) if row.get(f'mainaccord{i}', '')])}."
    return f"{row['Perfume']} by {row['Brand']}, {row['Gender']} fragrance. {notes} {accords}"

df['semantic_text'] = df.apply(create_text, axis=1)

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------
print(f"Loading embedding model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

# -------------------------------
# GENERATE EMBEDDINGS (BATCHED)
# -------------------------------
print("Generating embeddings ...")
embeddings = model.encode(df['semantic_text'].to_list(), batch_size=BATCH_SIZE, show_progress_bar=True)
embeddings = np.vstack(embeddings)  # convert to contiguous NumPy array

# -------------------------------
# SAVE EMBEDDINGS AND METADATA SEPARATELY
# -------------------------------
print(f"Saving embeddings to {EMBEDDINGS_NPY} ...")
np.save(EMBEDDINGS_NPY, embeddings)

print(f"Saving metadata to {METADATA_CSV} ...")
metadata_cols = ['Perfume','Brand','Gender','Top','Middle','Base',
                 'mainaccord1','mainaccord2','mainaccord3','mainaccord4','mainaccord5',
                 'Rating Value','Rating Count']
df[metadata_cols].to_csv(METADATA_CSV, sep=";", index=False)

print(f"Done! Saved {len(df)} perfumes and embeddings of shape {embeddings.shape}.")
