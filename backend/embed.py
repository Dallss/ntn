# embed_perfumes_scaled_v1.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "dataset/fra_cleaned.csv"         # input CSV
EMBEDDINGS_NPY = "embeddings.npy"    # embeddings file
METADATA_CSV = "metadata_v1.csv"     # new metadata file with URL
ENCODING = "windows-1252"
SEPARATOR = ";"                       # semicolon CSV
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

# -------------------------------
# LOAD CSV / Clean Data
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
    # 1. Clean and collect accords into a natural list
    accords = [row.get(f'mainaccord{i}', '').strip() for i in range(1, 6)]
    accords_list = ", ".join([a for a in accords if a])
    
    # 2. Handle Gender naturally
    gender_map = {
        'women': 'for women',
        'men': 'for men',
        'unisex': 'for both men and women'
    }
    gender = gender_map.get(str(row['Gender']).lower(), f"a {row['Gender']} fragrance")

    # 3. Construct a narrative "Vibe" sentence
    # This helps the vectorizer connect 'fresh' or 'warm' to the specific perfume
    narrative = (
        f"{row['Perfume']} by {row['Brand']} is a {gender}. "
        f"It is primarily defined by {accords_list} accords, giving it a distinct character. "
        f"The experience begins with top notes of {row['Top']}, "
        f"evolves into a heart of {row['Middle']}, "
        f"and leaves a lasting impression with base notes of {row['Base']}."
    )
    
    return narrative

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
# SAVE EMBEDDINGS
# -------------------------------
print(f"Saving embeddings to {EMBEDDINGS_NPY} ...")
np.save(EMBEDDINGS_NPY, embeddings)

# -------------------------------
# ADD URL COLUMN
# -------------------------------
if 'url' not in df.columns:
    print("Adding placeholder 'url' column ...")
    df['url'] = ""  # optionally, fill with actual URLs if available

# -------------------------------
# SAVE METADATA WITH URL
# -------------------------------
metadata_cols = ['url', 'Perfume','Brand','Gender','Top','Middle','Base',
                 'mainaccord1','mainaccord2','mainaccord3','mainaccord4','mainaccord5',
                 'Rating Value','Rating Count']

df[metadata_cols].to_csv(METADATA_CSV, sep=";", index=False)

print(f"Done! Saved {len(df)} perfumes and embeddings of shape {embeddings.shape} with URL column.")