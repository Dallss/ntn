# recommender.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class SemanticPerfumeRecommender:
    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        print("Loading embeddings and metadata...")
        self.embeddings = np.load(embeddings_path)
        self.metadata = pd.read_csv(metadata_path, sep=";")

        print(
            f"Loaded {len(self.metadata)} perfumes "
            f"with embeddings shape {self.embeddings.shape}"
        )

        self.model = SentenceTransformer(model_name)
        print(f"Embedding model '{model_name}' loaded.")

    def recommend(self, query: str, top_n: int = 5) -> list[dict]:
        # Embed user query
        query_embedding = self.model.encode(query)

        # Cosine similarity
        similarities = util.cos_sim(query_embedding, self.embeddings)[0].cpu().numpy()

        df = self.metadata.copy()
        df["similarity"] = similarities

        top = df.sort_values("similarity", ascending=False).head(top_n)

        results = []
        for _, row in top.iterrows():
            main_accords = [
                str(row[f"mainaccord{i}"])
                for i in range(1, 6)
                if f"mainaccord{i}" in row
                and pd.notna(row[f"mainaccord{i}"])
                and row[f"mainaccord{i}"] != ""
            ]

            results.append({
                "perfume": row["Perfume"],
                "brand": row["Brand"],
                "gender": row.get("Gender"),
                "rating_value": row.get("Rating Value"),
                "rating_count": row.get("Rating Count"),
                "top_notes": row.get("Top"),
                "middle_notes": row.get("Middle"),
                "base_notes": row.get("Base"),
                "main_accords": main_accords,
                "similarity_score": float(row["similarity"])
            })

        return results
