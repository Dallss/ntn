import time
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# Custom modules
from modules.recommender import SemanticPerfumeRecommender
from modules.ollama_client import OllamaHandler

# -------------------------------
# CONFIGURATION
# -------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"
_OLLAMA_MODEL = "perfume-extractor:latest"  # include :latest for Ollama
TOP_N = 5

# -------------------------------
# GLOBAL APP STATE
# -------------------------------
APP_STATE = {
    "perfume": {
        "details": {
            "gender": None,
            "price_range": None,
            "accords": [],
            "descriptive_words": []
        },
        "llm-intent": None
    }
}

# -------------------------------
# FLASK APP
# -------------------------------
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# -------------------------------
# GLOBALS (initialized later)
# -------------------------------
MODEL = None
OLLAMA = None
RECOMMENDER = None

# -------------------------------
# UTILS
# -------------------------------
def wait_for_ollama(timeout=30):
    """Ensure Ollama is awake before Flask starts accepting traffic."""
    print(f"Waiting for Ollama model '{_OLLAMA_MODEL}'...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            res = OLLAMA.ask_model("ping")
            if "error" not in res:
                print("Ollama is ready!")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("Warning: Ollama not detected. Server starting anyway...")

# -------------------------------
# ENDPOINTS
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400

    user_query = data["query"]

    # Build query dictionary for keeping state (optional)
    ollama_query = {
        "details": APP_STATE["perfume"]["details"],
        "user-query": user_query
    }

    # 1. Extract perfume info using Ollama
    ollama_data = OLLAMA.ask_model(user_query)

    print(ollama_data)
    # Update APP_STATE with the latest details if available
    if "details" in ollama_data:
        APP_STATE["perfume"]["details"] = ollama_data["details"]


    details = APP_STATE['perfume']['details']

    # Convert to string for embedding
    details_str = f"Gender: {details['gender']}, Price range: {details['price_range']}, " \
                f"Accords: {', '.join(details['accords'])}, " \
                f"Descriptive words: {', '.join(details['descriptive_words'])}"

    # 2. Get recommendations based on raw query
    results = RECOMMENDER.recommend(
        query=details_str,
        top_n=data.get("top_n", TOP_N)
    )

    # Filter by gender if set
    gender_filter = APP_STATE["perfume"]["details"].get("gender")

    gender_map = {
        "male": ["male", "men"],
        "female": ["female", "women"]
    }

    if gender_filter:
        gender_filter = gender_filter.lower().strip()

    if gender_filter in gender_map:
        allowed_genders = gender_map[gender_filter]

        results = [
            row for row in results
            if str(row.get("gender", "")).lower() in allowed_genders
        ]


    # Use hard filters for gender
    return jsonify({
        "text": ollama_data.get("llm-response") or ollama_data.get("text") or "failed to get ollama response",
        "extracted_data": ollama_data.get("details") or ollama_data.get("data") or {},
        "recommendations": results
    })

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # Initialize heavy objects once
    MODEL = SentenceTransformer(_MODEL_NAME)
    print(f"Embedding model '{_MODEL_NAME}' loaded.")

    OLLAMA = OllamaHandler(model_file=_OLLAMA_MODEL)
    print("Ollama handler initialized.")

    RECOMMENDER = SemanticPerfumeRecommender(
        embeddings_path="embeddings.npy",
        metadata_path="metadata_v1.csv",
        model=MODEL
    )
    print("Recommender system initialized.")

    # Wait for Ollama to be ready
    wait_for_ollama()

    # Run Flask
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
