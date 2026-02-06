from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

from modules.recommender import SemanticPerfumeRecommender
import modules.process_input_utils as ProcessInput
from modules.ollama_client import OllamaHandler


# -------------------------------
# CONFIG
# -------------------------------

# MODEL
_model_name = "all-MiniLM-L6-v2"

# RECOMENDER
_embeddings_npy = "embeddings.npy"
_metadata_csv = "metadata_v1.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_N = 5

# OLLAMA
_ollama_base_url = "http://localhost:11434"
_ollama_model = "llama3.2:3b"



# -------------------------------
# INIT RECOMMENDER
# -------------------------------

# MODEL
MODEL = SentenceTransformer(_model_name)
print(f"Embedding model '{_model_name}' loaded.")

# OLLAMA
OLLAMA = OllamaHandler()
print("Ollama loaded.")

# RECOMENDER
RECOMMENDER = SemanticPerfumeRecommender(
    embeddings_path=_embeddings_npy,
    metadata_path=_metadata_csv,
    model=MODEL
)
print("Recommender loaded.")


# -------------------------------
# FLASK APP
# -------------------------------
app = Flask(__name__)
CORS(app)  # <-- enable CORS for all routes
CORS(app, origins=["http://localhost:5173"])

# -------------------------------
# RECOMMEND ENDPOINT
# -------------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400

    results = RECOMMENDER.recommend(
        query=data["query"],
        top_n=data.get("top_n", TOP_N)
    )

    return jsonify({"recommendations": results})



# @app.route("/is-related", methods=["POST"])
# def isRelated():
#     data = request.get_json()
#     if not data or "query" not in data:
#         return jsonify({"error": "Missing 'query' in request"}), 400

#     result = ProcessInput.is_related_to_perfume(data["query"])
#     return jsonify({"is-related": result})

# -------------------------------
# CHAT ENDPOINT
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "querry" not in data:
        return jsonify({"error": "Missing 'querry' in request"}), 400

    user_input = data["querry"]
    response_text = f"You said: {user_input}"
    perfume_results = RECOMMENDER.recommend(query=user_input, top_n=TOP_N)

    return jsonify({
        "text": response_text,
        "results": perfume_results
    })

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)