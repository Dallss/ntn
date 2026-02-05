from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- import this
from recommender import SemanticPerfumeRecommender

# -------------------------------
# CONFIG
# -------------------------------
EMBEDDINGS_NPY = "embeddings.npy"
METADATA_CSV = "metadata.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_N = 5

# -------------------------------
# INIT RECOMMENDER
# -------------------------------
recommender = SemanticPerfumeRecommender(
    embeddings_path=EMBEDDINGS_NPY,
    metadata_path=METADATA_CSV,
    model_name=MODEL_NAME
)

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

    results = recommender.recommend(
        query=data["query"],
        top_n=data.get("top_n", TOP_N)
    )

    return jsonify({"recommendations": results})

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
    perfume_results = recommender.recommend(query=user_input, top_n=TOP_N)

    return jsonify({
        "text": response_text,
        "results": perfume_results
    })

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)