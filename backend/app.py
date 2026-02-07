import time
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# Custom modules
from modules.recommender import SemanticPerfumeRecommender
from modules.ollama_client import OllamaHandler
from modules.space import extract_descriptions

# -------------------------------
# CONFIGURATION
# -------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"
_OLLAMA_MODEL = "perfume-extractor:beta"
TOP_N = 12

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
        "llm-intent": "inquire"
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
# LOGGING SETUP
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------
# UTILS
# -------------------------------
def normalize_gender(raw_gender: str) -> str:
    if not raw_gender:
        return "unisex"
    g = raw_gender.lower().strip()
    if g in ["male", "man", "men", "boy"]:
        return "male"
    elif g in ["female", "woman", "women", "girl"]:
        return "female"
    else:
        return "unisex"

def wait_for_ollama(timeout=30):
    logger.info(f"Waiting for Ollama model '{_OLLAMA_MODEL}'...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            res = OLLAMA.ask_model("ping")
            if "error" not in res:
                logger.info("Ollama is ready!")
                return True
        except Exception as e:
            logger.warning(f"Ollama ping failed: {e}")
        time.sleep(2)
    logger.warning("Warning: Ollama not detected. Server starting anyway...")

# -------------------------------
# ENDPOINTS
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "query" not in data:
        logger.error("Missing 'query' in request")
        return jsonify({"error": "Missing 'query' in request"}), 400

    user_query = data["query"]
    logger.info(f"Received query: {user_query}")

    # Extract descriptions
    spaceyextract = extract_descriptions(user_query)
    logger.info(f"Extracted descriptions: {spaceyextract}")

    # Update descriptive words
    new_descriptive = set()
    for category in ["descriptive_words", "mood", "texture", "setting", "common_noun_places"]:
        if spaceyextract.get(category):
            new_descriptive.update(spaceyextract[category])
    
    if new_descriptive:
        existing = set(APP_STATE["perfume"]["details"].get("descriptive_words", []))
        APP_STATE["perfume"]["details"]["descriptive_words"] = list(existing.union(new_descriptive))
        logger.info(f"Updated descriptive words: {APP_STATE['perfume']['details']['descriptive_words']}")

    # Prepare Ollama query
    ollama_query = {
        "details": APP_STATE["perfume"]["details"],
        "user-query": user_query,
        "llm-intent": APP_STATE["perfume"]["llm-intent"]
    }
    ollama_query_str = json.dumps(ollama_query, indent=2)

    # -------------------
    # Extract perfume info using Ollama
    # -------------------
    try:
        ollama_data = OLLAMA.ask_model(ollama_query_str)
        logger.info(f"Ollama response: {ollama_data}")

        # Update APP_STATE from Ollama's 'details'
        if "details" in ollama_data:
            for key in ["gender", "price_range", "accords", "descriptive_words"]:
                value = ollama_data["details"].get(key)
                if value:
                    if key == "gender":
                        value = normalize_gender(value)
                    APP_STATE["perfume"]["details"][key] = value
                    logger.info(f"Updated {key} from Ollama: {value}")

    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        ollama_data = {"llm-response": None}

    # Update conversation state
    details = APP_STATE['perfume']['details']
    if not details.get("gender"):
        APP_STATE["perfume"]["llm-intent"] = "ask for gender"
    elif not details.get("descriptive_words"):
        APP_STATE["perfume"]["llm-intent"] = "ask for descriptive words"
    else:
        APP_STATE["perfume"]["llm-intent"] = "end-conversation"

    logger.info(f"APP_STATE after processing: {APP_STATE}")

    # Prepare string for recommendation
    parts = []
    if details.get("accords"):
        parts.append(f"Accords: {', '.join(details['accords'])}")
    if details.get("descriptive_words"):
        parts.append(f"Descriptive words: {', '.join(details['descriptive_words'])}")
    details_str = "; ".join(parts)
    # logger.info(f"Details string for recommendation: {details_str}")

    # Get recommendations
    results = RECOMMENDER.recommend(
        query=details_str,
        top_n=data.get("top_n", TOP_N)
    )
    logger.info(f"Initial recommendations: {results}")

    # ----- ROBUST GENDER FILTER -----
    gender_filter = details.get("gender")
    gender_map = {
        "male": ["male", "men", "unisex"],
        "female": ["female", "women", "unisex"]
    }

    if gender_filter:
        gender_filter = gender_filter.lower().strip()
        allowed_genders = gender_map.get(gender_filter, [])
        logger.info(f"Filtering recommendations for gender: {gender_filter}, allowed: {allowed_genders}")

        filtered_results = []
        for row in results:
            row_gender = str(row.get("gender", "")).lower().strip()
            if row_gender in allowed_genders:
                filtered_results.append(row)
            else:
                logger.debug(f"Excluding row with gender: {row_gender}")

        results = filtered_results
        logger.info(f"Recommendations after gender filter: {results}")

    return jsonify({
        "text": ollama_data.get("llm-response") or ollama_data.get("text") or "failed to get ollama response",
        "extracted_data": APP_STATE['perfume']['details'],
        "recommendations": results
    })

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # Initialize heavy objects once
    MODEL = SentenceTransformer(_MODEL_NAME)
    logger.info(f"Embedding model '{_MODEL_NAME}' loaded.")

    OLLAMA = OllamaHandler(model_file=_OLLAMA_MODEL)
    logger.info("Ollama handler initialized.")

    RECOMMENDER = SemanticPerfumeRecommender(
        embeddings_path="embeddings.npy",
        metadata_path="metadata_v1.csv",
        model=MODEL
    )
    logger.info("Recommender system initialized.")

    # Wait for Ollama to be ready
    wait_for_ollama()

    # Run Flask
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
