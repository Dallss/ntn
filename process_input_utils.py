from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------------
# CONFIG / CONSTANTS
# -------------------------------

DOMAIN_THRESHOLD = 0.35

# -------------------------------
# MODULE-LEVEL VARIABLES
# -------------------------------
_model = None       # will be injected from main
_domain_emb = None  # domain embedding

DOMAIN_ANCHOR = """
Choosing, comparing, and recommending perfumes and fragrances,
including scent notes, accords, brands, prices, occasions, constraints and preferences.
"""

# -------------------------------
# SETTER FUNCTION
# -------------------------------
def set_model(model: SentenceTransformer):
    """Inject the already-loaded model from main."""
    global _model, _domain_emb
    _model = model
    # Compute domain embedding once using the injected model
    _domain_emb = _model.encode(DOMAIN_ANCHOR.strip().lower(), normalize_embeddings=True)

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def compute_domain_score(text: str) -> float:
    """Compute similarity of text to perfume domain."""
    if _model is None or _domain_emb is None:
        raise RuntimeError("Model not set. Call set_model(model) first.")
    text_emb = _model.encode(text.strip().lower(), normalize_embeddings=True)
    return float(np.dot(text_emb, _domain_emb))  # cosine similarity

def is_related_to_perfume(text: str) -> bool:
    """
    Returns True if the text is related to perfumes,
    False if it is likely noise/off-topic.
    """
    return compute_domain_score(text) >= DOMAIN_THRESHOLD