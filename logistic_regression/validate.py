# validate_perfume_classifier.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------------------
# CONFIG
# -------------------------------
VALIDATION_FILE = "validation_data_large.csv"  # CSV with columns: message,label
MODEL_FILE = "logreg_perfume_model.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------------------
# LOAD VALIDATION DATA
# -------------------------------
df_val = pd.read_csv(VALIDATION_FILE)
val_messages = df_val['message'].tolist()
val_labels = df_val['label'].tolist()

# -------------------------------
# LOAD MODEL
# -------------------------------
clf = joblib.load(MODEL_FILE)

# -------------------------------
# ENCODE VALIDATION MESSAGES
# -------------------------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
X_val = embed_model.encode(val_messages, normalize_embeddings=True)

# -------------------------------
# PREDICT
# -------------------------------
pred_labels = clf.predict(X_val)
pred_probs = clf.predict_proba(X_val)[:, 1]

# -------------------------------
# METRICS
# -------------------------------
accuracy = accuracy_score(val_labels, pred_labels)
print(f"Validation Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(val_labels, pred_labels))

# -------------------------------
# OPTIONAL: PRINT MISCLASSIFIED MESSAGES
# -------------------------------
print("\nMisclassified Messages:")
for msg, true_label, pred_label, prob in zip(val_messages, val_labels, pred_labels, pred_probs):
    if true_label != pred_label:
        print(f"- Message: {msg}")
        print(f"  True Label: {true_label}, Predicted: {pred_label}, Probability On-Topic: {prob:.2f}\n")