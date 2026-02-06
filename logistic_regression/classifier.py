# perfume_topic_classifier.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------------------
# CONFIG
# -------------------------------
DATA_FILE = "off_topic_identifier_dataset.csv"      # CSV with columns: message,label
MODEL_NAME = "all-MiniLM-L6-v2"
SAVED_MODEL_FILE = "logreg_perfume_model.pkl"

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_FILE)
messages = df['message'].tolist()
labels = df['label'].tolist()

# -------------------------------
# EMBEDDINGS
# -------------------------------
print("Encoding messages...")
embed_model = SentenceTransformer(MODEL_NAME)
X = embed_model.encode(messages, normalize_embeddings=True)  # L2 normalize for better performance

# -------------------------------
# TRAIN/VALIDATION SPLIT
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# -------------------------------
# TRAIN LOGISTIC REGRESSION
# -------------------------------
print("Training logistic regression...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# -------------------------------
# EVALUATE
# -------------------------------
y_pred = clf.predict(X_val)
y_prob = clf.predict_proba(X_val)[:, 1]

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# -------------------------------
# SAVE MODEL
# -------------------------------
joblib.dump(clf, SAVED_MODEL_FILE)
print(f"Trained model saved to {SAVED_MODEL_FILE}")

# -------------------------------
# TEST ON NEW MESSAGES
# -------------------------------
test_messages = [
    "I need a fragrance for prom tonight",
    "Did you watch the game yesterday?",
    "Looking for a fresh scent for daily wear",
    "I love pizza and movies"
]

test_embeddings = embed_model.encode(test_messages, normalize_embeddings=True)
pred_probs = clf.predict_proba(test_embeddings)[:, 1]
pred_labels = clf.predict(test_embeddings)

for msg, label, prob in zip(test_messages, pred_labels, pred_probs):
    print(f"\nMessage: {msg}\nPredicted Label: {label}, Probability On-Topic: {prob:.2f}")