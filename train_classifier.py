"""
Trains a lightweight TF-IDF + Logistic Regression intent classifier
on the Roman Urdu / Urdu script intent dataset (data/intents.json).

This is the fast, no-GPU-required baseline. See bert_classifier.py
for the upgrade path to a fine-tuned multilingual BERT model.

Usage:
    python train_classifier.py
"""

import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = "data/intents.json"
MODEL_PATH = "models/intent_classifier.joblib"


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent, phrases in data.items():
        for phrase in phrases:
            texts.append(phrase)
            labels.append(intent)
    return texts, labels


def train():
    texts, labels = load_dataset(DATA_PATH)
    print(f"Loaded {len(texts)} examples across {len(set(labels))} intents.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",   # char n-grams handle Roman Urdu spelling variation well
            ngram_range=(2, 4),
            lowercase=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    print("\n--- Evaluation on held-out test set ---")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Refit on the FULL dataset before saving. The split above is only for
    # honest evaluation — the deployed model should learn from every example
    # we have, not leave 20% of it unused.
    pipeline.fit(texts, labels)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nFinal model refit on full dataset ({len(texts)} examples) and saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
