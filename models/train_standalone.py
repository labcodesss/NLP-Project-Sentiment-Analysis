# models/train_standalone.py
"""
Standalone training script. Reads data/data_movie_reviews.csv (created by build_dataset.py).
Saves artifacts into models/ (tfidf_vectorizer.joblib, sentiment_model.joblib).
Run:
    python models/train_standalone.py
"""

import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../movie_sentiment/models
ROOT = os.path.dirname(THIS_DIR)                         # .../movie_sentiment
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "movie_reviews.csv")
VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CLF_PATH  = os.path.join(MODELS_DIR, "sentiment_model.joblib")

def simple_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}. Run data_pipeline/build_dataset.py first.")
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label_bin"] = df["label"].map({"pos": 1, "neg": 0})
    return df

def train(save=True):
    df = load_data()
    df["clean"] = df["text"].apply(simple_clean)

    X = df["clean"].values
    y = df["label_bin"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vect = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2)
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf = vect.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_tfidf, y_train)

    preds = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print("Evaluation on test set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (binary): {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["neg", "pos"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    if save:
        joblib.dump(vect, VECT_PATH)
        joblib.dump(clf, CLF_PATH)
        print(f"Saved vectorizer and model into {MODELS_DIR}")

    return vect, clf

if __name__ == "__main__":
    train()
