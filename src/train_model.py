# src/train_model.py
import os
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "movie_reviews.csv")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # map labels to binary 1=pos, 0=neg
    df = df.dropna(subset=["text", "label"])
    df["label_bin"] = df["label"].map({"pos": 1, "neg": 0})
    return df

def simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train(save=True):
    df = load_data()
    df["clean"] = df["text"].apply(simple_clean)

    X = df["clean"].values
    y = df["label_bin"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorizer (you can tune max_features / ngram_range)
    vect = TfidfVectorizer(max_features=20_000, ngram_range=(1,2))
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf = vect.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
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
        joblib.dump(vect, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        joblib.dump(clf, os.path.join(MODELS_DIR, "sentiment_model.joblib"))
        print(f"Saved vectorizer and model into {MODELS_DIR}")

    return vect, clf

if __name__ == "__main__":
    train()
