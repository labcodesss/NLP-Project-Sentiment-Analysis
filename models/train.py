# models/train.py
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
from data_pipeline.preprocess import preprocess_text
from utils.config import DATA_DIR, ARTIFACTS_DIR
from utils.logger import get_logger

logger = get_logger("models.train")

DATA_PATH = os.path.join(DATA_DIR, "movie_reviews.csv")
VECT_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
CLF_PATH = os.path.join(ARTIFACTS_DIR, "sentiment_model.joblib")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label_bin"] = df["label"].map({"pos": 1, "neg": 0})
    return df

def train(save=True):
    df = load_data()
    logger.info("Preprocessing text (lemmatization)...")
    df["clean"] = df["text"].apply(preprocess_text)

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
    logger.info(f"Accuracy: {acc:.4f}; F1: {f1:.4f}")
    logger.info("\n" + classification_report(y_test, preds, target_names=["neg", "pos"]))

    if save:
        joblib.dump(vect, VECT_PATH)
        joblib.dump(clf, CLF_PATH)
        logger.info(f"Saved artifacts to {ARTIFACTS_DIR}")

    return vect, clf

if __name__ == "__main__":
    train()
