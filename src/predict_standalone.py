# src/predict_standalone.py
"""
Standalone CLI that loads models/*.joblib and predicts a given text.
Run:
    python src/predict_standalone.py --text "I loved the acting!"
"""

import os
import re
import joblib
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../movie_sentiment/src
ROOT = os.path.dirname(THIS_DIR)
MODELS_DIR = os.path.join(ROOT, "models")

VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CLF_PATH  = os.path.join(MODELS_DIR, "sentiment_model.joblib")

POSITIVE_OVERRIDES = {"amazing","excellent","awesome","fantastic","loved","love","best","brilliant","wonderful","perfect","great","gripping"}

def simple_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_artifacts():
    if not os.path.exists(VECT_PATH) or not os.path.exists(CLF_PATH):
        raise FileNotFoundError("Model files not found. Run training first.")
    vect = joblib.load(VECT_PATH)
    clf = joblib.load(CLF_PATH)
    return vect, clf

def predict_text(text: str, override_lexicon=True):
    vect, clf = load_artifacts()
    clean = simple_clean(text)
    X = vect.transform([clean])
    probs = clf.predict_proba(X)[0]

    classes = list(clf.classes_)
    pos_prob = None
    if "pos" in classes:
        pos_prob = float(probs[classes.index("pos")])
    elif 1 in classes:
        pos_prob = float(probs[classes.index(1)])
    else:
        pred_idx = int(clf.predict(X)[0])
        pos_prob = float(probs[pred_idx]) if pred_idx==1 else 1.0 - float(probs[pred_idx])

    label = "Positive" if pos_prob >= 0.5 else "Negative"
    if override_lexicon and label == "Negative":
        words = set(clean.split())
        if words & POSITIVE_OVERRIDES:
            label = "Positive"

    neg_prob = 1.0 - pos_prob
    conf = pos_prob if label == "Positive" else neg_prob
    return label, float(conf), float(pos_prob), float(neg_prob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text to classify")
    args = parser.parse_args()
    label, conf, pos_prob, neg_prob = predict_text(args.text)
    print(f"Prediction: {label} (confidence: {conf:.3f})")
    print(f"Positive: {pos_prob:.3f} | Negative: {neg_prob:.3f}")
