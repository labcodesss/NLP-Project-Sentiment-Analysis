# src/predict.py
import os
import joblib
import argparse
import re

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

POSITIVE_OVERRIDES = {"amazing","excellent","awesome","fantastic","loved","love","best","brilliant","wonderful","perfect","great","gripping"}

def simple_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_artifacts():
    vect = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    clf = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))
    return vect, clf

def predict_text(text: str, override_lexicon=True):
    vect, clf = load_artifacts()
    clean = simple_clean(text)
    X = vect.transform([clean])
    probs = clf.predict_proba(X)[0]

    # get pos probability robustly
    classes = list(clf.classes_)
    pos_prob = None
    if "pos" in classes:
        pos_prob = float(probs[classes.index("pos")])
    elif 1 in classes:
        pos_prob = float(probs[classes.index(1)])
    else:
        # fallback: use predicted class probability
        pred = int(clf.predict(X)[0])
        pos_prob = float(probs[pred]) if pred==1 else 1.0 - float(probs[pred])

    # Decide label by pos_prob threshold
    label = "Positive" if pos_prob >= 0.5 else "Negative"

    # small lexicon override: if extremely obvious positive word present, prefer Positive
    if override_lexicon and label == "Negative":
        words = set(clean.split())
        if words & POSITIVE_OVERRIDES:
            label = "Positive"

    # confidence: use pos_prob for Positive otherwise neg_prob
    conf = pos_prob if label == "Positive" else 1.0 - pos_prob

    # also return both probs if caller wants them
    neg_prob = 1.0 - pos_prob
    return label, float(conf), float(pos_prob), float(neg_prob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Review text to classify")
    args = parser.parse_args()
    label, conf, pos_prob, neg_prob = predict_text(args.text)
    print(f"Prediction: {label} (confidence: {conf:.3f})")
    print(f"Positive: {pos_prob:.3f} | Negative: {neg_prob:.3f}")
