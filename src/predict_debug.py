# drop into src/predict_debug.py (temporary)
import os, joblib, re
from pprint import pprint

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

def simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

vect = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
clf  = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))

print("Model classes (order):", clf.classes_)   # <-- VERY important

tests = [
    "This movie was amazing.",
    "I loved the acting and the plot was gripping!",
    "Terrible movie. I hated it.",
    "It was okay, not great."
]

for t in tests:
    c = simple_clean(t)
    X = vect.transform([c])
    probs = clf.predict_proba(X)[0]   # array like [prob_class0, prob_class1]
    pred = clf.predict(X)[0]
    print("\nTEXT:", t)
    print("CLEAN:", c)
    print("PRED index:", pred, "-> label according to classes_:", clf.classes_[pred])
    print("PROBs:", probs)
    # check if 'amazing' exists in vocabulary
    v = "amazing"
    print(f"'amazing' in vectorizer.vocabulary_? -> {v in vect.vocabulary_}")
    if v in vect.vocabulary_:
        print("vocab index:", vect.vocabulary_[v])
