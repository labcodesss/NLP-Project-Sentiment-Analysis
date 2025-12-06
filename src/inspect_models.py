# src/inspect_models.py
import os, joblib, time, argparse, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CLF_PATH  = os.path.join(MODELS_DIR, "sentiment_model.joblib")

def simple_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def print_file_info(path):
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        return
    st = os.stat(path)
    print(f"Path: {path}")
    print(f" - size: {st.st_size} bytes")
    print(f" - mtime: {time.ctime(st.st_mtime)}")
    print()

def load_and_inspect(sample_texts):
    # Existence
    print("=== FILE METADATA ===")
    print_file_info(VECT_PATH)
    print_file_info(CLF_PATH)
    print("=====================\n")

    # Load
    try:
        vect = joblib.load(VECT_PATH)
        print("Loaded vectorizer from", VECT_PATH)
    except Exception as e:
        print("Failed to load vectorizer:", e)
        return
    try:
        clf = joblib.load(CLF_PATH)
        print("Loaded classifier from", CLF_PATH)
    except Exception as e:
        print("Failed to load classifier:", e)
        return
    print()

    # classes
    print("Classifier classes_ (order):", clf.classes_)
    print("Classifier type:", type(clf))
    # if logistic regression, coef_ shaped (1, n_features)
    is_lr = hasattr(clf, "coef_") and hasattr(clf, "predict_proba")
    print("Has coef_:", is_lr)
    print()

    # vocabulary
    try:
        feat_names = vect.get_feature_names_out()
        print("Vectorizer: number of features:", len(feat_names))
    except Exception as e:
        print("Could not get feature names:", e)
        feat_names = None

    # check some tokens
    tokens = ["amazing", "great", "terrible", "love", "horrible", "gripping", "plot"]
    print("\nToken presence in vocabulary:")
    for t in tokens:
        present = (t in vect.vocabulary_) if hasattr(vect, "vocabulary_") else False
        idx = vect.vocabulary_[t] if present else None
        print(f" - {t:10s} -> present: {present}, index: {idx}")
    print()

    # top positive / negative features if LR
    if is_lr and feat_names is not None:
        coefs = clf.coef_[0]  # binary
        top_pos = np.argsort(coefs)[-20:][::-1]
        top_neg = np.argsort(coefs)[:20]
        print("Top positive features (word : coef):")
        for i in top_pos[:20]:
            print(f"  {feat_names[i]} : {coefs[i]:.4f}")
        print("\nTop negative features (word : coef):")
        for i in top_neg[:20]:
            print(f"  {feat_names[i]} : {coefs[i]:.4f}")
    print()

    # For each sample, show vector nonzero elements and contributions
    for s in sample_texts:
        print("=== SAMPLE ===")
        print("TEXT:", s)
        c = simple_clean(s)
        print("CLEAN:", c)
        X = vect.transform([c])
        probs = clf.predict_proba(X)[0]
        pred = clf.predict(X)[0]
        print("Raw pred index:", pred, "-> label according to classes_:", clf.classes_[int(pred)])
        print("PROBs (in classes_ order):", probs)

        # show non-zero features in transform
        nz = X.nonzero()
        # sklearn sparse: X.tocsr()
        Xc = X.tocoo()
        items = list(zip(Xc.col, Xc.data))
        if not items:
            print("TRANSFORM: vector is all zeros (no known tokens matched).")
        else:
            # show feature, tfidf value, coef*tfidf (contribution)
            print("\nNon-zero features and contributions (feature, tfidf, coef, tfidf*coef):")
            for col, tfidf_val in items[:200]:  # limit output to first 200
                fname = feat_names[col] if feat_names is not None else str(col)
                coef = clf.coef_[0][col] if is_lr else 0.0
                contrib = coef * tfidf_val
                print(f"  {fname:20s} | tfidf: {tfidf_val:.4f} | coef: {coef:.4f} | contrib: {contrib:.4f}")
        print("\n----\n")
    print("INSPECTION COMPLETE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="I loved the acting and the plot was gripping!", help="Sample text to inspect")
    args = parser.parse_args()
    # provide a couple samples
    samples = [
        args.text,
        "This movie was amazing.",
        "Terrible, I hated every minute."
    ]
    load_and_inspect(samples)
