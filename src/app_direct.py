# src/app_direct.py
"""
Streamlit app that loads model artifacts from models/ and predicts locally.
No network, no API. Use this when FastAPI connectivity gives trouble.
Run:
    streamlit run src/app_direct.py
"""

import os
import re
import joblib
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CLF_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")

st.set_page_config(page_title="Movie Review Sentiment (Direct)", page_icon="🎬", layout="centered")
st.title("Movie Review Sentiment Classifier — Direct Mode")
st.write("This UI loads the model files directly from the `models/` folder (no network needed).")

def simple_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# Check artifacts exist and show helpful messages
missing = []
if not os.path.exists(VECT_PATH):
    missing.append(VECT_PATH)
if not os.path.exists(CLF_PATH):
    missing.append(CLF_PATH)

if missing:
    st.error("Model artifact(s) not found. Please train the model first with:\n\n"
             "`python models/train_standalone.py`\n\n"
             "Missing files:\n" + "\n".join(missing))
    st.stop()

# Load artifacts (with caching so UI is snappy)
@st.cache_resource
def load_artifacts():
    vect = joblib.load(VECT_PATH)
    clf = joblib.load(CLF_PATH)
    return vect, clf

vect, clf = load_artifacts()

st.caption(f"Loaded: {os.path.basename(VECT_PATH)} and {os.path.basename(CLF_PATH)}")
st.caption(f"Model classes (order): {list(clf.classes_)}")

review = st.text_area("Enter review:", height=160, placeholder="This movie was amazing because...")
if st.button("Predict"):
    if not review.strip():
        st.info("Please enter a review.")
    else:
        clean = simple_clean(review)
        X = vect.transform([clean])
        probs = clf.predict_proba(X)[0]

        # robust extraction of pos prob
        classes = list(clf.classes_)
        pos_prob = None
        if "pos" in classes:
            pos_prob = float(probs[classes.index("pos")])
        elif 1 in classes:
            pos_prob = float(probs[classes.index(1)])
        else:
            # fallback: assume last index is positive
            pos_prob = float(probs[-1])

        neg_prob = 1.0 - pos_prob
        label = "Positive" if pos_prob >= 0.5 else "Negative"

        st.markdown(f"## Prediction: **{label}**")
        st.write(f"Positive: {pos_prob:.3f}   |   Negative: {neg_prob:.3f}")
        st.progress(min(100, int(pos_prob * 100)))
        # show raw probs and cleaned text for debugging
        with st.expander("Debug info"):
            st.write("Cleaned:", clean)
            st.write("Raw probs (model.classes_ order):", probs)
            st.write("classes_: ", classes)
