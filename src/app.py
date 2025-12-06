# src/app.py
import os
import sys
import time
import shlex
import subprocess
import streamlit as st

# Project root (one level up from src/)
ROOT = os.path.dirname(os.path.dirname(__file__))
PREDICT_SCRIPT = os.path.join(ROOT, "src", "predict.py")
MODELS_DIR = os.path.join(ROOT, "models")
VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CLF_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")
st.title("Movie Review Sentiment Classifier")
st.write("Type a movie review and press Predict. The app calls the same CLI used in your terminal, so results will match.")

# Basic model existence check
if not os.path.exists(PREDICT_SCRIPT):
    st.error(f"predict.py not found at: {PREDICT_SCRIPT}\nMake sure you placed predict.py in src/")
    st.stop()

if not (os.path.exists(VECT_PATH) and os.path.exists(CLF_PATH)):
    st.error("Model files not found in the models/ directory. Run `python src/train_model.py` first to generate them.")
    st.stop()

# Show small info about model files to help debug caching issues
st.caption(f"Model files: {os.path.basename(VECT_PATH)} (mtime: {time.ctime(os.path.getmtime(VECT_PATH))}), "
           f"{os.path.basename(CLF_PATH)} (mtime: {time.ctime(os.path.getmtime(CLF_PATH))})")

placeholder = st.empty()
user_input = placeholder.text_area("Enter review", height=160, placeholder="This movie was amazing because...")

col1, col2 = st.columns([1, 3])
with col1:
    predict_btn = st.button("Predict")
with col2:
    st.write("Tip: short sentences can be ambiguous. This app uses the same CLI code as your terminal.")

def run_predict_cli(text: str, timeout: int = 8):
    """
    Run the CLI predict.py and return (stdout, stderr, returncode).
    We call Python executable directly for consistent environment.
    """
    if not text.strip():
        return "", "Empty text", 1

    # Build argument list to avoid shell quoting issues
    args = [sys.executable, PREDICT_SCRIPT, "--text", text]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "predict.py timed out", 2
    except Exception as e:
        return "", f"Failed to run predict.py: {e}", 3

def parse_predict_output(stdout: str):
    """
    Parse common lines from predict.py output.
    Expected lines (example):
      Prediction: Positive (confidence: 0.853)
      Positive: 0.853 | Negative: 0.147
    We'll extract prediction label and numeric probs if present.
    """
    out_lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    parsed = {"raw": stdout, "label_line": None, "pos_prob": None, "neg_prob": None}
    for ln in out_lines:
        if ln.lower().startswith("prediction:"):
            parsed["label_line"] = ln
        if ln.lower().startswith("positive:"):
            # format: "Positive: 0.853 | Negative: 0.147"
            parts = ln.replace(",", ".").split("|")
            try:
                left = parts[0].strip()  # "Positive: 0.853"
                right = parts[1].strip() if len(parts) > 1 else ""
                parsed["pos_prob"] = float(left.split(":")[1].strip())
                if right:
                    parsed["neg_prob"] = float(right.split(":")[1].strip())
            except Exception:
                pass
    return parsed

if predict_btn:
    text = user_input.strip()
    if not text:
        st.info("Please type a review first.")
    else:
        with st.spinner("Running predict.py ..."):
            stdout, stderr, code = run_predict_cli(text)
        if stderr:
            st.error("predict.py stderr:\n" + stderr)
        if code != 0 and not stdout:
            st.error(f"predict.py failed with code {code}. See stderr above.")
        else:
            parsed = parse_predict_output(stdout)
            # Show parsed label (big) and probabilities if available
            if parsed["label_line"]:
                # show label_line as markdown (make label larger)
                label_text = parsed["label_line"].replace("Prediction:", "").strip()
                st.markdown(f"## Prediction: **{label_text}**")
            else:
                # fallback to raw
                st.markdown("## Prediction (raw output)")
                st.code(stdout)

            if parsed["pos_prob"] is not None and parsed["neg_prob"] is not None:
                st.write(f"Positive: {parsed['pos_prob']:.2f}   |   Negative: {parsed['neg_prob']:.2f}")
                st.progress(min(100, int(parsed["pos_prob"] * 100)))
            else:
                # show raw output if probs not parsed
                st.code(stdout)

            # show raw output collapsed for debugging
            with st.expander("Raw predict.py output"):
                st.text(stdout or "<no stdout>")
                if stderr:
                    st.text("STDERR:")
                    st.text(stderr)

# Footer / small notes
st.markdown("---")
st.write("If results still look off for short sentences, we can either:")
st.write("- Expand the small lexicon used by the CLI (fast rule-based fix), or")
st.write("- Retrain with lemmatization or a larger TF-IDF vocabulary, or")
st.write("- Switch to a sentence-embedding model (heavier but better for short text).")

