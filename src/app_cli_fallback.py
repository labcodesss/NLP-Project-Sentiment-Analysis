# src/app_cli_fallback.py
"""
Streamlit UI fallback that runs the local CLI predictor script (predict_standalone.py).
This avoids any FastAPI/network issues and matches terminal CLI results exactly.
Run:
    streamlit run src/app_cli_fallback.py
"""
import os
import sys
import shlex
import subprocess
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICT_SCRIPT = os.path.join(ROOT, "src", "predict_standalone.py")  # uses models/*.joblib

st.set_page_config(page_title="Movie Review Sentiment (CLI)", page_icon="🎬", layout="centered")
st.title("Movie Review Sentiment Classifier — CLI Fallback")
st.write("This UI runs the same local CLI you used earlier, so it will always match the terminal output.")

if not os.path.exists(PREDICT_SCRIPT):
    st.error(f"predict_standalone.py not found at: {PREDICT_SCRIPT}. Ensure you have the CLI script in src/")
    st.stop()

text = st.text_area("Enter review:", height=160, placeholder="This movie was amazing because...")
if st.button("Predict"):
    if not text.strip():
        st.info("Please type a review first.")
    else:
        # Build subprocess args
        cmd = [sys.executable, PREDICT_SCRIPT, "--text", text]
        try:
            with st.spinner("Running local predictor..."):
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            if stderr:
                st.error("predict_standalone stderr:\n" + stderr)
            if stdout:
                # Parse output lines to show nicely
                lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
                # show prediction first line if present
                st.markdown("### Result")
                for ln in lines:
                    st.text(ln)
                # also show raw output in expander
                with st.expander("Raw CLI output"):
                    st.text(stdout)
            else:
                st.warning("No output from predict script. Check models exist in models/ and that script runs in terminal.")
        except subprocess.TimeoutExpired:
            st.error("predict_standalone timed out.")
        except Exception as e:
            st.error(f"Failed to run predictor: {e}")
