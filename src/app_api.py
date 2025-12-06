import os
import requests
import streamlit as st

# FORCE IPv4 explicitly
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Sentiment Classifier (API)", page_icon="🎬", layout="centered")
st.title("Movie Review Sentiment Classifier (API)")
st.caption(f"API endpoint: {API_URL}/predict")

text = st.text_area("Enter review:", height=160)

if st.button("Predict"):
    if not text.strip():
        st.info("Please enter some text.")
    else:
        with st.spinner("Calling API..."):
            try:
                # Use IPv4-only adapters
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text},
                    timeout=5
                )
                resp.raise_for_status()
                data = resp.json()

                st.markdown(f"## Prediction: **{data['label']}**")
                st.write(f"Positive: {data['pos_prob']:.2f} | Negative: {data['neg_prob']:.2f}")
                st.progress(int(data['pos_prob'] * 100))

                with st.expander("Raw API Response"):
                    st.json(data)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API at 127.0.0.1:8000. "
                         "Make sure FastAPI server is running:\n"
                         "`uvicorn api.api_server:app --host 0.0.0.0 --port 8000`")
            except Exception as e:
                st.error(f"API error: {e}")
