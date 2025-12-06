# api/api_server.py

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import joblib
import os
import re

# paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  
ROOT = os.path.dirname(THIS_DIR)
MODELS_DIR = os.path.join(ROOT, "models")

VECT_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CLF_PATH  = os.path.join(MODELS_DIR, "sentiment_model.joblib")

# 🔥 THIS MUST EXIST — the FastAPI app
app = FastAPI(title="Movie Review Sentiment API")

class PredictRequest(BaseModel):
    text: str

def simple_clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@app.on_event("startup")
def load_model():
    """Load model on startup"""
    global vect, clf
    vect = joblib.load(VECT_PATH)
    clf  = joblib.load(CLF_PATH)
    print("Models loaded.")

@app.get("/")
def root():
    return {"message": "API is working"}
@app.get("/favicon.ico")
def favicon():
    path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return Response(status_code=204) 

@app.post("/predict")
def predict(payload: PredictRequest):
    clean = simple_clean(payload.text)
    X = vect.transform([clean])
    probs = clf.predict_proba(X)[0]

    # assume classes = ['neg','pos']
    pos_prob = float(probs[1])
    neg_prob = float(probs[0])

    label = "Positive" if pos_prob >= 0.5 else "Negative"

    return {
        "label": label,
        "pos_prob": pos_prob,
        "neg_prob": neg_prob
    }
