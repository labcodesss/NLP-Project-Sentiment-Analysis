# utils/config.py
import os

ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root when used from src or top-level scripts
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")  # for versioned artifacts
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
