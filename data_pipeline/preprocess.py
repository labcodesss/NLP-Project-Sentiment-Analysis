# data_pipeline/preprocess.py
import re
from typing import List
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def simple_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)    # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_lemmatize(text: str) -> List[str]:
    text = simple_clean(text)
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
    return lemmas

def preprocess_text(text: str) -> str:
    return " ".join(tokenize_and_lemmatize(text))
