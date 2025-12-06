# data_pipeline/build_dataset.py
"""
Standalone script to build movie_reviews.csv using NLTK.
Does NOT import other project modules; computes paths relative to this file.
Run from project root OR from inside the project folder:
    python data_pipeline/build_dataset.py
"""

import os
import csv
import nltk

# Ensure NLTK corpora
nltk_packs = ["movie_reviews", "punkt", "wordnet", "omw-1.4"]
for p in nltk_packs:
    try:
        nltk.data.find(p)
    except LookupError:
        nltk.download(p)

from nltk.corpus import movie_reviews

# Paths (data folder will be sibling to this script's parent 'movie_sentiment' root)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../movie_sentiment/data_pipeline
ROOT = os.path.dirname(THIS_DIR)                               # .../movie_sentiment
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUT_PATH = os.path.join(DATA_DIR, "movie_reviews.csv")

def build_csv(out_path=OUT_PATH):
    with open(out_path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for fileid in movie_reviews.fileids():
            label = movie_reviews.categories(fileid)[0]
            words = movie_reviews.words(fileid)
            text = " ".join(words)
            writer.writerow([text, label])
    print(f"Wrote dataset to {out_path} (rows: {len(movie_reviews.fileids())})")

if __name__ == "__main__":
    build_csv()
