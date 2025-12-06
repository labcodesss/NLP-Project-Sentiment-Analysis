# src/data_prep.py
import os
import csv
import nltk

# ensure required NLTK data
nltk_data = ["movie_reviews", "punkt", "wordnet", "omw-1.4"]
for pack in nltk_data:
    try:
        nltk.data.find(pack)
    except LookupError:
        nltk.download(pack)

from nltk.corpus import movie_reviews

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "movie_reviews.csv")

def build_csv(out_path=OUT_PATH):
    """
    Create CSV with columns: text,label
    labels: 'pos' or 'neg'
    """
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
