import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

IN_PATH = "data/processed/easy/easy_subset_en_bn.csv"
OUT_DIR = "data/processed/easy"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH)
    print("Loaded:", df.shape)
    print("Columns:", list(df.columns))

    texts = df["lyrics"].astype(str).tolist()
    labels = df["language"].astype(str).tolist()

    # Encode labels (Bangla/English)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print("Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    X = tfidf.fit_transform(texts).toarray().astype(np.float32)

    np.save(os.path.join(OUT_DIR, "X_easy.npy"), X)
    np.save(os.path.join(OUT_DIR, "y_easy.npy"), y)

    print("✅ Saved:", os.path.join(OUT_DIR, "X_easy.npy"), X.shape)
    print("✅ Saved:", os.path.join(OUT_DIR, "y_easy.npy"), y.shape)

if __name__ == "__main__":
    main()
