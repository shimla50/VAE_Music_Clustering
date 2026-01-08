import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

# === Hard task feature output directory ===
FEATURES_DIR = "data/hard/features"
os.makedirs(FEATURES_DIR, exist_ok=True)


# ====== paths ======
CSV_PATH = r"data/raw/Hard task dataset/songs.csv"

OUT_PROCESSED = os.path.join("data", "hard", "processed")
OUT_FEATURES = os.path.join("data", "hard", "features")
os.makedirs(OUT_PROCESSED, exist_ok=True)
os.makedirs(OUT_FEATURES, exist_ok=True)

RANDOM_SEED = 42

# ====== hard-task CPU friendly settings ======
TOP_GENRES = 10
TARGET_MAX = 8000          # keep 5k–8k (CPU friendly)
MIN_LYRICS_WORDS = 30
MIN_DURATION_SEC = 30
TFIDF_MAX_FEATURES = 3000  # 3000–5000 good on CPU

# ====== helper ======
def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # keep English + Bangla letters + space
    text = re.sub(r"[^a-zA-Z\u0980-\u09FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def purity_score(y_true, y_pred):
    # y_true: true labels (int), y_pred: cluster labels (int)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        labels, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()
    return total / len(y_true)

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # ====== 1) required columns check ======
    if "lyrics" not in df.columns or "genre" not in df.columns:
        raise ValueError(
            "Required columns missing. Need at least: lyrics, genre. "
            f"Available columns: {list(df.columns)[:40]} ..."
        )

    # ====== 2) duration column (best effort) ======
    duration_col = None
    for cand in ["duration_ms", "duration", "track_duration_ms"]:
        if cand in df.columns:
            duration_col = cand
            break
    if duration_col:
        print("Using duration column:", duration_col)
    else:
        print("WARNING: no duration column found, skipping duration filter.")

    # ====== 3) audio numeric columns selection ======
    # We'll auto-pick numeric columns but drop obvious non-feature IDs if present
    drop_cols = set(["lyrics", "genre"])
    numeric_cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    # safety: remove columns with too many missing values
    if numeric_cols:
        na_ratio = df[numeric_cols].isna().mean()
        numeric_cols = [c for c in numeric_cols if na_ratio[c] < 0.2]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric audio/features columns found. Cannot build X_audio.")

    print("Numeric feature columns count:", len(numeric_cols))

    # ====== 4) clean text + filter ======
    df["lyrics"] = df["lyrics"].apply(clean_lyrics)
    df["lyrics_wc"] = df["lyrics"].str.split().apply(len)

    df = df.dropna(subset=["genre"])
    df["genre"] = df["genre"].astype(str).str.strip()
    df = df[df["genre"].str.len() > 0]

    df = df[df["lyrics_wc"] >= MIN_LYRICS_WORDS]

    if duration_col:
        if duration_col.endswith("_ms"):
            df["duration_sec"] = pd.to_numeric(df[duration_col], errors="coerce") / 1000.0
        else:
            df["duration_sec"] = pd.to_numeric(df[duration_col], errors="coerce")
        df = df[df["duration_sec"] >= MIN_DURATION_SEC]

    print("After base filters:", len(df))

    # ====== 5) keep top genres ======
    top = df["genre"].value_counts().head(TOP_GENRES)
    keep_genres = top.index.tolist()
    df = df[df["genre"].isin(keep_genres)].copy()

    print("Top genres kept:")
    print(top)

    # ====== 6) balanced sampling ======
    per_genre = int(TARGET_MAX / TOP_GENRES)
    parts = []
    for g in keep_genres:
        sub = df[df["genre"] == g]
        n = min(len(sub), per_genre)
        parts.append(sub.sample(n=n, random_state=RANDOM_SEED))
    df_bal = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if len(df_bal) > TARGET_MAX:
        df_bal = df_bal.sample(n=TARGET_MAX, random_state=RANDOM_SEED).reset_index(drop=True)

    print("Final N:", len(df_bal))

    # ====== 7) save processed csv ======
    processed_path = os.path.join(OUT_PROCESSED, "clean_balanced_songs.csv")
    df_bal[["genre", "lyrics"] + numeric_cols].to_csv(processed_path, index=False, encoding="utf-8")
    print("Saved processed:", processed_path)

    # ====== 8) build labels ======
    le = LabelEncoder()
    y = le.fit_transform(df_bal["genre"].values)
    genre_map = {int(i): g for i, g in enumerate(le.classes_)}
    with open(os.path.join(OUT_FEATURES, "genre_map.json"), "w", encoding="utf-8") as f:
        json.dump(genre_map, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(OUT_FEATURES, "y_genre.npy"), y)
    print("Saved y_genre.npy")

    # ====== 9) build X_audio (scaled) ======
    X_audio_raw = df_bal[numeric_cols].copy()
    X_audio_raw = X_audio_raw.fillna(X_audio_raw.median(numeric_only=True))
    scaler = StandardScaler()
    X_audio = scaler.fit_transform(X_audio_raw.values).astype(np.float32)
    np.save(os.path.join(OUT_FEATURES, "X_audio.npy"), X_audio)
    joblib.dump(scaler, os.path.join(OUT_FEATURES, "audio_scaler.joblib"))
    print("Saved X_audio.npy")

    # ====== 10) build TF-IDF ======
    tfidf = TfidfVectorizer(max_features=3000, norm="l2")
    X_lyrics = tfidf.fit_transform(df_bal["lyrics"].values)
    sparse.save_npz(os.path.join(OUT_FEATURES, "X_lyrics_tfidf.npz"), X_lyrics)
    joblib.dump(tfidf, os.path.join(OUT_FEATURES, "tfidf_vectorizer.joblib"))
    print("Saved X_lyrics_tfidf.npz")

    # ====== 11) build fusion ======
    # Keep fusion as dense but small enough: N<=8000, TFIDF<=5000 => OK
    X_lyrics_dense = X_lyrics.toarray().astype(np.float32)
    X_fusion = np.concatenate([X_audio, X_lyrics_dense], axis=1).astype(np.float32)
    np.save(os.path.join(OUT_FEATURES, "X_fusion.npy"), X_fusion)
    print("Saved X_fusion.npy", X_fusion.shape)

    print("\nDONE ✅ Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
