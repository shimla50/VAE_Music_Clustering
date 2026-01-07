import os
import re
import pandas as pd


CSV_PATH = r"C:\Users\HP\Documents\music_vae_project\data\raw\Medium task dataset\song_lyrics.csv "   # from PowerShell output

OUT_PATH = os.path.join("data", "processed", "easy", "easy_subset_en_bn.csv")

def is_bangla_text(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return bool(re.search(r"[\u0980-\u09FF]", s))

def looks_english(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return bool(re.search(r"[A-Za-z]", s))

def main():
    df = pd.read_csv(CSV_PATH)
    print("Loaded:", df.shape)
    print("Columns:", list(df.columns))

    # --------- find lyrics column ----------
    lyr_candidates = ["lyrics", "lyric", "text", "clean_lyrics"]
    lyr_col = None
    for c in lyr_candidates:
        if c in df.columns:
            lyr_col = c
            break
    if lyr_col is None:
        raise ValueError(f"No lyrics column found. Checked: {lyr_candidates}")

    # --------- clean basic ----------
    df = df.dropna(subset=[lyr_col]).copy()
    df[lyr_col] = df[lyr_col].astype(str)

    # drop very short lyrics
    df["word_count"] = df[lyr_col].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 30].copy()

    # --------- language column exists? ----------
    lang_candidates = ["language", "lang", "lyrics_language"]
    lang_col = None
    for c in lang_candidates:
        if c in df.columns:
            lang_col = c
            break

    if lang_col:
        print("✅ Found language column:", lang_col)
        df[lang_col] = df[lang_col].astype(str).str.lower().str.strip()

        df_bn = df[df[lang_col].isin(["bn", "bangla", "bengali"])].copy()
        df_en = df[df[lang_col].isin(["en", "english"])].copy()

    else:
        print("⚠️ No language column. Using Bangla unicode detection...")
        df["is_bangla"] = df[lyr_col].apply(is_bangla_text)
        df["looks_english"] = df[lyr_col].apply(looks_english)

        df_bn = df[df["is_bangla"] == True].copy()
        df_en = df[(df["is_bangla"] == False) & (df["looks_english"] == True)].copy()

    print("Bangla candidates:", df_bn.shape)
    print("English candidates:", df_en.shape)

    # --------- balanced sampling ----------
    N_BN = 1000
    N_EN = 1000

    df_bn_s = df_bn.sample(min(N_BN, len(df_bn)), random_state=42)
    df_en_s = df_en.sample(min(N_EN, len(df_en)), random_state=42)

    df_sub = pd.concat([df_bn_s, df_en_s], ignore_index=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_sub.to_csv(OUT_PATH, index=False)

    print("✅ Saved subset:", OUT_PATH)
    print("Final subset:", df_sub.shape)

if __name__ == "__main__":
    main()
