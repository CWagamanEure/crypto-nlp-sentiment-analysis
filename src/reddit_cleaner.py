import re, html
from pathlib import Path
import pandas as pd
INPUT  = Path("~/cpsc470/crypto-sentiment-corpus/data/reddit_crypto_corpus.csv").expanduser()
OUTDIR = Path("~/cpsc470/crypto-sentiment-corpus/data/processed").expanduser()
OUTPUT = OUTDIR / "reddit_crypto_corpus_clean_min.csv"

url_re       = re.compile(r"https?://\S+|www\.\S+")
md_link_re   = re.compile(r"\[(.*?)\]\((.*?)\)")
code_inline  = re.compile(r"`{1,3}.*?`{1,3}")

BOT_PHRASE = re.compile(
    r"(auto-?\s*generated|summary\s*bot|replace\s*reading|always\s*dyor)",
    re.IGNORECASE,
)

def clean(s: str)-> str:
    s = "" if s is None else str(s)
    s = html.unescape(s)
    s = url_re.sub(" ", s)
    s = md_link_re.sub(r"\1", s)
    s = code_inline.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    df = pd.read_csv(INPUT)

    

    title = df["title"].fillna("") if "title" in df.columns else pd.Series("", index=df.index)
    text  = df["text"].fillna("")  if "text"  in df.columns else pd.Series("", index=df.index)

    body = (title.astype(str) + " " + text.astype(str)).str.replace(r"\s+", " ", regex=True).str.strip()
    df["text_raw"] = body

    rm = df["text_raw"].str.strip().str.lower().isin({"[removed]", "[deleted]"})
    df = df[~rm].copy()

    if "author" in df.columns:
        df = df[~df["author"].fillna("").str.contains(r"^automoderator$", case=False)].copy()

    df = df[~df["text_raw"].str.contains(BOT_PHRASE, na=False)].copy()
    

    df["text"] = df["text_raw"].apply(clean)

    df = df[df["text"].str.len() >= 10].drop_duplicates(subset=["text"]).copy()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"clean rows: {len(df):,} → {OUTPUT}")

if __name__ == "__main__":
    main()
