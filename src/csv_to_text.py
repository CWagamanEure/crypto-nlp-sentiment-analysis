from pathlib import Path
import pandas as pd, textwrap, random

IN = Path("../data/processed/reddit_crypto_corpus_clean_min.csv")
OUT = Path("../data/processed/corpus_reddit.txt")

df = pd.read_csv(IN, usecols=["text"])
df["text"] = df["text"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

with OUT.open("w", encoding="utf-8") as f:
    for t in df["text"]:
        if t:
            f.write(t + "\n\n")

size_mb = OUT.stat().st_size / (1024**2)
print(f"Wrote: {OUT} | Size: {size_mb:.2f} MB | Docs: {len(df)}")



samples = [t for t in df["text"] if 100 <= len(t.split()) <= 1000]
random.seed(470)
for i, s in enumerate(random.sample(samples, k=min(2, len(samples))), 1):
    print(f"\n--- Passage {i} ({len(s.split())} words) ---\n")
    print(textwrap.fill(s, width=100))
