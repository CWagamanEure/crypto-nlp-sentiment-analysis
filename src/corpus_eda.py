import re
from collections import Counter
from pathlib import Path
import pandas as pd

# ---- INPUT (edit if needed) ----
INPUT = Path("~/cpsc470/crypto-sentiment-corpus/data/processed/reddit_crypto_corpus_clean_min.csv").expanduser()

df = pd.read_csv(INPUT)

# Tokenize (keeps tickers like $BTC, hashtags like #Bitcoin)
WORD_RE = re.compile(r"[#$]?[A-Za-z][A-Za-z0-9_'\-]*")
df["tokens"]  = df["text"].fillna("").str.findall(WORD_RE)
df["n_words"] = df["tokens"].str.len()
df["n_chars"] = df["text"].fillna("").str.len()

print("=== Size ===")
print(f"Docs: {len(df):,}")
print(f"Total words: {int(df['n_words'].sum()):,}")
print(f"Avg words/doc: {df['n_words'].mean():.1f}")
print(f"Median words/doc: {int(df['n_words'].median())}")
print(df["n_words"].describe(percentiles=[.1,.25,.5,.75,.9]).round(1))

# Time coverage (if present)
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    span = (df["created_at"].max() - df["created_at"].min())
    per_day = df.set_index("created_at").resample("D").size()
    print("\n=== Time ===")
    print(f"Span: {df['created_at'].min()} → {df['created_at'].max()}  (~{span.days} days)")
    if len(per_day):
        print(f"Median docs/day: {int(per_day.median())}")

# Subreddit mix (if present)
if "subreddit" in df.columns:
    vc = df["subreddit"].value_counts()
    print("\n=== Top subreddits by docs ===")
    print(vc.head(12))
    share_top1 = vc.div(len(df)).iloc[0]
else:
    share_top1 = None

# Top tokens & bigrams (very small stoplist)
STOP = set("""
a an the and or of to in for on with is are was were be been being at by from as this that it it's i you he she we they them my our your his her their me us
will would should could about into not no yes if then than vs s t d re ve ll m
""".split())

uni, bi = Counter(), Counter()
for toks in df["tokens"]:
    toks = [t.lower() for t in toks]
    toks = [t for t in toks if t not in STOP]
    uni.update(toks)
    bi.update(zip(toks, toks[1:]))

print("\n=== Top 25 tokens ===")
for w,c in uni.most_common(25):
    print(f"{w:15} {c}")

print("\n=== Top 25 bigrams ===")
for (a,b),c in bi.most_common(25):
    print(f"{a} {b:12} {c}")

