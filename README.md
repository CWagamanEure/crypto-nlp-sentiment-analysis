# IN PROGRESS: Semester Long Project for NLP class
## will be updated with all asignments


## Repository Structure

```
.
├── NGramProbAndGen.py     # Provide a corpus to type a sentence and receive 1-3 gram prob, Also generates grams from the corpus
├── corpus_eda.py          # Quick stats/plots: tokens, vocab size, top n‑grams, lengths
├── csv_to_text.py         # Convert one or more CSV files/columns into a plain‑text corpus
├── reddit_cleaner.py      # Normalize & clean raw Reddit JSON/CSV into newline‑delimited text
├── reddit_client.py       # Fetch posts/comments from Reddit API and save to CSV
└── regex.py               # Script to take regex and test on corpus
```
## Reddit-client CSV format:
```
"platform": "reddit",
                "subreddit": sub,
                "id": c.id,
                "author": str(c.author) if c.author else None,
                "created_at": pd.to_datetime(c.created_utc, unit="s", utc=True),
                "title": "",
                "text": c.body,
                "score": c.score,
                "num_comments": None,
                "url": f"https://reddit.com{c.permalink}",
                "type": "comment"

```
---

## Quickstart

1) **Create & activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) **Install deps (suggested)**
```bash
pip install -U pandas numpy nltk matplotlib praw tqdm click
# If you plot with Jupyter later:
pip install jupyter
```

3) **Collect data → clean → EDA → n‑gram**
```bash
# 1) Collect
python reddit_client.py --subreddit crypto --limit 1000 --comments --out data/raw/crypto.jsonl

# 2) Clean
python reddit_cleaner.py --in data/raw/crypto.jsonl --out data/clean/crypto.txt

# (Alternative) Convert CSV -> text (choose text column)
python csv_to_text.py --in data/raw/reddit_dump.csv --text-col body --out data/clean/reddit_body.txt

# 3) Explore
python corpus_eda.py --in data/clean/crypto.txt --top 30 --ngram 1 --ngram 2 --savefig out/eda/

# 4) Train & sample from an n‑gram LM
python NGramProbAndGen.py --in data/clean/crypto.txt --n 3 --k 1.0 --samples 5 --max-tokens 60 \
  --export-probs out/models/trigram_probs.json --out out/samples/trigram_samples.txt
```

---

