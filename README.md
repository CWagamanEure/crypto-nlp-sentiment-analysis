# IN PROGRESS: Semester Long Project for NLP class
## will be updated with all asignments

## How to run the custom neural network:
### Requirements:
```
pip install -r requirements.txt
```
1. cd through src into the neural-net directory and run: 
```
python3 load_embeds.py
```
2. To prep your corpus run the following command pointing to
your own corpora and custom labels in order:
```
python3 prep.py -a y \
--corpora ./crypto-reddit-corpus.txt \
./KendrickLamarCorpus.txt \
./ipadgettCorpus-final.txt \
--labels reddit_crypto kendrick Isaiahs_corpus \
-o ./parsed_labels.txt

```
3. Then run the train.py script with optional parmaters:
 - number of hidden dimensions: -H
 - learning rate: -l
 - number of training epochs: -e
 ```
python3 train.py -H 512 -l .2 -e 1000
```
4. Then you can evaluate the model by running:
```
python3 eval.py
```
5. You can then interact with the model through:
```
python3 eval.py
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

