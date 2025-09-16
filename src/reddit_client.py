import os, time
import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")
PATH = "~/cpsc470/crypto-sentiment-corpus/data"

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)

SUBS = ["investing", "stocks", "technology", "economics", "wallstreetbets", "CryptoCurrency", "Bitcoin", "ethtrader", "CryptoMarkets", "CryptoCurrencyTrading", "ethfinance", "ethereum", "BitcoinBeginners", "defi", "Solana", "CryptoTechnology"]
QUERY = "bitcoin OR ethereum OR crypto OR altcoin OR solana"

rows = []
for sub in SUBS:
    for s in reddit.subreddit(sub).search(QUERY, sort="new", time_filter="week", limit=300):
        rows.append({
            "platform": "reddit",
            "subreddit": sub,
            "id": s.id,
            "author": str(s.author) if s.author else None,
            "created_at": pd.to_datetime(s.created_utc, unit="s", utc=True),
            "title": s.title,
            "text": s.selftext or "",
            "score": s.score,
            "num_comments": s.num_comments,
            "url": f"https://reddit.com{s.permalink}",
            "type": "submission"
        })

        s.comments.replace_more(limit=0)
        for c in s.comments[:100]:
            rows.append({
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
            })
        time.sleep(0.3)
df_reddit = pd.DataFrame(rows)
file_name = "reddit_crypto_corpus.csv"
save_path = os.path.join(PATH, file_name)
os.makedirs(PATH, exist_ok=True)
df_reddit.to_csv(save_path, index=False)
print("Reddit rows:", len(df_reddit))
