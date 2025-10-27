import argparse, os, re, sys
from typing import Dict, List, Set, Tuple
import pandas as pd

ALIASES: Dict[str, List[str]] = {
    "BTC":  [r"\bbitcoin\b", r"\bxbt\b", r"\bsats?\b"],
    "ETH":  [r"\bethereum\b", r"\bether\b"],
    "SOL":  [r"\bsolana\b"],
    "XRP":  [r"\bripple\b", r"\bxrp\b"],
    "ADA":  [r"\bcardano\b", r"\bada\b"],
    "AVAX": [r"\bavalanche\b", r"\bavax\b"],
    "DOGE": [r"\bdogecoin\b", r"\bdoge\b"],
    "LINK": [r"\bchainlink\b"],                    # avoid generic "link"
    "LTC":  [r"\blitecoin\b", r"\bltc\b"],
    "BNB":  [r"\bbinance coin\b", r"\bbnb\b"],
    "MATIC":[r"\bpolygon\b", r"\bmatic\b"],
    "DOT":  [r"\bpolkadot\b"],                     # avoid bare "dot"
    "OP":   [r"\boptimism\b"],                     # avoid bare "op"
    "ARB":  [r"\barbitrum\b"],
    "ATOM": [r"\bcosmos\b", r"\batom\b"],
    "TRX":  [r"\btron\b", r"\btrx\b"],
    "NEAR": [r"\bnear protocol\b"],                # avoid bare "near"
    "APT":  [r"\baptos\b", r"\bapt\b"],
    "SUI":  [r"\bsui\b"],
    "XLM":  [r"\bstellar\b", r"\bxlm\b"],
    "BCH":  [r"\bbitcoin cash\b", r"\bbch\b"],
    "UNI":  [r"\buniswap\b", r"\buni\b"],
    "AAVE": [r"\baave\b"],
    "MKR":  [r"\bmaker\b", r"\bmkr\b"],
    "INJ":  [r"\binjective\b", r"\binj\b"],
    "USDT": [r"\btether\b", r"\busdt\b"],
    "USDC": [r"\busdc\b", r"\bcircle\b(?=.*\b(usdc|stable)\b)"],
    "DAI":  [r"\bdai\b(?=.*\b(stable|maker|cdp|vault)\b)"],
}


def build_patterns():
    patterns = {}
    for sym, kws in ALIASES.items():
        cashtag = rf"\${sym}\b"
        body = "(?:" + "|".join([cashtag] + kws) + ")"
        patterns[sym] = re.compile(body, re.I)
    return patterns

CRYPTO_CUES = re.compile(
    r"\b(crypto|blockchain|defi|web3|token|coin|airdrop|wallet|exchange|nft|staking|hashrate|"
    r"smart contract|layer ?2|l2|gas fees|on-?chain|dex|bridge|yield farming)\b",
    re.I
)

CRYPTO_SUBS = {
    "cryptocurrency","cryptomarkets","bitcoin","btc","ethereum","ethtrader","ethfinance",
    "solana","cardano","dogecoin","chainlink","binance","defi"
}


def detect_assets_and_flag(title: str, text: str, subreddit: str, patterns: Dict[str, re.Pattern]):
    t = f"{title or ''} {text or ''}"
    found: Set[str] = set()
    for sym, pat in patterns.items():
        if pat.search(t):
            found.add(sym)
    has_cues = bool(CRYPTO_CUES.search(t))
    sub = (subreddit or "").lower()
    in_crypto_sub = sub in CRYPTO_SUBS
    # is_crypto = explicit asset OR generic crypto cues OR clearly crypto subreddit
    is_crypto = bool(found) or has_cues or in_crypto_sub
    return sorted(found), is_crypto

def main():
    ap = argparse.ArgumentParser(description="Filter a Reddit CSV down to crypto-related rows.")
    ap.add_argument("csv_path", help="Input CSV path (expects columns: subreddit, title, text, ...)")
    ap.add_argument("-o", "--out", default=None, help="Output CSV path (default: <input>_crypto.csv)")
    ap.add_argument("--mode", choices=["assets_or_cues","assets_only"], default="assets_or_cues",
                    help="assets_only keeps rows only if a specific asset is detected (no generic cues).")
    args = ap.parse_args()

    out_path = args.out or os.path.splitext(args.csv_path)[0] + "_crypto.csv"

    try:
        df = pd.read_csv(args.csv_path, low_memory=False)
    except Exception as e:
        print(f"[error] Could not read CSV: {e}", file=sys.stderr); sys.exit(1)

    for col in ("title","text","subreddit"):
        if col not in df.columns: df[col] = ""

    patterns = build_patterns()

    assets_col, flags = [], []
    for ttl, txt, sub in zip(df["title"], df["text"], df["subreddit"]):
        assets, is_crypto = detect_assets_and_flag(ttl, txt, sub, patterns)
        assets_col.append(assets)
        flags.append(is_crypto)

    df["assets"] = assets_col
    df["is_crypto"] = flags

    if args.mode == "assets_only":
        mask = df["assets"].map(lambda a: len(a) > 0)
    else:
        mask = df["is_crypto"].astype(bool)

    out_df = df[mask].copy()

    try:
        out_df.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[error] Could not write CSV: {e}", file=sys.stderr); sys.exit(1)

    kept = len(out_df); total = len(df)
    print(f"[done] Kept {kept:,} of {total:,} rows → {out_path}")
    if kept:
        counts = (out_df.explode("assets")
                          .query("assets.notna() and assets != ''")
                          .groupby("assets").size()
                          .sort_values(ascending=False)
                          .head(15))
        if len(counts):
            print("\nTop asset mentions in output:")
            for sym, n in counts.items():
                print(f"  {sym:>5s} : {int(n)}")

if __name__ == "__main__":
    main()
