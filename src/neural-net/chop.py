import argparse
import random
import re
from pathlib import Path
import spacy

def main():
    ap = argparse.ArgumentParser(description="Sentencize a corpus and keep up to N sentences.")
    ap.add_argument("input", help="Path to the raw text file")
    ap.add_argument("-o", "--output", default="ipadgettCorpus-final.txt", help="Output path (default: reduced-corpus.txt)")
    ap.add_argument("-n", "--num", type=int, default=10000, help="Max sentences to keep (default: 10000)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling (default: 42)")
    ap.add_argument("--newline-only", action="store_true",
                    help="Treat ONLY newlines as sentence boundaries (useful for lyrics/chats)")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")

    punct = ["\n"] if args.newline_only else ["\n", ".", "!", "?"]

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer", config={"punct_chars": punct})
    nlp.max_length = max(nlp.max_length, len(text) + 10_000)

    doc = nlp(text)
    sentences = [re.sub(r"\s+", " ", s.text).strip()
                 for s in doc.sents if s.text.strip()]

    random.seed(args.seed)
    random.shuffle(sentences)

    keep = sentences[: args.num]
    Path(args.output).write_text("\n".join(keep) + "\n", encoding="utf-8")

    print(f"Wrote {len(keep)} sentences to {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()

