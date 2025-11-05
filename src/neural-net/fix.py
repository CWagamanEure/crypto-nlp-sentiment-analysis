# split_on_markers.py
import re
import argparse
from pathlib import Path

# Common sentence markers you mentioned (extend if needed)
DEFAULT_MARKERS = [
    r"<s>", r"</s>", r"<S>", r"</S>",
    r"<eos>", r"</eos>", r"<EOS>", r"</EOS>",
    r"<sep>", r"<SEP>", r"\[SEP\]", r"\[EOD\]",
    r"<\|endoftext\|>", r"<\|eot\|>"
]

def main():
    ap = argparse.ArgumentParser(description="Split a single-line corpus into one sentence per line using special markers.")
    ap.add_argument("input", help="Path to the raw corpus (likely one very long line).")
    ap.add_argument("-o", "--output", default="sentences.txt", help="Where to write one-sentence-per-line output.")
    ap.add_argument("--markers", nargs="*", help="Override markers (regex). If omitted, uses a sensible default set.")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="ignore")

    markers = args.markers if args.markers else DEFAULT_MARKERS
    pattern = re.compile("|".join(markers))

    # Split on markers, normalize whitespace, drop empties
    parts = pattern.split(text)
    lines = [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]

    out_path = Path(args.output)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} sentences to {out_path.resolve()}")

if __name__ == "__main__":
    main()

