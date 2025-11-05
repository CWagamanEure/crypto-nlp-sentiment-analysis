with open("../data/processed/neural-stuff/reduced-corpus.txt") as f:
    line_count = sum(1 for line in f)
print(line_count)
