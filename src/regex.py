import re

with open("../data/processed/corpus_reddit.txt", "r", encoding="utf-8") as f: text = f.read()

while (True):
    regex = input("Enter regex: ")
    if regex == "done":
        break
    if not regex:
        print("None found")
        continue
    try:
        pat = re.compile(regex, re.I)
    except re.error as e:
        print("error")
        continue

    if pat.match("") is not None:
        print("skipped")
        continue

    hits = 0
    for i, line in enumerate(text.splitlines(), start=1):
        m = pat.search(line)
        if not m:
            continue
        hits += 1


        a, b = m.span()
        pre = line[:a]
        mid = line[a:b]
        post = line[b:]
        print(f"{i:6} | {pre}\x1b[1;31m{mid}\x1b[0m{post}\n")

    total_matches = sum(1 for _ in pat.finditer(text))
    print(f"Found {hits} matching lines; {total_matches} total matches.\n"+ "-"*60)

