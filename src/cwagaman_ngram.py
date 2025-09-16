corpus = input("Enter the path to your corpus: ")
if corpus == '':
    corpus = "../data/processed/corpus_reddit.txt"
f = open(corpus)
lines = f.readlines()


ABBR = [
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
    "etc", "e.g", "i.e", "al", "fig", "cf",
    "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec",
    "mon","tue","wed","thu","fri","sat","sun",
    "u.s","u.k","u.n"
]

print(lines)
