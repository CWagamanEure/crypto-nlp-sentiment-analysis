import re



corpus = input("Enter the path to your corpus: ")
if corpus == '':
    corpus = "../data/processed/corpus_reddit.txt"
f = open(corpus)
lines = f.read()



def normalize(lines):
    lines = lines.lower()
    lines = re.sub(r'[^a-z\s]', '', lines)
    lines = re.sub(r'\s+', ' ', lines).strip()
    return lines

def separate_sentences(text):
    word_list = []
    for s in re.split(r'[.!?]+', text):
        s = s.strip()
        word_list.append(s)
    return word_list


def make_tokens(text):
    tokens = []
    for s in separate_sentences(text):
        s = normalize(s)
        if not s:
            continue
        words = s.split()
        tokens.append('<s>')
        tokens.extend(words)
        tokens.append('</s>')
    return tokens

print(make_tokens(lines))

