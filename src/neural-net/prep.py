import spacy
import torch
import numpy as np
from gensim.models import KeyedVectors
import argparse
from pathlib import Path
import random
import re

def parse_sentences(inFilePath, outFilePath, label=None, append=False):
    """
    Parse a raw text file into one sentence per line.
    If label is provided, each line is 'label\\t<sentence>'.
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer", config={"punct_chars": ["\n", ".", "!", "?"]})  
    text = Path(inFilePath).read_text(encoding="utf-8")
    nlp.max_length = max(nlp.max_length, len(text) + 10_000)  
    doc = nlp(text)

    mode = "a" if append else "w"
    with open(outFilePath, mode, encoding="utf-8") as f:
        for sent in doc.sents:
            clean = re.sub(r"\s+", " ", sent.text).strip()
            if not clean:
                continue
            prefix = f"{label}\t" if label else ""
            f.write(f"{prefix}{clean}\n")  

def parse_sentences_many(in_paths, labels, outFilePath):
    if len(in_paths) != len(labels):
        raise ValueError("what are you doing, lock in")
    for i, (p, lab) in enumerate(zip(in_paths, labels)):
        parse_sentences(p, outFilePath, label=lab, append=(i > 0))

def shuffle_and_split(inFilePath):
    """
    Shuffles and splits lines into 80/20 train/test.
    Lines may be either raw sentences or 'label\\ttext'.
    """
    random.seed(42)
    text = Path(inFilePath).read_text(encoding="utf-8")
    sentences = [ln.strip() for ln in text.splitlines() if ln.strip()]
    random.shuffle(sentences)
    n_total = len(sentences)
    n_test = int(n_total * 0.2)
    test = sentences[:n_test]
    train = sentences[n_test:]
    return {"train": train, "test": test}

def sentence_vector(sentence, kv):
    """
    Mean-pooled word vectors
    """
    tokens = sentence.split()
    vecs = [kv[w] for w in tokens if w in kv.key_to_index]
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

def encoding(data, embeddings_path):
    """
    Encodes train/test lists.
    """
    def split_label_text(s):
        if "\t" in s:
            lab, txt = s.split("\t", 1)
            return lab, txt
        return "unlabeled", s

    label2id = {}
    def lab_id(lab):
        if lab not in label2id:
            label2id[lab] = len(label2id)
        return label2id[lab]

    kv = KeyedVectors.load(embeddings_path)

    X_train_txt, y_train = [], []
    for s in data["train"]:
        lab, txt = split_label_text(s)
        y_train.append(lab_id(lab))
        X_train_txt.append(txt)

    X_test_txt, y_test = [], []
    for s in data["test"]:
        lab, txt = split_label_text(s)
        y_test.append(lab_id(lab))
        X_test_txt.append(txt)

    encoded_train = [sentence_vector(s, kv) for s in X_train_txt]
    encoded_test  = [sentence_vector(s, kv) for s in X_test_txt]

    return {
        "train": np.stack(encoded_train),
        "test": np.stack(encoded_test),
        "y_train": np.array(y_train, dtype=np.int64),
        "y_test": np.array(y_test, dtype=np.int64),
        "label2id": label2id,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="+", help="List of raw corpus files to parse")
    parser.add_argument("--labels", nargs="+", help="List of labels")
    parser.add_argument("-o", "--outputPath", required=True, help="Where to write parsed lines")
    parser.add_argument("-a", "--parse_sentences", choices=["y","n"], default="n")
    parser.add_argument("-p", "--inputPath", help="Path to labeled or raw file")
    args = parser.parse_args()

    if args.parse_sentences == "y":
        if args.corpora and args.labels:
            parse_sentences_many(args.corpora, args.labels, args.outputPath)
            data_source = args.outputPath
        elif args.inputPath:
            parse_sentences(args.inputPath, args.outputPath, label=args.corpusLabel, append=False)
            data_source = args.outputPath
        else: raise SystemExit("need to give corpora and labels or just -p with -a y")

    else:
        if args.inputPath:
            data_source = args.inputPath
        elif args.corpora:
            data_source = args.corpora[0]
        else: 
            raise SystemExit("need -p or corpora file with -a n")



    data = shuffle_and_split(data_source)
    enc = encoding(data, "./glove_embeddings.data")

    torch.save(
        {
            "train": torch.from_numpy(enc["train"]),
            "test": torch.from_numpy(enc["test"]),
            "y_train": torch.from_numpy(enc["y_train"]),
            "y_test": torch.from_numpy(enc["y_test"]),
            "label2id": enc["label2id"],
        },
        "./encodings.pt",
    )

