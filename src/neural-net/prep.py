import spacy
import torch
import numpy as np
from gensim.models import KeyedVectors
import argparse
from pathlib import Path
import random
import json

def parse_sentences(inFilePath, outFilePath):
    '''
    Goes through a text file given by 'inFilePath'
    saves to an output file all text parsed by sentence.
    '''
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    text = Path(inFilePath).read_text(encoding="utf-8")
    nlp.max_length = len(text) + 10_000
    doc = nlp(text)
    with open(outFilePath, "a", encoding="utf-8") as f:
        for sent in doc.sents:
            f.write(f"{sent.text}")

def shuffle_and_split(inFilePath):
    '''
    Shuffles and splits the sentences from the input file
    into train and test sets
    Returns dict 
    '''
    random.seed(42)
    text = Path(inFilePath).read_text(encoding="utf-8")
    sentences = [sentence.strip() for sentence in text.splitlines()]
    random.shuffle(sentences)
    n_total = len(sentences)
    n_test = int(n_total*0.2)
    test = sentences[:n_test]
    train = sentences[n_test:]
    return {
        "train": train,
        "test": test
    }

def sentence_vector(sentence, kv):
    '''
    takes in a sentence and KeyedVectors embeddings
    returns mean of word vectors for that sentence
    '''
    tokens = sentence.split() 
    sentence_vector = [kv[word] for word in tokens if word in kv.key_to_index]
    if not sentence_vector:
        return np.zeros(kv.vector_size, dtype=np.float32)
    return np.mean(sentence_vector, axis=0)

def encoding(data, embeddings_path):
    train = data["train"]
    test = data["test"]
    embeds = KeyedVectors.load(embeddings_path)
    encoded_train = [sentence_vector(sentence, embeds) for sentence in train]
    encoded_test = [sentence_vector(sentence, embeds) for sentence in test]
    return {
        "train": encoded_train,
        "test": encoded_test
    } 


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--inputPath")
    parser.add_argument("-o", "--outputPath")
    parser.add_argument("-a", "--parse_sentences")
    args = parser.parse_args()
    if args.parse_sentences == "y":
        parse_sentences(args.inputPath, args.outputPath)
    data = shuffle_and_split(args.outputPath)
    encodings = encoding(data, "../../data/embeddings/glove_embeddings.data")
    torch.save(encodings, "../../data/processed/neural-stuff/encodings.pt")





