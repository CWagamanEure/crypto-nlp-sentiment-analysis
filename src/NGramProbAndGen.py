import re
import sys
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("corpus", help="path to corpus file")
parser.add_argument("--k", type=float, default=0.0, help="add da k smoothing")
args = parser.parse_args()
with open(args.corpus) as f:
    lines =f.read()
K = args.k

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
        tokens.extend(['<s>', '<s>'])
        tokens.extend(words)
        tokens.append('</s>')
    return tokens

def unigram(tokens):
    count_dict = {}
    for token in tokens:
        count_dict[token] = count_dict.get(token, 0) + 1
    return count_dict

def bigram(tokens):
    count_dict = {}
    for i in range(len(tokens) -1):
        gram = (tokens[i], tokens[i+1])
        count_dict[gram] = count_dict.get(gram, 0) + 1
    return count_dict

def trigram(tokens):
    count_dict = {}
    for i in range(len(tokens) -2):
        gram = (tokens[i], tokens[i+1], tokens[i+2])
        count_dict[gram] = count_dict.get(gram, 0) + 1
    return count_dict


class Prob:
    def __init__(self, uni_sentence, bi_sentence, tri_sentence, uni_count_dict, bi_count_dict, tri_count_dict):    
        self.sentence = ""
        self.uni_count_size = 0
        self.bi_count_size = 0
        self.tri_count_size = 0
        self.uni_count_dict = uni_count_dict
        self.bi_count_dict = bi_count_dict
        self.tri_count_dict = tri_count_dict
        self.uni_sentence = uni_sentence
        self.bi_sentence = bi_sentence
        self.tri_sentence = tri_sentence
        self.V = 0


    def compute_size_corpus(self):
        self.uni_count_size = sum(self.uni_count_dict.values())
        self.bi_count_size = sum(self.bi_count_dict.values())
        self.tri_count_size = sum(self.tri_count_dict.values())

    def comput_v(self):
        self.V = len([w for w in self.uni_count_dict if w != "<s>"])

    def compute_unigram_prob(self, k=0):
        self.compute_size_corpus()
        cum_prob = 1 
        self.comput_v() 
        for word in self.uni_sentence.split():
            prob = (self.uni_count_dict.get(word, 0)+k) / (self.uni_count_size + k * self.V) 
            cum_prob *= prob
            if cum_prob == 0.0: return 0.0
        return cum_prob

    def compute_bigram_prob(self, k=0):
        self.compute_size_corpus()
        self.comput_v()
        if len(self.tri_sentence.split()) < 2: return 0
        cum_prob = 1
        sentence = self.bi_sentence.split()
        for i in range(len(sentence) -1):
            previous, next = sentence[i], sentence[i+1]
            gram = (previous, next)
            row_total = sum(self.bi_count_dict.get((previous, w), 0) for w in self.uni_count_dict if w!="<s>")
            denom = row_total + k * self.V 
            if denom == 0: return 0.0
            prob = (self.bi_count_dict.get((previous, next), 0) + k) / denom
            
            cum_prob *= prob
            if cum_prob == 0.0: return 0.0
        return cum_prob

    def compute_trigram_prob(self, k=0):
        self.comput_v()
        if len(self.tri_sentence.split()) < 3: return 0
        cum_prob = 1
        sentence = self.tri_sentence.split()
        for i in range(len(sentence) -2):
            g1, g2, g3 = sentence[i], sentence[i+1], sentence[i+2]
            gram = (g1, g2, g3)

            row_total = sum(self.tri_count_dict.get((g1, g2, w), 0) for w in self.uni_count_dict if w!="<s>")
            denom = row_total + k * self.V
            if denom == 0: return 0
            prob = (self.tri_count_dict.get((g1, g2, g3), 0) + k) / denom 
            cum_prob *= prob
        return cum_prob

    def generate_random_unigram_sentence(self, max_length=9, k=0):
        sentence = ["<s>"]
        vocab = [w for w in self.uni_count_dict if w != "<s>"]
        weights = [self.uni_count_dict.get(w, 0)+1 for w in vocab]
        for i in range(0, max_length):
            word = random.choices(vocab, weights=weights, k=1)[0]
            sentence.append(word)
            if i == max_length-1 and sentence[-1] != "</s>": sentence.append("</s>")
            if sentence[-1] == "</s>":
                break
        barf = " "
        self.uni_sentence = barf.join(sentence)

    def generate_random_bigram_sentence(self, max_length=9, k=0):
        sentence = ["<s>" ]
        vocab = [w for w in self.uni_count_dict.keys() if w!="<s>"]
        for i in range(0, max_length):
            last_word = sentence[-1]
            
            #word_pairs = [(p,w) for (p, w) in self.bi_count_dict.keys() if p == last_word]
            #candidates = [w for (_,w) in word_pairs if w != "<s>"]
            candidates = vocab
            weights = [self.bi_count_dict.get((last_word, w),0)+1 for w in candidates]

            word = random.choices(candidates, weights=weights, k=1)[0]
            sentence.append(word)
            if i == max_length-1 and sentence[-1] != "</s>": sentence.append("</s>")
            if sentence[-1] == "</s>": break
        barf = " "
        self.bi_sentence = barf.join(sentence)


    def generate_random_trigram_sentence(self, max_length=9, k=0):
        sentence = ["<s>", "<s>"]
        vocab = [w for w in self.uni_count_dict.keys() if w!="<s>"]
        for i in range(0, max_length):
            last_gram = (sentence[-2], sentence[-1])

            #word_pairs = [(p,w,m) for (p,w, m) in self.tri_count_dict.keys() if (p, w) == last_gram]
            #candidates = [m for (_,_,m) in word_pairs if m != "<s>"]
            candidates = vocab
            weights = [self.tri_count_dict.get((sentence[-2], sentence[-1], m), 0) + k for m in candidates]

            word = random.choices(candidates, weights=weights, k=1)[0]
            sentence.append(word)
            if i == max_length-1 and sentence[-1] != "</s>": sentence.append("</s>")
            if sentence[-1] == "</s>": break
        barf = " "
        self.tri_sentence = barf.join(sentence)

tokens = make_tokens(lines)
uni_count = unigram(tokens)
bi_count = bigram(tokens)
tri_count = trigram(tokens)
while True:
    sentence = input("Enter a sentence (or 'random', or 'done'): ").lower()
    if sentence == "done": break
    if sentence == "random":
        prob = Prob(sentence, sentence, sentence, uni_count, bi_count, tri_count)  
        prob.generate_random_unigram_sentence(k=K)
        prob.generate_random_bigram_sentence(k=K)
        prob.generate_random_trigram_sentence(k=K)
        print(f"Random unigram sentence: {prob.uni_sentence}")
        print(f"Random bigram sentence: {prob.bi_sentence}")
        print(f"Random trigram sentence: {prob.tri_sentence}")
        unigram_prob= prob.compute_unigram_prob(k=K)
        bigram_prob = prob.compute_bigram_prob(k=K)
        trigram_prob = prob.compute_trigram_prob(k=K)
        print(f"Unigram prob: {unigram_prob}")
        print(f"Bigram prob: {bigram_prob}")
        print(f"Trigram prob: {trigram_prob}")
        continue
    user = normalize(sentence)
    prob = Prob(user, user, user, uni_count, bi_count, tri_count)
    unigram_prob= prob.compute_unigram_prob(k=K)
    bigram_prob = prob.compute_bigram_prob(k=K)
    trigram_prob = prob.compute_trigram_prob(k=K)
    print(f"Unigram prob: {unigram_prob}")
    print(f"Bigram prob: {bigram_prob}")
    print(f"Trigram prob: {trigram_prob}")




