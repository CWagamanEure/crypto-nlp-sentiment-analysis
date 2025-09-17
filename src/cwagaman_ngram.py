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


class Prob:
    def __init__(self, sentence, uni_count_dict, bi_count_dict):    
        self.sentence = sentence
        self.uni_count_size = 0
        self.bi_count_size = 0
        self.uni_count_dict = uni_count_dict
        self.bi_count_dict = bi_count_dict

    def compute_size_corpus(self):
        for index in self.uni_count_dict:
            self.uni_count_size += self.uni_count_dict[index]
        for index in self.bi_count_dict:
            self.bi_count_size += self.bi_count_dict[index]



    def compute_unigram_prob(self):
        self.compute_size_corpus()
        cum_prob = 1 
        for word in self.sentence.split():
            if word in self.uni_count_dict:
                prob = self.uni_count_dict[word] / self.uni_count_size 
                cum_prob *= prob
            else: 
                cum_prob=0
        return cum_prob

    def compute_bigram_prob(self):
        self.compute_size_corpus()
        cum_prob = 1
        sentence = self.sentence.split()
        sentence_gram_list = []
        for i in range(len(sentence) - 1):
            gram = (sentence[i], sentence[i+1])
            sentence_gram_list.append(gram)
        for gram in sentence_gram_list:
            if gram in self.bi_count_dict:
                prob = self.bi_count_dict[gram] / self.bi_count_size
                cum_prob *= prob
            else:
                cum_prob=0
        return cum_prob

tokens = make_tokens(lines)
uni_count = unigram(tokens)
bi_count = bigram(tokens)
while True:
    sentence = input("Enter a sentence (or 'random', or 'done'): ").lower()
    prob = Prob(sentence, uni_count, bi_count)  
    if sentence == "done": break
    unigram_prob= prob.compute_unigram_prob()
    bigram_prob = prob.compute_bigram_prob()
    print(f"Unigram prob: {unigram_prob}")
    print(f"Bigram prob: {bigram_prob}")



