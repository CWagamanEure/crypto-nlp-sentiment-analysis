#!/usr/bin/env python
'''
Core functionality to compute co-occurrence matrices and PPMI matrices for an
arbitrary corpus. The functions in this module can be used in conjunction with
user-facing programs interact_cooccur.py and visualize_cooccur.py to illuminate
word similarity/relatedness for that corpus.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List
import random
from itertools import product
import argparse

import torch
from tqdm import tqdm
import spacy
# Need to "python -m spacy download en_core_web_sm" once to make this work.


def build_cooccurrence_matrix(
    text: str,
    max_vocab: int,   # maximum number of tokens to keep
    window_size: int, # size of co-occurrence window
    normalize: bool = True,  # make rows sum to 1?
    verbose: bool = False,   # print verbose output during computation?
    use_saved: bool = False, # reuse previously saved tensor if exists, & save?
    save_name: str = None,   # if use_saved, filename to store (w/ .pt ext)
    nlp = spacy.load("en_core_web_sm")
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Given a long text string, return a co-occurrence matrix of terms that occur
    more often than a given frequency.

    Returns:
    The dense matrix (as a PyTorch tensor) and a vocabulary dict mapping token
    strings to numeric ids.
    """
    sn = save_name + ".pt" if not save_name.endswith(".pt") else save_name
    if use_saved  and  Path(sn).is_file():
        if verbose: print(f"Returning previously saved matrix/vocab ({sn}).")
        saved = torch.load(sn)
        pair_freq, vocab = saved['pair_freq'], saved['vocab']

    else:
        # Tokenize.
        if verbose: print("Tokenizing...", end="", flush=True)
        token_stream = tokenize(text, nlp)
        print(f"{len(token_stream):,} tokens.")

        # Count 'em up.
        if verbose: print("Counting...", end="", flush=True)
        tok_freq = Counter(token_stream)
        print(f"{len(tok_freq):,} types.")

        # Filter out too-rare, and too common, tokens.
        if verbose: print("Building vocab...", end="", flush=True)
        filtered_tokens = [
            t for t, _ in tok_freq.most_common(max_vocab)
            if t not in nlp.Defaults.stop_words
        ]

        if verbose: print("Assigning vocab indices...")
        # Assign indices to vocab words.
        vocab = { token: idx for idx, token in enumerate(filtered_tokens) }

        # Count up all co-occurring word-pairs. (Tuples of token ids.)
        if verbose: print("Counting word pairs...", flush=True)
        pair_freq = Counter()
        w = window_size
        center_iter = range(0 + w, len(token_stream) - w)
        if verbose: center_iter = tqdm(center_iter)
        for center in center_iter:
            tokens_in_range = (
                token_stream[center-w:center] +
                token_stream[center+1:center+w+1]
            )
            center_tok = token_stream[center]
            for tok in tokens_in_range:
                if center_tok in vocab and tok in vocab:
                    pair_freq[(vocab[center_tok], vocab[tok])] += 1
                    pair_freq[(vocab[tok], vocab[center_tok])] += 1
        if verbose: print(f"...{len(pair_freq):,} pairs.", flush=True)

        if verbose: print(f"Saving sparse matrix for future use ({sn})...")
        torch.save({'pair_freq':pair_freq, 'vocab':vocab}, sn)

    # To do full vector math (even for things like normalizing rows) we need to
    # make a dense tensor. PyTorch doesn't do those operations well with sparse
    # matrices. From this point forward, we'll always be dense.
    if verbose: print("Creating tensor...")
    co_mat = to_tensor(pair_freq)
    if normalize:
        if verbose: print("Normalizing matrix...")
        co_mat = normalize_cooccurrence_matrix(co_mat)
    return co_mat, vocab


def build_ppmi_matrix(
    text: str,
    max_vocab: int,   # maximum number of tokens to keep
    window_size: int, # size of co-occurrence window
    verbose: bool = False,   # print verbose output during computation?
    use_saved: bool = False, # reuse previously saved tensor if exists, & save?
    save_name: str = None,   # if use_saved, filename to store (w/ .pt ext)
    nlp = spacy.load("en_core_web_sm"),
    eps: float = 1e-10       # small epsilon to smooth dividing by zero
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Given a long text string, return a PPMI matrix of the terms that co-occur
    together.

    Returns:
    The dense matrix (as a PyTorch tensor) and a vocabulary dict mapping token
    strings to numeric ids.
    """
    co_mat, vocab = build_cooccurrence_matrix(text, max_vocab, window_size,
        normalize=False, verbose=verbose, use_saved=use_saved,
        save_name=save_name, nlp=nlp)
    ppmi_mat = torch.empty_like(co_mat)

    # Each entry (r,c) in the PPMI matrix is a conditional probability: the
    # probability of seeing word c in a context where word r appears.
    total_count = co_mat.sum(dim=(0,1))
    for r, c in product(range(co_mat.size(0)), range(co_mat.size(1))):
        marg_prob_r = co_mat[r,:].sum() / total_count
        marg_prob_c = co_mat[:,c].sum() / total_count
        joint_prob = co_mat[r,c] / total_count
        cond_prob = joint_prob / (marg_prob_r * marg_prob_c + eps)
        ppmi_mat[r][c] = max(0.0, torch.log(cond_prob).item())
    return ppmi_mat, vocab


def normalize_cooccurrence_matrix(
    co_mat: torch.Tensor    # This must be a dense tensor, not sparse
) -> torch.Tensor:
    row_sums = co_mat.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0   # defensively avoid divide-by-zero
    return co_mat / row_sums


def tokenize(text: str, nlp=spacy.load("en_core_web_sm")) -> List[str]:
    """
    Given a long text string, return a list of the lowercase tokens it is
    comprised of, with punctuation and whitespace removed.
    """
    nlp.max_length = len(text)    # Accommodate large corpora
    return [
        t.text.lower()
        for t in nlp.tokenizer(text)  # way faster than calling nlp directly
        if not t.is_punct and not t.is_space and t.is_alpha
    ]


def to_tensor(
    pair_freq: Dict[Tuple[int, int], int],
) -> torch.Tensor:
    """
    Given a dictionary (e.g., a Counter) mapping index pairs to counts, return
    the corresponding PyTorch dense tensor with this info.
    """
    max_index = max(max(i, j) for (i, j) in pair_freq.keys())
    mat = torch.zeros((max_index + 1, max_index + 1), dtype=torch.float32)

    for (i, j), count in pair_freq.items():
        mat[i, j] = count

    return mat


def prompt_word(
    vocab: Dict[str,int],
    num_ex: int = 10    # Number of examples to show each time
) -> str:
    num_ex = min(num_ex, len(vocab))   # Tolerate tiny vocabs
    w = input("Enter a word (or 'ex' for examples, or 'done'): ")
    while w not in ['ex','done'] and w not in vocab:
        print(f"No such word '{w}' in corpus!")
        w = input("Enter a word (or 'ex' for examples, or 'done'): ")
    if w == 'ex':
        print("Here are some common words in the corpus:")
        examples = sorted(random.sample(list(vocab.keys()), k=num_ex))
        for i, example in enumerate(examples):
            print(f"({i+1:2>d}) {example}")
        ex_choice = input("Choose one: ")
        try:
            return examples[int(ex_choice)-1]
        except ValueError:
            if ex_choice in vocab:
                return ex_choice
            elif ex_choice == 'done':
                return 'done'
            else:
                print(f"No such word '{ex_choice}' in corpus!")
                return prompt_word(vocab, num_examples_to_show)
    return w


def get_similar_words(
    word: str,                # a word in the vocab
    co_mat: torch.Tensor,     # dense, normalized, indexed by word id in vocab
    vocab: Dict[str,int],     # words to word ids
    inv_vocab: Dict[int,str], # word ids to words
    k: int = 10,              # number of words to return
    omit_zeros: bool = True   # don't return anything with neg or 0 similarity
) -> List[Tuple[str,float]]:
    """
    Return the top-k most, and top-k least, similar words to the given word,
    based on cosine similarity of vectors in the matrix (which could be a
    co-occurrence matrix, PPMI, etc.)
    """
    k = min(k, len(vocab))   # Tolerate tiny vocabs

    assert word in vocab
    this_words_vector = co_mat[vocab[word]].unsqueeze(0)    # (1,|V|)
    sims = torch.cosine_similarity(this_words_vector, co_mat, dim=1)  # (|V|,)
    sims[vocab[word]] = -1    # don't include ourself of course!

    vals_top, idxs_top = torch.topk(sims, k=k)
    vals_bot, idxs_bot = torch.topk(sims, k=k+1, largest=False)
    vals_bot = vals_bot[1:]   # don't include ourself of course!
    idxs_bot = idxs_bot[1:]
    vals = torch.concat([vals_top, reversed(vals_bot)])
    idxs = torch.concat([idxs_top, reversed(idxs_bot)])
    return [
        (inv_vocab[idx.item()], val.item())
        for idx, val in zip(idxs, vals)
    ]


def create_cooccur_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "inputfile",
        type=str,
        help="Path to input file from which to compute co-occurrence matrix"
    )
    parser.add_argument(
        "--matrix-type",
        choices=["counts","ppmi"],
        default="counts",
        help="Which kind of co-occurrence matrix to build"
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=2000,
        help="Maximum number of vocab words to keep (based on frequency)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Size of window, to left and right of a token (default +/-3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during computation"
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Use previously cached matrix/vocab if available"
    )
    return parser


def set_display():
    from utils import set_torch_width
    set_torch_width()
    torch.set_printoptions(precision=2,sci_mode=False)


def get_matrix_and_vocabs(args, verbose=True):
    if not Path(args.inputfile).is_file():
        sys.exit(f"No such file {args.inputfile}.")
    with open(args.inputfile, "r", encoding="utf-8") as f:
        text = f.read()

    if args.matrix_type == "ppmi":
        build_func = build_ppmi_matrix
    else:
        build_func = build_cooccurrence_matrix
    mat, vocab = build_func(text, args.max_vocab, args.window_size,
        verbose=args.verbose, use_saved=args.cached,
        save_name=f"{Path(args.inputfile).name.replace('.txt','')}-"
            f"{args.max_vocab}-{args.window_size}-{args.matrix_type}")
    inv_vocab = { v:k for k,v in vocab.items() }
    num_elements = mat.shape[0] * mat.shape[1]
    num_non_zeros = torch.count_nonzero(mat).item()
    if verbose:
        print(f"The matrix is size {mat.shape[0]}x{mat.shape[1]} "
            f"with {num_non_zeros} entries "
            f"({num_non_zeros / num_elements * 100:.1f}% full)")
    return mat, vocab, inv_vocab
