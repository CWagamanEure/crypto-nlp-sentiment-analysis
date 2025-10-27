#!/usr/bin/env python
'''
Interactive program that lets a user interact with the words in a manually
computed co-occurrence matrix.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import argparse
import sys
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
import numpy as np
import torch
import polars as pl

from cooccur import *


if __name__ == "__main__":
    set_display()
    parser = create_cooccur_arg_parser(
        "Interact with terms in a manually computed co-occurrence matrix.")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of most, and least, similar words to display"
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=0,
        help="Dimensions of word embeddings (if this is 0, use PPMI directly)"
    )

    args = parser.parse_args()
    pl.Config.set_tbl_rows(args.n*2+10)

    assert (
        args.dimensions <= args.max_vocab
    ), f"Can't reduce dimensionality lower than {args.max_vocab}."

    mat, vocab, inv_vocab = get_matrix_and_vocabs(args)

    units = "words"
    if args.dimensions > 0:
        if args.verbose:
            print(f"Computing {args.dimensions}-dim embeddings...")
        units = "embeds"
        mat = mat.numpy()
        words = list(vocab.keys())[:args.max_vocab]
        top_indices = [vocab[w] for w in words]
        mat = mat[top_indices][:, top_indices]

        svd = TruncatedSVD(n_components=args.dimensions, n_iter=10)
        mat = torch.from_numpy(svd.fit_transform(mat))
        # Apply customary sqrt(Σ) weighting (to give more weight to dimensions
        # with more signal).
        mat *= torch.from_numpy(np.sqrt(svd.singular_values_))

    word = prompt_word(vocab)
    while word != "done":
        print(f"\nMost/least similar {units} to {word.upper()}:")
        sims = get_similar_words(word, mat, vocab, inv_vocab, args.n)
        df = pl.DataFrame(sims, schema=[units[:-1],'similarity'], orient='row')
        df_print = pl.concat([
            df[:len(df)//2],
            pl.DataFrame({units[:-1]:'--------', 'similarity':None}),
            df[len(df)//2:],
        ], how="vertical")
        print(df_print)
        word = prompt_word(vocab)
