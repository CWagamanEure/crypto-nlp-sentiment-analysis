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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import polars as pl

from cooccur import *


if __name__ == "__main__":
    set_display()
    parser = create_cooccur_arg_parser(
        "Plot word relationships from a manually computed co-occurrence matrix.")
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of words to display"
    )
    parser.add_argument(
        "--projection",
        choices=["PCA","t-SNE"],
        default="PCA",
        help="Dimensionality reduction algorithm to apply"
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        choices=[2,3],
        default=2,
        help="Number of dimensions/components to retain for visualization"
    )

    args = parser.parse_args()

    mat, vocab, inv_vocab = get_matrix_and_vocabs(args)

    mat = mat.numpy()
    words = list(vocab.keys())[:args.n]
    top_indices = [vocab[w] for w in words]

    # This makes mat non-square, which is fine for exploring similarity between
    # words. (If we wanted to run SVD or spectral methods, then we'd want to do
    # mat = mat[top_indices][:, top_indices] instead.)
    mat = mat[top_indices]

    if args.projection == "PCA":
        pca = PCA(n_components=args.dimensions)
        mat_reduced = pca.fit_transform(mat)
    else:
        tsne = TSNE(n_components=args.dimensions, perplexity=30)
        mat_reduced = tsne.fit_transform(mat)

    fig, ax = plt.subplots(figsize=(10,10))
    if args.dimensions == 2:
        ax.scatter(mat_reduced[:, 0], mat_reduced[:, 1], s=15)
        for i, word in enumerate(words):
            ax.text(mat_reduced[i, 0], mat_reduced[i, 1], word,
                     fontsize=10, alpha=0.7)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mat_reduced[:,0], mat_reduced[:,1], mat_reduced[:,2], s=15)
        for i, word in enumerate(words):
            ax.text(mat_reduced[i,0], mat_reduced[i,1], mat_reduced[i,2],
                    word, fontsize=10, alpha=0.7)

    plt.tight_layout()
    plt.show()
