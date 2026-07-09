"""Tools to complete missing data from other sources of the same data.

Merging/joining tables is very common instance, yet only a small part of what is
possible, and often needed. Without moving away from tabular data:

- Find the columns to match (join keys) by comparing how well the values of the column
match.
- Comparing the values of the columns with something more flexible than hard equality.
For example, correlation, similarity, etc.
- Find near duplicate columns
- Find rows to align, based on flexible comparison of fuzzily matched cells

.. note::
   This module is a stale early duplicate of ``equate.util`` and is slated to be
   rewritten as a thin facade over the redesigned stages (roadmap #13). Its ``grub``
   import is lazy (behind the optional ``equate[grub]`` extra) so importing the module
   does not fail on a default numpy/scipy-only install.
"""

from functools import partial
from itertools import chain
from scipy.optimize import linear_sum_assignment
import numpy as np

from equate._dependencies import require
from equate._vector import cosine_similarity


def transform_text(text, transformer):
    if isinstance(text, str):
        return transformer([text])[0]
    else:
        return transformer(text)


def mk_text_to_vect(*learn_texts):
    grub = require("grub", extra="grub", purpose="the legacy grub TF-IDF featurizer")

    docs = dict(enumerate(chain(*learn_texts)))
    s = grub.SearchStore(docs)
    return partial(transform_text, transformer=s.tfidf.transform)


def similarity_matrix(
    keys, values, *, text_to_vect=None, similarity_func=cosine_similarity
):
    """Return a matrix of the similarity of the keys to the values.
    By default, the `text_to_vect` will be learned from the keys and values,
    using the tfidf vectorizer from `grub`.
    You can pass in a different `text_to_vect` function to use a different vectorizer:
    For example, pre-computed embeddings such as word2vec or fasttext, or openai
    embeddings.

    >>> keys = ['apple pie', 'apple crumble', 'banana split']
    >>> values = ['american pie', 'big apple', 'american girl', 'banana republic']
    >>> m = similarity_matrix(keys, values)
    >>> m.round(2).tolist()
    [[0.54, 0.38, 0.0, 0.0], [0.0, 0.33, 0.0, 0.0], [0.0, 0.0, 0.0, 0.41]]
    """
    text_to_vect = text_to_vect or mk_text_to_vect(keys, values)
    key_vectors, value_vectors = map(text_to_vect, (keys, values))
    return similarity_func(key_vectors, value_vectors)


def greedy_matching(similarity_matrix):
    """
    Greedy Algorithm: Iteratively picks the highest similarity pair and removes the
    matched items from the pool.

    :param similarity_matrix: A sparse matrix of similarities.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    matches = []
    for i in range(similarity_matrix.shape[0]):
        if similarity_matrix[i].nnz == 0:
            continue
        j = similarity_matrix[i].argmax()
        matches.append((i, j))
        similarity_matrix[:, j] = 0  # Remove this column for subsequent iterations
    return matches


def hungarian_matching(similarity_matrix):
    """
    Hungarian Algorithm (Optimal Matching): Finds the optimal matching, minimizing the
    total cost.
    :param similarity_matrix: A sparse matrix of similarities.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    cost_matrix = similarity_matrix.max() - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))


def maximal_matching(similarity_matrix):
    """
    Maximal Matching in Bipartite Graph: Finds a maximal matching in a bipartite graph.
    See https://www.geeksforgeeks.org/maximum-bipartite-matching/.

    :param similarity_matrix: A sparse matrix of similarities.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    import networkx as nx

    G = nx.Graph()
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            G.add_edge(f"key_{i}", f"value_{j}", weight=similarity_matrix[i, j])
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return [(int(u.split("_")[1]), int(v.split("_")[1])) for u, v in matching]


def stable_marriage_matching(similarity_matrix):
    """
    Stable Marriage Problem (Gale-Shapley Algorithm):
    Solves the stable marriage problem, ensuring a stable matching.
    See: https://en.wikipedia.org/wiki/Stable_marriage_problem

    :param similarity_matrix: A sparse matrix of similarities.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    distance_matrix = 1 - similarity_matrix
    men_prefs = np.argsort(distance_matrix, axis=1)
    women_prefs = np.argsort(distance_matrix, axis=0).T
    couples = {}
    while len(couples) < similarity_matrix.shape[0]:
        for man in range(similarity_matrix.shape[0]):
            if man not in couples:
                woman = men_prefs[man][0]
                if woman not in couples.values():
                    couples[man] = woman
                else:
                    current_man = list(couples.keys())[
                        list(couples.values()).index(woman)
                    ]
                    if list(women_prefs[woman]).index(man) < list(
                        women_prefs[woman]
                    ).index(current_man):
                        del couples[current_man]
                        couples[man] = woman
        for couple in couples.items():
            men_prefs[couple[0]] = men_prefs[couple[0]][1:]
    return list(couples.items())


def kuhn_munkres_matching(similarity_matrix):
    """
    Kuhn-Munkres Algorithm: Another implementation of the Hungarian algorithm using
    the `networkx` package.

    :param similarity_matrix: A sparse matrix of similarities.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    import networkx as nx

    G = nx.Graph()
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            G.add_edge(f"key_{i}", f"value_{j}", weight=-similarity_matrix[i, j])
    matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
        G, weight="weight"
    )
    return [
        (int(u.split("_")[1]), int(v.split("_")[1]))
        for u, v in matching.items()
        if u.startswith("key_")
    ]
