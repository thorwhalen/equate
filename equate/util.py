"""Utils for equate."""

from typing import Callable
from functools import lru_cache, partial
from itertools import chain

from grub import SearchStore
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse, csr_matrix
import numpy as np


def ensure_sparse(matrix):
    if not issparse(matrix):
        matrix = csr_matrix(matrix)
    return matrix


def match_greedily(keys, values, *, score_func=None, minimum_score=-float('infinity')):
    """Generates the best (key, value) matches of each key of keys with a value of
    values. The score_func is used to gauge the strength of a match. If not provided,
    the default score_func is the ratio of the longest common subsequence to the
    length of the longest of the two strings.

    >>> keys = ['apple', 'banana', 'carrot']
    >>> values = ['car', 'app', 'carob', 'cabana']
    >>> dict(match_greedily(keys, values))
    {'apple': 'app', 'banana': 'cabana', 'carrot': 'carob'}

    """
    if score_func is None:
        from difflib import SequenceMatcher

        score_func = lambda a, b: SequenceMatcher(None, a, b).ratio()

    _values = values.copy()

    for key in keys:
        best_score = 0
        best_value = None
        for value in _values:
            score = score_func(key, value)
            if score > best_score and score >= minimum_score:
                best_score = score
                best_value = value
        if best_value is not None:
            yield key, best_value
            _values.remove(best_value)  # Ensure no endpoint is used more than once


def transform_text(text, transformer):
    if isinstance(text, str):
        return transformer([text])[0]
    else:
        return transformer(text)


# @lru_cache
def mk_text_to_vect(*learn_texts):
    docs = dict(enumerate(chain(*learn_texts)))
    s = SearchStore(docs)
    return partial(transform_text, transformer=s.tfidf.transform)


def similarity_matrix(
    keys, values, *, obj_to_vect=None, similarity_func=cosine_similarity
):
    """Return a matrix of the similarity of the keys to the values.
    By default, the `obj_to_vect` will be learned from the keys and values,
    using the tfidf vectorizer from `grub`.

    You can pass in a different `obj_to_vect` function to use a different vectorizer:
    For example, pre-computed embeddings such as word2vec or fasttext, or openai
    embeddings.

    >>> keys = ['apple pie', 'apple crumble', 'banana split']
    >>> values = ['american pie', 'big apple', 'american girl', 'banana republic']
    >>> m = similarity_matrix(keys, values)
    >>> m.round(2).tolist()
    [[0.54, 0.38, 0.0, 0.0], [0.0, 0.33, 0.0, 0.0], [0.0, 0.0, 0.0, 0.41]]
    """
    obj_to_vect = obj_to_vect or mk_text_to_vect(keys, values)
    key_vectors, value_vectors = map(obj_to_vect, (keys, values))
    return similarity_func(key_vectors, value_vectors)


def greedy_matching(similarity_matrix):
    """
    Greedy Algorithm: Iteratively picks the highest similarity pair and removes the
    matched items from the pool.

    :param similarity_matrix: A sparse matrix of similarities.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    similarity_matrix = ensure_sparse(similarity_matrix)

    for i in range(similarity_matrix.shape[0]):
        if similarity_matrix[i].nnz == 0:
            continue
        j = similarity_matrix[i].argmax()
        yield i, j
        similarity_matrix[:, j] = 0  # Remove this column for subsequent iterations


def hungarian_matching(similarity_matrix, *, cost_matrix=None):
    """
    Hungarian Algorithm (Optimal Matching): Finds the optimal matching, minimizing the
    total cost.
    :param similarity_matrix: A sparse matrix of similarities.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    if cost_matrix is None:
        cost_matrix = similarity_matrix.max() - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return zip(row_ind, col_ind)


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
            G.add_edge(f'key_{i}', f'value_{j}', weight=similarity_matrix[i, j])
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return ((int(u.split('_')[1]), int(v.split('_')[1])) for u, v in matching)


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
            G.add_edge(f'key_{i}', f'value_{j}', weight=-similarity_matrix[i, j])
    matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
        G, weight='weight'
    )
    return [
        (int(u.split('_')[1]), int(v.split('_')[1]))
        for u, v in matching.items()
        if u.startswith('key_')
    ]


def match_keys_to_values(
    keys,
    values,
    obj_to_vect: Callable = None,
    similarity_func: Callable = cosine_similarity,
    matcher: Callable = greedy_matching,
):
    similarity_matrix_ = similarity_matrix(
        keys, values, obj_to_vect=obj_to_vect, similarity_func=similarity_func
    )
    for key_idx, value_idx in matcher(similarity_matrix_):
        yield keys[key_idx], values[value_idx]
