"""Utils for equate."""

from collections.abc import Callable
from functools import partial
from itertools import chain

from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse, csr_matrix
import numpy as np

from equate.base import to_cost
from equate._dependencies import require
from equate._vector import cosine_similarity

# The default text path is now pure numpy/scipy: char-n-gram TF-IDF
# (equate.featurize) + cosine similarity (equate._vector). scikit-learn and grub are
# optional extras, imported lazily only where still offered (mk_text_to_vect).


def ensure_sparse(matrix):
    if not issparse(matrix):
        matrix = csr_matrix(matrix)
    return matrix


def match_greedily(keys, values, *, score_func=None, minimum_score=-float("infinity")):
    """Generates the best (key, value) matches of each key of keys with a value of
    values. The score_func is used to gauge the strength of a match. If not provided,
    the default score_func is ``difflib.SequenceMatcher(None, a, b).ratio()`` — a
    normalized measure of matching subsequences in [0, 1] (higher is more alike).

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


def mk_text_to_vect(*learn_texts):
    """Legacy grub-backed TF-IDF featurizer (requires the optional ``equate[grub]``
    extra). The default text featurizer is now equate's pure numpy/scipy char-n-gram
    TF-IDF — see :func:`equate.featurize.mk_tfidf`.
    """
    grub = require("grub", extra="grub", purpose="the legacy grub TF-IDF featurizer")

    docs = dict(enumerate(chain(*learn_texts)))
    s = grub.SearchStore(docs)
    return partial(transform_text, transformer=s.tfidf.transform)


def similarity_matrix(keys, values, *, obj_to_vect=None, similarity_func=None):
    """Return a matrix of the similarity of the keys to the values.

    By default the featurizer is learned from ``keys`` and ``values`` using equate's
    pure numpy/scipy char-n-gram TF-IDF (:mod:`equate.featurize`), and similarity is
    cosine (:func:`equate._vector.cosine_similarity`) — no scikit-learn or grub needed.
    Pass ``obj_to_vect`` to use a different batch featurizer (precomputed embeddings, or
    a registered name resolved via ``equate.featurize.resolve_featurizer``), and
    ``similarity_func`` to use a different similarity.

    >>> keys = ['apple pie', 'apple crumble', 'banana split']
    >>> values = ['american pie', 'big apple', 'american girl', 'banana republic']
    >>> m = similarity_matrix(keys, values)
    >>> m.round(2).tolist()
    [[0.32, 0.42, 0.01, 0.0], [0.04, 0.35, 0.01, 0.02], [0.03, 0.04, 0.02, 0.44]]
    """
    keys, values = list(keys), list(values)
    if similarity_func is None:
        similarity_func = cosine_similarity
    if obj_to_vect is None:
        from equate.featurize import resolve_featurizer

        obj_to_vect = resolve_featurizer("tfidf", corpus=keys + values)
    key_vectors, value_vectors = obj_to_vect(keys), obj_to_vect(values)
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


def hungarian_matching(similarity_matrix, *, sense="maximize", cost_matrix=None):
    """
    Hungarian Algorithm (Optimal Matching): Finds the optimal 1:1 assignment,
    minimizing the total cost (equivalently, maximizing the total similarity).

    :param similarity_matrix: A matrix of scores. ``sense='maximize'`` treats them as
        similarities (higher = better, the default); ``sense='minimize'`` treats them
        as costs/distances. The conversion goes through the ``to_cost`` SSOT.
    :param cost_matrix: An explicit cost matrix to use instead (overrides ``sense``).
    :return: An iterator of (row_index, col_index) tuples for matched pairs.
    """
    if cost_matrix is None:
        cost_matrix = to_cost(similarity_matrix, sense=sense)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return zip(row_ind, col_ind)


def maximal_matching(similarity_matrix):
    """
    Maximal Matching in Bipartite Graph: Finds a maximal matching in a bipartite graph.
    See https://www.geeksforgeeks.org/maximum-bipartite-matching/.

    :param similarity_matrix: A sparse matrix of similarities.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    nx = require(
        "networkx", extra="graph", purpose="the maximal_matching graph matcher"
    )

    G = nx.Graph()
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            G.add_edge(f"key_{i}", f"value_{j}", weight=similarity_matrix[i, j])
    matching = nx.max_weight_matching(G, maxcardinality=True)

    def _oriented(u, v):
        # networkx returns each matched edge as an UNORDERED tuple, so the key_ (row) node
        # may be either endpoint. Orient to (row, col) or pairs come out transposed — which
        # silently corrupts rectangular/asymmetric results (and, with candidate masking, can
        # index out of bounds). kuhn_munkres_matching guards the same way.
        if u.startswith("key_"):
            return int(u.split("_")[1]), int(v.split("_")[1])
        return int(v.split("_")[1]), int(u.split("_")[1])

    return (_oriented(u, v) for u, v in matching)


def stable_marriage_matching(similarity_matrix, *, sense="maximize"):
    """
    Stable Marriage Problem (Gale-Shapley Algorithm):
    Solves the stable marriage problem, ensuring a stable matching. Note this
    optimizes *stability* (no blocking pair), not total score, and is
    proposer-optimal / receiver-pessimal — so it can differ from the optimal
    assignment found by ``hungarian_matching``.
    See: https://en.wikipedia.org/wiki/Stable_marriage_problem

    :param similarity_matrix: A matrix of scores (see ``sense``); preferences are
        ranked by the ``to_cost`` distance, so higher similarity is more preferred.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    distance_matrix = np.asarray(to_cost(similarity_matrix, sense=sense), dtype=float)
    n_men, n_women = distance_matrix.shape
    # Each man ranks women by ascending distance (most-preferred first).
    men_prefs = np.argsort(distance_matrix, axis=1)
    # women_rank[m, w] = the rank of man m in woman w's preference list (0 = best).
    women_rank = np.argsort(np.argsort(distance_matrix, axis=0), axis=0)

    next_proposal = [0] * n_men  # index into men_prefs[man] of the next woman to try
    woman_partner: dict = {}  # woman -> currently engaged man
    free_men = list(range(n_men))
    while free_men:
        man = free_men.pop(0)
        if next_proposal[man] >= n_women:
            continue  # exhausted his list (possible when there are fewer women)
        woman = int(men_prefs[man][next_proposal[man]])
        next_proposal[man] += 1
        current = woman_partner.get(woman)
        if current is None:
            woman_partner[woman] = man
        elif women_rank[man, woman] < women_rank[current, woman]:
            woman_partner[woman] = man
            free_men.append(current)  # the jilted man is free again
        else:
            free_men.append(man)  # rejected; he will try his next choice
    return [(man, woman) for woman, man in woman_partner.items()]


def kuhn_munkres_matching(similarity_matrix, *, sense="maximize"):
    """
    Kuhn-Munkres Algorithm: Another implementation of the Hungarian algorithm using
    the `networkx` package (requires the optional ``equate[graph]`` extra).

    :param similarity_matrix: A matrix of scores (see ``sense``); converted to edge
        costs via the ``to_cost`` SSOT.
    :return: List of tuples (row_index, col_index) for matched pairs.
    """
    nx = require("networkx", extra="graph", purpose="the kuhn_munkres_matching matcher")

    cost = to_cost(similarity_matrix, sense=sense)
    G = nx.Graph()
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            G.add_edge(f"key_{i}", f"value_{j}", weight=cost[i, j])
    matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
        G, weight="weight"
    )
    return [
        (int(u.split("_")[1]), int(v.split("_")[1]))
        for u, v in matching.items()
        if u.startswith("key_")
    ]


def match_keys_to_values(
    keys,
    values,
    obj_to_vect: Callable = None,
    similarity_func: Callable = None,
    matcher: Callable = hungarian_matching,
):
    """Match each key to a value via a similarity matrix and a matcher.

    Builds ``similarity_matrix(keys, values, ...)`` (TF-IDF + cosine by default) and
    applies ``matcher`` (optimal Hungarian assignment by default), yielding
    ``(key, value)`` pairs. ``obj_to_vect`` and ``similarity_func`` default to the
    lazily-resolved TF-IDF featurizer and cosine comparator when left as ``None``.
    """
    keys, values = list(keys), list(values)  # need indexable sequences below
    similarity_matrix_ = similarity_matrix(
        keys, values, obj_to_vect=obj_to_vect, similarity_func=similarity_func
    )
    for key_idx, value_idx in matcher(similarity_matrix_):
        yield keys[key_idx], values[value_idx]
