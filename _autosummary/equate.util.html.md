# equate.util

Utils for equate.

### Functions

| `ensure_sparse`(matrix)                                                                           |                                                                                                                                                       |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`greedy_matching`](#equate.util.greedy_matching)(similarity_matrix)               | Greedy Algorithm: Iteratively picks the highest similarity pair and removes the matched items from the pool.                                          |
| [`hungarian_matching`](#equate.util.hungarian_matching)(similarity_matrix, \*[, ...]) | Hungarian Algorithm (Optimal Matching): Finds the optimal 1:1 assignment, minimizing the total cost (equivalently, maximizing the total similarity).  |
| [`kuhn_munkres_matching`](#equate.util.kuhn_munkres_matching)(similarity_matrix, \*)     | Kuhn-Munkres Algorithm: Another implementation of the Hungarian algorithm using the `networkx` package (requires the optional `equate[graph]` extra). |
| [`match_greedily`](#equate.util.match_greedily)(keys, values, \*[, ...])          | Generates the best (key, value) matches of each key of keys with a value of values.                                                                   |
| [`match_keys_to_values`](#equate.util.match_keys_to_values)(keys, values[, ...])        | Match each key to a value via a similarity matrix and a matcher.                                                                                      |
| [`maximal_matching`](#equate.util.maximal_matching)(similarity_matrix)              | Maximal Matching in Bipartite Graph: Finds a maximal matching in a bipartite graph.                                                                   |
| [`mk_text_to_vect`](#equate.util.mk_text_to_vect)(\*learn_texts)                   | Legacy grub-backed TF-IDF featurizer (requires the optional `equate[grub]` extra).                                                                    |
| [`similarity_matrix`](#equate.util.similarity_matrix)(keys, values, \*[, ...])       | Return a matrix of the similarity of the keys to the values.                                                                                          |
| [`stable_marriage_matching`](#equate.util.stable_marriage_matching)(similarity_matrix, \*)  | Stable Marriage Problem (Gale-Shapley Algorithm): Solves the stable marriage problem, ensuring a stable matching.                                     |
| `transform_text`(text, transformer)                                                               |                                                                                                                                                       |

### equate.util.greedy_matching(similarity_matrix)

Greedy Algorithm: Iteratively picks the highest similarity pair and removes the
matched items from the pool.

* **Parameters:**
  **similarity_matrix** – A sparse matrix of similarities.
* **Returns:**
  List of tuples (row_index, col_index) for matched pairs.

### equate.util.hungarian_matching(similarity_matrix, , sense='maximize', cost_matrix=None)

Hungarian Algorithm (Optimal Matching): Finds the optimal 1:1 assignment,
minimizing the total cost (equivalently, maximizing the total similarity).

* **Parameters:**
  * **similarity_matrix** – A matrix of scores. `sense='maximize'` treats them as
    similarities (higher = better, the default); `sense='minimize'` treats them
    as costs/distances. The conversion goes through the `to_cost` SSOT.
  * **cost_matrix** – An explicit cost matrix to use instead (overrides `sense`).
* **Returns:**
  An iterator of (row_index, col_index) tuples for matched pairs.

### equate.util.kuhn_munkres_matching(similarity_matrix, , sense='maximize')

Kuhn-Munkres Algorithm: Another implementation of the Hungarian algorithm using
the `networkx` package (requires the optional `equate[graph]` extra).

* **Parameters:**
  **similarity_matrix** – A matrix of scores (see `sense`); converted to edge
  costs via the `to_cost` SSOT.
* **Returns:**
  List of tuples (row_index, col_index) for matched pairs.

### equate.util.match_greedily(keys, values, , score_func=None, minimum_score=-inf)

Generates the best (key, value) matches of each key of keys with a value of
values. The score_func is used to gauge the strength of a match. If not provided,
the default score_func is `difflib.SequenceMatcher(None, a, b).ratio()` — a
normalized measure of matching subsequences in [0, 1] (higher is more alike).

```pycon
>>> keys = ['apple', 'banana', 'carrot']
>>> values = ['car', 'app', 'carob', 'cabana']
>>> dict(match_greedily(keys, values))
{'apple': 'app', 'banana': 'cabana', 'carrot': 'carob'}
```

### equate.util.match_keys_to_values(keys, values, obj_to_vect=None, similarity_func=None, matcher=<function hungarian_matching>)

Match each key to a value via a similarity matrix and a matcher.

Builds `similarity_matrix(keys, values, ...)` (TF-IDF + cosine by default) and
applies `matcher` (optimal Hungarian assignment by default), yielding
`(key, value)` pairs. `obj_to_vect` and `similarity_func` default to the
lazily-resolved TF-IDF featurizer and cosine comparator when left as `None`.

### equate.util.maximal_matching(similarity_matrix)

Maximal Matching in Bipartite Graph: Finds a maximal matching in a bipartite graph.
See [https://www.geeksforgeeks.org/maximum-bipartite-matching/](https://www.geeksforgeeks.org/maximum-bipartite-matching/).

Accepts a dense array, a **sparse (blocked)** matrix, or a
[`ScoreMatrix`](equate.base.html.md#equate.base.ScoreMatrix). Only *candidate* cells become graph edges, so a
structurally-absent cell is not an edge at all: it can be neither preferred nor
assigned, and max-cardinality is maximized over real candidates (D11). Reading the
matrix cell-by-cell instead would turn every hole into a real weight-`0.0` edge —
which silently beats any negative similarity.

* **Parameters:**
  **similarity_matrix** – A dense/sparse matrix of similarities, or a `ScoreMatrix`.
* **Returns:**
  Iterator of tuples (row_index, col_index) for matched pairs.

### equate.util.mk_text_to_vect(\*learn_texts)

Legacy grub-backed TF-IDF featurizer (requires the optional `equate[grub]`
extra). The default text featurizer is now equate’s pure numpy/scipy char-n-gram
TF-IDF — see [`equate.featurize.mk_tfidf()`](equate.featurize.html.md#equate.featurize.mk_tfidf).

### equate.util.similarity_matrix(keys, values, , obj_to_vect=None, similarity_func=None)

Return a matrix of the similarity of the keys to the values.

By default the featurizer is learned from `keys` and `values` using equate’s
pure numpy/scipy char-n-gram TF-IDF ([`equate.featurize`](equate.featurize.html.md#module-equate.featurize)), and similarity is
cosine (`equate._vector.cosine_similarity()`) — no scikit-learn or grub needed.
Pass `obj_to_vect` to use a different batch featurizer (precomputed embeddings, or
a registered name resolved via `equate.featurize.resolve_featurizer`), and
`similarity_func` to use a different similarity.

```pycon
>>> keys = ['apple pie', 'apple crumble', 'banana split']
>>> values = ['american pie', 'big apple', 'american girl', 'banana republic']
>>> m = similarity_matrix(keys, values)
>>> m.round(2).tolist()
[[0.32, 0.42, 0.01, 0.0], [0.04, 0.35, 0.01, 0.02], [0.03, 0.04, 0.02, 0.44]]
```

### equate.util.stable_marriage_matching(similarity_matrix, , sense='maximize')

Stable Marriage Problem (Gale-Shapley Algorithm):
Solves the stable marriage problem, ensuring a stable matching. Note this
optimizes *stability* (no blocking pair), not total score, and is
proposer-optimal / receiver-pessimal — so it can differ from the optimal
assignment found by `hungarian_matching`.
See: [https://en.wikipedia.org/wiki/Stable_marriage_problem](https://en.wikipedia.org/wiki/Stable_marriage_problem)

* **Parameters:**
  **similarity_matrix** – A matrix of scores (see `sense`); preferences are
  ranked by the `to_cost` distance, so higher similarity is more preferred.
* **Returns:**
  List of tuples (row_index, col_index) for matched pairs.
