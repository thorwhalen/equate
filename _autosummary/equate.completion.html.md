# equate.completion

Tabular completion: find join keys, near-duplicate columns, and fill a table from another.

Merging tables on a known key is a small part of what is needed. Without leaving tables:

- **find the columns to join on** ([`find_join_keys()`](#equate.completion.find_join_keys)) by how much one column’s
  values are *contained* in another’s — no header names needed;
- **find near-duplicate columns** ([`find_near_duplicate_columns()`](#equate.completion.find_near_duplicate_columns)) in one table;
- **complete a table** ([`complete_table()`](#equate.completion.complete_table)): discover the join key, align rows by
  fuzzy matching of the key values, and fill the target’s missing cells from the source.

These are facades over the same seams as everything else (research doc 07 §8, doc 10
§8.7) — not a parallel implementation: a column is an object featurized as its value set
(`featurize='value_set'`), compared with a set comparator (`'containment'`,
`'jaccard'`, `'overlap'`), and selected with the top-k `
CandidateStore`; rows are aligned with `equate.match()` / `equate.score_matrix()`.

A *table* is any of: a mapping of column name to values (`{'name': [...], ...}`), a list
of row mappings, or anything DataFrame-like (`.columns` and `table[column]`). Results
are structured dataclasses; [`Completion`](#equate.completion.Completion) gives back columns (`.table`) or rows
(`.rows()`).

```pycon
>>> left = {'cust': ['c1', 'c2', 'c3'], 'city': [None, 'Lyon', None]}
>>> right = {'id': ['c3', 'c1', 'c2', 'c9'], 'city': ['Nice', 'Paris', 'Lyon', 'Lille']}
>>> [(m.left, m.right, m.score) for m in find_join_keys(left, right)][0]
('cust', 'id', 1.0)
>>> complete_table(left, right).table['city']
['Paris', 'Lyon', 'Nice']
```

The functions further down (`similarity_matrix`, `mk_text_to_vect`, …) are the
pre-redesign API, kept for back-compat; new code should use `equate.score_matrix()`.

### Functions

| [`as_columns`](#equate.completion.as_columns)(table)                                  | `{column name: list of values}` from a column mapping, a list of row mappings, or a DataFrame-like object (`.columns` + `table[column]`).             |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`find_join_keys`](#equate.completion.find_join_keys)(left, right, \*[, compare, k, ...]) | Columns of `right` that `left`'s columns could be joined on, best first.                                                                              |
| [`find_near_duplicate_columns`](#equate.completion.find_near_duplicate_columns)(table, \*[, ...])      | Pairs of columns of one table whose value sets are near-identical.                                                                                    |
| [`complete_table`](#equate.completion.complete_table)(target, source, \*[, on, ...])      | Fill `target`'s missing cells from `source`.                                                                                                          |
| [`similarity_matrix`](#equate.completion.similarity_matrix)(keys, values, \*[, ...])         | Return a matrix of the similarity of the keys to the values.                                                                                          |
| `mk_text_to_vect`(\*learn_texts)                                                                    |                                                                                                                                                       |
| `transform_text`(text, transformer)                                                                 |                                                                                                                                                       |
| [`greedy_matching`](#equate.completion.greedy_matching)(similarity_matrix)                 | Greedy Algorithm: Iteratively picks the highest similarity pair and removes the matched items from the pool.                                          |
| [`hungarian_matching`](#equate.completion.hungarian_matching)(similarity_matrix, \*[, ...])   | Hungarian Algorithm (Optimal Matching): Finds the optimal 1:1 assignment, minimizing the total cost (equivalently, maximizing the total similarity).  |
| [`maximal_matching`](#equate.completion.maximal_matching)(similarity_matrix)                | Maximal Matching in Bipartite Graph: Finds a maximal matching in a bipartite graph.                                                                   |
| [`stable_marriage_matching`](#equate.completion.stable_marriage_matching)(similarity_matrix, \*)    | Stable Marriage Problem (Gale-Shapley Algorithm): Solves the stable marriage problem, ensuring a stable matching.                                     |
| [`kuhn_munkres_matching`](#equate.completion.kuhn_munkres_matching)(similarity_matrix, \*)       | Kuhn-Munkres Algorithm: Another implementation of the Hungarian algorithm using the `networkx` package (requires the optional `equate[graph]` extra). |

### Classes

| [`ColumnMatch`](#equate.completion.ColumnMatch)(left, right, score[, explanation])   | A scored correspondence between a `left` and a `right` column.   |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| [`Completion`](#equate.completion.Completion)(table, key, rows, filled[, columns])  | The completed table and how it was completed.                    |

### *class* equate.completion.ColumnMatch(left, right, score, explanation=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A scored correspondence between a `left` and a `right` column.

### *class* equate.completion.Completion(table, key, rows, filled, columns=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The completed table and how it was completed.

`table` is the target as columns, with missing cells filled; `key` the join key
used; `rows` the aligned `(target_row, source_row, score)` triples (`score` is
`None` for an exact key match, the fuzzy score otherwise); `filled` the number of
cells filled per column; `columns` the `{target: source}` columns used.

#### as_rows()

The completed table as a list of row dicts.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### equate.completion.as_columns(table)

`{column name: list of values}` from a column mapping, a list of row mappings,
or a DataFrame-like object (`.columns` + `table[column]`).

Column names are kept as they are (an `int` stays an `int`). Rows missing a
column get `None` there; column order is first-seen order. A column must be a
sequence of cells, not a string.

```pycon
>>> as_columns([{'a': 1}, {'a': 2, 'b': 'x'}])
{'a': [1, 2], 'b': [None, 'x']}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### equate.completion.complete_table(target, source, , on=None, columns=None, key_compare='containment', min_key_score=0.5, row_compare='ratio', how='best', min_row_score=0.9)

Fill `target`’s missing cells from `source`.

1. **Key** — `on=(target_column, source_column)`, or discovered with
   [`find_join_keys()`](#equate.completion.find_join_keys) (`key_compare`, `min_key_score`).
2. **Rows** — key values are normalized (case, spacing, `1.0` = `1`) and joined
   **exactly** first. Only target keys with no exact partner are matched
   **fuzzily** with `row_compare` (default the `'ratio'` edit similarity, so a
   stray character still meets its key; `None` turns fuzzy matching off) and kept only
   when they score at least `min_row_score` (default `0.9`; for a *distance*
   comparator it is a maximum) and are not tied between two source keys.
   `how='best'` (default) is a lookup: each target row takes its source key, so
   many target rows may share one. A matcher name (`'optimal'`, `'greedy'`,
   `'stable'`) aligns rows one-to-one instead.
3. **Fill** — for each `{target_column: source_column}` in `columns` (default:
   the columns both tables share, except the key), a missing target cell (`None`,
   blank, NaN, pandas `NA`) takes the value of the first aligned source row that
   has one. A target column that does not exist yet is created. Present values are
   never overwritten.

* **Return type:**
  [`Completion`](#equate.completion.Completion)

### equate.completion.find_join_keys(left, right, , compare='containment', k=1, min_score=0.5)

Columns of `right` that `left`’s columns could be joined on, best first.

Each column is featurized as its value set and compared with `compare` (default
`'containment'`: the share of the left column’s distinct values found in the right
one — a foreign key scores 1.0 against its primary key whatever else that holds).
Each left column keeps its top-`k` right columns scoring at least `min_score`.
Returns [`ColumnMatch`](#equate.completion.ColumnMatch) es sorted by score; ties go to the pair sharing more
distinct values (a real key over a low-cardinality column), then to column order.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### equate.completion.find_near_duplicate_columns(table, , compare='jaccard', threshold=0.9)

Pairs of columns of one table whose value sets are near-identical.

Each column is featurized as its value set and every pair compared with `compare`
(default `'jaccard'`); pairs scoring at least `threshold` are returned as
[`ColumnMatch`](#equate.completion.ColumnMatch) es, best first — earlier column on the left for a symmetric
comparator, both directions for a directional one (`'containment'`).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### equate.completion.greedy_matching(similarity_matrix)

Greedy Algorithm: Iteratively picks the highest similarity pair and removes the
matched items from the pool.

* **Parameters:**
  **similarity_matrix** – A sparse matrix of similarities.
* **Returns:**
  List of tuples (row_index, col_index) for matched pairs.

### equate.completion.hungarian_matching(similarity_matrix, , sense='maximize', cost_matrix=None)

Hungarian Algorithm (Optimal Matching): Finds the optimal 1:1 assignment,
minimizing the total cost (equivalently, maximizing the total similarity).

* **Parameters:**
  * **similarity_matrix** – A matrix of scores. `sense='maximize'` treats them as
    similarities (higher = better, the default); `sense='minimize'` treats them
    as costs/distances. The conversion goes through the `to_cost` SSOT.
  * **cost_matrix** – An explicit cost matrix to use instead (overrides `sense`).
* **Returns:**
  An iterator of (row_index, col_index) tuples for matched pairs.

### equate.completion.kuhn_munkres_matching(similarity_matrix, , sense='maximize')

Kuhn-Munkres Algorithm: Another implementation of the Hungarian algorithm using
the `networkx` package (requires the optional `equate[graph]` extra).

* **Parameters:**
  **similarity_matrix** – A matrix of scores (see `sense`); converted to edge
  costs via the `to_cost` SSOT.
* **Returns:**
  List of tuples (row_index, col_index) for matched pairs.

### equate.completion.maximal_matching(similarity_matrix)

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

### equate.completion.similarity_matrix(keys, values, \*, text_to_vect=None, similarity_func=<function cosine_similarity>)

Return a matrix of the similarity of the keys to the values.
By default, the `text_to_vect` will be learned from the keys and values,
using the tfidf vectorizer from `grub`.
You can pass in a different `text_to_vect` function to use a different vectorizer:
For example, pre-computed embeddings such as word2vec or fasttext, or openai
embeddings.

```pycon
>>> keys = ['apple pie', 'apple crumble', 'banana split']
>>> values = ['american pie', 'big apple', 'american girl', 'banana republic']
>>> m = similarity_matrix(keys, values)
>>> m.round(2).tolist()
[[0.54, 0.38, 0.0, 0.0], [0.0, 0.33, 0.0, 0.0], [0.0, 0.0, 0.0, 0.41]]
```

### equate.completion.stable_marriage_matching(similarity_matrix, , sense='maximize')

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
