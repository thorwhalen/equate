# equate.evaluate.labels

Convert between the two granularities: pairs and cluster labels.

Pairwise metrics take pair sets; cluster metrics take a label per record. The
conversion is a deliberate act (research doc 14, §2.2): [`pairs_to_labels()`](#equate.evaluate.labels.pairs_to_labels) takes
the **transitive closure** of a pair set (union-find), which can merge records a matcher
never linked directly, so a user who evaluates a 1:1 matcher with a cluster metric sees
exactly what they are asking for.

### Functions

| [`cross_pairs_to_labels`](#equate.evaluate.labels.cross_pairs_to_labels)(pairs, \*, n_a, n_b)   | Labels over `A + B` (B's records offset by `n_a`) from two-table `(i, j)` pairs.   |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`labels_to_pairs`](#equate.evaluate.labels.labels_to_pairs)(labels)                      | Every within-cluster pair `(i, j)` with `i < j`.                                   |
| [`pairs_to_labels`](#equate.evaluate.labels.pairs_to_labels)(pairs, \*, n)                | Cluster label per record `0..n-1` from the transitive closure of `pairs`.          |

### equate.evaluate.labels.cross_pairs_to_labels(pairs, , n_a, n_b)

Labels over `A + B` (B’s records offset by `n_a`) from two-table `(i, j)` pairs.

The bridge from a matching between two collections to the entity view a cluster
metric or an entity split needs.

```pycon
>>> cross_pairs_to_labels([(0, 1)], n_a=2, n_b=2)
[0, 1, 2, 0]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]

### equate.evaluate.labels.labels_to_pairs(labels)

Every within-cluster pair `(i, j)` with `i < j`.

```pycon
>>> sorted(labels_to_pairs([0, 0, 1, 0]))
[(0, 1), (0, 3), (1, 3)]
```

* **Return type:**
  [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]]

### equate.evaluate.labels.pairs_to_labels(pairs, , n)

Cluster label per record `0..n-1` from the transitive closure of `pairs`.

Labels are canonical: numbered in order of each cluster’s smallest record, so two
pair sets that induce the same partition give equal label lists. A record no pair
mentions is a singleton.

```pycon
>>> pairs_to_labels([(0, 1), (1, 2), (4, 5)], n=6)
[0, 0, 0, 1, 2, 2]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]
