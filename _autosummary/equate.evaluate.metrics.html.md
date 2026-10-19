# equate.evaluate.metrics

Pairwise and cluster-level match metrics, dependency-light.

Two granularities, kept explicit (research doc 14, §2): [`pairwise_prf()`](#equate.evaluate.metrics.pairwise_prf) scores pair
sets; [`bcubed()`](#equate.evaluate.metrics.bcubed), [`ceaf()`](#equate.evaluate.metrics.ceaf) and [`cluster_metrics()`](#equate.evaluate.metrics.cluster_metrics) score a label per record.
**B³ is the default cluster metric** — it alone satisfies all of Amigó’s formal
constraints (§2.4) — with CEAF, ARI and V-measure as complements. ARI and V-measure
delegate to `sklearn.metrics` when it is installed and fall back to the numpy
implementations here otherwise; CEAF’s cluster alignment goes through equate’s own
matcher seam rather than a second Hungarian.

### Functions

| [`bcubed`](#equate.evaluate.metrics.bcubed)(pred_labels, true_labels)                   | B³ precision / recall / F1 — the default cluster metric.                                                                                                           |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ceaf`](#equate.evaluate.metrics.ceaf)(pred_labels, true_labels, \*[, variant, ...]) | CEAF: score the best one-to-one alignment of predicted to true clusters.                                                                                           |
| [`cluster_metrics`](#equate.evaluate.metrics.cluster_metrics)(pred_labels, true_labels, \*)      | B³ (the headline), CEAF-e, ARI, V-measure, homogeneity and completeness.                                                                                           |
| [`gmd`](#equate.evaluate.metrics.gmd)(pred_labels, true_labels, \*[, ...])           | Generalized merge distance: the cost of the splits and merges that turn the predicted clustering into the true one (Menestrina, Whang & Garcia-Molina, VLDB 2010). |
| [`linear_cost`](#equate.evaluate.metrics.linear_cost)([k1, k2])                              | A cost `k1 + k2 * x * y` — the order-independent family GMD is defined for.                                                                                        |
| [`pair_cost`](#equate.evaluate.metrics.pair_cost)(x, y)                                    | Cost `x * y`: the record pairs a split separates or a merge joins — with it, GMD equals the number of pairwise errors (false positives + false negatives).         |
| [`pairwise_prf`](#equate.evaluate.metrics.pairwise_prf)(pred_pairs, true_pairs, \*[, ...])    | Pairwise precision / recall / F1 of predicted pairs against gold pairs.                                                                                            |
| [`unit_cost`](#equate.evaluate.metrics.unit_cost)(x, y)                                    | Cost 1 per split or merge, whatever the sizes (GMD counts operations).                                                                                             |

### equate.evaluate.metrics.bcubed(pred_labels, true_labels)

B³ precision / recall / F1 — the default cluster metric.

Per record: precision is the share of its predicted cluster that truly belongs with
it, recall the share of its true cluster that the prediction kept with it; both are
averaged over records.

```pycon
>>> s = bcubed([0, 0, 1, 1], [0, 0, 0, 1])
>>> round(s.precision, 3), round(s.recall, 3)
(0.75, 0.667)
```

* **Return type:**
  [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)

### equate.evaluate.metrics.ceaf(pred_labels, true_labels, , variant='entity', matcher='optimal')

CEAF: score the best one-to-one alignment of predicted to true clusters.

`variant='entity'` (CEAF-e, φ4) weighs each aligned pair by
`2|K∩R| / (|K|+|R|)` and normalises by the number of clusters;
`variant='mention'` (CEAF-m, φ3) weighs by `|K∩R|` and normalises by the
number of records. The alignment is a maximum-weight assignment, resolved through
equate’s [`matchers`](equate.matching.html.md#equate.matching.matchers) registry (`matcher`), never a
private Hungarian.

```pycon
>>> round(ceaf([0, 0, 1, 1], [0, 0, 0, 1]).f1, 3)
0.733
```

* **Return type:**
  [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)

### equate.evaluate.metrics.cluster_metrics(pred_labels, true_labels, , backend='auto')

B³ (the headline), CEAF-e, ARI, V-measure, homogeneity and completeness.

`backend='auto'` uses `sklearn.metrics` for ARI / V-measure when sklearn is
installed and the numpy implementations otherwise; `'numpy'` forces the latter,
`'sklearn'` requires the former. The two agree; the choice only exists so the
core install needs neither.

```pycon
>>> m = cluster_metrics([0, 0, 1, 1], [0, 0, 0, 1], backend='numpy')
>>> sorted(m)
['ari', 'b3_f1', 'b3_precision', 'b3_recall', 'ceaf_e_f1', 'completeness',
 'homogeneity', 'v_measure']
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### equate.evaluate.metrics.gmd(pred_labels, true_labels, \*, merge_cost=<function unit_cost>, split_cost=<function unit_cost>)

Generalized merge distance: the cost of the splits and merges that turn the
predicted clustering into the true one (Menestrina, Whang & Garcia-Molina, VLDB 2010).

`split_cost(x, y)` prices splitting a cluster of `x + y` records into `x` and
`y`; `merge_cost(x, y)` merging an `x`- and a `y`-record cluster. The defaults
([`unit_cost()`](#equate.evaluate.metrics.unit_cost)) count operations; [`pair_cost()`](#equate.evaluate.metrics.pair_cost) for both gives the pairwise
error count; [`linear_cost()`](#equate.evaluate.metrics.linear_cost) `(k1, k2)` is `k1 + k2·x·y`. Unlike F-measures it
is a distance (`0` = identical), and its costs let an application say whether a
wrong merge hurts more than a missed one. Linear time.

As in the paper, this is the cost of the *legal* path — split each predicted cluster
along the true clusters, then merge the pieces of each true cluster — and the costs
must be **operation-order independent** (constant, `k·x·y`, or `k1 + k2·x·y`):
for other costs the result can depend on the labels’ names. With unit costs a path
that merges first can occasionally be cheaper; GMD is defined as the legal path.

```pycon
>>> gmd([0, 0, 1, 1], [0, 0, 0, 1])   # split {2, 3}, merge {2} into {0, 1}
2.0
>>> gmd([0, 0, 1, 1], [0, 0, 0, 1], merge_cost=pair_cost, split_cost=pair_cost)
3.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### equate.evaluate.metrics.linear_cost(k1=0.0, k2=1.0)

A cost `k1 + k2 * x * y` — the order-independent family GMD is defined for.

```pycon
>>> linear_cost(1, 0)(3, 4), linear_cost(0, 1)(3, 4)
(1.0, 12.0)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

### equate.evaluate.metrics.pair_cost(x, y)

Cost `x * y`: the record pairs a split separates or a merge joins — with it,
GMD equals the number of pairwise errors (false positives + false negatives).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### equate.evaluate.metrics.pairwise_prf(pred_pairs, true_pairs, , self_join=False)

Pairwise precision / recall / F1 of predicted pairs against gold pairs.

`self_join=True` (deduplication within one collection) treats `(i, j)` and
`(j, i)` as one pair and drops `(i, i)`; otherwise pairs are `(row of A, row
of B)` and orientation matters.

```pycon
>>> s = pairwise_prf([(0, 0), (1, 2)], [(0, 0), (1, 1)])
>>> s.precision, s.recall, s.support
(0.5, 0.5, 2)
```

* **Return type:**
  [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)

### equate.evaluate.metrics.unit_cost(x, y)

Cost 1 per split or merge, whatever the sizes (GMD counts operations).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
