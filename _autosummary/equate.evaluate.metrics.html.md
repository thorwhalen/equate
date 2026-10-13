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

| [`bcubed`](#equate.evaluate.metrics.bcubed)(pred_labels, true_labels)                   | B³ precision / recall / F1 — the default cluster metric.                 |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`ceaf`](#equate.evaluate.metrics.ceaf)(pred_labels, true_labels, \*[, variant, ...]) | CEAF: score the best one-to-one alignment of predicted to true clusters. |
| [`cluster_metrics`](#equate.evaluate.metrics.cluster_metrics)(pred_labels, true_labels, \*)      | B³ (the headline), CEAF-e, ARI, V-measure, homogeneity and completeness. |
| [`pairwise_prf`](#equate.evaluate.metrics.pairwise_prf)(pred_pairs, true_pairs, \*[, ...])    | Pairwise precision / recall / F1 of predicted pairs against gold pairs.  |

### equate.evaluate.metrics.bcubed(pred_labels, true_labels)

B³ precision / recall / F1 — the default cluster metric.

Per record: precision is the share of its predicted cluster that truly belongs with
it, recall the share of its true cluster that the prediction kept with it; both are
averaged over records.

* **Return type:**
  [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)

```pycon
>>> s = bcubed([0, 0, 1, 1], [0, 0, 0, 1])
>>> round(s.precision, 3), round(s.recall, 3)
(0.75, 0.667)
```

### equate.evaluate.metrics.ceaf(pred_labels, true_labels, , variant='entity', matcher='optimal')

CEAF: score the best one-to-one alignment of predicted to true clusters.

`variant='entity'` (CEAF-e, φ4) weighs each aligned pair by
`2|K∩R| / (|K|+|R|)` and normalises by the number of clusters;
`variant='mention'` (CEAF-m, φ3) weighs by `|K∩R|` and normalises by the
number of records. The alignment is a maximum-weight assignment, resolved through
equate’s [`matchers`](equate.matching.html.md#equate.matching.matchers) registry (`matcher`), never a
private Hungarian.

* **Return type:**
  [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)

```pycon
>>> round(ceaf([0, 0, 1, 1], [0, 0, 0, 1]).f1, 3)
0.733
```

### equate.evaluate.metrics.cluster_metrics(pred_labels, true_labels, , backend='auto')

B³ (the headline), CEAF-e, ARI, V-measure, homogeneity and completeness.

`backend='auto'` uses `sklearn.metrics` for ARI / V-measure when sklearn is
installed and the numpy implementations otherwise; `'numpy'` forces the latter,
`'sklearn'` requires the former. The two agree; the choice only exists so the
core install needs neither.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

```pycon
>>> m = cluster_metrics([0, 0, 1, 1], [0, 0, 0, 1], backend='numpy')
>>> sorted(m)
['ari', 'b3_f1', 'b3_precision', 'b3_recall', 'ceaf_e_f1', 'completeness',
 'homogeneity', 'v_measure']
```

### equate.evaluate.metrics.pairwise_prf(pred_pairs, true_pairs, , self_join=False)

Pairwise precision / recall / F1 of predicted pairs against gold pairs.

`self_join=True` (deduplication within one collection) treats `(i, j)` and
`(j, i)` as one pair and drops `(i, i)`; otherwise pairs are `(row of A, row
of B)` and orientation matters.

* **Return type:**
  [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)

```pycon
>>> s = pairwise_prf([(0, 0), (1, 2)], [(0, 0), (1, 1)])
>>> s.precision, s.recall, s.support
(0.5, 0.5, 2)
```
