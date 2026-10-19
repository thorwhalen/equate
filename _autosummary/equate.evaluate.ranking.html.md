# equate.evaluate.ranking

Threshold-free and ranking metrics, and threshold selection as its own step.

[`pr_auc()`](#equate.evaluate.ranking.pr_auc) (average precision) summarises a scorer without choosing a threshold and
stays honest under the heavy class imbalance of matching, where ROC-AUC flatters
(research doc 14, §4). [`select_threshold()`](#equate.evaluate.ranking.select_threshold) is deliberately separate from scoring:
tuning on the test set should be something a user does on purpose, never by accident
(§3, §7.2).

### Functions

| [`pr_auc`](#equate.evaluate.ranking.pr_auc)(scores, y_true)                           | Average precision: the area under the precision-recall step curve.             |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`ranking_metrics`](#equate.evaluate.ranking.ranking_metrics)(ranked, gold, \*[, ks])          | Recall@k, MRR and MAP for ranked candidate lists.                              |
| [`select_threshold`](#equate.evaluate.ranking.select_threshold)(scores, y_true, \*[, optimize]) | The score threshold (`score >= t` predicts a match) that maximises `optimize`. |

### equate.evaluate.ranking.pr_auc(scores, y_true)

Average precision: the area under the precision-recall step curve.

Records with tied scores enter the ranking together (one threshold step), so the
figure does not depend on the input order of ties. No positives → `nan`.

```pycon
>>> round(pr_auc([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0]), 3)
0.833
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### equate.evaluate.ranking.ranking_metrics(ranked, gold, , ks=(1, 5, 10))

Recall@k, MRR and MAP for ranked candidate lists.

`ranked[q]` is query `q`’s candidates, best first; `gold[q]` is its true match
or a collection of them. Queries with no gold are skipped; a query whose gold never
appears contributes 0.

```pycon
>>> m = ranking_metrics({'a': ['x', 'y'], 'b': ['z', 'w']}, {'a': 'y', 'b': 'z'}, ks=(1,))
>>> m['recall_at_1'], m['mrr']
(0.5, 0.75)
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### equate.evaluate.ranking.select_threshold(scores, y_true, , optimize='f1')

The score threshold (`score >= t` predicts a match) that maximises `optimize`.

Pass **validation** scores and labels. `optimize` is `'f1'`, `'precision'` or
`'recall'`; ties go to the higher threshold (fewer predicted matches). Returns the
threshold and the `Scores` it achieves on the data it was chosen on — which
are therefore optimistic, and are not a test result.

```pycon
>>> t, s = select_threshold([0.9, 0.8, 0.4, 0.3], [1, 1, 0, 1])
>>> t, round(s.f1, 3)
(0.3, 0.857)
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)]
