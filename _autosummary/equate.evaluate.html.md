# equate.evaluate

Evaluate matchers honestly: metrics, leakage-aware splits, and a benchmark harness.

Dependency-light (numpy/scipy only; sklearn is used for ARI / V-measure when present).
Precision and recall always travel together ([`Scores`](#equate.evaluate.Scores)), the pairwise and cluster
granularities are explicit ([`pairs_to_labels()`](#equate.evaluate.pairs_to_labels) converts on purpose), B³ is the
default cluster metric, thresholds are selected on validation data by a separate call,
and [`benchmark_matcher()`](#equate.evaluate.benchmark_matcher) splits by entity so test entities are unseen.
See `docs/research/14-evaluation-benchmarks-and-methodology.md`.

```pycon
>>> from equate.evaluate import pairwise_prf, bcubed
>>> pairwise_prf([(0, 0), (1, 1)], [(0, 0), (1, 2)]).f1
0.5
```

### Functions

| [`pairwise_prf`](#equate.evaluate.pairwise_prf)(pred_pairs, true_pairs, \*[, ...])    | Pairwise precision / recall / F1 of predicted pairs against gold pairs.          |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`bcubed`](#equate.evaluate.bcubed)(pred_labels, true_labels)                   | B³ precision / recall / F1 — the default cluster metric.                         |
| [`ceaf`](#equate.evaluate.ceaf)(pred_labels, true_labels, \*[, variant, ...]) | CEAF: score the best one-to-one alignment of predicted to true clusters.         |
| [`cluster_metrics`](#equate.evaluate.cluster_metrics)(pred_labels, true_labels, \*)      | B³ (the headline), CEAF-e, ARI, V-measure, homogeneity and completeness.         |
| [`blocking_metrics`](#equate.evaluate.blocking_metrics)(candidate_pairs, ...[, n_b])      | Return `{pair_completeness, reduction_ratio, pairs_quality, n_candidates}`.      |
| [`pr_auc`](#equate.evaluate.pr_auc)(scores, y_true)                             | Average precision: the area under the precision-recall step curve.               |
| [`select_threshold`](#equate.evaluate.select_threshold)(scores, y_true, \*[, optimize])   | The score threshold (`score >= t` predicts a match) that maximises `optimize`.   |
| [`ranking_metrics`](#equate.evaluate.ranking_metrics)(ranked, gold, \*[, ks])            | Recall@k, MRR and MAP for ranked candidate lists.                                |
| [`pairs_to_labels`](#equate.evaluate.pairs_to_labels)(pairs, \*, n)                      | Cluster label per record `0..n-1` from the transitive closure of `pairs`.        |
| [`labels_to_pairs`](#equate.evaluate.labels_to_pairs)(labels)                            | Every within-cluster pair `(i, j)` with `i < j`.                                 |
| [`cross_pairs_to_labels`](#equate.evaluate.cross_pairs_to_labels)(pairs, \*, n_a, n_b)         | Labels over `A + B` (B's records offset by `n_a`) from two-table `(i, j)` pairs. |
| [`entity_split`](#equate.evaluate.entity_split)(entity_ids, \*[, ratios, seed])       | Split records into train / val / test by entity, never by record or pair.        |
| [`assert_no_leakage`](#equate.evaluate.assert_no_leakage)(entity_ids, \*parts)             | Raise `AssertionError` if any entity has records in more than one of `parts`.    |
| [`list_benchmarks`](#equate.evaluate.list_benchmarks)()                                  | Names of the registered benchmarks.                                              |
| [`load_benchmark`](#equate.evaluate.load_benchmark)(name, \*\*config)                   | Build a registered benchmark (`config` goes to its loader).                      |
| [`benchmark_matcher`](#equate.evaluate.benchmark_matcher)([matcher, on, n_seeds, ...])     | Score `matcher` on unseen-entity test splits of each benchmark in `on`.          |

### Classes

| [`Scores`](#equate.evaluate.Scores)(precision, recall, f1[, support])      | Precision, recall and their harmonic mean, plus what they were computed over.   |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`Split`](#equate.evaluate.Split)(train, val, test)                       | Record indices of each part; the entity ids of a part never occur in another.   |
| [`Benchmark`](#equate.evaluate.Benchmark)(name, a, b, gold[, meta])           | Two collections and the gold correspondence between them.                       |
| [`BenchmarkReport`](#equate.evaluate.BenchmarkReport)(rows[, meta])                 | All runs, plus a per-benchmark summary of mean ± std over seeds.                |
| [`BenchmarkRow`](#equate.evaluate.BenchmarkRow)(benchmark, seed, precision, ...) | One (benchmark, seed) run: the scores, their context, and what they cost.       |

### *class* equate.evaluate.Benchmark(name, a, b, gold, meta=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Two collections and the gold correspondence between them.

`gold` holds `(i, j)` pairs: `a[i]` corresponds to `b[j]`. `meta` carries
what a reader needs next to a score — at least `source` and `difficulty`.

#### *property* base_rate *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Share of all `a × b` pairs that are true matches.

### *class* equate.evaluate.BenchmarkReport(rows, meta=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

All runs, plus a per-benchmark summary of mean ± std over seeds.

#### summary()

`{benchmark: {metric: (mean, std), 'base_rate': mean, 'n_seeds': k}}`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### to_markdown()

One row per benchmark: precision, recall, F1, seconds, base rate, difficulty.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* equate.evaluate.BenchmarkRow(benchmark, seed, precision, recall, f1, seconds, n_a, n_b, n_gold, base_rate)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One (benchmark, seed) run: the scores, their context, and what they cost.

### *class* equate.evaluate.Scores(precision, recall, f1, support=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Precision, recall and their harmonic mean, plus what they were computed over.

```pycon
>>> s = Scores.from_counts(tp=3, n_pred=4, n_true=6)
>>> s.precision, s.recall, round(s.f1, 3)
(0.75, 0.5, 0.6)
>>> s.support
6
```

#### as_dict()

The fields as a plain dict (for tables, JSON, a dataframe row).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### *classmethod* from_counts(, tp, n_pred, n_true)

Build from true-positive, predicted and gold counts.

An empty prediction has precision 1.0 (it asserted nothing false); an empty gold
set has recall 1.0 (there was nothing to find). Both empty is a perfect score.

* **Return type:**
  [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)

#### *classmethod* from_pr(precision, recall, , support=None)

Build from precision and recall; F1 is derived, so it can never disagree.

* **Return type:**
  [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)

#### support *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/builtins/functions.html#int)]* *= None*

the size of the gold side (true pairs, or records for a cluster metric)

### *class* equate.evaluate.Split(train, val, test)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Record indices of each part; the entity ids of a part never occur in another.

### equate.evaluate.assert_no_leakage(entity_ids, \*parts)

Raise `AssertionError` if any entity has records in more than one of `parts`.

```pycon
>>> assert_no_leakage([0, 0, 1], [0], [1])
Traceback (most recent call last):
    ...
AssertionError: entity 0 appears in parts 0 and 1 — the split leaks it
```

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### equate.evaluate.bcubed(pred_labels, true_labels)

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

### equate.evaluate.benchmark_matcher(matcher=None, , on=('toy_names',), n_seeds=3, ratios=(0.6, 0.2, 0.2), fit=None)

Score `matcher` on unseen-entity test splits of each benchmark in `on`.

`matcher(a, b)` returns `(i, j)` pairs (a [`Matching`](equate.base.html.md#equate.base.Matching)
qualifies); the default is `equate.match()`. `on` is a benchmark name, a
[`Benchmark`](#equate.evaluate.Benchmark), or an iterable of either. Scores are pairwise; a seed whose test
part has no gold pair is skipped with a warning, and a benchmark with no scorable
seed is an error. For each of `n_seeds` seeds the records of `a + b`
are split by entity (`ratios` = train/val/test); only the **test** part is
matched and scored, so a test entity was never seen in training.

`fit(train_a, train_b, train_pairs, val_a, val_b, val_pairs) -> matcher` makes the
harness train per seed: it gets the train and validation parts only (the split is
asserted leak-free first), and the matcher it returns is scored on test. Choose a
threshold on the validation part there ([`select_threshold()`](#equate.evaluate.select_threshold)),
never on test.

```pycon
>>> report = benchmark_matcher(n_seeds=2)
>>> s = report.summary()['toy_names']
>>> s['f1'][0] > 0.8, s['n_seeds']
(True, 2)
```

* **Return type:**
  [`BenchmarkReport`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.BenchmarkReport)

### equate.evaluate.blocking_metrics(candidate_pairs, true_pairs, , n_a, n_b=None)

Return `{pair_completeness, reduction_ratio, pairs_quality, n_candidates}`.

- **pair_completeness (PC)** = `|candidates ∩ true| / |true|` — blocking *recall*
  (upper-bounds end-to-end recall);
- **reduction_ratio (RR)** = `1 - |candidates| / |all pairs|` — how much blocking pruned;
- **pairs_quality (PQ)** = `|candidates ∩ true| / |candidates|` — blocking *precision*.

`n_b=None` means a self-join (dedup) over `n_a` items, so
`|all pairs| = n_a*(n_a-1)/2`; otherwise `|all pairs| = n_a*n_b`. For a self-join,
pairs are canonicalized to unordered `(min, max)` so orientation doesn’t matter and
`(i, j)` / `(j, i)` count once. Empty `true_pairs` yields `pair_completeness=1.0`
(vacuously perfect recall); `reduction_ratio` is clamped to `[0, 1]`.

```pycon
>>> m = blocking_metrics([(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)], n_a=3, n_b=3)
>>> round(m['pair_completeness'], 2), round(m['pairs_quality'], 2)
(0.67, 1.0)
```

### equate.evaluate.ceaf(pred_labels, true_labels, , variant='entity', matcher='optimal')

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

### equate.evaluate.cluster_metrics(pred_labels, true_labels, , backend='auto')

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

### equate.evaluate.cross_pairs_to_labels(pairs, , n_a, n_b)

Labels over `A + B` (B’s records offset by `n_a`) from two-table `(i, j)` pairs.

The bridge from a matching between two collections to the entity view a cluster
metric or an entity split needs.

```pycon
>>> cross_pairs_to_labels([(0, 1)], n_a=2, n_b=2)
[0, 1, 2, 0]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]

### equate.evaluate.entity_split(entity_ids, , ratios=(0.6, 0.2, 0.2), seed=0)

Split records into train / val / test by entity, never by record or pair.

`entity_ids[i]` is record `i`’s true entity (e.g. from
[`cross_pairs_to_labels()`](#equate.evaluate.cross_pairs_to_labels) over a two-table gold matching).
Entities, not records, are shuffled (`seed`) and cut by `ratios`, so the parts
are disjoint in entities and roughly — not exactly — proportional in records.

```pycon
>>> s = entity_split([0, 0, 1, 1, 2, 3, 4, 4, 5, 6], seed=1)
>>> assert_no_leakage([0, 0, 1, 1, 2, 3, 4, 4, 5, 6], s.train, s.test)
>>> sorted(s.train + s.val + s.test) == list(range(10))
True
```

* **Return type:**
  [`Split`](equate.evaluate.split.html.md#equate.evaluate.split.Split)

### equate.evaluate.labels_to_pairs(labels)

Every within-cluster pair `(i, j)` with `i < j`.

```pycon
>>> sorted(labels_to_pairs([0, 0, 1, 0]))
[(0, 1), (0, 3), (1, 3)]
```

* **Return type:**
  [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]]

### equate.evaluate.list_benchmarks()

Names of the registered benchmarks.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### equate.evaluate.load_benchmark(name, \*\*config)

Build a registered benchmark (`config` goes to its loader).

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)

### equate.evaluate.pairs_to_labels(pairs, , n)

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

### equate.evaluate.pairwise_prf(pred_pairs, true_pairs, , self_join=False)

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

### equate.evaluate.pr_auc(scores, y_true)

Average precision: the area under the precision-recall step curve.

Records with tied scores enter the ranking together (one threshold step), so the
figure does not depend on the input order of ties. No positives → `nan`.

```pycon
>>> round(pr_auc([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0]), 3)
0.833
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### equate.evaluate.ranking_metrics(ranked, gold, , ks=(1, 5, 10))

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

### equate.evaluate.select_threshold(scores, y_true, , optimize='f1')

The score threshold (`score >= t` predicts a match) that maximises `optimize`.

Pass **validation** scores and labels. `optimize` is `'f1'`, `'precision'` or
`'recall'`; ties go to the higher threshold (fewer predicted matches). Returns the
threshold and the [`Scores`](#equate.evaluate.Scores) it achieves on the data it was chosen on — which
are therefore optimistic, and are not a test result.

```pycon
>>> t, s = select_threshold([0.9, 0.8, 0.4, 0.3], [1, 1, 0, 1])
>>> t, round(s.f1, 3)
(0.3, 0.857)
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`Scores`](equate.evaluate.scores.html.md#equate.evaluate.scores.Scores)]

### Modules

| [`benchmark`](equate.evaluate.benchmark.html.md#module-equate.evaluate.benchmark)   | Self-benchmarking: a benchmark registry and an honest-by-default harness.      |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`labels`](equate.evaluate.labels.html.md#module-equate.evaluate.labels)         | Convert between the two granularities: pairs and cluster labels.               |
| [`metrics`](equate.evaluate.metrics.html.md#module-equate.evaluate.metrics)       | Pairwise and cluster-level match metrics, dependency-light.                    |
| [`ranking`](equate.evaluate.ranking.html.md#module-equate.evaluate.ranking)       | Threshold-free and ranking metrics, and threshold selection as its own step.   |
| [`scores`](equate.evaluate.scores.html.md#module-equate.evaluate.scores)         | The one score type every metric returns: precision and recall, never F1 alone. |
| [`split`](equate.evaluate.split.html.md#module-equate.evaluate.split)           | Leakage-aware splitting: split over entities, so test entities are unseen.     |
