# equate.evaluate

Evaluate matchers honestly: metrics, leakage-aware splits, and a benchmark harness.

Dependency-light (numpy/scipy only; sklearn is used for ARI / V-measure when present).
Precision and recall always travel together ([`Scores`](#equate.evaluate.Scores)), the pairwise and cluster
granularities are explicit ([`pairs_to_labels()`](#equate.evaluate.pairs_to_labels) converts on purpose), B³ is the
default cluster metric, thresholds are selected on validation data by a separate call,
and [`benchmark_matcher()`](#equate.evaluate.benchmark_matcher) splits by entity so test entities are unseen. The standard
suites (`abt_buy`, `amazon_google`, `dblp_acm`, `wdc_products`) download on first
use into a store outside the package ([`equate.evaluate.store`](equate.evaluate.store.html.md#module-equate.evaluate.store)).
See `docs/research/14-evaluation-benchmarks-and-methodology.md`.

```pycon
>>> from equate.evaluate import pairwise_prf, bcubed
>>> pairwise_prf([(0, 0), (1, 1)], [(0, 0), (1, 2)]).f1
0.5
```

### Functions

| [`pairwise_prf`](#equate.evaluate.pairwise_prf)(pred_pairs, true_pairs, \*[, ...])    | Pairwise precision / recall / F1 of predicted pairs against gold pairs.                                                                                            |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`bcubed`](#equate.evaluate.bcubed)(pred_labels, true_labels)                   | B³ precision / recall / F1 — the default cluster metric.                                                                                                           |
| [`ceaf`](#equate.evaluate.ceaf)(pred_labels, true_labels, \*[, variant, ...]) | CEAF: score the best one-to-one alignment of predicted to true clusters.                                                                                           |
| [`cluster_metrics`](#equate.evaluate.cluster_metrics)(pred_labels, true_labels, \*)      | B³ (the headline), CEAF-e, ARI, V-measure, homogeneity and completeness.                                                                                           |
| [`gmd`](#equate.evaluate.gmd)(pred_labels, true_labels, \*[, ...])           | Generalized merge distance: the cost of the splits and merges that turn the predicted clustering into the true one (Menestrina, Whang & Garcia-Molina, VLDB 2010). |
| [`unit_cost`](#equate.evaluate.unit_cost)(x, y)                                    | Cost 1 per split or merge, whatever the sizes (GMD counts operations).                                                                                             |
| [`pair_cost`](#equate.evaluate.pair_cost)(x, y)                                    | Cost `x * y`: the record pairs a split separates or a merge joins — with it, GMD equals the number of pairwise errors (false positives + false negatives).         |
| [`linear_cost`](#equate.evaluate.linear_cost)([k1, k2])                              | A cost `k1 + k2 * x * y` — the order-independent family GMD is defined for.                                                                                        |
| [`bootstrap_ci`](#equate.evaluate.bootstrap_ci)(pred, gold, left_entities, \*[, ...]) | Percentile bootstrap interval of pairwise precision, recall and F1.                                                                                                |
| [`blocking_metrics`](#equate.evaluate.blocking_metrics)(candidate_pairs, ...[, n_b])      | Return `{pair_completeness, reduction_ratio, pairs_quality, n_candidates}`.                                                                                        |
| [`pr_auc`](#equate.evaluate.pr_auc)(scores, y_true)                             | Average precision: the area under the precision-recall step curve.                                                                                                 |
| [`select_threshold`](#equate.evaluate.select_threshold)(scores, y_true, \*[, optimize])   | The score threshold (`score >= t` predicts a match) that maximises `optimize`.                                                                                     |
| [`ranking_metrics`](#equate.evaluate.ranking_metrics)(ranked, gold, \*[, ks])            | Recall@k, MRR and MAP for ranked candidate lists.                                                                                                                  |
| [`pairs_to_labels`](#equate.evaluate.pairs_to_labels)(pairs, \*, n)                      | Cluster label per record `0..n-1` from the transitive closure of `pairs`.                                                                                          |
| [`labels_to_pairs`](#equate.evaluate.labels_to_pairs)(labels)                            | Every within-cluster pair `(i, j)` with `i < j`.                                                                                                                   |
| [`cross_pairs_to_labels`](#equate.evaluate.cross_pairs_to_labels)(pairs, \*, n_a, n_b)         | Labels over `A + B` (B's records offset by `n_a`) from two-table `(i, j)` pairs.                                                                                   |
| [`entity_split`](#equate.evaluate.entity_split)(entity_ids, \*[, ratios, seed])       | Split records into train / val / test by entity, never by record or pair.                                                                                          |
| [`assert_no_leakage`](#equate.evaluate.assert_no_leakage)(entity_ids, \*parts)             | Raise `AssertionError` if any entity has records in more than one of `parts`.                                                                                      |
| [`list_benchmarks`](#equate.evaluate.list_benchmarks)()                                  | Names of the registered benchmarks.                                                                                                                                |
| [`load_benchmark`](#equate.evaluate.load_benchmark)(name, \*\*config)                   | Build a registered benchmark (`config` goes to its loader).                                                                                                        |
| [`benchmark_matcher`](#equate.evaluate.benchmark_matcher)([matcher, on, n_seeds, ...])     | Score `matcher` on unseen-entity test splits of each benchmark in `on`.                                                                                            |
| [`subset_pairs`](#equate.evaluate.subset_pairs)(gold, rows_a, rows_b)                 | Gold pairs with both ends in the subset, re-indexed into the subset.                                                                                               |
| [`linearity`](#equate.evaluate.linearity)(a, b, gold, \*[, max_pairs])             | Best pairwise F1 of a single threshold on TF-IDF cosine similarity over `a × b`.                                                                                   |
| [`difficulty_profile`](#equate.evaluate.difficulty_profile)(a, b, gold)                     | `{'linearity', 'level', 'base_rate', 'n_a', 'n_b', 'n_gold'}` for a matching task.                                                                                 |
| [`data_dir`](#equate.evaluate.data_dir)()                                         | The default root of the benchmark cache (not created until something is written).                                                                                  |
| [`default_store`](#equate.evaluate.default_store)()                                    | The default benchmark cache: a [`FileStore`](#equate.evaluate.FileStore) over [`data_dir()`](#equate.evaluate.data_dir).        |
| [`fetch`](#equate.evaluate.fetch)(key, source, \*[, store, download, get])     | The bytes of `source`, from `store` if cached, else downloaded and cached.                                                                                         |
| [`abt_buy`](#equate.evaluate.abt_buy)(\*\*config)                                | Abt-Buy: 1081 × 1092 product offers, 1097 matches (Leipzig; textual, hard).                                                                                        |
| [`amazon_google`](#equate.evaluate.amazon_google)(\*\*config)                          | Amazon-Google: 1363 × 3226 software products, 1300 matches (Leipzig; hard).                                                                                        |
| [`dblp_acm`](#equate.evaluate.dblp_acm)(\*\*config)                               | DBLP-ACM: 2616 × 2294 publications, 2224 matches (Leipzig; easy).                                                                                                  |
| [`wdc_products`](#equate.evaluate.wdc_products)(\*[, corner_cases, unseen, ...])      | WDC Products as a two-table benchmark with its canonical train/valid/test split.                                                                                   |

### Classes

| [`Scores`](#equate.evaluate.Scores)(precision, recall, f1[, support])         | Precision, recall and their harmonic mean, plus what they were computed over.   |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`Split`](#equate.evaluate.Split)(train, val, test)                          | Record indices of each part; the entity ids of a part never occur in another.   |
| [`Benchmark`](#equate.evaluate.Benchmark)(name, a, b, gold[, meta, splits, ...]) | Two collections and the gold correspondence between them.                       |
| [`BenchmarkReport`](#equate.evaluate.BenchmarkReport)(rows[, meta])                    | All runs, plus a per-benchmark summary of mean ± std over seeds.                |
| [`BenchmarkRow`](#equate.evaluate.BenchmarkRow)(benchmark, seed, precision, ...)    | One (benchmark, seed) run: the scores, their context, and what they cost.       |
| [`FileStore`](#equate.evaluate.FileStore)(root)                                  | A `MutableMapping[str, bytes]` over the files under `root` (dol-style).         |
| [`Source`](#equate.evaluate.Source)(url[, sha256])                            | A remote file: where it is, and the SHA-256 its bytes must have.                |

### *class* equate.evaluate.Benchmark(name, a, b, gold, meta=dict(), splits=None, pairs=None, ids=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Two collections and the gold correspondence between them.

`gold` holds `(i, j)` pairs: `a[i]` corresponds to `b[j]`. `meta` carries
what a reader needs next to a score — at least `source` and `difficulty`.

Optional, when the suite defines them:

- `splits` — the suite’s canonical record split, `{'train'|'val'|'test':
  (a_indices, b_indices)}`;
- `pairs` — the suite’s labelled candidate pairs, `{'train'|'val'|'test':
  ((i, j, label), ...)}` (the DeepMatcher pair-classification task);
- `ids` — the publisher’s record ids, `{'a': [...], 'b': [...]}` (kept out of
  `meta`, which every report copies).

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

### *class* equate.evaluate.BenchmarkRow(benchmark, seed, precision, recall, f1, seconds, n_a, n_b, n_gold, base_rate, ci=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One (benchmark, seed) run: the scores, their context, and what they cost.

### *class* equate.evaluate.FileStore(root)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

A `MutableMapping[str, bytes]` over the files under `root` (dol-style).

Keys are relative POSIX paths (`'wdc_products/80multi.zip'`) of plain names: no
`..`/`.`/empty segment, backslash, drive or `.part` suffix, so a key cannot escape
`root` on any platform. Writes are atomic even with concurrent writers (a unique
temporary file in the same directory, then a rename), so a reader never sees a
truncated file.

```pycon
>>> import tempfile
>>> s = FileStore(tempfile.mkdtemp())
>>> s['suite/file.csv'] = b'a,b'
>>> list(s), s['suite/file.csv']
(['suite/file.csv'], b'a,b')
```

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

### *class* equate.evaluate.Source(url, sha256=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A remote file: where it is, and the SHA-256 its bytes must have.

`sha256=None` skips the integrity check (for a file whose publisher re-issues it).

### *class* equate.evaluate.Split(train, val, test)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Record indices of each part; the entity ids of a part never occur in another.

### equate.evaluate.abt_buy(\*\*config)

Abt-Buy: 1081 × 1092 product offers, 1097 matches (Leipzig; textual, hard).

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)

### equate.evaluate.amazon_google(\*\*config)

Amazon-Google: 1363 × 3226 software products, 1300 matches (Leipzig; hard).

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)

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

### equate.evaluate.benchmark_matcher(matcher=None, , on=('toy_names',), n_seeds=3, ratios=(0.6, 0.2, 0.2), fit=None, split='auto', ci='auto', n_boot=1000, ci_seed=0)

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

`split` chooses the parts: `'entity'` is the seeded entity split above;
`'canonical'` uses the benchmark’s own `splits` (one run, `n_seeds` and
`ratios` ignored; no entity-leakage assertion, since a suite’s *seen-entity*
variants share entities across parts by design); `'auto'` (default) is canonical
when the benchmark has `splits`, else entity.

`ci` adds a **bootstrap confidence interval** to each row (doc 14 §7.4): the test
*entities* are resampled `n_boot` times (a cluster bootstrap: an entity’s pairs move
together) and the percentile interval of precision, recall and F1 is kept in
`BenchmarkRow.ci`. `'auto'` (default) computes a 95 % interval only for a
single-run benchmark — a canonical split has no seed spread, and a 500-pair test set
leaves a few points of F1 within noise; a level (`0.9`) computes it for every row;
`None` never. The resampling is seeded (`ci_seed`), so a report is reproducible.
A percentile interval cannot show uncertainty at a 0/1 boundary: a perfect run reads
`1.000 [1.000, 1.000]` — that is “no resample found an error”, not certainty.

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

### equate.evaluate.bootstrap_ci(pred, gold, left_entities, , level=0.95, n_boot=1000, seed=0)

Percentile bootstrap interval of pairwise precision, recall and F1.

`pred` / `gold` are `(i, j)` pairs; `left_entities[i]` is the entity of left
record `i`. Entities (not pairs) are resampled with replacement and each pair is
weighted by how often its **left** record’s entity was drawn, so the pairs of one
entity succeed or fail together (a false positive counts against the entity of its
left record; entities with no left record in the test part are not resampled).

A resample with no predicted pair has no precision, and one with no gold pair no
recall: those resamples are left out of that metric’s interval (scoring them 1.0, as
[`Scores`](#equate.evaluate.Scores) does for a whole empty run, would inflate it). An
interval with no valid resample is `(nan, nan)`. Returns `{'precision': (lo,
hi), 'recall': ..., 'f1': ...}`. Memory is bounded: resamples are drawn in chunks.

```pycon
>>> ci = bootstrap_ci({(0, 0), (1, 1)}, {(0, 0), (1, 2)}, ['e0', 'e1'], n_boot=200)
>>> ci['precision'][0] <= 0.5 <= ci['precision'][1]
True
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

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

### equate.evaluate.data_dir()

The default root of the benchmark cache (not created until something is written).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### equate.evaluate.dblp_acm(\*\*config)

DBLP-ACM: 2616 × 2294 publications, 2224 matches (Leipzig; easy).

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)

### equate.evaluate.default_store()

The default benchmark cache: a [`FileStore`](#equate.evaluate.FileStore) over [`data_dir()`](#equate.evaluate.data_dir).

* **Return type:**
  [`FileStore`](equate.evaluate.store.html.md#equate.evaluate.store.FileStore)

### equate.evaluate.difficulty_profile(a, b, gold)

`{'linearity', 'level', 'base_rate', 'n_a', 'n_b', 'n_gold'}` for a matching task.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

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

### equate.evaluate.fetch(key, source, , store=None, download=True, get=None)

The bytes of `source`, from `store` if cached, else downloaded and cached.

The bytes are checked against `source.sha256` **every time**, cached or not: a
cached copy that no longer matches (corrupted on disk, or a pin updated for a
re-issued file) is downloaded again; a download that does not match raises
`ValueError` and nothing is written. With `download=False` a missing or stale
key raises `FileNotFoundError` instead of reaching the network. `get(url) ->
bytes` is the downloader (default `http_get()`).

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### equate.evaluate.gmd(pred_labels, true_labels, \*, merge_cost=<function unit_cost>, split_cost=<function unit_cost>)

Generalized merge distance: the cost of the splits and merges that turn the
predicted clustering into the true one (Menestrina, Whang & Garcia-Molina, VLDB 2010).

`split_cost(x, y)` prices splitting a cluster of `x + y` records into `x` and
`y`; `merge_cost(x, y)` merging an `x`- and a `y`-record cluster. The defaults
([`unit_cost()`](#equate.evaluate.unit_cost)) count operations; [`pair_cost()`](#equate.evaluate.pair_cost) for both gives the pairwise
error count; [`linear_cost()`](#equate.evaluate.linear_cost) `(k1, k2)` is `k1 + k2·x·y`. Unlike F-measures it
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

### equate.evaluate.labels_to_pairs(labels)

Every within-cluster pair `(i, j)` with `i < j`.

```pycon
>>> sorted(labels_to_pairs([0, 0, 1, 0]))
[(0, 1), (0, 3), (1, 3)]
```

* **Return type:**
  [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]]

### equate.evaluate.linear_cost(k1=0.0, k2=1.0)

A cost `k1 + k2 * x * y` — the order-independent family GMD is defined for.

```pycon
>>> linear_cost(1, 0)(3, 4), linear_cost(0, 1)(3, 4)
(1.0, 12.0)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)

### equate.evaluate.linearity(a, b, gold, , max_pairs=10000000)

Best pairwise F1 of a single threshold on TF-IDF cosine similarity over `a × b`.

Records (mappings) are compared through their joined values. Ties are cut together
(a threshold cannot split equal scores). `0.0` when there is nothing to find.
Every pair is scored densely (peak ~130 bytes per pair), so inputs over `max_pairs`
pairs raise `ValueError` rather than exhaust memory; raise the limit knowingly.

```pycon
>>> round(linearity(['apple pie', 'kiwi'], ['kiwi', 'apple pie'], {(0, 1), (1, 0)}), 2)
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### equate.evaluate.list_benchmarks()

Names of the registered benchmarks.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### equate.evaluate.load_benchmark(name, \*\*config)

Build a registered benchmark (`config` goes to its loader).

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)

### equate.evaluate.pair_cost(x, y)

Cost `x * y`: the record pairs a split separates or a merge joins — with it,
GMD equals the number of pairwise errors (false positives + false negatives).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

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

### equate.evaluate.subset_pairs(gold, rows_a, rows_b)

Gold pairs with both ends in the subset, re-indexed into the subset.

```pycon
>>> sorted(subset_pairs({(0, 1), (2, 0)}, [2, 0], [0, 1]))
[(0, 0), (1, 1)]
```

### equate.evaluate.unit_cost(x, y)

Cost 1 per split or merge, whatever the sizes (GMD counts operations).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### equate.evaluate.wdc_products(, corner_cases=80, unseen=100, dev_size='large', seed=0, store=None, download=True, get=None, verify=True, profile=True)

WDC Products as a two-table benchmark with its canonical train/valid/test split.

The defaults pick the hardest generalization variant — 80 % corner cases, every test
entity unseen in training (research doc 14, §7.6) — with the large development set.
Gold pairs every `a`/`b` offer of the same product, across parts too (a seen test
product matches its training offers); only the test-part gold is scored. Row order
within each part is shuffled with `seed`. `meta['leakage']` counts test offers
that also appear, identically, in train or valid. WDC’s own files have such overlap:
hundreds in the 50 %-unseen variants, a few in valid for 0 %. A warning says so.

* **Return type:**
  [`Benchmark`](equate.evaluate.benchmark.html.md#equate.evaluate.benchmark.Benchmark)

### Modules

| [`suites`](equate.evaluate.suites.html.md#module-equate.evaluate.suites)         | Loaders for the standard matching benchmarks, fetched on first use into a store.        |
|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`benchmark`](equate.evaluate.benchmark.html.md#module-equate.evaluate.benchmark)   | Self-benchmarking: a benchmark registry and an honest-by-default harness.               |
| [`difficulty`](equate.evaluate.difficulty.html.md#module-equate.evaluate.difficulty) | How hard is a benchmark? A measured difficulty profile to print next to a score.        |
| [`labels`](equate.evaluate.labels.html.md#module-equate.evaluate.labels)         | Convert between the two granularities: pairs and cluster labels.                        |
| [`metrics`](equate.evaluate.metrics.html.md#module-equate.evaluate.metrics)       | Pairwise and cluster-level match metrics, dependency-light.                             |
| [`ranking`](equate.evaluate.ranking.html.md#module-equate.evaluate.ranking)       | Threshold-free and ranking metrics, and threshold selection as its own step.            |
| [`scores`](equate.evaluate.scores.html.md#module-equate.evaluate.scores)         | The one score type every metric returns: precision and recall, never F1 alone.          |
| [`split`](equate.evaluate.split.html.md#module-equate.evaluate.split)           | Leakage-aware splitting: split over entities, so test entities are unseen.              |
| [`store`](equate.evaluate.store.html.md#module-equate.evaluate.store)           | Where downloaded benchmark data lives: a store (a `MutableMapping`), never the package. |
