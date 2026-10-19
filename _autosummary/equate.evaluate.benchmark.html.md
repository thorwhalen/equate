# equate.evaluate.benchmark

Self-benchmarking: a benchmark registry and an honest-by-default harness.

[`benchmark_matcher()`](#equate.evaluate.benchmark.benchmark_matcher) is where “measure honestly” becomes the path of least
resistance (research doc 14, §8.4): it splits **by entity** so test entities are unseen,
fits a trainable matcher on train only (and asserts the split did not leak), scores the
test part only, reports the test **base rate** and a **cost column** (wall-clock seconds)
next to every score, and repeats over seeds so a number comes with its spread. A seed
whose test part holds no gold pair is skipped with a warning rather than scored as a
vacuous perfect 1.0.

The scores are **pairwise** (precision, recall, F1). B³ is deliberately not reported for
a two-table matching: every unmatched record is a singleton there, and singletons score
well under B³ whatever the matcher did — an empty prediction would read ~0.75.

Benchmarks are a registry, so adding one is registering a loader, not editing the core.
The built-in `toy_names` is generated (no download, deterministic): enough to exercise a
matcher and this harness in CI. The standard suites — `abt_buy`, `amazon_google`,
`dblp_acm` and `wdc_products` — are registered by [`equate.evaluate.suites`](equate.evaluate.suites.html.md#module-equate.evaluate.suites) and
download on first use into a store ([`equate.evaluate.store`](equate.evaluate.store.html.md#module-equate.evaluate.store)), never the package.

A benchmark that ships its own **canonical split** (WDC Products’ unseen-entity test sets)
carries it in `Benchmark.splits`, and [`benchmark_matcher()`](#equate.evaluate.benchmark.benchmark_matcher) uses it by default
(`split='auto'`): the suite’s designers chose those test sets to control difficulty, and
published numbers are measured on them.

### Module Attributes

| [`benchmarks`](#equate.evaluate.benchmark.benchmarks)   | name -> factory(\*\*config) returning a [`Benchmark`](#equate.evaluate.benchmark.Benchmark)   |
|---------------------------------------------------------------|------------------------------------------------------------------------------------------------------|

### Functions

| [`benchmark_matcher`](#equate.evaluate.benchmark.benchmark_matcher)([matcher, on, n_seeds, ...])   | Score `matcher` on unseen-entity test splits of each benchmark in `on`.   |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`list_benchmarks`](#equate.evaluate.benchmark.list_benchmarks)()                                | Names of the registered benchmarks.                                       |
| [`load_benchmark`](#equate.evaluate.benchmark.load_benchmark)(name, \*\*config)                 | Build a registered benchmark (`config` goes to its loader).               |
| [`subset_pairs`](#equate.evaluate.benchmark.subset_pairs)(gold, rows_a, rows_b)               | Gold pairs with both ends in the subset, re-indexed into the subset.      |

### Classes

| [`Benchmark`](#equate.evaluate.benchmark.Benchmark)(name, a, b, gold[, meta, splits, ...])   | Two collections and the gold correspondence between them.                 |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`BenchmarkReport`](#equate.evaluate.benchmark.BenchmarkReport)(rows[, meta])                      | All runs, plus a per-benchmark summary of mean ± std over seeds.          |
| [`BenchmarkRow`](#equate.evaluate.benchmark.BenchmarkRow)(benchmark, seed, precision, ...)      | One (benchmark, seed) run: the scores, their context, and what they cost. |

### *class* equate.evaluate.benchmark.Benchmark(name, a, b, gold, meta=dict(), splits=None, pairs=None, ids=None)

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

### *class* equate.evaluate.benchmark.BenchmarkReport(rows, meta=dict())

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

### *class* equate.evaluate.benchmark.BenchmarkRow(benchmark, seed, precision, recall, f1, seconds, n_a, n_b, n_gold, base_rate)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One (benchmark, seed) run: the scores, their context, and what they cost.

### equate.evaluate.benchmark.benchmark_matcher(matcher=None, , on=('toy_names',), n_seeds=3, ratios=(0.6, 0.2, 0.2), fit=None, split='auto')

Score `matcher` on unseen-entity test splits of each benchmark in `on`.

`matcher(a, b)` returns `(i, j)` pairs (a [`Matching`](equate.base.html.md#equate.base.Matching)
qualifies); the default is `equate.match()`. `on` is a benchmark name, a
[`Benchmark`](#equate.evaluate.benchmark.Benchmark), or an iterable of either. Scores are pairwise; a seed whose test
part has no gold pair is skipped with a warning, and a benchmark with no scorable
seed is an error. For each of `n_seeds` seeds the records of `a + b`
are split by entity (`ratios` = train/val/test); only the **test** part is
matched and scored, so a test entity was never seen in training.

`fit(train_a, train_b, train_pairs, val_a, val_b, val_pairs) -> matcher` makes the
harness train per seed: it gets the train and validation parts only (the split is
asserted leak-free first), and the matcher it returns is scored on test. Choose a
threshold on the validation part there ([`select_threshold()`](equate.evaluate.html.md#equate.evaluate.select_threshold)),
never on test.

`split` chooses the parts: `'entity'` is the seeded entity split above;
`'canonical'` uses the benchmark’s own `splits` (one run, `n_seeds` and
`ratios` ignored; no entity-leakage assertion, since a suite’s *seen-entity*
variants share entities across parts by design); `'auto'` (default) is canonical
when the benchmark has `splits`, else entity.

```pycon
>>> report = benchmark_matcher(n_seeds=2)
>>> s = report.summary()['toy_names']
>>> s['f1'][0] > 0.8, s['n_seeds']
(True, 2)
```

* **Return type:**
  [`BenchmarkReport`](#equate.evaluate.benchmark.BenchmarkReport)

### equate.evaluate.benchmark.benchmarks *= <equate.registry.Registry object>*

name -> factory(\*\*config) returning a [`Benchmark`](#equate.evaluate.benchmark.Benchmark)

### equate.evaluate.benchmark.list_benchmarks()

Names of the registered benchmarks.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### equate.evaluate.benchmark.load_benchmark(name, \*\*config)

Build a registered benchmark (`config` goes to its loader).

* **Return type:**
  [`Benchmark`](#equate.evaluate.benchmark.Benchmark)

### equate.evaluate.benchmark.subset_pairs(gold, rows_a, rows_b)

Gold pairs with both ends in the subset, re-indexed into the subset.

```pycon
>>> sorted(subset_pairs({(0, 1), (2, 0)}, [2, 0], [0, 1]))
[(0, 0), (1, 1)]
```
