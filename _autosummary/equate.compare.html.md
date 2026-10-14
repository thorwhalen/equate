# equate.compare

The compare stage (②): score how alike two (featurized) items are.

There are two routes (decision register / doc 00-taxonomy):

- **featurize-then-compare** ([`featurized()`](#equate.compare.featurized)) — an *indexable* vector metric over
  independent representations (what the legacy `similarity_matrix` does); only this
  route can drive an ANN/LSH index;
- **direct pairwise compare** ([`direct()`](#equate.compare.direct)) — an *opaque* scorer `h(a, b) -> float`
  (edit distance, a cross-encoder/LLM); it can only re-rank a candidate set.

Pairwise comparators are `(a, b) -> float` callables; select one by name via the
[`comparators`](#equate.compare.comparators) registry or pass a callable. Each carries a
[`equate.base.ComparatorMeta`](equate.base.html.md#equate.base.ComparatorMeta) (polarity / bounded / is_metric / is_symmetric) so
the framework adapts — never treating a non-metric as a metric, keeping directional
comparators (Monge-Elkan, containment) directional until the matcher boundary (D3).

### Module Attributes

| [`comparators`](#equate.compare.comparators)   | the compare-stage strategy registry (name -> lazy factory)   |
|----------------------------------------------------------------|--------------------------------------------------------------|

### Functions

| [`resolve_comparator`](#equate.compare.resolve_comparator)(spec, \*\*config)           | Resolve `spec` to a pairwise comparator.                                                                                             |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [`direct`](#equate.compare.direct)(h, \*[, meta])                          | Wrap an opaque pairwise scorer `h(a, b) -> float` into a score-matrix builder `build(A, B) -> ndarray`.                              |
| [`polarity_sense`](#equate.compare.polarity_sense)(comparator, \*[, default])      | The score `sense` a comparator's declared polarity implies.                                                                          |
| [`featurized`](#equate.compare.featurized)([featurizer, metric, meta])         | Build a score-matrix builder that featurizes both collections then applies a vector `metric` (default cosine over the two matrices). |
| [`comparison_vector`](#equate.compare.comparison_vector)(a, b, field_comparators, \*) | Return `{field: comparator(a[field], b[field])}` for each field comparator.                                                          |
| [`weighted_sum`](#equate.compare.weighted_sum)([weights])                        | Combiner: weighted sum of a comparison vector (default weight 1.0 per field).                                                        |
| [`mean`](#equate.compare.mean)(cv)                                       | Combiner: arithmetic mean of a comparison vector.                                                                                    |
| [`max_combiner`](#equate.compare.max_combiner)(cv)                               | Combiner: maximum field score in a comparison vector.                                                                                |
| [`fellegi_sunter`](#equate.compare.fellegi_sunter)(m_probs, u_probs, \*[, eps])    | Fellegi-Sunter combiner (decision register D5): sum of per-field match weights.                                                      |
| [`threshold`](#equate.compare.threshold)(t, \*[, hard])                       | Return a calibrator: `hard` -> boolean `score >= t`; else pass the score through.                                                    |
| [`platt`](#equate.compare.platt)(scores, labels)                          | Fit Platt scaling (logistic regression on 1-D scores) -> `P(match)`.                                                                 |
| [`isotonic`](#equate.compare.isotonic)(scores, labels)                       | Fit isotonic regression -> a monotone calibrated probability.                                                                        |

### equate.compare.comparators *= <equate.registry.Registry object>*

the compare-stage strategy registry (name -> lazy factory)

### equate.compare.comparison_vector(a, b, field_comparators, , missing=0.0)

Return `{field: comparator(a[field], b[field])}` for each field comparator.

`field_comparators` maps a field name to a pairwise comparator; records may be
mappings or objects (attribute access). A field whose value is *missing* (`None`)
in either record scores `missing` (default 0.0) rather than being passed to the
comparator — many comparators (edit distance, ratio) raise on `None`.

```pycon
>>> from equate.compare.string import levenshtein
>>> cv = comparison_vector({'name': 'jon'}, {'name': 'john'}, {'name': levenshtein})
>>> round(cv['name'], 2)
0.75
```

### equate.compare.direct(h, , meta=None)

Wrap an opaque pairwise scorer `h(a, b) -> float` into a score-matrix builder
`build(A, B) -> ndarray`. Marked **not indexable** — usable only to re-rank a
candidate set, never to drive an ANN/LSH index.

### equate.compare.featurized(featurizer='tfidf', metric=None, , meta=None)

Build a score-matrix builder that featurizes both collections then applies a
vector `metric` (default cosine over the two matrices). Marked **indexable** — the
featurize-then-compare route; this is what the legacy `similarity_matrix` does.

### equate.compare.fellegi_sunter(m_probs, u_probs, , eps=1e-06)

Fellegi-Sunter combiner (decision register D5): sum of per-field match weights.

For each field with agreement `s` in [0, 1], contributes
`s*log(m/u) + (1-s)*log((1-m)/(1-u))` — the log-likelihood-ratio “match weight”.
`m_probs` / `u_probs` are the m- and u-probabilities per field (from labels or
EM; the EM estimation lives in the classify stage, roadmap #8). Returns a log-odds
score (higher = more likely a match). Probabilities are clamped to `[eps, 1-eps]`
so `m`/`u` of exactly 0 or 1 can’t cause `log(0)` / division-by-zero.

### equate.compare.isotonic(scores, labels)

Fit isotonic regression -> a monotone calibrated probability.

Requires `equate[sklearn]`. Returns a calibrator mapping a raw score to a
probability.

### equate.compare.max_combiner(cv)

Combiner: maximum field score in a comparison vector.

### equate.compare.mean(cv)

Combiner: arithmetic mean of a comparison vector.

### equate.compare.platt(scores, labels)

Fit Platt scaling (logistic regression on 1-D scores) -> `P(match)`.

Requires `equate[sklearn]`. Returns a calibrator mapping a raw score to a
probability.

### equate.compare.polarity_sense(comparator, , default='maximize')

The score `sense` a comparator’s declared polarity implies.

A `'distance'` comparator (DTW, edit distance) is minimized, a `'similarity'`
maximized; a comparator with no declared metadata gets `default`.

```pycon
>>> polarity_sense(resolve_comparator('dtw'))
'minimize'
>>> polarity_sense(resolve_comparator('ratio'))
'maximize'
```

### equate.compare.resolve_comparator(spec, \*\*config)

Resolve `spec` to a pairwise comparator.

A callable passes through; a registered name (`str`) is built via its factory
(`**config` forwarded, so an unknown option raises instead of being ignored).

### equate.compare.threshold(t, , hard=True)

Return a calibrator: `hard` -> boolean `score >= t`; else pass the score through.

### equate.compare.weighted_sum(weights=None)

Combiner: weighted sum of a comparison vector (default weight 1.0 per field).

### Modules

| [`calibrate`](equate.compare.calibrate.html.md#module-equate.compare.calibrate)     | Score calibration (stage ②): map raw comparator scores to a decision or a probability.                                                                      |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`numeric_geo`](equate.compare.numeric_geo.html.md#module-equate.compare.numeric_geo) | Numeric, date, and geospatial comparators (stage ②): turn a domain distance into a [0, 1] similarity via a decay function.                                  |
| [`sequence`](equate.compare.sequence.html.md#module-equate.compare.sequence)       | Sequence-alignment comparators (stage ②): score two *sequences* by aligning them.                                                                           |
| [`string`](equate.compare.string.html.md#module-equate.compare.string)           | String comparators (stage ②): pairwise similarity between two strings.                                                                                      |
| [`vector`](equate.compare.vector.html.md#module-equate.compare.vector)           | Vector comparators (stage ②): pairwise similarity/distance between two vectors.                                                                             |
| [`vectorize`](equate.compare.vectorize.html.md#module-equate.compare.vectorize)     | Multi-field record comparison (stage ②): build a per-pair *comparison vector* from per-field comparators, then reduce it to a single score with a combiner. |
