# equate.compare.vectorize

Multi-field record comparison (stage ②): build a per-pair *comparison vector* from
per-field comparators, then reduce it to a single score with a combiner.

This is the record-matching bridge: the same field comparators can feed a raw score, a
probabilistic Fellegi-Sunter weight, or (later) a learned classifier interchangeably —
the “key” architectural move of decision register D5.

### Functions

| [`comparison_vector`](#equate.compare.vectorize.comparison_vector)(a, b, field_comparators, \*)   | Return `{field: comparator(a[field], b[field])}` for each field comparator.     |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`weighted_sum`](#equate.compare.vectorize.weighted_sum)([weights])                          | Combiner: weighted sum of a comparison vector (default weight 1.0 per field).   |
| [`mean`](#equate.compare.vectorize.mean)(cv)                                         | Combiner: arithmetic mean of a comparison vector.                               |
| [`max_combiner`](#equate.compare.vectorize.max_combiner)(cv)                                 | Combiner: maximum field score in a comparison vector.                           |
| [`fellegi_sunter`](#equate.compare.vectorize.fellegi_sunter)(m_probs, u_probs, \*[, eps])      | Fellegi-Sunter combiner (decision register D5): sum of per-field match weights. |

### equate.compare.vectorize.comparison_vector(a, b, field_comparators, , missing=0.0)

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

### equate.compare.vectorize.fellegi_sunter(m_probs, u_probs, , eps=1e-06)

Fellegi-Sunter combiner (decision register D5): sum of per-field match weights.

For each field with agreement `s` in [0, 1], contributes
`s*log(m/u) + (1-s)*log((1-m)/(1-u))` — the log-likelihood-ratio “match weight”.
`m_probs` / `u_probs` are the m- and u-probabilities per field (from labels or
EM; the EM estimation lives in the classify stage, roadmap #8). Returns a log-odds
score (higher = more likely a match). Probabilities are clamped to `[eps, 1-eps]`
so `m`/`u` of exactly 0 or 1 can’t cause `log(0)` / division-by-zero.

### equate.compare.vectorize.max_combiner(cv)

Combiner: maximum field score in a comparison vector.

### equate.compare.vectorize.mean(cv)

Combiner: arithmetic mean of a comparison vector.

### equate.compare.vectorize.weighted_sum(weights=None)

Combiner: weighted sum of a comparison vector (default weight 1.0 per field).
