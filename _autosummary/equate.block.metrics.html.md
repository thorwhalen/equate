# equate.block.metrics

Blocking evaluation metrics (stage ②): quantify the recall/efficiency trade-off of a
candidate set. Pair completeness (blocking recall) upper-bounds the whole system’s
recall, so it is the metric to protect (doc 02-blocking).

### Functions

| [`blocking_metrics`](#equate.block.metrics.blocking_metrics)(candidate_pairs, ...[, n_b])   | Return `{pair_completeness, reduction_ratio, pairs_quality, n_candidates}`.   |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|

### equate.block.metrics.blocking_metrics(candidate_pairs, true_pairs, , n_a, n_b=None)

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
