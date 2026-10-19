# equate.evaluate.difficulty

How hard is a benchmark? A measured difficulty profile to print next to a score.

The popular matching benchmarks are mostly *easy*: a single similarity threshold already
separates their matches from their non-matches (Papadakis et al. 2023, “degree of
linearity”; Primpeli & Bizer 2020 — research doc 14, §7.6). So a 98% F1 means little
until you know what a trivial matcher gets. [`linearity()`](#equate.evaluate.difficulty.linearity) is that trivial matcher’s
best case: the highest pairwise F1 any **one** threshold on TF-IDF cosine similarity
reaches over **all** `a × b` pairs. Near 1 the benchmark cannot tell matchers apart;
well below it, it can.

It is adapted from Papadakis et al., who threshold the *labelled* pairs of a split; over
the full cross product it also charges the threshold for every non-candidate pair, which
is the setting [`benchmark_matcher()`](equate.evaluate.html.md#equate.evaluate.benchmark_matcher) scores.

### Functions

| [`linearity`](#equate.evaluate.difficulty.linearity)(a, b, gold, \*[, max_pairs])            | Best pairwise F1 of a single threshold on TF-IDF cosine similarity over `a × b`.                              |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [`difficulty_profile`](#equate.evaluate.difficulty.difficulty_profile)(a, b, gold)                    | `{'linearity', 'level', 'base_rate', 'n_a', 'n_b', 'n_gold'}` for a matching task.                            |
| [`difficulty_label`](#equate.evaluate.difficulty.difficulty_label)(lin, \*[, easy_at, moderate_at]) | `'easy'` / `'moderate'` / `'hard'` from a [`linearity()`](#equate.evaluate.difficulty.linearity) value. |

### equate.evaluate.difficulty.difficulty_label(lin, , easy_at=0.9, moderate_at=0.7)

`'easy'` / `'moderate'` / `'hard'` from a [`linearity()`](#equate.evaluate.difficulty.linearity) value.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### equate.evaluate.difficulty.difficulty_profile(a, b, gold)

`{'linearity', 'level', 'base_rate', 'n_a', 'n_b', 'n_gold'}` for a matching task.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### equate.evaluate.difficulty.linearity(a, b, gold, , max_pairs=10000000)

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
