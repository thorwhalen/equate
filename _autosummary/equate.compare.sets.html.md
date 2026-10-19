# equate.compare.sets

Set comparators: how much two *value sets* overlap (stage ②, for columns and bags).

When the objects are sets — a column’s distinct values, a record’s tokens — overlap is
the signal (research doc 07, §4-5). Three standard measures, all in `[0, 1]`:

- [`jaccard()`](#equate.compare.sets.jaccard) `|A ∩ B| / |A ∪ B|` — symmetric; near-duplicate detection;
- [`overlap()`](#equate.compare.sets.overlap) `|A ∩ B| / min(|A|, |B|)` — symmetric; forgives a size mismatch;
- [`containment()`](#equate.compare.sets.containment) `|A ∩ B| / |A|` — **directional** (how much of `A` is in `B`):
  the joinability score of LSH Ensemble / JOSIE, so `containment(fk, pk)` is high when
  a foreign key’s values all appear in the primary key, whatever else the latter holds.

Two empty sets score `0.0` (no evidence of correspondence), never `nan`.

### Functions

| [`jaccard`](#equate.compare.sets.jaccard)(a, b)     | `|A ∩ B| / |A ∪ B|`.                                                   |
|--------------------------------------------------------------------|------------------------------------------------------------------------|
| [`overlap`](#equate.compare.sets.overlap)(a, b)     | `|A ∩ B| / min(|A|, |B|)` (the overlap coefficient).                   |
| [`containment`](#equate.compare.sets.containment)(a, b) | `|A ∩ B| / |A|`: the share of `a`'s values found in `b` (directional). |

### equate.compare.sets.containment(a, b)

`|A ∩ B| / |A|`: the share of `a`’s values found in `b` (directional).

```pycon
>>> containment({1, 2}, {1, 2, 3, 4}), containment({1, 2, 3, 4}, {1, 2})
(1.0, 0.5)
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### equate.compare.sets.jaccard(a, b)

`|A ∩ B| / |A ∪ B|`.

```pycon
>>> jaccard({1, 2, 3}, {2, 3, 4})
0.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### equate.compare.sets.overlap(a, b)

`|A ∩ B| / min(|A|, |B|)` (the overlap coefficient).

```pycon
>>> overlap({1, 2}, {1, 2, 3, 4})
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
