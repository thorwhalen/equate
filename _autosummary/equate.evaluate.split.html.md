# equate.evaluate.split

Leakage-aware splitting: split over entities, so test entities are unseen.

The most common way a matching benchmark lies is a pair-level split: the same real-world
entity shows up in train and test, and the model memorises it (research doc 14, §7.2).
[`entity_split()`](#equate.evaluate.split.entity_split) makes the right split the easy one — it partitions **entity ids**,
so every record of an entity lands in exactly one part — and [`assert_no_leakage()`](#equate.evaluate.split.assert_no_leakage)
is the cheap post-hoc guard for a split made some other way.

### Functions

| [`assert_no_leakage`](#equate.evaluate.split.assert_no_leakage)(entity_ids, \*parts)       | Raise `AssertionError` if any entity has records in more than one of `parts`.   |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`entity_split`](#equate.evaluate.split.entity_split)(entity_ids, \*[, ratios, seed]) | Split records into train / val / test by entity, never by record or pair.       |

### Classes

| [`Split`](#equate.evaluate.split.Split)(train, val, test)   | Record indices of each part; the entity ids of a part never occur in another.   |
|----------------------------------------------------------------------------|---------------------------------------------------------------------------------|

### *class* equate.evaluate.split.Split(train, val, test)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Record indices of each part; the entity ids of a part never occur in another.

### equate.evaluate.split.assert_no_leakage(entity_ids, \*parts)

Raise `AssertionError` if any entity has records in more than one of `parts`.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> assert_no_leakage([0, 0, 1], [0], [1])
Traceback (most recent call last):
    ...
AssertionError: entity 0 appears in parts 0 and 1 — the split leaks it
```

### equate.evaluate.split.entity_split(entity_ids, , ratios=(0.6, 0.2, 0.2), seed=0)

Split records into train / val / test by entity, never by record or pair.

`entity_ids[i]` is record `i`’s true entity (e.g. from
[`cross_pairs_to_labels()`](equate.evaluate.html.md#equate.evaluate.cross_pairs_to_labels) over a two-table gold matching).
Entities, not records, are shuffled (`seed`) and cut by `ratios`, so the parts
are disjoint in entities and roughly — not exactly — proportional in records.

* **Return type:**
  [`Split`](#equate.evaluate.split.Split)

```pycon
>>> s = entity_split([0, 0, 1, 1, 2, 3, 4, 4, 5, 6], seed=1)
>>> assert_no_leakage([0, 0, 1, 1, 2, 3, 4, 4, 5, 6], s.train, s.test)
>>> sorted(s.train + s.val + s.test) == list(range(10))
True
```
