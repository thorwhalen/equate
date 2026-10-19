# equate.featurize.columns

Column featurizers: a column (a sequence of cell values) as a comparable object.

Lifting the pipeline from cells to columns (research doc 07, §8.1): a column’s values are
its signal when headers are missing or untrustworthy. [`value_set()`](#equate.featurize.columns.value_set) represents each
column as the set of its distinct, normalized, non-missing values — the input of the set
comparators (`jaccard`, `overlap`, `containment`) used for join-key discovery and
near-duplicate columns.

### Functions

| [`value_set`](#equate.featurize.columns.value_set)(columns, \*[, normalize])   | Each column as the `frozenset` of its normalized, non-missing values (batched).                                                                     |
|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| [`normalize_value`](#equate.featurize.columns.normalize_value)(v)                    | The default cell normalization: text, stripped and lower-cased; `1.0` and `1` read alike (an integral real drops its `.0`, numpy scalars included). |
| [`is_missing`](#equate.featurize.columns.is_missing)(v)                         | `None`, an empty/blank string, a NaN of any float type, or pandas' `NA` / `NaT`.                                                                    |

### equate.featurize.columns.is_missing(v)

`None`, an empty/blank string, a NaN of any float type, or pandas’ `NA` / `NaT`.

```pycon
>>> [is_missing(v) for v in (None, ' ', float('nan'), 0, 'x')]
[True, True, True, False, False]
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### equate.featurize.columns.normalize_value(v)

The default cell normalization: text, stripped and lower-cased; `1.0` and `1`
read alike (an integral real drops its `.0`, numpy scalars included).

```pycon
>>> normalize_value('  Paris '), normalize_value(3.0), normalize_value(3)
('paris', '3', '3')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### equate.featurize.columns.value_set(columns, \*, normalize=<function normalize_value>)

Each column as the `frozenset` of its normalized, non-missing values (batched).

```pycon
>>> [sorted(s) for s in value_set([['Paris', 'paris ', None], [1, 2.0]])]
[['paris'], ['1', '2']]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)
