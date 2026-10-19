# equate.evaluate.scores

The one score type every metric returns: precision and recall, never F1 alone.

[`Scores`](#equate.evaluate.scores.Scores) is the SSOT for how a match quality figure is represented (research
doc 14, §2.5): an F1 without the precision/recall it averages hides whether a matcher
is over- or under-linking, so the type carries all three plus the support they were
computed over.

### Classes

| [`Scores`](#equate.evaluate.scores.Scores)(precision, recall, f1[, support])   | Precision, recall and their harmonic mean, plus what they were computed over.   |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|

### *class* equate.evaluate.scores.Scores(precision, recall, f1, support=None)

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
  [`Scores`](#equate.evaluate.scores.Scores)

#### *classmethod* from_pr(precision, recall, , support=None)

Build from precision and recall; F1 is derived, so it can never disagree.

* **Return type:**
  [`Scores`](#equate.evaluate.scores.Scores)

#### support *: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/builtins/functions.html#int)]* *= None*

the size of the gold side (true pairs, or records for a cluster metric)
