# equate.facade

The equate facade: the one-line `match(A, B)` entry point.

Wires the pipeline — featurize → compare (over the blocker’s candidates) → match — with
sensible defaults, and returns a structured [`equate.base.Matching`](equate.base.html.md#equate.base.Matching). Every stage is
swappable by one keyword (decision register D4): `featurize=`, `compare=`, `block=`,
`how=`. This is the canonical case (doc 10 §1): two collections -> optimally-matched
(and scored) pairs, zero config.

### Functions

| [`match`](#equate.facade.match)(A[, B, featurize, compare, block, ...])     | Match two collections into a [`Matching`](equate.base.html.md#equate.base.Matching) of scored `(i, j)` pairs.                                  |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`score_matrix`](#equate.facade.score_matrix)(A, B, \*[, featurize, compare, ...]) | Score `A` against `B` into a [`ScoreMatrix`](equate.base.html.md#equate.base.ScoreMatrix) — stages ① and ②.                                    |
| [`dedupe`](#equate.facade.dedupe)(A, \*[, compare, block, threshold, ...])   | Deduplicate a single collection into entity groups.                                                                                                                    |
| [`resolve`](#equate.facade.resolve)(\*collections, \*\*kwargs)                | Resolve entities across multiple collections: pool them (in order) and [`dedupe()`](#equate.facade.dedupe) -> a Partition over the pooled items. |

### equate.facade.dedupe(A, , compare='ratio', block=None, threshold=0.7, cluster='connected_components', sense=None)

Deduplicate a single collection into entity groups.

Self-compares `A`’s candidate pairs, keeps those passing `threshold`, and clusters
them -> a [`Partition`](equate.base.html.md#equate.base.Partition) over `A` (iterate it for the duplicate
`(i, j)` pairs; `.groups()` for the clusters). `block` (a blocker) restricts the
candidate pairs (self-join); the default compares all `i < j` pairs. `cluster` is
`'connected_components'` (transitive closure) or `'correlation'`. `sense` defaults
to the comparator’s declared polarity — a *distance* comparator (e.g.
`levenshtein_distance`) uses `'minimize'` so *lower* scores link, a similarity uses
`'maximize'`.

```pycon
>>> from equate import dedupe
>>> dedupe(['Jon Smith', 'Jon Smyth', 'Kate Doe'], threshold=0.8).groups()
{0: [0, 1], 1: [2]}
```

### equate.facade.match(A, B=None, , featurize=None, compare=None, block=None, how='assign', threshold=None, cluster='connected_components', sense=None)

Match two collections into a [`Matching`](equate.base.html.md#equate.base.Matching) of scored `(i, j)` pairs.

Pipeline (each stage swappable by one keyword):

- **featurize / compare** — with `compare=None` (default), the *vector route*:
  featurize both sides with `featurize` (TF-IDF by default) and score by cosine.
  With a `compare` comparator (a registered name or a callable), the \*direct
  pairwise route\* on the raw items.
- **block** — `None` (default) scores all pairs (dense); a blocker (name/callable)
  scores only the candidate pairs into a sparse matrix, and the matcher stays sparse-aware.
- **sense** — `None` (default) reads it off the comparator’s declared polarity: a
  *distance* comparator (`'dtw'`, `'levenshtein_distance'`) is minimized, anything
  else maximized. Pass `'maximize'` / `'minimize'` to override.
- **how** — the matcher objective: `'assign'` (optimal 1:1, the default),
  `'greedy'`, `'stable'`, or `'soft'` (an optimal-transport plan; `.plan` is set).

Returns a `Matching`: iterate it for `(i, j)` pairs, `dict(...)` it for a mapping,
`.labeled_pairs()` for `(A[i], B[j])`; `.scores` holds each matched pair’s score.

```pycon
>>> keys = ['apple pie', 'banana split', 'cherry tart']
>>> values = ['cherry tarte', 'aple pie', 'banana split']
>>> m = match(keys, values, compare='ratio')
>>> dict(m.labeled_pairs())
{'apple pie': 'aple pie', 'banana split': 'banana split', 'cherry tart': 'cherry tarte'}
```

### equate.facade.resolve(\*collections, \*\*kwargs)

Resolve entities across multiple collections: pool them (in order) and
[`dedupe()`](#equate.facade.dedupe) -> a Partition over the pooled items. Generalizes cross-collection
matching to n-way linkage. Accepts the same keywords as [`dedupe()`](#equate.facade.dedupe).

### equate.facade.score_matrix(A, B, , featurize=None, compare=None, block=None, sense=None)

Score `A` against `B` into a [`ScoreMatrix`](equate.base.html.md#equate.base.ScoreMatrix) — stages ① and ②.

The scoring half of [`match()`](#equate.facade.match), exposed on its own for callers that keep the scores
and match later or repeatedly (the interactive loop re-solves from them on every edit;
see [`equate.service`](equate.service.html.md#module-equate.service)). The keywords mean exactly what they mean in [`match()`](#equate.facade.match):
the default is the TF-IDF + cosine vector route; `compare=` takes the direct pairwise
route; `block=` scores only the blocker’s candidates into a sparse matrix; `sense`
defaults to the comparator’s declared polarity.

```pycon
>>> sm = score_matrix(['apple', 'kiwi'], ['kiwis', 'apples'], compare='ratio')
>>> sm.shape, sm.sense
((2, 2), 'maximize')
```
