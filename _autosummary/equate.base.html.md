# equate.base

Foundational contracts and data structures for equate.

This module is the stable spine of the matching framework. It defines:

- the four **stage protocols** (`Featurizer`, `Comparator`, `Blocker`,
  `Matcher`) that every strategy satisfies — each is just a callable with a
  documented contract, so a plain lambda qualifies;
- the `ScoreMatrix` that flows from the compare stage to the match stage (the
  single source of truth carrying the scores, their `sense`, and row/col labels);
- the structured result types (`Candidate`, `Matching`, `Explanation`,
  `Partition`) returned at every public boundary — never printouts — so the same
  core dispatches cleanly to a CLI, an HTTP service, and a declarative UI;
- `to_cost` — the single similarity->cost conversion that every cost-minimizing
  matcher routes through, so they can never disagree.

The three-stage decomposition (featurize -> compare -> match/resolve) and the design
decisions realized here are documented in `docs/research/` — start with
`00-taxonomy-and-terminology.md` and the decision register
`11-design-decisions-and-open-questions.md`.

### Module Attributes

| [`Sense`](#equate.base.Sense)            | Whether higher scores are better (`'maximize'`, a similarity) or lower scores are better (`'minimize'`, a distance/cost).               |
|-------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [`Featurizer`](#equate.base.Featurizer)       | ① Featurize — map each object to a comparable representation (batched).                                                                 |
| [`Comparator`](#equate.base.Comparator)       | ② Compare — score how alike two representations are (higher = more alike by the canonical `'maximize'` sense).                          |
| [`Blocker`](#equate.base.Blocker)          | yield `(i, j)` index pairs lazily, never a materialized n×m matrix.                                                                     |
| [`Matcher`](#equate.base.Matcher)          | ③ Match — turn a score structure into matched `(i, j)` pairs.                                                                           |
| [`QuadraticMatcher`](#equate.base.QuadraticMatcher) | align the nodes of \*\*two structured objects\*\* (graphs) — `QuadraticMatcher(G1, G2, *, node_affinity=None, seeds=None) -> Matching`. |

### Functions

| [`scorematrix_matcher`](#equate.base.scorematrix_matcher)(fn)      | Mark `fn` as a **ScoreMatrix-native** matcher (`Matcher(ScoreMatrix) -> pairs`).   |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`to_cost`](#equate.base.to_cost)(scores, \*[, sense]) | Convert a score array to a **cost** array for minimization-based matchers.         |

### Classes

| [`FeaturizerMeta`](#equate.base.FeaturizerMeta)([output_kinds, normalize, ...])    | Declared properties of a featurizer (decision register D9), so the framework can pick a legal index/metric, auto-apply a required query/passage prefix, refuse a non-commercial model in a commercial context, and offer Matryoshka truncation.   |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ComparatorMeta`](#equate.base.ComparatorMeta)([polarity, bounded, ...])          | Declared properties of a comparator, so the framework adapts instead of the caller hard-coding conversions (decision register D2/D3/D6).                                                                                                          |
| [`ScoreMatrix`](#equate.base.ScoreMatrix)(data[, sense, row_labels, ...])       | Scored candidate structure flowing compare -> match (the SSOT).                                                                                                                                                                                   |
| [`Alignment`](#equate.base.Alignment)(score[, path, sense])                   | A structured comparator's score **plus** the internal correspondence behind it.                                                                                                                                                                   |
| [`Explanation`](#equate.base.Explanation)([summary, details])                   | Render-agnostic justification for a decision (rules / attributions / diffs).                                                                                                                                                                      |
| [`Candidate`](#equate.base.Candidate)(left, right, score[, explanation])      | One scored correspondence between a `left` item and a `right` item.                                                                                                                                                                               |
| [`Matching`](#equate.base.Matching)(pairs[, scores, plan, probability, ...]) | Default (hard) matcher output: matched index pairs, with retained scores.                                                                                                                                                                         |
| [`Partition`](#equate.base.Partition)(labels)                                 | Clustering / collective output: a map from item index to entity (cluster) id.                                                                                                                                                                     |

### *class* equate.base.Alignment(score, path=None, sense='minimize')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A structured comparator’s score **plus** the internal correspondence behind it.

Sequence/structure comparators (DTW, Needleman-Wunsch, Smith-Waterman, …) match
the *sub-parts* of two objects before scoring them — matching, one level down. The
scalar `score` is what the collection-level matcher consumes; `path` keeps the
element-level correspondence for explanation, visualization and element-level
matching (doc `12` §2.6, decision register D8):

- `path`: a tuple of `(i, j)` element-index pairs, in order. A gap is `None` on
  the side that skips (`(i, None)` / `(None, j)`). `None` when the method has no
  single path (soft-DTW averages over all of them) or the backend does not report one.
- `sense`: how to read `score` — `'minimize'` for a dissimilarity (DTW),
  `'maximize'` for an alignment score (Needleman-Wunsch / Smith-Waterman).

`float(alignment)` is its score, so an `Alignment` drops in where a number is
expected.

### equate.base.Blocker

yield `(i, j)` index pairs lazily, never a
materialized n×m matrix. A default `all_pairs` blocker (roadmap #5) will be the
correctness baseline.

* **Type:**
  ② Block — candidate generation

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[…], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)]]

### *class* equate.base.Candidate(left, right, score, explanation=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One scored correspondence between a `left` item and a `right` item.

Iterating a `Candidate` yields `(left, right)`, so a list of candidates is
`dict`-able into a mapping — keeping the “richer iterates as simpler” contract
consistent across the result types.

### equate.base.Comparator

② Compare — score how alike two representations are (higher = more alike by the
canonical `'maximize'` sense).

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### *class* equate.base.ComparatorMeta(polarity='similarity', bounded=False, is_metric=False, is_symmetric=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Declared properties of a comparator, so the framework adapts instead of the
caller hard-coding conversions (decision register D2/D3/D6).

- `polarity`: `'similarity'` (higher = more alike) or `'distance'` (lower =
  more alike);
- `bounded`: whether scores lie in a known bounded range (e.g. `[0, 1]`);
- `is_metric`: obeys identity / symmetry / triangle-inequality (cosine does NOT —
  so a triangle-inequality index like a VP/ball tree must not be used on it);
- `is_symmetric`: `g(a, b) == g(b, a)` (Monge-Elkan, containment do NOT — such
  comparators stay directional and are only symmetrized at the matcher boundary).

A comparator can carry this on a `.meta` attribute; the `compare` stage
(roadmap #4) attaches it when registering strategies.

### *class* equate.base.Explanation(summary='', details=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Render-agnostic justification for a decision (rules / attributions / diffs).

A structured payload, never a printed string, so any UI can render it declaratively.

### equate.base.Featurizer

① Featurize — map each object to a comparable representation (batched). `id` and
any `key=` function (`str.lower`, `operator.attrgetter('name')`) are valid.

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### *class* equate.base.FeaturizerMeta(output_kinds=('vector',), normalize=False, dim=None, license=None, max_seq_len=None, query_prefix=None, passage_prefix=None, truncatable_to=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Declared properties of a featurizer (decision register D9), so the framework
can pick a legal index/metric, auto-apply a required query/passage prefix, refuse a
non-commercial model in a commercial context, and offer Matryoshka truncation.

- `output_kinds`: what the representation is — `'vector'` / `'set'` /
  `'bitstring'` / `'scalar'` / `'structured'` (a multi-functional embedder may
  emit several, e.g. dense + sparse);
- `normalize`: whether output vectors are already L2-normalized;
- `dim` / `max_seq_len`: dimensionality and max input length, when fixed;
- `license`: SPDX/label, so a commercial context can refuse non-commercial models;
- `query_prefix` / `passage_prefix`: instruction prefixes some embedders require
  (e.g. E5’s `"query: "` / `"passage: "`) — a silent-omission footgun;
- `truncatable_to`: Matryoshka dims the embedding can be safely truncated to.

### equate.base.Matcher

③ Match — turn a score structure into matched `(i, j)` pairs. The **native**
contract is `Matcher(ScoreMatrix) -> pairs`: the matcher reads `sense` off the
[`ScoreMatrix`](#equate.base.ScoreMatrix) and densifies **only** through its sanctioned views
(`dense_cost` / `dense_similarity` / `candidate_mask` / `stored_entries`), so
the sparse hole-worst-casing can never be bypassed (decision register D11). The
legacy `(scores, *, sense) -> pairs` raw-array contract is still accepted — see
[`scorematrix_matcher()`](#equate.base.scorematrix_matcher) and `equate.matching.resolve_matcher`.

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[…], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)]]

### *class* equate.base.Matching(pairs, scores=None, plan=None, probability=None, row_labels=None, col_labels=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Default (hard) matcher output: matched index pairs, with retained scores.

Iterating a `Matching` yields `(i, j)` index tuples, so `dict(matching)` and
`list(matching)` work. Richer tiers are *optional attributes* rather than
separate return types: `plan` holds a soft transport plan, `probability` holds
per-pair calibrated match probabilities. `Partition` (clustering) and, later,
`PartitionPosterior` (Bayesian) also iterate as their point-estimate pairs, so a
caller can always `dict(...)` a result regardless of the `how=` strategy.

#### labeled_pairs()

Yield `(row_label, col_label)`, using labels when present, else indices.

### *class* equate.base.Partition(labels)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Clustering / collective output: a map from item index to entity (cluster) id.

Iterating yields the induced within-cluster `(i, j)` pairs (the point estimate),
so a `Partition` is drop-in where matched pairs are expected. This is the
canonical *clustering* output — it subsumes pairwise links, 1:1 assignment and
dedup, and generalizes `match(A, B)` to `resolve(*collections)`.

#### groups()

Return `{cluster_id: [item_index, ...]}`.

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)]

### equate.base.QuadraticMatcher

align the nodes of \*\*two
structured objects\*\* (graphs) — `QuadraticMatcher(G1, G2, *, node_affinity=None,
seeds=None) -> Matching`. Unlike a flat [`Matcher`](#equate.base.Matcher) it cannot start from a
precomputed score matrix: it produces the node scores *and* the correspondence
jointly (the cost of `u <-> v` depends on how their neighbors match — a quadratic
assignment problem), reusing a flat matcher as its inner extraction step. See
`equate.matching.quadratic`.

* **Type:**
  ③ Match, the *second arity* (decision register D8)

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[…], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### *class* equate.base.ScoreMatrix(data, sense='maximize', row_labels=None, col_labels=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Scored candidate structure flowing compare -> match (the SSOT).

`data` is a dense `numpy` array or a `scipy.sparse` matrix of pairwise
scores; `sense` records the polarity (see [`Sense`](#equate.base.Sense)); `row_labels` /
`col_labels` name the two collections so a downstream matcher never guesses
orientation or polarity. A sparse `data` is exactly what a blocker produces:
the cells it chose not to score are simply absent.

#### candidate_mask()

Boolean dense mask, `True` exactly on stored (candidate) cells.

A dense matrix has no holes (all `True`); a sparse one marks only the cells the
blocker chose to score, so a matcher can drop any assignment that lands on a hole.

#### *classmethod* coerce(scores, , sense=None)

Return a `ScoreMatrix`, wrapping a raw dense/sparse `scores` if needed.

The single entry point every matcher uses so a raw array *or* a `ScoreMatrix`
both work (back-compat) yet the matcher body only ever touches the sanctioned
views. A `ScoreMatrix` passes through unchanged (a conflicting `sense` is an
error, never a silent override); a raw array is wrapped with `sense` (default
`'maximize'`).

#### dense_cost()

Worst-cased dense **cost** for min-solvers — the sanctioned densify.

Routes through the [`to_cost()`](#equate.base.to_cost) SSOT, so a sparse (blocked) matrix has its
structurally-absent cells worst-cased (never cheaper than a real pair). \*\*This
is why a matcher must never `.toarray()` the raw scores and then call\*\*
`to_cost` — that takes the dense branch and silently loses the worst-casing
(the recurring bug class this method exists to make unreachable; D11).

#### dense_similarity()

Worst-cased dense **similarity** (higher = better) for argmax / max-weight.

The negated [`dense_cost()`](#equate.base.ScoreMatrix.dense_cost), so it is `sense`-correct for both polarities
and absent cells are strictly the *lowest* value — never argmaxed, never a
preferred edge. Real cells keep their relative order.

#### drop_holes(pairs, , keep=())

Drop any `(i, j)` landing on a non-candidate (absent) cell — the D11 filter.

The single place the hole-dropping invariant lives, so **every** matcher/resolve
path routes its output through it instead of re-implementing the mask filter (the
soft/OT and `reoptimize` paths originally forgot to, and leaked holes). A dense
matrix has no holes (returned unchanged). `keep` is an allow-list of pairs to
retain even if absent — e.g. user-*forced* interactive edges, which are assertions
that override blocking.

#### *property* is_sparse *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the underlying `data` is a `scipy.sparse` (blocked) matrix.

#### legacy_view()

Dense array for a *legacy* raw-array matcher: **raw** scores, holes worst-cased.

A legacy matcher may read *absolute* score values (a threshold, a sign test, a
`[0, 1]` assumption), so the stored values are handed over **unchanged** — a dense
matrix, having no holes, is returned exactly as a pre-`ScoreMatrix` matcher used to
receive it. Only the structurally-absent cells of a sparse matrix are rewritten, to a
sentinel strictly worse than any real assignment in this matrix’s own orientation
(`_hole_fill()`), so the D11 guarantee survives the trip through a matcher that
never sees the sparse structure.

Routing this through `dense_similarity()` instead — i.e. `S - S.max()` — would
silently rescale a *hole-free* matrix to be all-nonpositive and break every legacy
matcher that reads absolute scores.

#### score_at(i, j)

The retained (raw) score at `(i, j)` — for reporting a matched pair.

A matched pair is always a stored cell, so reading the raw value is correct;
this is *not* a densify-then-match path (it never feeds `to_cost`).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### stored_entries()

Yield `(i, j, score)` for each candidate cell (every cell if dense).

The sanctioned way to iterate a blocked matrix’s real candidates without
densifying — a sparse matrix yields only its stored cells.

#### to_cost()

back-compat alias for [`dense_cost()`](#equate.base.ScoreMatrix.dense_cost).

### equate.base.Sense

Whether higher scores are better (`'maximize'`, a similarity) or lower scores
are better (`'minimize'`, a distance/cost).

alias of [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[‘maximize’, ‘minimize’]

### equate.base.scorematrix_matcher(fn)

Mark `fn` as a **ScoreMatrix-native** matcher (`Matcher(ScoreMatrix) -> pairs`).

`equate.matching.resolve_matcher` hands a marked matcher the [`ScoreMatrix`](#equate.base.ScoreMatrix)
itself (so it can worst-case holes via the sanctioned views); an unmarked callable
that declares a `sense` parameter is treated as a *legacy* raw-array matcher and is
handed a pre-worst-cased dense array instead ([`ScoreMatrix.legacy_view()`](#equate.base.ScoreMatrix.legacy_view)). This
is the seam that lets the deep contract change stay back-compatible.

### equate.base.to_cost(scores, , sense='maximize')

Convert a score array to a **cost** array for minimization-based matchers.

The single source of truth for the similarity->cost conversion, so every matcher
agrees. Historically the matchers each converted differently (`hungarian` used
`S.max() - S`, `kuhn_munkres` used `-S`, `stable` used `1 - S`) — wrong
for unbounded or non-`[0, 1]` scores. Route every cost-minimizing matcher through
this instead.

- `sense='maximize'` (a *similarity*, higher = better): return the complement to
  the maximum, `S.max() - S` — non-negative (safe for min-weight solvers), and
  for a **fixed-cardinality** assignment it yields the same optimum as `-S`.
- `sense='minimize'` (already a *distance*/cost, lower = better): returned
  unchanged. An edit/DTW/geographic distance is never forced through `1 - S`.

A **sparse** input is treated as blocker output: structurally-absent cells are
*non-candidates* and become a worst-case cost (never preferred over a real pair) —
*not* a score of 0, which would let unscored holes win when stored similarities are
negative. This densifies for correctness; the efficient path that never
materializes the full matrix lands with the blocking / sparse-LAP work (roadmap #6).
Empty inputs return an empty array (an empty collection yields an empty matching).

```pycon
>>> import numpy as np
>>> S = np.array([[0.9, 0.1], [0.2, 0.8]])
>>> to_cost(S).round(2).tolist()
[[0.0, 0.8], [0.7, 0.1]]
>>> to_cost(S, sense='minimize') is S
True
```
