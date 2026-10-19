# equate.compare.sequence

Sequence-alignment comparators (stage ②): score two *sequences* by aligning them.

Matching, one level down: these comparators first find a correspondence between the
*elements* of two sequences, then score it. They are `direct(h)` comparators — opaque,
**not ANN-indexable**, usable to score all pairs or re-rank a blocker’s candidates — and
the collection-level matcher above them is unchanged (doc `12` §2.6, decision D8).

- **DTW** ([`dtw()`](#equate.compare.sequence.dtw)) — dynamic time warping for real-valued series (scalars or vectors
  per step). The default is an **exact DTW inside a Sakoe-Chiba band**, *not* FastDTW
  (Wu & Keogh: banded exact DTW is faster in realistic settings). A *distance*.
- **Subsequence DTW** ([`subsequence_dtw()`](#equate.compare.sequence.subsequence_dtw)) — the best-matching *sub-span* of a long
  series for a short query (open begin and end). A directional distance.
- **Soft-DTW** ([`soft_dtw()`](#equate.compare.sequence.soft_dtw)) — Cuturi & Blondel’s differentiable soft-minimum DTW.
  A smoothed discrepancy, not a distance (it can be negative).
- **Needleman-Wunsch** ([`needleman_wunsch()`](#equate.compare.sequence.needleman_wunsch)) / **Smith-Waterman**
  ([`smith_waterman()`](#equate.compare.sequence.smith_waterman)) — global / local alignment of symbolic sequences (strings,
  token lists, DNA). Alignment *scores*: higher = better.

Each comparator returns a `float`; its `*_alignment` twin (also reachable as
`dtw.align` etc.) returns an [`equate.base.Alignment`](equate.base.html.md#equate.base.Alignment) carrying the element-level
`path`. The core implementations are pure numpy; a faster engine is one keyword away
(`backend='dtaidistance'` / `'tslearn'` / `'biopython'`, behind the
`equate[timeseries]` / `equate[bio]` extras, imported only on use).

```pycon
>>> dtw([0, 1, 2, 3], [0, 0, 1, 2, 3])  # a warped copy is at distance 0
0.0
>>> needleman_wunsch('GATTACA', 'GATACA')  # 6 matches, 1 gap
5.0
>>> smith_waterman('xxGATTACAyy', 'GATTACA')  # the shared fragment
7.0
```

### Module Attributes

| [`DFLT_DTW_WINDOW`](#equate.compare.sequence.DFLT_DTW_WINDOW)     | Sakoe-Chiba band half-width (inclusive).               |
|----------------------------------------------------------------------|--------------------------------------------------------|
| [`DFLT_SOFT_DTW_GAMMA`](#equate.compare.sequence.DFLT_SOFT_DTW_GAMMA) | soft-DTW temperature (`gamma -> 0` recovers hard DTW). |
| [`DFLT_MATCH_SCORE`](#equate.compare.sequence.DFLT_MATCH_SCORE)    | default symbolic-alignment scoring (linear gaps).      |

### Functions

| [`dtw`](#equate.compare.sequence.dtw)(a, b, \*[, window, local_cost, backend])       | DTW distance (lower = more alike) — the score of [`dtw_alignment()`](#equate.compare.sequence.dtw_alignment).                |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [`dtw_alignment`](#equate.compare.sequence.dtw_alignment)(a, b, \*[, window, local_cost, ...]) | Banded exact DTW between `a` and `b` -> [`Alignment`](equate.base.html.md#equate.base.Alignment).         |
| [`subsequence_dtw`](#equate.compare.sequence.subsequence_dtw)(query, series, \*[, ...])          | Subsequence-DTW distance of `query` within `series` (directional).                                                                |
| [`subsequence_dtw_alignment`](#equate.compare.sequence.subsequence_dtw_alignment)(query, series, \*)       | Best match of `query` inside any sub-span of `series` -> `Alignment`.                                                             |
| [`soft_dtw`](#equate.compare.sequence.soft_dtw)(a, b, \*[, gamma, backend])               | Soft-DTW discrepancy (Cuturi & Blondel 2017) with temperature `gamma`.                                                            |
| [`soft_dtw_alignment`](#equate.compare.sequence.soft_dtw_alignment)(a, b, \*[, gamma])              | The soft-DTW **expected alignment** of `a` and `b`: an `n x m` soft plan.                                                         |
| [`needleman_wunsch`](#equate.compare.sequence.needleman_wunsch)(a, b, \*\*kwargs)                 | Global alignment score (higher = more alike) — see [`needleman_wunsch_alignment()`](#equate.compare.sequence.needleman_wunsch_alignment). |
| [`needleman_wunsch_alignment`](#equate.compare.sequence.needleman_wunsch_alignment)(a, b, \*[, ...])        | Global (Needleman-Wunsch) alignment of two symbolic sequences -> `Alignment`.                                                     |
| [`smith_waterman`](#equate.compare.sequence.smith_waterman)(a, b, \*\*kwargs)                   | Local alignment score (higher = more alike) — see [`smith_waterman_alignment()`](#equate.compare.sequence.smith_waterman_alignment).    |
| [`smith_waterman_alignment`](#equate.compare.sequence.smith_waterman_alignment)(a, b, \*[, ...])          | Local (Smith-Waterman) alignment -> `Alignment` of the best-matching fragments.                                                   |

### equate.compare.sequence.DFLT_DTW_WINDOW *= 0.1*

Sakoe-Chiba band half-width (inclusive). A `float` is a fraction of the longer
series’ length (at least one cell), an `int` a number of cells (`0` = no warping),
`None` unconstrained. The band is always widened by the length difference so two
series of different lengths can still be aligned.

### equate.compare.sequence.DFLT_MATCH_SCORE *= 1.0*

default symbolic-alignment scoring (linear gaps). Swap in a `substitution` callable
for a domain matrix (BLOSUM, keyboard distance, token similarity, …).

### equate.compare.sequence.DFLT_SOFT_DTW_GAMMA *= 1.0*

soft-DTW temperature (`gamma -> 0` recovers hard DTW).

### equate.compare.sequence.dtw(a, b, , window=0.1, local_cost=None, backend='numpy')

DTW distance (lower = more alike) — the score of [`dtw_alignment()`](#equate.compare.sequence.dtw_alignment).

```pycon
>>> dtw([0, 1, 2], [0, 1, 2])
0.0
>>> dtw([0, 0, 0], [1, 1, 1])  # sqrt(1 + 1 + 1)
1.7320508075688772
```

### equate.compare.sequence.dtw_alignment(a, b, , window=0.1, local_cost=None, backend='numpy')

Banded exact DTW between `a` and `b` -> [`Alignment`](equate.base.html.md#equate.base.Alignment).

`a`/`b` are sequences of numbers or of equal-length vectors (multivariate series).
`window` is the Sakoe-Chiba half-width (see [`DFLT_DTW_WINDOW`](#equate.compare.sequence.DFLT_DTW_WINDOW); `None` is
unconstrained DTW). `local_cost(x, y)` replaces the default squared-Euclidean step
cost (then the total is not square-rooted). `backend='dtaidistance'` (`equate
[timeseries]`) computes the same quantity in C for the default cost and reports no
path.

```pycon
>>> al = dtw_alignment([1, 2, 3], [1, 2, 2, 3], window=None)
>>> al.score, al.path
(0.0, ((0, 0), (1, 1), (1, 2), (2, 3)))
```

### equate.compare.sequence.needleman_wunsch(a, b, \*\*kwargs)

Global alignment score (higher = more alike) — see [`needleman_wunsch_alignment()`](#equate.compare.sequence.needleman_wunsch_alignment).

### equate.compare.sequence.needleman_wunsch_alignment(a, b, , substitution=None, match=1.0, mismatch=-1.0, gap=-1.0, backend='numpy')

Global (Needleman-Wunsch) alignment of two symbolic sequences -> `Alignment`.

Aligns the *entireties* of `a` and `b` (strings, token lists, …), maximizing
the summed `substitution(x, y)` (default `match`/`mismatch`) plus a linear
`gap` per skipped element. Gaps appear in the path as `None`.

```pycon
>>> needleman_wunsch_alignment('ACT', 'AT').path
((0, 0), (1, None), (2, 1))
```

### equate.compare.sequence.smith_waterman(a, b, \*\*kwargs)

Local alignment score (higher = more alike) — see [`smith_waterman_alignment()`](#equate.compare.sequence.smith_waterman_alignment).

### equate.compare.sequence.smith_waterman_alignment(a, b, , substitution=None, match=1.0, mismatch=-1.0, gap=-1.0, backend='numpy')

Local (Smith-Waterman) alignment -> `Alignment` of the best-matching fragments.

Like [`needleman_wunsch_alignment()`](#equate.compare.sequence.needleman_wunsch_alignment) but floored at zero, so only the
highest-scoring aligned *sub-regions* count; the path covers just that fragment pair.

```pycon
>>> al = smith_waterman_alignment('xxABCyy', 'zABCz')
>>> al.score, al.path
(3.0, ((2, 1), (3, 2), (4, 3)))
```

### equate.compare.sequence.soft_dtw(a, b, , gamma=1.0, backend='numpy')

Soft-DTW discrepancy (Cuturi & Blondel 2017) with temperature `gamma`.

Replaces DTW’s hard `min` by a log-sum-exp soft-minimum over all warping paths (on
*squared*-Euclidean step costs, no final root), so it is differentiable — but it is
**not a distance**: it can be negative and `soft_dtw(x, x) != 0`. `gamma -> 0`
recovers (squared) DTW. `backend='tslearn'` (`equate[timeseries]`) uses
`tslearn.metrics.soft_dtw`, which computes the same quantity.

```pycon
>>> round(soft_dtw([0, 1, 2], [0, 1, 2], gamma=1e-3), 6)
0.0
>>> soft_dtw([0, 1, 2], [0, 1, 2]) < 0  # smoothing makes it negative
True
```

### equate.compare.sequence.soft_dtw_alignment(a, b, , gamma=1.0)

The soft-DTW **expected alignment** of `a` and `b`: an `n x m` soft plan.

`E[i, j]` is the probability-weighted share of warping paths through `(i, j)`
under soft-DTW’s Gibbs distribution — the gradient of [`soft_dtw()`](#equate.compare.sequence.soft_dtw) with respect to
the step costs (Cuturi & Blondel 2017). It is the element-level soft correspondence
soft-DTW has instead of a single path: `gamma -> 0` concentrates it on the DTW path,
larger `gamma` spreads it. `E[0, 0] == E[-1, -1] == 1` (every path starts and ends
there). Same squared-Euclidean steps as [`soft_dtw()`](#equate.compare.sequence.soft_dtw).

```pycon
>>> E = soft_dtw_alignment([0, 1, 2], [0, 1, 1, 2], gamma=1e-3)
>>> E.round(2).tolist()
[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
```

### equate.compare.sequence.subsequence_dtw(query, series, , local_cost=None, backend='numpy')

Subsequence-DTW distance of `query` within `series` (directional).

### equate.compare.sequence.subsequence_dtw_alignment(query, series, , local_cost=None, backend='numpy')

Best match of `query` inside any sub-span of `series` -> `Alignment`.

Open-begin, open-end DTW: the query must be consumed whole, the series need not be.
The path’s `j` range is the matched span. Directional — `(query, series)` order
matters. A custom `local_cost` must be non-negative (a negative step cost would make
“start later” ill-defined). `backend='dtaidistance'` (`equate[timeseries]`) runs
`dtaidistance.subsequence` (with no step penalty, so it computes the same score and
path) for the default cost.

```pycon
>>> al = subsequence_dtw_alignment([5, 6, 7], [0, 0, 5, 6, 7, 0])
>>> al.score, al.path[0], al.path[-1]
(0.0, (0, 2), (2, 4))
```
