# equate.matching

The match stage (③): turn a score matrix into a correspondence.

A *Matcher* is a strategy over a score structure — the choice is the **objective**, not
the algorithm (decision register D4). Select by name via the [`matchers`](#equate.matching.matchers) registry or
pass a callable `(scores, *, sense) -> pairs`. The optimal matcher is sparse-aware
(routes a blocked matrix to the sparse LAP); soft matching (optimal transport) lives in a
parallel [`soft_match()`](#equate.matching.soft_match) / [`harden()`](#equate.matching.harden) seam.

### Module Attributes

| [`matchers`](#equate.matching.matchers)   | the match-stage strategy registry (name -> lazy factory)   |
|-------------------------------------------------------------|------------------------------------------------------------|

### Functions

| [`resolve_matcher`](#equate.matching.resolve_matcher)([spec])                    | Resolve `spec` to a runner `(scores_or_scorematrix, *, sense=None) -> pairs`.   |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`optimal_matching`](#equate.matching.optimal_matching)(scores, \*[, sense])      | Globally optimal 1:1 assignment (the linear assignment problem).                |
| [`greedy_matching`](#equate.matching.greedy_matching)(scores, \*[, sense])       | Greedy 1:1: repeatedly take the best still-available `(i, j)`, removing both.   |
| [`stable_matching`](#equate.matching.stable_matching)(scores, \*[, sense])       | Gale-Shapley stable matching (optimizes stability, not total score).            |
| [`soft_match`](#equate.matching.soft_match)(scores, \*[, sense, reg, a, b]) | Entropic optimal-transport (Sinkhorn) plan over the cost derived from `scores`. |
| [`harden`](#equate.matching.harden)(plan)                               | Recover a hard 1:1 assignment from a transport `plan` (LAP maximizing mass).    |

### equate.matching.greedy_matching(scores, , sense=None)

Greedy 1:1: repeatedly take the best still-available `(i, j)`, removing both.

Order-dependent and not globally optimal, but fast. Iterates
[`stored_entries()`](equate.base.html.md#equate.base.ScoreMatrix.stored_entries), so for a sparse (blocked) matrix only
the stored candidate cells are considered — an absent cell is never a match.

### equate.matching.harden(plan)

Recover a hard 1:1 assignment from a transport `plan` (LAP maximizing mass).

### equate.matching.matchers *= <equate.registry.Registry object>*

the match-stage strategy registry (name -> lazy factory)

### equate.matching.optimal_matching(scores, , sense=None)

Globally optimal 1:1 assignment (the linear assignment problem).

Accepts a [`ScoreMatrix`](equate.base.html.md#equate.base.ScoreMatrix) (native) or a raw dense/sparse array with
`sense=` (back-compat). Solves `linear_sum_assignment` on
[`dense_cost()`](equate.base.html.md#equate.base.ScoreMatrix.dense_cost) — so a **sparse (blocked)** matrix has its
absent cells worst-cased (never preferred over a real pair) and any assignment landing
on a non-candidate cell is dropped via [`candidate_mask()`](equate.base.html.md#equate.base.ScoreMatrix.candidate_mask):
a row with no candidate is left unmatched (a *partial* matching), an all-absent matrix
yields no matches. Densifying the cost is a correctness stopgap; a non-densifying
sparse LAP is a future optimization (scipy’s sparse solver silently drops
explicitly-stored zero-score candidates, so it is not used here).

### equate.matching.resolve_matcher(spec='optimal', \*\*config)

Resolve `spec` to a runner `(scores_or_scorematrix, *, sense=None) -> pairs`.

A registered name (or a facade `how=` alias like `'assign' -> 'optimal'`) is built
via the registry; a callable is used as-is. The returned runner coerces its argument to
a [`ScoreMatrix`](equate.base.html.md#equate.base.ScoreMatrix) and then either hands that matrix to a **native**
matcher (so it worst-cases holes through the sanctioned views) or hands a **legacy**
raw-array matcher a hole-worst-cased dense array — so the sparse hole-worst-casing holds
no matter which contract the matcher follows (D11).

Note this returns a *runner*, not `spec` itself: unlike the pre-D11 version it is no
longer an identity passthrough for a callable (`resolve_matcher(f) is f` was true, and
is not any more). The runner carries the wrapped matcher’s `__name__`/`__doc__`, and
the original is recoverable via `__wrapped__`.

### equate.matching.soft_match(scores, , sense=None, reg=0.1, a=None, b=None)

Entropic optimal-transport (Sinkhorn) plan over the cost derived from `scores`.

Accepts a [`ScoreMatrix`](equate.base.html.md#equate.base.ScoreMatrix) (native) or a raw array with `sense=`.
Returns a dense transport-plan matrix whose rows/cols sum to the marginals `a`/`b`
(uniform by default). Requires `equate[ot]` (POT). `reg` (entropic regularization)
must be > 0 (`reg=0` divides by zero inside Sinkhorn).

### equate.matching.stable_matching(scores, , sense=None)

Gale-Shapley stable matching (optimizes stability, not total score).

See [`equate.util.stable_marriage_matching()`](equate.util.html.md#equate.util.stable_marriage_matching). Ranks preferences by
[`dense_cost()`](equate.base.html.md#equate.base.ScoreMatrix.dense_cost) (absent cells worst-cased), so a sparse
(blocked) cell never out-ranks a real candidate; hole-assignments are then dropped via
[`candidate_mask()`](equate.base.html.md#equate.base.ScoreMatrix.candidate_mask).

### Modules

| [`assign`](equate.matching.assign.html.md#module-equate.matching.assign)   | Assignment matchers (stage ③): turn a score matrix into matched `(i, j)` pairs.                                                                                                                                                                  |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`soft`](equate.matching.soft.html.md#module-equate.matching.soft)       | Soft / optimal-transport matching (stage ③): a fractional transport plan rather than a hard assignment — the home for graded, partial, unequal-mass and cross-modal matching (and, later, soft-DTW / Gromov-Wasserstein — decision register D8). |
