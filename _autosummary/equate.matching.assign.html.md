# equate.matching.assign

Assignment matchers (stage ③): turn a score matrix into matched `(i, j)` pairs.

Expose the *objective* (optimal / greedy / stable), not the algorithm. The optimal
matcher is **sparse-aware**: a blocked (sparse) score matrix routes to the sparse LAP
solver instead of densifying. All route similarity→cost through the `to_cost` SSOT
(decision register D2), so a `sense` of `'maximize'` (similarity) or `'minimize'`
(distance) is honored consistently.

### Functions

| [`optimal_matching`](#equate.matching.assign.optimal_matching)(scores, \*[, sense])   | Globally optimal 1:1 assignment (the linear assignment problem).              |
|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`greedy_matching`](#equate.matching.assign.greedy_matching)(scores, \*[, sense])    | Greedy 1:1: repeatedly take the best still-available `(i, j)`, removing both. |
| [`stable_matching`](#equate.matching.assign.stable_matching)(scores, \*[, sense])    | Gale-Shapley stable matching (optimizes stability, not total score).          |

### equate.matching.assign.greedy_matching(scores, , sense=None)

Greedy 1:1: repeatedly take the best still-available `(i, j)`, removing both.

Order-dependent and not globally optimal, but fast. Iterates
[`stored_entries()`](equate.base.html.md#equate.base.ScoreMatrix.stored_entries), so for a sparse (blocked) matrix only
the stored candidate cells are considered — an absent cell is never a match.

### equate.matching.assign.optimal_matching(scores, , sense=None)

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

### equate.matching.assign.stable_matching(scores, , sense=None)

Gale-Shapley stable matching (optimizes stability, not total score).

See [`equate.util.stable_marriage_matching()`](equate.util.html.md#equate.util.stable_marriage_matching). Ranks preferences by
[`dense_cost()`](equate.base.html.md#equate.base.ScoreMatrix.dense_cost) (absent cells worst-cased), so a sparse
(blocked) cell never out-ranks a real candidate; hole-assignments are then dropped via
[`candidate_mask()`](equate.base.html.md#equate.base.ScoreMatrix.candidate_mask).
