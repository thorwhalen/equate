# equate.matching.soft

Soft / optimal-transport matching (stage ③): a fractional transport plan rather than a
hard assignment — the home for graded, partial, unequal-mass and cross-modal matching
(and, later, soft-DTW / Gromov-Wasserstein — decision register D8). Requires `equate[ot]`.

### Functions

| [`soft_match`](#equate.matching.soft.soft_match)(scores, \*[, sense, reg, a, b])   | Entropic optimal-transport (Sinkhorn) plan over the cost derived from `scores`.   |
|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`harden`](#equate.matching.soft.harden)(plan)                                 | Recover a hard 1:1 assignment from a transport `plan` (LAP maximizing mass).      |

### equate.matching.soft.harden(plan)

Recover a hard 1:1 assignment from a transport `plan` (LAP maximizing mass).

### equate.matching.soft.soft_match(scores, , sense=None, reg=0.1, a=None, b=None)

Entropic optimal-transport (Sinkhorn) plan over the cost derived from `scores`.

Accepts a [`ScoreMatrix`](equate.base.html.md#equate.base.ScoreMatrix) (native) or a raw array with `sense=`.
Returns a dense transport-plan matrix whose rows/cols sum to the marginals `a`/`b`
(uniform by default). Requires `equate[ot]` (POT). `reg` (entropic regularization)
must be > 0 (`reg=0` divides by zero inside Sinkhorn).
