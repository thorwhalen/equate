# equate.interactive.constraints

Interactive constraints and re-optimization: every user edit is a constraint, not a
recompute.

A *confirm* forces an edge into the assignment; a *reject* forbids one. [`reoptimize()`](#equate.interactive.constraints.reoptimize)
re-solves the optimal assignment honoring the accumulated constraints — the primitive
behind “edit a match and the system updates the rest from the remembered alternatives”
(doc 08-interactive). The constraint set is append-only so a session’s edit history is a
replayable log.

### Functions

| [`reoptimize`](#equate.interactive.constraints.reoptimize)(scores[, constraints, sense])   | Re-solve the optimal 1:1 assignment honoring `constraints` (confirmed/rejected).   |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|

### Classes

| [`ConstraintSet`](#equate.interactive.constraints.ConstraintSet)([forced, forbidden])   | Append-only record of user edits: `forced` (confirmed) and `forbidden` (rejected) `(i, j)` edges.   |
|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|

### *class* equate.interactive.constraints.ConstraintSet(forced=list(), forbidden=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Append-only record of user edits: `forced` (confirmed) and `forbidden`
(rejected) `(i, j)` edges.

#### confirm(i, j)

Force edge `(i, j)` into the assignment (a user-confirmed match).

#### reject(i, j)

Forbid edge `(i, j)` from the assignment (a user-rejected match).

### equate.interactive.constraints.reoptimize(scores, constraints=None, , sense='maximize')

Re-solve the optimal 1:1 assignment honoring `constraints` (confirmed/rejected).

Returns the matched `(i, j)` pairs. A rejected pair is a *hole* (a non-candidate, the
blocked-matching semantics of D11), not an `inf` cost: the re-solve matches as many
non-rejected pairs as possible, then the best score, so rejecting every candidate of an
item leaves it unmatched. Raises `ValueError` only if the constraints contradict each
other (the same pair both confirmed and rejected, or two confirms sharing a row or
column). This is O(one LAP): the whole point is that an edit is a cheap constrained
re-solve, not a from-scratch recompute.
