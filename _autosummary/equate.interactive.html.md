# equate.interactive

The interactive stage: human-in-the-loop review and re-optimization.

The payoff of retaining **top-k candidates with scores** (doc 08-interactive): a user can
edit a match and the system re-solves the rest from the remembered alternatives. The pieces:

- [`CandidateStore`](#equate.interactive.CandidateStore) — the per-item top-k SSOT (built from a score matrix; serializable);
- [`k_best_assignments()`](#equate.interactive.k_best_assignments) — Murty’s k-best global matchings, in order (in-house, pure scipy);
- [`ConstraintSet`](#equate.interactive.ConstraintSet) + [`reoptimize()`](#equate.interactive.reoptimize) — every edit is a force/forbid constraint,
  re-solved with one constrained LAP, not a from-scratch recompute;
- [`review_queue()`](#equate.interactive.review_queue) / [`uncertainty_sampling()`](#equate.interactive.uncertainty_sampling) — rank items by confidence margin for
  human review and active-learning query selection.

Structured explanations use [`equate.base.Explanation`](equate.base.html.md#equate.base.Explanation). Heavier active-learning
backends (modAL / scikit-activeml) are optional (`equate[active]`); the margin-based
query strategy here needs no extra.

### Functions

| [`k_best_assignments`](#equate.interactive.k_best_assignments)(cost, k)                    | The `k` lowest-cost assignments of `cost`, in increasing-cost order (Murty).                          |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [`solve_constrained`](#equate.interactive.solve_constrained)(cost[, forced, forbidden])   | Minimum-cost assignment of `cost` with edge constraints.                                              |
| [`reoptimize`](#equate.interactive.reoptimize)(scores[, constraints, sense])       | Re-solve the optimal 1:1 assignment honoring `constraints` (confirmed/rejected).                      |
| [`review_queue`](#equate.interactive.review_queue)(candidate_store, \*[, ...])       | Order the left items of a [`CandidateStore`](#equate.interactive.CandidateStore) for review. |
| [`uncertainty_sampling`](#equate.interactive.uncertainty_sampling)(candidate_store, \*[, n]) | Active-learning query: the `n` most uncertain items to label next.                                    |

### Classes

| [`CandidateStore`](#equate.interactive.CandidateStore)(candidates[, sense, ...])   | Per left index `i`, its top-k `(right_index, score)` candidates, best-first.                      |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [`ConstraintSet`](#equate.interactive.ConstraintSet)([forced, forbidden])         | Append-only record of user edits: `forced` (confirmed) and `forbidden` (rejected) `(i, j)` edges. |

### *class* equate.interactive.CandidateStore(candidates, sense='maximize', row_labels=None, col_labels=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per left index `i`, its top-k `(right_index, score)` candidates, best-first.

Iterating yields the top-1 `(i, j)` pairs, so `dict(store)` gives the greedy 1:n
match. `row_labels` / `col_labels` name the two collections.

#### *classmethod* from_scores(scores, , k=5, sense='maximize', row_labels=None, col_labels=None)

Build a store of each row’s top-`k` candidates from a (dense or sparse) score
matrix.

A **sparse** matrix considers only its stored (candidate) cells — which is exactly
the blocking semantics (an absent cell is not a candidate), so for a blocked matrix
this intentionally differs from the dense all-cells behavior regardless of `sense`.

#### margin(i)

Confidence gap between item `i`’s top-1 and top-2 candidate scores.

A *small* margin means the top choice is uncertain (two close candidates) — the
signal for review triage and active-learning query selection. An item with 0 or 1
candidates is unambiguous (no rival to be confused with), so its margin is `+inf`
— it sorts LAST in a most-uncertain-first queue, never spuriously to the front.

#### top(i, n=1)

The top-`n` `(right_index, score)` candidates for left item `i`.

### *class* equate.interactive.ConstraintSet(forced=list(), forbidden=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Append-only record of user edits: `forced` (confirmed) and `forbidden`
(rejected) `(i, j)` edges.

#### confirm(i, j)

Force edge `(i, j)` into the assignment (a user-confirmed match).

#### reject(i, j)

Forbid edge `(i, j)` from the assignment (a user-rejected match).

### equate.interactive.k_best_assignments(cost, k)

The `k` lowest-cost assignments of `cost`, in increasing-cost order (Murty).

Returns a list of `(assignment: {i: j}, total_cost)` — fewer than `k` if there are
not that many feasible assignments.

```pycon
>>> [c for _, c in k_best_assignments([[1, 2], [2, 1]], 2)]
[2.0, 4.0]
```

### equate.interactive.reoptimize(scores, constraints=None, , sense='maximize')

Re-solve the optimal 1:1 assignment honoring `constraints` (confirmed/rejected).

Returns the matched `(i, j)` pairs. A rejected pair is a *hole* (a non-candidate, the
blocked-matching semantics of D11), not an `inf` cost: the re-solve matches as many
non-rejected pairs as possible, then the best score, so rejecting every candidate of an
item leaves it unmatched. Raises `ValueError` only if the constraints contradict each
other (the same pair both confirmed and rejected, or two confirms sharing a row or
column). This is O(one LAP): the whole point is that an edit is a cheap constrained
re-solve, not a from-scratch recompute.

### equate.interactive.review_queue(candidate_store, , most_uncertain_first=True)

Order the left items of a [`CandidateStore`](#equate.interactive.CandidateStore) for review.

By default the most-uncertain items (smallest top-1/top-2 margin) come first — the
decisions a human should look at soonest.

### equate.interactive.solve_constrained(cost, forced=(), forbidden=())

Minimum-cost assignment of `cost` with edge constraints.

`forced` = `(i, j)` edges that MUST be in the assignment; `forbidden` = `(i, j)`
edges that must NOT. Returns `(assignment: {i: j}, total_cost)` or `None` if the
constraints admit no valid assignment.

### equate.interactive.uncertainty_sampling(candidate_store, , n=10)

Active-learning query: the `n` most uncertain items to label next.

Uncertainty sampling — label the items whose top choice is least certain (smallest
margin), since those labels are the most informative for training a matcher.

### Modules

| [`candidate_store`](equate.interactive.candidate_store.html.md#module-equate.interactive.candidate_store)   | The CandidateStore: per left-item top-k candidates with scores — the SSOT for review and interactive re-optimization.   |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| [`constraints`](equate.interactive.constraints.html.md#module-equate.interactive.constraints)           | Interactive constraints and re-optimization: every user edit is a constraint, not a recompute.                          |
| [`kbest`](equate.interactive.kbest.html.md#module-equate.interactive.kbest)                       | k-best assignments via Murty's algorithm (in-house; pure numpy/scipy).                                                  |
| [`review`](equate.interactive.review.html.md#module-equate.interactive.review)                     | Human review triage and active-learning query selection.                                                                |
