# equate.interactive.kbest

k-best assignments via Murty’s algorithm (in-house; pure numpy/scipy).

Enumerating the k lowest-cost assignments *in order* is the substrate for interactive
re-optimization: retain the top-k global matchings so that when a user confirms or rejects
an edge, the next-best consistent assignment is already known (decision register D10 /
doc 08-interactive). No well-maintained pure-Python Murty exists on PyPI, so equate ships
one. `solve_constrained` (a LAP with forced / forbidden edges) is also the primitive the
interactive re-optimization uses directly.

### Functions

| [`solve_constrained`](#equate.interactive.kbest.solve_constrained)(cost[, forced, forbidden])   | Minimum-cost assignment of `cost` with edge constraints.                     |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`k_best_assignments`](#equate.interactive.kbest.k_best_assignments)(cost, k)                    | The `k` lowest-cost assignments of `cost`, in increasing-cost order (Murty). |

### equate.interactive.kbest.k_best_assignments(cost, k)

The `k` lowest-cost assignments of `cost`, in increasing-cost order (Murty).

Returns a list of `(assignment: {i: j}, total_cost)` — fewer than `k` if there are
not that many feasible assignments.

```pycon
>>> [c for _, c in k_best_assignments([[1, 2], [2, 1]], 2)]
[2.0, 4.0]
```

### equate.interactive.kbest.solve_constrained(cost, forced=(), forbidden=())

Minimum-cost assignment of `cost` with edge constraints.

`forced` = `(i, j)` edges that MUST be in the assignment; `forbidden` = `(i, j)`
edges that must NOT. Returns `(assignment: {i: j}, total_cost)` or `None` if the
constraints admit no valid assignment.
