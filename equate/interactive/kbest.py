"""k-best assignments via Murty's algorithm (in-house; pure numpy/scipy).

Enumerating the k lowest-cost assignments *in order* is the substrate for interactive
re-optimization: retain the top-k global matchings so that when a user confirms or rejects
an edge, the next-best consistent assignment is already known (decision register D10 /
doc 08-interactive). No well-maintained pure-Python Murty exists on PyPI, so equate ships
one. ``solve_constrained`` (a LAP with forced / forbidden edges) is also the primitive the
interactive re-optimization uses directly.
"""

import heapq

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ['solve_constrained', 'k_best_assignments']


def solve_constrained(cost, forced=(), forbidden=()):
    """Minimum-cost assignment of ``cost`` with edge constraints.

    ``forced`` = ``(i, j)`` edges that MUST be in the assignment; ``forbidden`` = ``(i, j)``
    edges that must NOT. Returns ``(assignment: {i: j}, total_cost)`` or ``None`` if the
    constraints admit no valid assignment.
    """
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    constrained = cost.copy()
    for i, j in forbidden:
        constrained[i, j] = np.inf

    forced = list(forced)
    forced_rows = {i for i, _ in forced}
    forced_cols = {j for _, j in forced}
    assignment = {i: j for i, j in forced}
    total = sum(cost[i, j] for i, j in forced)

    free_rows = [i for i in range(n) if i not in forced_rows]
    free_cols = [j for j in range(m) if j not in forced_cols]
    if free_rows and free_cols:
        sub = constrained[np.ix_(free_rows, free_cols)]
        if not np.isfinite(sub).any(axis=1).all():  # a free row with no finite column
            return None
        try:
            rows, cols = linear_sum_assignment(sub)
        except ValueError:
            return None
        for ri, ci in zip(rows.tolist(), cols.tolist()):
            gi, gj = free_rows[ri], free_cols[ci]
            if not np.isfinite(constrained[gi, gj]):
                return None
            assignment[gi] = gj
            total += cost[gi, gj]
    return assignment, float(total)


def k_best_assignments(cost, k):
    """The ``k`` lowest-cost assignments of ``cost``, in increasing-cost order (Murty).

    Returns a list of ``(assignment: {i: j}, total_cost)`` — fewer than ``k`` if there are
    not that many feasible assignments.

    >>> [c for _, c in k_best_assignments([[1, 2], [2, 1]], 2)]
    [2.0, 4.0]
    """
    cost = np.asarray(cost, dtype=float)
    root = solve_constrained(cost)
    if root is None:
        return []

    results = []
    seen = set()
    counter = 0
    # heap entries: (total_cost, tiebreak, forced, forbidden, assignment)
    heap = [(root[1], counter, (), (), root[0])]
    while heap and len(results) < k:
        total, _, forced, forbidden, assignment = heapq.heappop(heap)
        key = tuple(sorted(assignment.items()))
        if key in seen:
            continue
        seen.add(key)
        results.append((assignment, total))

        # Murty partitioning: split the solution space on each non-forced edge in turn —
        # forbid it (a new subproblem), then force it for the subsequent splits.
        forced_set = set(forced)
        cur_forced = list(forced)
        for edge in [(i, j) for i, j in assignment.items() if (i, j) not in forced_set]:
            child = solve_constrained(cost, cur_forced, list(forbidden) + [edge])
            if child is not None:
                counter += 1
                heapq.heappush(
                    heap,
                    (
                        child[1],
                        counter,
                        tuple(cur_forced),
                        tuple(list(forbidden) + [edge]),
                        child[0],
                    ),
                )
            cur_forced.append(edge)
    return results
