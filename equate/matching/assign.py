"""Assignment matchers (stage ③): turn a score matrix into matched ``(i, j)`` pairs.

Expose the *objective* (optimal / greedy / stable), not the algorithm. The optimal
matcher is **sparse-aware**: a blocked (sparse) score matrix routes to the sparse LAP
solver instead of densifying. All route similarity→cost through the ``to_cost`` SSOT
(decision register D2), so a ``sense`` of ``'maximize'`` (similarity) or ``'minimize'``
(distance) is honored consistently.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse

from equate.base import to_cost

__all__ = ['optimal_matching', 'greedy_matching', 'stable_matching']


def optimal_matching(scores, *, sense='maximize'):
    """Globally optimal 1:1 assignment (the linear assignment problem).

    Dense input -> ``scipy.optimize.linear_sum_assignment`` on ``to_cost(scores, sense)``.
    Sparse input -> the sparse solver ``min_weight_full_bipartite_matching`` (no densify)
    when the blocked graph admits a full matching, else a densify fallback.
    """
    if issparse(scores):
        try:
            from scipy.sparse.csgraph import min_weight_full_bipartite_matching

            row, col = min_weight_full_bipartite_matching(
                scores, maximize=(sense == 'maximize')
            )
            return list(zip(row.tolist(), col.tolist()))
        except ValueError:
            scores = scores.toarray()  # no full matching in the blocked graph -> densify
    cost = to_cost(scores, sense=sense)
    row, col = linear_sum_assignment(cost)
    return list(zip(row.tolist(), col.tolist()))


def greedy_matching(scores, *, sense='maximize'):
    """Greedy 1:1: repeatedly take the best still-available ``(i, j)``, removing both.

    Order-dependent and not globally optimal, but fast. For a sparse matrix only the
    stored (candidate) cells are considered — an absent cell is never a match.
    """
    if issparse(scores):
        coo = scores.tocoo()
        entries = list(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()))
        shape = scores.shape
    else:
        S = np.asarray(scores, dtype=float)
        shape = S.shape
        entries = [(i, j, S[i, j]) for i in range(shape[0]) for j in range(shape[1])]
    entries.sort(key=lambda e: e[2], reverse=(sense == 'maximize'))
    used_i, used_j, pairs = set(), set(), []
    for i, j, _ in entries:
        if i in used_i or j in used_j:
            continue
        pairs.append((i, j))
        used_i.add(i)
        used_j.add(j)
        if len(pairs) == min(shape):
            break
    return pairs


def stable_matching(scores, *, sense='maximize'):
    """Gale-Shapley stable matching (optimizes stability, not total score).

    See :func:`equate.util.stable_marriage_matching`; a sparse matrix is densified.
    """
    from equate.util import stable_marriage_matching

    S = scores.toarray() if issparse(scores) else scores
    return stable_marriage_matching(S, sense=sense)
