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

    A dense matrix goes straight to ``linear_sum_assignment`` on ``to_cost(scores, sense)``.
    A **sparse (blocked)** matrix honors the *candidate* semantics: absent cells are
    worst-cased (via ``to_cost``/``_sparse_to_cost``), the LAP is solved on that cost, and
    any assignment landing on a non-candidate (absent) cell is dropped — so a row with no
    candidate is left unmatched (a *partial* matching), and an all-absent matrix yields no
    matches. This densifies the cost; a non-densifying sparse LAP is a future optimization
    (scipy's sparse solver silently drops explicitly-stored zero-score candidates, so it is
    not used here).
    """
    cost = to_cost(scores, sense=sense)  # handles sparse via _sparse_to_cost (holes worst-cased)
    row, col = linear_sum_assignment(cost)
    if issparse(scores):
        from equate.base import _stored_mask

        stored = _stored_mask(scores)  # True only on real candidate cells
        return [(int(i), int(j)) for i, j in zip(row, col) if stored[i, j]]
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

    See :func:`equate.util.stable_marriage_matching`. A sparse (blocked) matrix routes
    through ``to_cost`` (absent cells worst-cased), is ranked by that cost, and
    hole-assignments are dropped — so an absent cell never out-ranks a real candidate.
    """
    from equate.util import stable_marriage_matching

    if issparse(scores):
        from equate.base import _stored_mask

        stored = _stored_mask(scores)
        cost = to_cost(scores, sense=sense)  # dense, absent cells worst-cased
        pairs = stable_marriage_matching(cost, sense='minimize')  # rank by cost ascending
        return [(i, j) for i, j in pairs if stored[i, j]]
    return stable_marriage_matching(scores, sense=sense)
