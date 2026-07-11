"""Assignment matchers (stage ③): turn a score matrix into matched ``(i, j)`` pairs.

Expose the *objective* (optimal / greedy / stable), not the algorithm. The optimal
matcher is **sparse-aware**: a blocked (sparse) score matrix routes to the sparse LAP
solver instead of densifying. All route similarity→cost through the ``to_cost`` SSOT
(decision register D2), so a ``sense`` of ``'maximize'`` (similarity) or ``'minimize'``
(distance) is honored consistently.
"""

from scipy.optimize import linear_sum_assignment

from equate.base import ScoreMatrix, scorematrix_matcher

__all__ = ["optimal_matching", "greedy_matching", "stable_matching"]


@scorematrix_matcher
def optimal_matching(scores, *, sense=None):
    """Globally optimal 1:1 assignment (the linear assignment problem).

    Accepts a :class:`~equate.base.ScoreMatrix` (native) or a raw dense/sparse array with
    ``sense=`` (back-compat). Solves ``linear_sum_assignment`` on
    :meth:`~equate.base.ScoreMatrix.dense_cost` — so a **sparse (blocked)** matrix has its
    absent cells worst-cased (never preferred over a real pair) and any assignment landing
    on a non-candidate cell is dropped via :meth:`~equate.base.ScoreMatrix.candidate_mask`:
    a row with no candidate is left unmatched (a *partial* matching), an all-absent matrix
    yields no matches. Densifying the cost is a correctness stopgap; a non-densifying
    sparse LAP is a future optimization (scipy's sparse solver silently drops
    explicitly-stored zero-score candidates, so it is not used here).
    """
    sm = ScoreMatrix.coerce(scores, sense=sense)
    row, col = linear_sum_assignment(sm.dense_cost())
    return sm.drop_holes([(int(i), int(j)) for i, j in zip(row, col)])


@scorematrix_matcher
def greedy_matching(scores, *, sense=None):
    """Greedy 1:1: repeatedly take the best still-available ``(i, j)``, removing both.

    Order-dependent and not globally optimal, but fast. Iterates
    :meth:`~equate.base.ScoreMatrix.stored_entries`, so for a sparse (blocked) matrix only
    the stored candidate cells are considered — an absent cell is never a match.
    """
    sm = ScoreMatrix.coerce(scores, sense=sense)
    shape = sm.shape
    entries = list(sm.stored_entries())
    entries.sort(key=lambda e: e[2], reverse=(sm.sense == "maximize"))
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


@scorematrix_matcher
def stable_matching(scores, *, sense=None):
    """Gale-Shapley stable matching (optimizes stability, not total score).

    See :func:`equate.util.stable_marriage_matching`. Ranks preferences by
    :meth:`~equate.base.ScoreMatrix.dense_cost` (absent cells worst-cased), so a sparse
    (blocked) cell never out-ranks a real candidate; hole-assignments are then dropped via
    :meth:`~equate.base.ScoreMatrix.candidate_mask`.
    """
    from equate.util import stable_marriage_matching

    sm = ScoreMatrix.coerce(scores, sense=sense)
    pairs = stable_marriage_matching(sm.dense_cost(), sense="minimize")
    return sm.drop_holes(pairs)
