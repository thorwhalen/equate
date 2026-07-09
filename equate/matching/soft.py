"""Soft / optimal-transport matching (stage ③): a fractional transport plan rather than a
hard assignment — the home for graded, partial, unequal-mass and cross-modal matching
(and, later, soft-DTW / Gromov-Wasserstein — decision register D8). Requires ``equate[ot]``.
"""

import numpy as np

from equate._dependencies import require
from equate.base import to_cost

__all__ = ["soft_match", "harden"]


def soft_match(scores, *, sense="maximize", reg=0.1, a=None, b=None):
    """Entropic optimal-transport (Sinkhorn) plan over the cost derived from ``scores``.

    Returns a dense transport-plan matrix whose rows/cols sum to the marginals ``a``/``b``
    (uniform by default). Requires ``equate[ot]`` (POT). ``reg`` (entropic regularization)
    must be > 0 (``reg=0`` divides by zero inside Sinkhorn).
    """
    if reg <= 0:
        raise ValueError(
            f"soft_match reg (entropic regularization) must be > 0, got {reg!r}"
        )
    ot = require("ot", extra="ot", purpose="soft/optimal-transport matching")
    # to_cost handles a sparse (blocked) matrix via _sparse_to_cost (holes worst-cased)
    cost = np.asarray(to_cost(scores, sense=sense), dtype=float)
    n, m = cost.shape
    a = np.ones(n) / n if a is None else np.asarray(a, dtype=float)
    b = np.ones(m) / m if b is None else np.asarray(b, dtype=float)
    return ot.sinkhorn(a, b, cost, reg)


def harden(plan):
    """Recover a hard 1:1 assignment from a transport ``plan`` (LAP maximizing mass)."""
    from equate.matching.assign import optimal_matching

    return optimal_matching(np.asarray(plan, dtype=float), sense="maximize")
