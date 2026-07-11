"""Soft / optimal-transport matching (stage ③): a fractional transport plan rather than a
hard assignment — the home for graded, partial, unequal-mass and cross-modal matching
(and, later, soft-DTW / Gromov-Wasserstein — decision register D8). Requires ``equate[ot]``.
"""

import numpy as np

from equate._dependencies import require
from equate.base import ScoreMatrix

__all__ = ["soft_match", "harden"]


def soft_match(scores, *, sense=None, reg=0.1, a=None, b=None):
    """Entropic optimal-transport (Sinkhorn) plan over the cost derived from ``scores``.

    Accepts a :class:`~equate.base.ScoreMatrix` (native) or a raw array with ``sense=``.
    Returns a dense transport-plan matrix whose rows/cols sum to the marginals ``a``/``b``
    (uniform by default). Requires ``equate[ot]`` (POT). ``reg`` (entropic regularization)
    must be > 0 (``reg=0`` divides by zero inside Sinkhorn).
    """
    if reg <= 0:
        raise ValueError(
            f"soft_match reg (entropic regularization) must be > 0, got {reg!r}"
        )
    ot = require("ot", extra="ot", purpose="soft/optimal-transport matching")
    # dense_cost worst-cases a sparse (blocked) matrix's absent cells (never densify first)
    cost = np.asarray(ScoreMatrix.coerce(scores, sense=sense).dense_cost(), dtype=float)
    n, m = cost.shape
    a = np.ones(n) / n if a is None else np.asarray(a, dtype=float)
    b = np.ones(m) / m if b is None else np.asarray(b, dtype=float)
    return ot.sinkhorn(a, b, cost, reg)


def harden(plan):
    """Recover a hard 1:1 assignment from a transport ``plan`` (LAP maximizing mass)."""
    from equate.matching.assign import optimal_matching

    return optimal_matching(np.asarray(plan, dtype=float), sense="maximize")
