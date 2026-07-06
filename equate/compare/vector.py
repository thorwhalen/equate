"""Vector comparators (stage ②): pairwise similarity/distance between two vectors."""

import math

import numpy as np

from equate.base import ComparatorMeta

__all__ = ['cosine', 'dot', 'angular_distance']


def cosine(u, v):
    """Cosine similarity between two vectors (0 if either is a zero vector).

    Note: cosine is a *similarity*, not a metric — do not use it with a
    triangle-inequality index (use :func:`angular_distance` when a metric is needed).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError(
            f"cosine: vectors must have the same shape, got {u.shape} and {v.shape}"
        )
    # Scale each vector by its max-abs component first, so squaring a large component
    # cannot overflow to inf/NaN. Cosine is scale-invariant, so this does not change
    # the value for in-range inputs but keeps huge-magnitude vectors correct.
    su = float(np.abs(u).max(initial=0.0))
    sv = float(np.abs(v).max(initial=0.0))
    if su == 0.0 or sv == 0.0:
        return 0.0
    u = u / su
    v = v / sv
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    result = float(u @ v / denom) if denom else 0.0
    return result if math.isfinite(result) else 0.0


cosine.meta = ComparatorMeta(
    polarity='similarity', bounded=True, is_metric=False, is_symmetric=True
)


def dot(u, v):
    """Dot product of two vectors (an unbounded similarity)."""
    return float(np.asarray(u, dtype=float) @ np.asarray(v, dtype=float))


dot.meta = ComparatorMeta(
    polarity='similarity', bounded=False, is_metric=False, is_symmetric=True
)


def angular_distance(u, v):
    """Angular distance in [0, 1] — a true metric derived from cosine similarity."""
    c = min(1.0, max(-1.0, cosine(u, v)))
    return math.acos(c) / math.pi


angular_distance.meta = ComparatorMeta(
    polarity='distance', bounded=True, is_metric=True, is_symmetric=True
)
