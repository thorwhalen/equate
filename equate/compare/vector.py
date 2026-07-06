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
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu and nv else 0.0


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
