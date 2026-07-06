"""Numeric, date, and geospatial comparators (stage ②): turn a domain distance into a
[0, 1] similarity via a decay function. Pure standard library.
"""

import math

from equate.base import ComparatorMeta

__all__ = [
    'exp_decay',
    'linear_decay',
    'gaussian_decay',
    'haversine',
    'geo_similarity',
]

_SIM = ComparatorMeta(polarity='similarity', bounded=True, is_metric=False, is_symmetric=True)


def _default_distance(x, y):
    return abs(x - y)


def exp_decay(scale=1.0, *, distance=_default_distance):
    """Comparator mapping distance ``d`` to ``exp(-d / scale)`` in ``(0, 1]``."""

    def cmp(x, y):
        return math.exp(-distance(x, y) / scale)

    cmp.meta = _SIM
    return cmp


def linear_decay(max_distance=1.0, *, distance=_default_distance):
    """Comparator mapping distance ``d`` to ``max(0, 1 - d / max_distance)``."""

    def cmp(x, y):
        return max(0.0, 1.0 - distance(x, y) / max_distance)

    cmp.meta = _SIM
    return cmp


def gaussian_decay(scale=1.0, *, distance=_default_distance):
    """Comparator mapping distance ``d`` to ``exp(-(d / scale)^2 / 2)``."""

    def cmp(x, y):
        d = distance(x, y) / scale
        return math.exp(-0.5 * d * d)

    cmp.meta = _SIM
    return cmp


def haversine(a, b):
    """Great-circle distance in km between two ``(lat, lon)`` points (degrees).

    >>> round(haversine((0.0, 0.0), (0.0, 1.0)), 1)   # 1 deg lon at the equator
    111.2
    """
    lat1, lon1 = a
    lat2, lon2 = b
    radius_km = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(h))


haversine.meta = ComparatorMeta(
    polarity='distance', bounded=False, is_metric=True, is_symmetric=True
)


def geo_similarity(scale_km=10.0):
    """Geospatial similarity comparator: exponential decay of haversine distance (km)."""
    return exp_decay(scale_km, distance=haversine)
