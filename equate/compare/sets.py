"""Set comparators: how much two *value sets* overlap (stage ②, for columns and bags).

When the objects are sets — a column's distinct values, a record's tokens — overlap is
the signal (research doc 07, §4-5). Three standard measures, all in ``[0, 1]``:

- :func:`jaccard` ``|A ∩ B| / |A ∪ B|`` — symmetric; near-duplicate detection;
- :func:`overlap` ``|A ∩ B| / min(|A|, |B|)`` — symmetric; forgives a size mismatch;
- :func:`containment` ``|A ∩ B| / |A|`` — **directional** (how much of ``A`` is in ``B``):
  the joinability score of LSH Ensemble / JOSIE, so ``containment(fk, pk)`` is high when
  a foreign key's values all appear in the primary key, whatever else the latter holds.

Two empty sets score ``0.0`` (no evidence of correspondence), never ``nan``.
"""

from equate.base import ComparatorMeta

__all__ = ["jaccard", "overlap", "containment"]

_SYMMETRIC = ComparatorMeta(polarity="similarity", bounded=True, is_symmetric=True)
_DIRECTIONAL = ComparatorMeta(polarity="similarity", bounded=True, is_symmetric=False)


def _as_set(x):
    return x if isinstance(x, (set, frozenset)) else set(x)


def jaccard(a, b) -> float:
    """``|A ∩ B| / |A ∪ B|``.

    >>> jaccard({1, 2, 3}, {2, 3, 4})
    0.5
    """
    a, b = _as_set(a), _as_set(b)
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def overlap(a, b) -> float:
    """``|A ∩ B| / min(|A|, |B|)`` (the overlap coefficient).

    >>> overlap({1, 2}, {1, 2, 3, 4})
    1.0
    """
    a, b = _as_set(a), _as_set(b)
    smaller = min(len(a), len(b))
    return len(a & b) / smaller if smaller else 0.0


def containment(a, b) -> float:
    """``|A ∩ B| / |A|``: the share of ``a``'s values found in ``b`` (directional).

    >>> containment({1, 2}, {1, 2, 3, 4}), containment({1, 2, 3, 4}, {1, 2})
    (1.0, 0.5)
    """
    a, b = _as_set(a), _as_set(b)
    return len(a & b) / len(a) if a else 0.0


jaccard.meta = overlap.meta = _SYMMETRIC
containment.meta = _DIRECTIONAL
