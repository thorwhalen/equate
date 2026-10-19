"""Column featurizers: a column (a sequence of cell values) as a comparable object.

Lifting the pipeline from cells to columns (research doc 07, §8.1): a column's values are
its signal when headers are missing or untrustworthy. :func:`value_set` represents each
column as the set of its distinct, normalized, non-missing values — the input of the set
comparators (``jaccard``, ``overlap``, ``containment``) used for join-key discovery and
near-duplicate columns.
"""

import numbers
from collections.abc import Callable, Iterable

from equate.base import FeaturizerMeta

__all__ = ["value_set", "normalize_value", "is_missing"]


def is_missing(v) -> bool:
    """``None``, an empty/blank string, a NaN of any float type, or pandas' ``NA`` / ``NaT``.

    >>> [is_missing(v) for v in (None, ' ', float('nan'), 0, 'x')]
    [True, True, True, False, False]
    """
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if type(v).__name__ in ("NAType", "NaTType"):  # pandas, without importing it
        return True
    try:
        return bool(v != v)  # NaN is the one value unequal to itself (numpy floats too)
    except (TypeError, ValueError):
        return False


def normalize_value(v) -> str:
    """The default cell normalization: text, stripped and lower-cased; ``1.0`` and ``1``
    read alike (an integral real drops its ``.0``, numpy scalars included).

    >>> normalize_value('  Paris '), normalize_value(3.0), normalize_value(3)
    ('paris', '3', '3')
    """
    if isinstance(v, numbers.Real) and not isinstance(v, (bool, numbers.Integral)):
        f = float(v)
        if f.is_integer():
            v = int(f)
    return str(v).strip().lower()


def value_set(columns: Iterable, *, normalize: Callable = normalize_value) -> list:
    """Each column as the ``frozenset`` of its normalized, non-missing values (batched).

    >>> [sorted(s) for s in value_set([['Paris', 'paris ', None], [1, 2.0]])]
    [['paris'], ['1', '2']]
    """
    return [
        frozenset(normalize(v) for v in column if not is_missing(v))
        for column in columns
    ]


value_set.meta = FeaturizerMeta(output_kinds=("set",), license="core")
