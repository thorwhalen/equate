"""Column featurizers: a column (a sequence of cell values) as a comparable object.

Lifting the pipeline from cells to columns (research doc 07, §8.1): a column's values are
its signal when headers are missing or untrustworthy. :func:`value_set` represents each
column as the set of its distinct, normalized, non-missing values — the input of the set
comparators (``jaccard``, ``overlap``, ``containment``) used for join-key discovery and
near-duplicate columns.
"""

import math
from collections.abc import Callable, Iterable

from equate.base import FeaturizerMeta

__all__ = ["value_set", "normalize_value", "is_missing"]


def is_missing(v) -> bool:
    """``None``, an empty/blank string, or a float NaN."""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    return isinstance(v, float) and math.isnan(v)


def normalize_value(v) -> str:
    """The default cell normalization: text, stripped and lower-cased; ``1.0`` and ``1``
    read alike (an integral float drops its ``.0``).

    >>> normalize_value('  Paris '), normalize_value(3.0), normalize_value(3)
    ('paris', '3', '3')
    """
    if isinstance(v, float) and v.is_integer():
        v = int(v)
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
