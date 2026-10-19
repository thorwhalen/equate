"""The text of an object, for text featurizers: a record (mapping) is its joined values.

Text featurizers (TF-IDF, embedders) need a string. A record — a dict from JSON, a CSV row
— has none of its own, so it is read as its non-empty values joined by spaces; anything
else as ``str(obj)``. This is what lets ``match(records_a, records_b)`` work on the
default vector route.
"""

from collections.abc import Mapping

__all__ = ["record_text"]


def record_text(obj) -> str:
    """Text of ``obj``: a mapping's non-empty values joined by spaces, else ``str(obj)``.

    >>> record_text({'name': 'Jon Smith', 'city': 'Paris', 'zip': None})
    'Jon Smith Paris'
    >>> record_text(42)
    '42'
    """
    if isinstance(obj, Mapping):
        return " ".join(str(v) for v in obj.values() if not _missing(v))
    return obj if isinstance(obj, str) else str(obj)


def _missing(v) -> bool:
    # identity / type checks only: `v in (None, "")` breaks on a numpy array value
    return v is None or (isinstance(v, str) and v == "")
