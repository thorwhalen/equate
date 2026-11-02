"""Native kernels for ``equate.compare.sequence`` (``backend="rust"``).

This package is an implementation detail of `equate <https://github.com/thorwhalen/equate>`_:
install it and pass ``backend="rust"`` to :func:`equate.compare.sequence.dtw`,
``subsequence_dtw``, ``needleman_wunsch``, ``smith_waterman`` (and their ``*_alignment``
twins). The functions here are private kernels with no defaults and no validation:
equate's facade owns the public signatures, the validation and the defaults, and calls
them with checked, fully specified arguments. They reproduce the numpy backend exactly,
tie-breaking included (equate's ``tests/test_differential.py`` checks it).
"""

from importlib.metadata import PackageNotFoundError, version

from equate_rust._core import (
    DEBUG_BUILD,
    align_codes,
    align_scores,
    dtw_accumulate,
    subsequence_dtw_accumulate,
    traceback_dtw,
)

__all__ = [
    "DEBUG_BUILD",
    "align_codes",
    "align_scores",
    "dtw_accumulate",
    "subsequence_dtw_accumulate",
    "traceback_dtw",
]

try:
    __version__ = version("equate-rust")
except PackageNotFoundError:  # a source tree without an install
    __version__ = "0.0.0"
