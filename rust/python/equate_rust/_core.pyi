"""Stub for the private native module ``equate_rust._core`` (hand-written; checked with stubtest)."""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEBUG_BUILD",
    "align_codes",
    "align_scores",
    "dtw_accumulate",
    "subsequence_dtw_accumulate",
    "traceback_dtw",
]

DEBUG_BUILD: bool

def dtw_accumulate(
    costs: NDArray[np.float64], /, *, window: int | None
) -> NDArray[np.float64]: ...
def subsequence_dtw_accumulate(
    costs: NDArray[np.float64], /
) -> NDArray[np.float64]: ...
def traceback_dtw(
    acc: NDArray[np.float64], i: int, j: int, /, *, stop_row0: bool
) -> tuple[tuple[int, int], ...]: ...
def align_scores(
    scores: NDArray[np.float64], /, *, gap: float, local: bool
) -> tuple[float, tuple[tuple[int | None, int | None], ...]]: ...
def align_codes(
    a: NDArray[np.int64],
    b: NDArray[np.int64],
    /,
    *,
    match_score: float,
    mismatch_score: float,
    gap: float,
    local: bool,
) -> tuple[float, tuple[tuple[int | None, int | None], ...]]: ...
