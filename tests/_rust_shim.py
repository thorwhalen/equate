"""A pure-Python stand-in for ``equate_rust`` (test harness only, never shipped).

It exposes the native module's interface, implemented with the numpy backend's own
kernels. With ``EQUATE_RUST_SHIM=1``, ``tests/test_differential.py`` installs it as
``equate_rust``, so the differential tests run reference against reference: the check
that the harness and the facade's ``backend='rust'`` routing are sound before (or
without) a native build.
"""

import numpy as np

from equate.compare import sequence as _Q

IS_SHIM = True
DEBUG_BUILD = False


def dtw_accumulate(costs, /, *, window):
    return _Q._dtw_accumulate(costs, window)


def subsequence_dtw_accumulate(costs, /):
    return _Q._subsequence_accumulate(costs)


def traceback_dtw(acc, i, j, /, *, stop_row0):
    return _Q._traceback_dtw(acc, i, j, stop_row0=stop_row0)


def _align(n, m, sub, *, gap, local):
    al = _Q._align_dp(
        range(n),
        range(m),
        local=local,
        substitution=sub,
        match=None,
        mismatch=None,
        gap=gap,
    )
    return al.score, al.path


def align_scores(scores, /, *, gap, local):
    n, m = scores.shape
    return _align(n, m, lambda i, j: scores[i, j], gap=gap, local=local)


def align_codes(a, b, /, *, match_score, mismatch_score, gap, local):
    a, b = np.asarray(a), np.asarray(b)
    sub = lambda i, j: match_score if a[i] == b[j] else mismatch_score  # noqa: E731
    return _align(len(a), len(b), sub, gap=gap, local=local)
