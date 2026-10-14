"""Sequence-alignment comparators (stage ②): score two *sequences* by aligning them.

Matching, one level down: these comparators first find a correspondence between the
*elements* of two sequences, then score it. They are ``direct(h)`` comparators — opaque,
**not ANN-indexable**, usable to score all pairs or re-rank a blocker's candidates — and
the collection-level matcher above them is unchanged (doc ``12`` §2.6, decision D8).

- **DTW** (:func:`dtw`) — dynamic time warping for real-valued series (scalars or vectors
  per step). The default is an **exact DTW inside a Sakoe-Chiba band**, *not* FastDTW
  (Wu & Keogh: banded exact DTW is faster in realistic settings). A *distance*.
- **Subsequence DTW** (:func:`subsequence_dtw`) — the best-matching *sub-span* of a long
  series for a short query (open begin and end). A directional distance.
- **Soft-DTW** (:func:`soft_dtw`) — Cuturi & Blondel's differentiable soft-minimum DTW.
  A smoothed discrepancy, not a distance (it can be negative).
- **Needleman-Wunsch** (:func:`needleman_wunsch`) / **Smith-Waterman**
  (:func:`smith_waterman`) — global / local alignment of symbolic sequences (strings,
  token lists, DNA). Alignment *scores*: higher = better.

Each comparator returns a ``float``; its ``*_alignment`` twin (also reachable as
``dtw.align`` etc.) returns an :class:`equate.base.Alignment` carrying the element-level
``path``. The core implementations are pure numpy; a faster engine is one keyword away
(``backend='dtaidistance'`` / ``'tslearn'`` / ``'biopython'``, behind the
``equate[timeseries]`` / ``equate[bio]`` extras, imported only on use).

>>> dtw([0, 1, 2, 3], [0, 0, 1, 2, 3])  # a warped copy is at distance 0
0.0
>>> needleman_wunsch('GATTACA', 'GATACA')  # 6 matches, 1 gap
5.0
>>> smith_waterman('xxGATTACAyy', 'GATTACA')  # the shared fragment
7.0
"""

import math

import numpy as np

from equate.base import Alignment, ComparatorMeta
from equate._dependencies import require

__all__ = [
    "dtw",
    "dtw_alignment",
    "subsequence_dtw",
    "subsequence_dtw_alignment",
    "soft_dtw",
    "needleman_wunsch",
    "needleman_wunsch_alignment",
    "smith_waterman",
    "smith_waterman_alignment",
    "DFLT_DTW_WINDOW",
    "DFLT_SOFT_DTW_GAMMA",
    "DFLT_MATCH_SCORE",
    "DFLT_MISMATCH_SCORE",
    "DFLT_GAP_SCORE",
]

#: Sakoe-Chiba band half-width. A ``float`` is a fraction of the longer series' length,
#: an ``int`` a number of cells, ``None`` unconstrained. The band is always widened by
#: the length difference so two series of different lengths can still be aligned.
DFLT_DTW_WINDOW = 0.1
#: soft-DTW temperature (``gamma -> 0`` recovers hard DTW).
DFLT_SOFT_DTW_GAMMA = 1.0
#: default symbolic-alignment scoring (linear gaps). Swap in a ``substitution`` callable
#: for a domain matrix (BLOSUM, keyboard distance, token similarity, ...).
DFLT_MATCH_SCORE = 1.0
DFLT_MISMATCH_SCORE = -1.0
DFLT_GAP_SCORE = -1.0

_DTW_META = ComparatorMeta(
    polarity="distance", bounded=False, is_metric=False, is_symmetric=True
)
_SUBSEQ_META = ComparatorMeta(
    polarity="distance", bounded=False, is_metric=False, is_symmetric=False
)
_ALIGN_SCORE_META = ComparatorMeta(
    polarity="similarity", bounded=False, is_metric=False, is_symmetric=True
)

_DTW_BACKENDS = ("numpy", "dtaidistance")
_SOFT_DTW_BACKENDS = ("numpy", "tslearn")
_ALIGN_BACKENDS = ("numpy", "biopython")


# --- shared helpers -----------------------------------------------------------------


def _check_backend(backend, allowed, *, what):
    if backend not in allowed:
        raise ValueError(f"{what}: backend must be one of {allowed}, got {backend!r}")


def _as_series(x, *, what):
    """A sequence of numbers (or of equal-length vectors) -> a float array ``(n, d)``."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            f"{what}: expected a sequence of scalars or of equal-length vectors, "
            f"got an array of shape {np.shape(x)}"
        )
    return arr


def _local_costs(a, b, local_cost, *, what):
    """The ``(n, m)`` local-cost matrix, and whether to take a final square root.

    The default cost is the *squared* Euclidean distance between steps, accumulated and
    square-rooted at the end — the convention of ``dtaidistance`` (so its backend is a
    drop-in speedup). A custom ``local_cost(x, y)`` is accumulated as-is (no root).
    """
    if local_cost is None:
        X, Y = _as_series(a, what=what), _as_series(b, what=what)
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"{what}: steps have different dimensions ({X.shape[1]} vs {Y.shape[1]})"
            )
        diff = X[:, None, :] - Y[None, :, :]
        return np.einsum("ijk,ijk->ij", diff, diff), True
    a, b = list(a), list(b)
    return np.array(
        [[float(local_cost(x, y)) for y in b] for x in a], dtype=float
    ), False


def _band(n, m, window):
    """Half-width of the Sakoe-Chiba band, or ``None`` for unconstrained."""
    if window is None:
        return None
    if isinstance(window, bool) or not isinstance(window, (int, float)):
        raise TypeError(
            f"window must be an int, a float fraction, or None, got {window!r}"
        )
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window!r}")
    if isinstance(window, float):
        return max(1, math.ceil(window * max(n, m)))
    return max(1, int(window))


def _in_band(i, j, n, m, w):
    """Cell ``(i, j)`` lies in the band: within ``w`` of the diagonal, widened by ``|n-m|``."""
    if w is None:
        return True
    return i - max(0, n - m) - w < j < i + max(0, m - n) + w


def _traceback_dtw(D, i, j, *, stop_row0=False):
    """Walk the accumulated-cost matrix back from ``(i, j)``; ``D`` is padded by one.

    ``stop_row0`` ends the walk on the first row (subsequence DTW's free start) instead of
    at ``(0, 0)``.
    """
    path = [(i, j)]
    while (i, j) != (0, 0):
        if i == 0 and stop_row0:
            break
        candidates = []
        if i > 0 and j > 0:
            candidates.append(
                (D[i, j], i - 1, j - 1)
            )  # D is padded: D[i, j] = cell (i-1, j-1)
        if i > 0:
            candidates.append((D[i, j + 1], i - 1, j))
        if j > 0 and not (stop_row0 and i == 0):
            candidates.append((D[i + 1, j], i, j - 1))
        _, i, j = min(candidates, key=lambda c: c[0])
        path.append((i, j))
    return tuple(reversed(path))


def _dtw_accumulate(C, w):
    """Padded accumulated-cost matrix ``D`` (``D[i+1, j+1]`` = best cost ending at ``(i, j)``)."""
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(n):
        for j in range(m):
            if not _in_band(i, j, n, m, w):
                continue
            D[i + 1, j + 1] = C[i, j] + min(D[i, j], D[i, j + 1], D[i + 1, j])
    return D


# --- DTW ----------------------------------------------------------------------------


def dtw_alignment(a, b, *, window=DFLT_DTW_WINDOW, local_cost=None, backend="numpy"):
    """Banded exact DTW between ``a`` and ``b`` -> :class:`~equate.base.Alignment`.

    ``a``/``b`` are sequences of numbers or of equal-length vectors (multivariate series).
    ``window`` is the Sakoe-Chiba half-width (see :data:`DFLT_DTW_WINDOW`; ``None`` is
    unconstrained DTW). ``local_cost(x, y)`` replaces the default squared-Euclidean step
    cost (then the total is not square-rooted). ``backend='dtaidistance'`` (``equate
    [timeseries]``) computes the same quantity in C for the default cost and reports no
    path.

    >>> al = dtw_alignment([1, 2, 3], [1, 2, 2, 3], window=None)
    >>> al.score, al.path
    (0.0, ((0, 0), (1, 1), (1, 2), (2, 3)))
    """
    _check_backend(backend, _DTW_BACKENDS, what="dtw")
    if backend == "dtaidistance":
        return _dtw_dtaidistance(a, b, window=window, local_cost=local_cost)
    C, root = _local_costs(a, b, local_cost, what="dtw")
    n, m = C.shape
    if n == 0 or m == 0:
        raise ValueError("dtw: cannot align an empty sequence")
    D = _dtw_accumulate(C, _band(n, m, window))
    total = float(D[n, m])
    path = _traceback_dtw(D, n - 1, m - 1)
    return Alignment(math.sqrt(total) if root else total, path, sense="minimize")


def _dtw_dtaidistance(a, b, *, window, local_cost):
    if local_cost is not None:
        raise ValueError(
            "dtw: backend='dtaidistance' supports only the default local cost"
        )
    dtaidistance = require("dtaidistance", extra="timeseries", purpose="fast DTW")
    from dtaidistance import dtw as _dtw, dtw_ndim as _dtw_ndim

    X, Y = _as_series(a, what="dtw"), _as_series(b, what="dtw")
    w = _band(len(X), len(Y), window)
    kw = {} if w is None else {"window": w}
    if X.shape[1] == 1:
        score = _dtw.distance(X[:, 0], Y[:, 0], **kw)
    else:
        score = _dtw_ndim.distance(X, Y, **kw)
    return Alignment(float(score), None, sense="minimize")


def dtw(a, b, *, window=DFLT_DTW_WINDOW, local_cost=None, backend="numpy"):
    """DTW distance (lower = more alike) — the score of :func:`dtw_alignment`.

    >>> dtw([0, 1, 2], [0, 1, 2])
    0.0
    >>> dtw([0, 0, 0], [1, 1, 1])  # sqrt(1 + 1 + 1)
    1.7320508075688772
    """
    return dtw_alignment(
        a, b, window=window, local_cost=local_cost, backend=backend
    ).score


dtw.meta = _DTW_META
dtw.align = dtw_alignment


# --- subsequence DTW ----------------------------------------------------------------


def subsequence_dtw_alignment(query, series, *, local_cost=None):
    """Best match of ``query`` inside any sub-span of ``series`` -> ``Alignment``.

    Open-begin, open-end DTW: the query must be consumed whole, the series need not be.
    The path's ``j`` range is the matched span. Directional — ``(query, series)`` order
    matters.

    >>> al = subsequence_dtw_alignment([5, 6, 7], [0, 0, 5, 6, 7, 0])
    >>> al.score, al.path[0], al.path[-1]
    (0.0, (0, 2), (2, 4))
    """
    C, root = _local_costs(query, series, local_cost, what="subsequence_dtw")
    n, m = C.shape
    if n == 0 or m == 0:
        raise ValueError("subsequence_dtw: cannot align an empty sequence")
    D = np.full((n + 1, m + 1), np.inf)
    D[0, :] = 0.0  # free start anywhere in the series
    for i in range(n):
        for j in range(m):
            D[i + 1, j + 1] = C[i, j] + min(D[i, j], D[i, j + 1], D[i + 1, j])
    end = int(np.argmin(D[n, 1:]))
    total = float(D[n, end + 1])
    path = _traceback_dtw(D, n - 1, end, stop_row0=True)
    return Alignment(math.sqrt(total) if root else total, path, sense="minimize")


def subsequence_dtw(query, series, *, local_cost=None):
    """Subsequence-DTW distance of ``query`` within ``series`` (directional)."""
    return subsequence_dtw_alignment(query, series, local_cost=local_cost).score


subsequence_dtw.meta = _SUBSEQ_META
subsequence_dtw.align = subsequence_dtw_alignment


# --- soft-DTW -----------------------------------------------------------------------


def soft_dtw(a, b, *, gamma=DFLT_SOFT_DTW_GAMMA, backend="numpy"):
    """Soft-DTW discrepancy (Cuturi & Blondel 2017) with temperature ``gamma``.

    Replaces DTW's hard ``min`` by a log-sum-exp soft-minimum over all warping paths (on
    *squared*-Euclidean step costs, no final root), so it is differentiable — but it is
    **not a distance**: it can be negative and ``soft_dtw(x, x) != 0``. ``gamma -> 0``
    recovers (squared) DTW. ``backend='tslearn'`` (``equate[timeseries]``) uses
    ``tslearn.metrics.soft_dtw``, which computes the same quantity.

    >>> round(soft_dtw([0, 1, 2], [0, 1, 2], gamma=1e-3), 6)
    0.0
    >>> soft_dtw([0, 1, 2], [0, 1, 2]) < 0  # smoothing makes it negative
    True
    """
    _check_backend(backend, _SOFT_DTW_BACKENDS, what="soft_dtw")
    if gamma <= 0:
        raise ValueError(f"soft_dtw: gamma must be positive, got {gamma!r}")
    if backend == "tslearn":
        require("tslearn", extra="timeseries", purpose="soft-DTW")
        from tslearn.metrics import soft_dtw as _soft_dtw

        X, Y = _as_series(a, what="soft_dtw"), _as_series(b, what="soft_dtw")
        return float(_soft_dtw(X, Y, gamma=gamma))
    C, _ = _local_costs(a, b, None, what="soft_dtw")
    n, m = C.shape
    if n == 0 or m == 0:
        raise ValueError("soft_dtw: cannot align an empty sequence")
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0.0
    for i in range(n):
        for j in range(m):
            prev = np.array([R[i, j], R[i, j + 1], R[i + 1, j]])
            z = -prev / gamma
            zmax = np.max(z)
            if not np.isfinite(zmax):  # all predecessors unreachable
                continue
            softmin = -gamma * (zmax + np.log(np.sum(np.exp(z - zmax))))
            R[i + 1, j + 1] = C[i, j] + softmin
    return float(R[n, m])


soft_dtw.meta = ComparatorMeta(
    polarity="distance", bounded=False, is_metric=False, is_symmetric=True
)


# --- symbolic alignment: Needleman-Wunsch / Smith-Waterman --------------------------


def _scorer(substitution, match, mismatch):
    if substitution is not None:
        return substitution
    return lambda x, y: match if x == y else mismatch


def _align_dp(a, b, *, local, substitution, match, mismatch, gap):
    """Shared NW/SW dynamic program with traceback (linear gap penalty)."""
    a, b = list(a), list(b)
    n, m = len(a), len(b)
    sub = _scorer(substitution, match, mismatch)
    H = np.zeros((n + 1, m + 1))
    if not local:
        H[:, 0] = gap * np.arange(n + 1)
        H[0, :] = gap * np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(
                H[i - 1, j - 1] + sub(a[i - 1], b[j - 1]),
                H[i - 1, j] + gap,
                H[i, j - 1] + gap,
            )
            H[i, j] = max(best, 0.0) if local else best
    if local:
        i, j = np.unravel_index(int(np.argmax(H)), H.shape)
        i, j = int(i), int(j)
    else:
        i, j = n, m
    score = float(H[i, j])
    path = []
    while i > 0 or j > 0:
        if local and H[i, j] == 0:
            break
        if i > 0 and j > 0 and H[i, j] == H[i - 1, j - 1] + sub(a[i - 1], b[j - 1]):
            path.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and H[i, j] == H[i - 1, j] + gap:
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1
    return Alignment(score, tuple(reversed(path)), sense="maximize")


def _align_biopython(a, b, *, mode, substitution, match, mismatch, gap):
    if substitution is not None:
        raise ValueError(
            "backend='biopython' takes match/mismatch/gap scores, not a substitution callable"
        )
    require("Bio", extra="bio", purpose="biosequence alignment")
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.gap_score = gap
    return Alignment(float(aligner.score(a, b)), None, sense="maximize")


def needleman_wunsch_alignment(
    a,
    b,
    *,
    substitution=None,
    match=DFLT_MATCH_SCORE,
    mismatch=DFLT_MISMATCH_SCORE,
    gap=DFLT_GAP_SCORE,
    backend="numpy",
):
    """Global (Needleman-Wunsch) alignment of two symbolic sequences -> ``Alignment``.

    Aligns the *entireties* of ``a`` and ``b`` (strings, token lists, ...), maximizing
    the summed ``substitution(x, y)`` (default ``match``/``mismatch``) plus a linear
    ``gap`` per skipped element. Gaps appear in the path as ``None``.

    >>> needleman_wunsch_alignment('ACT', 'AT').path
    ((0, 0), (1, None), (2, 1))
    """
    _check_backend(backend, _ALIGN_BACKENDS, what="needleman_wunsch")
    kw = dict(substitution=substitution, match=match, mismatch=mismatch, gap=gap)
    if backend == "biopython":
        return _align_biopython(a, b, mode="global", **kw)
    return _align_dp(a, b, local=False, **kw)


def needleman_wunsch(a, b, **kwargs):
    """Global alignment score (higher = more alike) — see :func:`needleman_wunsch_alignment`."""
    return needleman_wunsch_alignment(a, b, **kwargs).score


needleman_wunsch.meta = _ALIGN_SCORE_META
needleman_wunsch.align = needleman_wunsch_alignment


def smith_waterman_alignment(
    a,
    b,
    *,
    substitution=None,
    match=DFLT_MATCH_SCORE,
    mismatch=DFLT_MISMATCH_SCORE,
    gap=DFLT_GAP_SCORE,
    backend="numpy",
):
    """Local (Smith-Waterman) alignment -> ``Alignment`` of the best-matching fragments.

    Like :func:`needleman_wunsch_alignment` but floored at zero, so only the
    highest-scoring aligned *sub-regions* count; the path covers just that fragment pair.

    >>> al = smith_waterman_alignment('xxABCyy', 'zABCz')
    >>> al.score, al.path
    (3.0, ((2, 1), (3, 2), (4, 3)))
    """
    _check_backend(backend, _ALIGN_BACKENDS, what="smith_waterman")
    kw = dict(substitution=substitution, match=match, mismatch=mismatch, gap=gap)
    if backend == "biopython":
        return _align_biopython(a, b, mode="local", **kw)
    return _align_dp(a, b, local=True, **kw)


def smith_waterman(a, b, **kwargs):
    """Local alignment score (higher = more alike) — see :func:`smith_waterman_alignment`."""
    return smith_waterman_alignment(a, b, **kwargs).score


smith_waterman.meta = _ALIGN_SCORE_META
smith_waterman.align = smith_waterman_alignment
