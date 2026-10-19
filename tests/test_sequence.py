"""Tests for the sequence-alignment comparators (DTW family, Needleman-Wunsch,
Smith-Waterman): values against independent references, the band, paths, the
registry/facade wiring, and the optional backends (exercised with fake modules)."""

import math
import sys
import types
from functools import lru_cache

import numpy as np
import pytest

from equate import match, Alignment, MissingDependencyError
from equate.compare import comparators, resolve_comparator, polarity_sense
from equate.compare import sequence as Q


# --- independent references -----------------------------------------------------


def _ref_dtw(a, b, allowed=lambda i, j: True):
    """Memoized recursive DTW on squared differences (root at the end)."""
    a, b = list(map(float, a)), list(map(float, b))

    @lru_cache(maxsize=None)
    def D(i, j):
        if not allowed(i, j):
            return math.inf
        c = (a[i] - b[j]) ** 2
        if i == 0 and j == 0:
            return c
        prev = []
        if i > 0 and j > 0:
            prev.append(D(i - 1, j - 1))
        if i > 0:
            prev.append(D(i - 1, j))
        if j > 0:
            prev.append(D(i, j - 1))
        return c + min(prev)

    return math.sqrt(D(len(a) - 1, len(b) - 1))


def _ref_nw(a, b, match=1, mismatch=-1, gap=-1):
    @lru_cache(maxsize=None)
    def H(i, j):
        if i == 0:
            return gap * j
        if j == 0:
            return gap * i
        s = match if a[i - 1] == b[j - 1] else mismatch
        return max(H(i - 1, j - 1) + s, H(i - 1, j) + gap, H(i, j - 1) + gap)

    return float(H(len(a), len(b)))


def _path_cost(a, b, path):
    return math.sqrt(sum((a[i] - b[j]) ** 2 for i, j in path))


def _assert_warping_path(path, n, m):
    assert path[0] == (0, 0) and path[-1] == (n - 1, m - 1)
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        assert (i1 - i0, j1 - j0) in {(1, 1), (1, 0), (0, 1)}


# --- DTW ------------------------------------------------------------------------

rng = np.random.default_rng(0)
_SERIES = [
    (rng.normal(size=n).round(2).tolist(), rng.normal(size=m).round(2).tolist())
    for n, m in [(5, 5), (4, 7), (8, 3), (6, 6), (1, 4)]
]


@pytest.mark.parametrize("a,b", _SERIES)
def test_unconstrained_dtw_matches_reference(a, b):
    al = Q.dtw_alignment(a, b, window=None)
    assert al.score == pytest.approx(_ref_dtw(a, b))
    _assert_warping_path(al.path, len(a), len(b))
    # the reported path really achieves the reported score
    assert _path_cost(a, b, al.path) == pytest.approx(al.score)


@pytest.mark.parametrize("a,b", _SERIES)
@pytest.mark.parametrize("w", [0, 1, 2, 3])
def test_banded_dtw_matches_reference(a, b, w):
    n, m = len(a), len(b)

    def allowed(i, j):  # an inclusive half-width band, widened by |n - m|
        return i - max(0, n - m) - w <= j <= i + max(0, m - n) + w

    al = Q.dtw_alignment(a, b, window=w)
    assert al.score == pytest.approx(_ref_dtw(a, b, allowed))
    assert all(allowed(i, j) for i, j in al.path)
    # a band can only ever cost more than (or the same as) unconstrained DTW
    assert al.score >= Q.dtw(a, b, window=None) - 1e-12


def test_float_window_is_a_fraction_of_the_longer_series():
    a, b = list(range(20)), list(range(20))[::-1]
    assert Q.dtw(a, b, window=0.1) == Q.dtw(a, b, window=2)
    with pytest.raises(ValueError):
        Q.dtw(a, b, window=-1)
    with pytest.raises(TypeError):
        Q.dtw(a, b, window="wide")
    # numpy scalars are fine
    assert Q.dtw(a, b, window=np.int64(2)) == Q.dtw(a, b, window=2)
    assert Q.dtw(a, b, window=np.float32(0.1)) == Q.dtw(a, b, window=2)


def test_window_zero_is_euclidean_and_the_default_still_warps_short_series():
    a, b = [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]
    assert Q.dtw(a, b, window=0) == pytest.approx(math.sqrt(5))  # no warping
    # the default band (0.1 of a short series -> one cell) still allows a shift
    assert Q.dtw(a, b) == pytest.approx(Q.dtw(a, b, window=None))
    assert Q.dtw(a, b) < Q.dtw(a, b, window=0)


def test_dtw_accepts_iterators():
    assert Q.dtw(iter([1, 2, 3]), (x for x in [1, 2, 3])) == 0.0


def test_dtw_is_symmetric_and_zero_on_warped_copies():
    a, b = [0, 1, 2, 3, 2, 1], [0, 0, 1, 2, 3, 3, 2, 1]
    assert Q.dtw(a, b, window=None) == 0.0
    x, y = _SERIES[1]
    assert Q.dtw(x, y) == pytest.approx(Q.dtw(y, x))


def test_dtw_multivariate_and_custom_local_cost():
    a = [[0, 0], [1, 1], [2, 2]]
    b = [[0, 0], [1, 1], [1, 1], [2, 2]]
    assert Q.dtw(a, b, window=None) == 0.0
    with pytest.raises(ValueError, match="dimensions"):
        Q.dtw([[0, 0]], [[0, 0, 0]])
    # a custom local cost is accumulated without a square root (L1 DTW)
    assert Q.dtw([0, 0, 0], [1, 1, 1], local_cost=lambda x, y: abs(x - y)) == 3.0
    # ... and lets DTW align arbitrary objects (here: strings by length)
    assert (
        Q.dtw(
            ["a", "bb"],
            ["a", "bb", "bb"],
            local_cost=lambda x, y: abs(len(x) - len(y)),
            window=None,
        )
        == 0
    )


def test_dtw_rejects_empty_and_unknown_backend():
    with pytest.raises(ValueError, match="empty"):
        Q.dtw([], [1, 2])
    with pytest.raises(ValueError, match="backend"):
        Q.dtw([1], [1], backend="fastdtw")


def test_subsequence_dtw_finds_the_span_and_is_directional():
    series = [9, 9, 1, 2, 3, 9, 9]
    al = Q.subsequence_dtw_alignment([1, 2, 3], series)
    assert al.score == 0.0
    assert [j for _, j in al.path] == [2, 3, 4]
    # the query is consumed whole: every query index appears, in order
    assert sorted({i for i, _ in al.path}) == [0, 1, 2]
    assert Q.subsequence_dtw([1, 2, 3], series) < Q.dtw([1, 2, 3], series, window=None)
    assert Q.subsequence_dtw.meta.is_symmetric is False
    with pytest.raises(ValueError, match="non-negative"):
        Q.subsequence_dtw([0, 0], [0, 0, 0], local_cost=lambda x, y: -1.0)


def test_soft_dtw_limits():
    a, b = _SERIES[0]
    hard_squared = Q.dtw(a, b, window=None) ** 2
    assert Q.soft_dtw(a, b, gamma=1e-4) == pytest.approx(hard_squared, abs=1e-3)
    # the soft-min is below the hard min, so soft-DTW lower-bounds squared DTW
    assert Q.soft_dtw(a, b, gamma=1.0) <= hard_squared
    assert Q.soft_dtw(a, a) < 0  # not a distance
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            Q.soft_dtw(a, b, gamma=bad)


# --- Needleman-Wunsch / Smith-Waterman ------------------------------------------

_PAIRS = [
    ("GATTACA", "GCATGCU"),
    ("kitten", "sitting"),
    ("", "abc"),
    ("abc", ""),
    ("same", "same"),
]


@pytest.mark.parametrize("a,b", _PAIRS)
def test_needleman_wunsch_matches_reference(a, b):
    al = Q.needleman_wunsch_alignment(a, b)
    assert al.score == _ref_nw(a, b)
    # the global path covers every element of both sides exactly once, in order
    assert [i for i, _ in al.path if i is not None] == list(range(len(a)))
    assert [j for _, j in al.path if j is not None] == list(range(len(b)))
    # and re-scoring the path gives the score
    rescored = sum(
        -1 if i is None or j is None else (1 if a[i] == b[j] else -1)
        for i, j in al.path
    )
    assert rescored == al.score


@pytest.mark.parametrize("gap", [-0.7, -0.4, -0.1, -1 / 3])
@pytest.mark.parametrize(
    "a,b",
    [("ACGT" * 5, "A"), ("bbaabaa", ""), ("A" * 30, "AB"), ("GATTACA", "GCATGCU")],
)
def test_needleman_wunsch_fractional_scores(a, b, gap):
    al = Q.needleman_wunsch_alignment(a, b, gap=gap, mismatch=-0.3)
    assert al.score == pytest.approx(_ref_nw(a, b, mismatch=-0.3, gap=gap))
    assert [i for i, _ in al.path if i is not None] == list(range(len(a)))
    assert [j for _, j in al.path if j is not None] == list(range(len(b)))
    rescored = sum(
        gap if i is None or j is None else (1 if a[i] == b[j] else -0.3)
        for i, j in al.path
    )
    assert rescored == pytest.approx(al.score)
    local = Q.smith_waterman_alignment(a, b, gap=gap, mismatch=-0.3)
    assert all(i is None or 0 <= i < len(a) for i, _ in local.path)
    assert all(j is None or 0 <= j < len(b) for _, j in local.path)


def test_empty_alignment_score_is_plain_zero():
    assert str(Q.needleman_wunsch("", "")) == "0.0"


def test_smith_waterman_is_local_and_nonnegative():
    al = Q.smith_waterman_alignment("TTTTACGTTTT", "GGACGGG")
    assert al.score == 3.0
    assert al.path == ((4, 2), (5, 3), (6, 4))
    assert Q.smith_waterman("abc", "xyz") == 0.0
    assert Q.smith_waterman("abc", "xyz") >= Q.needleman_wunsch("abc", "xyz")


def test_symbolic_alignment_on_token_lists_with_custom_substitution():
    a = ["the", "quick", "brown", "fox"]
    b = ["the", "quik", "brown", "fox"]
    sub = lambda x, y: 2.0 if x == y else (1.0 if x[:3] == y[:3] else -2.0)  # noqa: E731
    assert Q.needleman_wunsch(a, b, substitution=sub) == 7.0
    assert Q.needleman_wunsch(a, b, gap=-5) == 2.0


def test_alignment_is_float_like():
    al = Q.dtw_alignment([0, 1], [0, 2])
    assert isinstance(al, Alignment) and float(al) == al.score
    assert Q.needleman_wunsch_alignment("a", "a").sense == "maximize"
    assert al.sense == "minimize"


# --- registry + facade ----------------------------------------------------------


@pytest.mark.parametrize(
    "name,sense",
    [
        ("dtw", "minimize"),
        ("subsequence_dtw", "minimize"),
        ("soft_dtw", "minimize"),
        ("needleman_wunsch", "maximize"),
        ("smith_waterman", "maximize"),
    ],
)
def test_registered_with_polarity(name, sense):
    assert name in comparators
    assert polarity_sense(resolve_comparator(name)) == sense
    # configuring a comparator keeps its declared polarity
    configured = resolve_comparator(
        name, **({"gap": -2} if sense == "maximize" else {})
    )
    assert polarity_sense(configured) == sense


def test_configured_comparator_applies_its_config():
    banded = resolve_comparator("dtw", window=None)
    assert banded([0, 1, 2, 3], [0, 0, 0, 1, 2, 3]) == 0.0
    assert banded.align([0, 1], [0, 0, 1]).path == ((0, 0), (0, 1), (1, 2))


def test_a_users_partial_keeps_the_polarity():
    from functools import partial

    assert polarity_sense(partial(Q.dtw, window=None)) == "minimize"
    A, B = [[0, 1, 2], [9, 9, 9]], [[9, 9, 8], [0, 1, 1, 2]]
    assert dict(match(A, B, compare=partial(Q.dtw, window=None))) == {0: 1, 1: 0}


def test_match_minimizes_a_distance_comparator_by_default():
    A = [[0, 1, 2, 3, 2], [5, 5, 6, 7, 7], [0, -1, -2, -1, 0]]
    B = [[5, 6, 6, 7, 7], [0, -1, -1, -2, 0], [0, 1, 1, 2, 3, 2]]
    m = match(A, B, compare="dtw")
    assert dict(m) == {0: 2, 1: 0, 2: 1}
    # an explicit sense still wins (here: deliberately the WORST assignment)
    assert dict(match(A, B, compare="dtw", sense="maximize")) != dict(m)


def test_match_with_alignment_scores_maximizes():
    A = ["GATTACA", "TTTTCCCC"]
    B = ["TTTCCCC", "GATACA"]
    assert dict(match(A, B, compare="needleman_wunsch")) == {0: 1, 1: 0}


def test_match_distance_comparator_through_a_blocker():
    A = [[0, 1, 2], [9, 9, 9]]
    B = [[9, 9, 8], [0, 1, 1, 2]]
    m = match(A, B, compare="dtw", block=lambda A, B: [(0, 0), (0, 1), (1, 0)])
    assert dict(m) == {0: 1, 1: 0}


# --- optional backends (fakes: no real dtaidistance / tslearn / Biopython) --------


def _install(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def test_dtaidistance_backend_wiring(monkeypatch):
    calls = []
    pkg = types.ModuleType("dtaidistance")
    dtw_mod = types.ModuleType("dtaidistance.dtw")
    ndim_mod = types.ModuleType("dtaidistance.dtw_ndim")
    dtw_mod.distance = lambda s1, s2, **kw: calls.append(("1d", list(s1), kw)) or 1.5
    ndim_mod.distance = lambda s1, s2, **kw: calls.append(("nd", s1.shape, kw)) or 2.5
    pkg.dtw, pkg.dtw_ndim = dtw_mod, ndim_mod
    for name, mod in [
        ("dtaidistance", pkg),
        ("dtaidistance.dtw", dtw_mod),
        ("dtaidistance.dtw_ndim", ndim_mod),
    ]:
        _install(monkeypatch, name, mod)

    al = Q.dtw_alignment([1, 2, 3], [1, 2], window=1, backend="dtaidistance")
    assert al == Alignment(1.5, None, sense="minimize")
    # dtaidistance's window is exclusive: our inclusive half-width 1 is its window=2
    assert calls[-1] == ("1d", [1.0, 2.0, 3.0], {"window": 2})
    assert Q.dtw([[0, 0]] * 3, [[1, 1]] * 3, window=None, backend="dtaidistance") == 2.5
    assert calls[-1] == ("nd", (3, 2), {})
    with pytest.raises(ValueError, match="local cost"):
        Q.dtw([1], [1], local_cost=lambda x, y: 0, backend="dtaidistance")


def test_tslearn_backend_wiring(monkeypatch):
    pkg = types.ModuleType("tslearn")
    metrics = types.ModuleType("tslearn.metrics")
    metrics.soft_dtw = lambda X, Y, gamma: X.shape[0] * 10 + gamma
    pkg.metrics = metrics
    _install(monkeypatch, "tslearn", pkg)
    _install(monkeypatch, "tslearn.metrics", metrics)
    assert Q.soft_dtw([1, 2, 3], [1], gamma=0.5, backend="tslearn") == 30.5


def test_biopython_backend_wiring(monkeypatch):
    seen = {}

    class FakeAligner:
        def score(self, a, b):
            seen.update(vars(self), a=a, b=b)
            return 4.0

    bio = types.ModuleType("Bio")
    align = types.ModuleType("Bio.Align")
    align.PairwiseAligner = FakeAligner
    bio.Align = align
    _install(monkeypatch, "Bio", bio)
    _install(monkeypatch, "Bio.Align", align)

    assert Q.smith_waterman("AC", "AC", gap=-2, backend="biopython") == 4.0
    assert seen == {
        "mode": "local",
        "match_score": 1.0,
        "mismatch_score": -1.0,
        "gap_score": -2,
        "a": "AC",
        "b": "AC",
    }
    al = Q.needleman_wunsch_alignment("AC", "AC", backend="biopython")
    assert seen["mode"] == "global" and al.path is None and al.sense == "maximize"
    with pytest.raises(ValueError, match="substitution"):
        Q.needleman_wunsch("a", "a", substitution=lambda x, y: 0, backend="biopython")


@pytest.mark.parametrize(
    "call,extra",
    [
        (lambda: Q.dtw([1], [1], backend="dtaidistance"), "timeseries"),
        (lambda: Q.soft_dtw([1], [1], backend="tslearn"), "timeseries"),
        (lambda: Q.smith_waterman("a", "a", backend="biopython"), "bio"),
    ],
)
def test_missing_backend_names_the_extra(monkeypatch, call, extra):
    for name in ["dtaidistance", "tslearn", "Bio"]:
        monkeypatch.setitem(sys.modules, name, None)  # import -> ModuleNotFoundError
    with pytest.raises(MissingDependencyError, match=rf"equate\[{extra}\]"):
        call()


# --- soft-DTW plan and the dtaidistance subsequence backend (equate#18) --------------

import numpy as _np  # noqa: E402


def test_soft_dtw_alignment_is_the_gradient_and_tends_to_the_dtw_path():
    from equate.compare.sequence import (
        _soft_dtw_backward,
        _soft_dtw_forward,
        dtw_alignment,
        soft_dtw_alignment,
    )

    rng = _np.random.default_rng(0)
    C, g = rng.random((4, 5)), 0.5
    R = _soft_dtw_forward(C, g)
    E = _soft_dtw_backward(C, R, g)
    h, numeric = 1e-6, _np.zeros_like(C)
    for i in range(4):
        for j in range(5):
            Cp = C.copy()
            Cp[i, j] += h
            numeric[i, j] = (_soft_dtw_forward(Cp, g)[4, 5] - R[4, 5]) / h
    assert _np.abs(numeric - E).max() < 1e-4
    a, b = rng.normal(size=6), rng.normal(size=8)
    hard = _np.zeros((6, 8))
    for i, j in dtw_alignment(a, b, window=None).path:
        hard[i, j] = 1
    assert _np.allclose(soft_dtw_alignment(a, b, gamma=1e-4).round(3), hard)
    soft = soft_dtw_alignment(a, b, gamma=1.0)
    assert soft[0, 0] == pytest.approx(1) and soft[-1, -1] == pytest.approx(1)
    assert ((soft > 0.01) & (hard == 0)).any()  # mass spreads off the hard path
    with pytest.raises(ValueError):
        soft_dtw_alignment(a, b, gamma=0)


def test_subsequence_dtw_dtaidistance_backend_agrees():
    pytest.importorskip("dtaidistance")
    from equate.compare.sequence import subsequence_dtw, subsequence_dtw_alignment

    rng = _np.random.default_rng(3)
    for _ in range(50):
        q = rng.normal(size=rng.integers(1, 6))
        s = rng.normal(size=rng.integers(len(q), 20))
        ours = subsequence_dtw_alignment(q, s)
        theirs = subsequence_dtw_alignment(q, s, backend="dtaidistance")
        assert theirs.score == pytest.approx(ours.score)
        assert theirs.path[-1] == ours.path[-1]
    assert subsequence_dtw([5, 6, 7], [0, 0, 5, 6, 7, 0], backend="dtaidistance") == 0.0
    with pytest.raises(ValueError, match="local cost"):
        subsequence_dtw([1], [1, 2], backend="dtaidistance", local_cost=lambda x, y: 0)
