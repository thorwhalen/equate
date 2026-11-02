"""Characterization tests for the Rust-port surface of ``equate.compare``.

These tests pin the behaviour of a few functions (``subsequence_dtw_alignment``,
``_soft_dtw_forward``, ``levenshtein_distance``, and the edge behaviour of the DTW /
Needleman-Wunsch / Smith-Waterman kernels) *as they are today*, including odd corner
cases (NaN / inf propagation, ties). Their purpose is that a Rust port can be checked
against them line by line. A change in any assertion here is a behaviour change and
needs a changelog entry.
"""

import math

import numpy as np
import pytest

from equate.compare.sequence import (
    _soft_dtw_forward,
    dtw_alignment,
    needleman_wunsch_alignment,
    smith_waterman_alignment,
    soft_dtw,
    subsequence_dtw_alignment,
)
from equate.compare.string import levenshtein_distance

NAN = float("nan")
INF = float("inf")


# --- subsequence_dtw_alignment --------------------------------------------------------


def test_subsequence_dtw_numpy_fixed_example():
    al = subsequence_dtw_alignment([5, 6, 7], [0, 0, 5, 6, 7, 0])
    assert al.score == 0.0
    assert al.path == ((0, 2), (1, 3), (2, 4))


@pytest.mark.parametrize(
    "query, series",
    [
        ([5, 6, 7], [0, 0, 5, 6, 7, 0]),
        ([1, 2, 2], [3, 1, 2, 4, 2, 2, 9]),
    ],
)
def test_subsequence_dtw_dtaidistance_matches_numpy(query, series):
    pytest.importorskip("dtaidistance")
    ref = subsequence_dtw_alignment(query, series, backend="numpy")
    got = subsequence_dtw_alignment(query, series, backend="dtaidistance")
    assert got.path == ref.path
    assert got.score == pytest.approx(ref.score)


def test_subsequence_dtw_dtaidistance_pinned_path():
    pytest.importorskip("dtaidistance")
    al = subsequence_dtw_alignment(
        [1, 2, 2], [3, 1, 2, 4, 2, 2, 9], backend="dtaidistance"
    )
    assert al.score == pytest.approx(0.0)
    assert al.path == ((0, 1), (1, 2), (2, 2))


@pytest.mark.parametrize("query, series", [([], []), ([], [1, 2]), ([1, 2], [])])
def test_subsequence_dtw_empty_raises(query, series):
    with pytest.raises(ValueError, match="empty"):
        subsequence_dtw_alignment(query, series, backend="numpy")


# --- _soft_dtw_forward / soft_dtw -----------------------------------------------------


def test_soft_dtw_forward_all_predecessors_unreachable():
    # C[0, 0] = inf makes R[1, 1] = inf; then R[1, 2], R[2, 1], R[2, 2] have only
    # unreachable (inf) predecessors, so the `continue` branch leaves them at inf.
    C = np.array([[np.inf, 0.0], [0.0, 0.0]])
    R = _soft_dtw_forward(C, 1.0)
    assert R.tolist() == [
        [0.0, INF, INF],
        [INF, INF, INF],
        [INF, INF, INF],
    ]


def test_soft_dtw_forward_finite_costs():
    R = _soft_dtw_forward(np.zeros((1, 1)), 1.0)
    assert R.tolist() == [[0.0, INF], [INF, 0.0]]


@pytest.mark.parametrize("a, b", [([INF, 0], [0, 0]), ([0, 0], [INF, 0])])
def test_soft_dtw_with_inf_step_is_inf(a, b):
    assert soft_dtw(a, b) == INF


# --- levenshtein_distance -------------------------------------------------------------


def test_levenshtein_empty_inputs():
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("ab", "") == 2
    assert levenshtein_distance("", "") == 0


def test_levenshtein_token_lists():
    assert levenshtein_distance(["a", "b"], []) == 2
    assert levenshtein_distance([], ["a", "b", "c"]) == 3
    assert levenshtein_distance(["a", "b"], ["a", "c"]) == 1


def test_levenshtein_regular():
    assert levenshtein_distance("kitten", "sitting") == 3


# --- dtw_alignment edge behaviour -----------------------------------------------------


@pytest.mark.parametrize("window", [None, 0])
def test_dtw_nan_propagates(window):
    al = dtw_alignment([0, NAN, 1], [0, 1], window=window)
    assert math.isnan(al.score)
    assert al.path == ((0, 0), (1, 0), (2, 1))


@pytest.mark.parametrize("window", [None, 0])
def test_dtw_inf_propagates(window):
    al = dtw_alignment([0, INF, 1], [0, 1], window=window)
    assert al.score == INF
    assert al.path == ((0, 0), (1, 0), (2, 1))


def test_dtw_length_one_vs_five_window0_band_widened():
    al = dtw_alignment([1], [0, 1, 2, 3, 4], window=0)
    assert al.score == pytest.approx(math.sqrt(15))
    assert al.path == ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))


def test_dtw_length_two_vs_five_window0():
    al = dtw_alignment([1, 2], [1, 2, 3, 4, 5], window=0)
    assert al.score == pytest.approx(math.sqrt(14))
    assert al.path == ((0, 0), (1, 1), (1, 2), (1, 3), (1, 4))


# --- Needleman-Wunsch / Smith-Waterman edge behaviour ---------------------------------


def _nan_sub(x, y):
    return NAN


def test_nw_nan_substitution():
    al = needleman_wunsch_alignment("ab", "b", substitution=_nan_sub)
    assert math.isnan(al.score)
    assert al.path == ((0, None), (1, 0))


def test_nw_nan_gap():
    al = needleman_wunsch_alignment("ab", "b", gap=NAN)
    assert math.isnan(al.score)
    assert al.path == ((0, None), (1, 0))


def test_nw_inf_match():
    al = needleman_wunsch_alignment("ab", "b", match=INF)
    assert al.score == INF
    assert al.path == ((0, None), (1, 0))


def test_sw_nan_substitution():
    al = smith_waterman_alignment("ab", "b", substitution=_nan_sub)
    assert math.isnan(al.score)
    assert al.path == ((0, 0),)


def test_sw_nan_gap():
    al = smith_waterman_alignment("ab", "b", gap=NAN)
    assert math.isnan(al.score)
    assert al.path == ((0, None),)


def test_sw_inf_match():
    al = smith_waterman_alignment("ab", "b", match=INF)
    assert al.score == INF
    assert al.path == ((1, 0),)


def test_sw_all_non_positive_is_empty():
    al = smith_waterman_alignment("abc", "xyz")
    assert al.score == 0.0
    assert al.path == ()


def test_nw_ties_gap0_mismatch0():
    al = needleman_wunsch_alignment("ab", "ba", gap=0, mismatch=0)
    assert al.score == 1.0
    assert al.path == ((None, 0), (0, 1), (1, None))
