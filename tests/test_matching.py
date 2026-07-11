"""Tests for the match stage (matchers registry, optimal/greedy/stable, sparse-LAP
routing, soft/OT) and the end-to-end match() facade."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from equate.matching import (
    matchers,
    resolve_matcher,
    optimal_matching,
    greedy_matching,
    stable_matching,
    soft_match,
    harden,
)
from equate import match
from equate.base import Matching


DIAG = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.2, 0.7]])
IDENTITY = {0: 0, 1: 1, 2: 2}


# --- registry & resolve -----------------------------------------------------------

def test_registry_has_matchers():
    for name in ("optimal", "hungarian", "greedy", "stable", "max_weight", "kuhn_munkres"):
        assert name in matchers


def test_resolve_matcher_name_callable_and_how_alias():
    assert dict(resolve_matcher("optimal")(DIAG)) == IDENTITY
    assert dict(resolve_matcher("assign")(DIAG)) == IDENTITY  # how alias -> optimal
    assert resolve_matcher(lambda s, *, sense="maximize": [(0, 0)])(DIAG) == [(0, 0)]


def test_matcher_rejects_config():
    with pytest.raises(TypeError):
        matchers.create("optimal", foo=1)


# --- assignment matchers ----------------------------------------------------------

def test_optimal_greedy_stable_agree_on_diagonal():
    assert dict(optimal_matching(DIAG)) == IDENTITY
    assert dict(greedy_matching(DIAG)) == IDENTITY
    assert dict(stable_matching(DIAG)) == IDENTITY


def test_optimal_minimize_sense():
    D = np.array([[0.1, 0.9], [0.9, 0.2]])
    assert dict(optimal_matching(D, sense="minimize")) == {0: 0, 1: 1}


def test_greedy_sparse_only_considers_stored_cells():
    # only the anti-diagonal is a candidate (stored); greedy must pick those
    S = csr_matrix(([0.9, 0.8], ([0, 1], [1, 0])), shape=(2, 2))
    assert dict(greedy_matching(S)) == {0: 1, 1: 0}


def test_optimal_sparse_lap_routing():
    # a blocked (sparse) score matrix with a full matching -> sparse solver
    S = csr_matrix(([0.9, 0.8, 0.7], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    assert dict(optimal_matching(S)) == IDENTITY


# --- soft / optimal transport -----------------------------------------------------

def test_soft_match_and_harden_if_ot_available():
    pytest.importorskip("ot")
    plan = soft_match(DIAG, reg=0.05)
    assert plan.shape == (3, 3)
    assert np.all(plan >= 0)
    assert dict(harden(plan)) == IDENTITY  # the plan's mass concentrates on the diagonal


def test_soft_match_requires_extra_when_absent():
    import importlib.util

    if importlib.util.find_spec("ot") is not None:
        pytest.skip("ot is installed; cannot test the missing-extra path")
    from equate import MissingDependencyError

    with pytest.raises(MissingDependencyError):
        soft_match(DIAG)


# --- the match() facade -----------------------------------------------------------

KEYS = ["apple pie", "banana split", "cherry tart"]
VALS = ["cherry tarte", "aple pie", "banana split"]
EXPECTED = {"apple pie": "aple pie", "banana split": "banana split", "cherry tart": "cherry tarte"}


def test_match_returns_matching_with_scores_and_labels():
    m = match(KEYS, VALS, compare="ratio")
    assert isinstance(m, Matching)
    assert dict(m.labeled_pairs()) == EXPECTED
    assert len(m.scores) == 3 and all(0.0 <= s <= 1.0 for s in m.scores)
    assert list(m) == [(0, 1), (1, 2), (2, 0)]  # (i, j) index pairs


def test_match_default_featurized_route():
    assert dict(match(KEYS, VALS).labeled_pairs()) == EXPECTED


def test_match_how_greedy_and_stable():
    assert dict(match(KEYS, VALS, compare="ratio", how="greedy").labeled_pairs()) == EXPECTED
    assert dict(match(KEYS, VALS, compare="ratio", how="stable").labeled_pairs()) == EXPECTED


def test_match_blocked_produces_sparse_and_still_matches():
    from equate.block import key_blocking, qgram_keys

    m = match(KEYS, VALS, compare="ratio", block=key_blocking(qgram_keys(2)))
    assert dict(m.labeled_pairs()) == EXPECTED


def test_match_how_soft_if_ot_available():
    pytest.importorskip("ot")
    m = match(KEYS, VALS, compare="ratio", how="soft")
    assert m.plan is not None and m.plan.shape == (3, 3)
    assert dict(m.labeled_pairs()) == EXPECTED
    # .scores are the similarities (from the score matrix), not transport-plan mass
    assert all(0.0 <= s <= 1.0 for s in m.scores) and max(m.scores) > 0.9


# --- review-driven fixes ----------------------------------------------------------

def test_optimal_sparse_keeps_explicit_zero_candidate_minimize():
    # (0,0)=0 is a perfect distance-0 candidate; scipy's sparse solver would drop it
    S = csr_matrix(([0.0, 10.0, 10.0, 1.0], ([0, 0, 1, 1], [0, 1, 0, 1])), shape=(2, 2))
    assert dict(optimal_matching(S, sense="minimize")) == {0: 0, 1: 1}


def test_optimal_sparse_partial_matching_drops_hole_assignments():
    # column 2 is absent everywhere; row 2's only candidates are taken by rows 0,1,
    # so row 2 has no free real candidate and is left unmatched (partial matching)
    S = csr_matrix(([-1.0, -1.0, -2.0, -2.0], ([0, 1, 2, 2], [0, 1, 0, 1])), shape=(3, 3))
    assert dict(optimal_matching(S, sense="maximize")) == {0: 0, 1: 1}


def test_optimal_all_absent_sparse_returns_empty():
    assert optimal_matching(csr_matrix((2, 2))) == []


def test_stable_sparse_prefers_real_candidate_over_hole():
    S = csr_matrix(([-5.0, -1.0, -2.0], ([0, 0, 1], [0, 1, 0])), shape=(2, 2))
    assert dict(stable_matching(S, sense="maximize")).get(1) == 0  # not the (1,1) hole


def test_soft_match_rejects_nonpositive_reg():
    with pytest.raises(ValueError):
        soft_match(DIAG, reg=0)


def test_match_requires_two_collections():
    with pytest.raises(ValueError):
        match(["a", "b"], None)


def test_match_featurize_and_compare_mutually_exclusive():
    with pytest.raises(ValueError):
        match(["a"], ["b"], featurize="tfidf", compare="ratio")


# --- registry-wide hole-worst-casing conformance (the backstop guarding D11) -------
# Every registered matcher must honour blocking: a structurally-absent (non-candidate)
# cell must never be selected. This is the ONE invariant that makes the recurring
# "densify-before-to_cost" bug class un-shippable for present AND future matchers — a
# matcher that .toarray()s the raw scores (filling holes with 0) fails these sweeps.

from equate.base import ScoreMatrix
from equate import MissingDependencyError


def _run_matcher_or_skip(name, sm):
    """Resolve+run a registered matcher, skipping if its optional extra is absent."""
    try:
        return list(resolve_matcher(name)(sm))
    except MissingDependencyError:
        pytest.skip(f"matcher {name!r} needs an optional extra not installed here")


def _assert_valid_partial_matching(name, pairs, sm):
    """No pair lands on a hole (D11) AND the matching is injective (no i or j reused)."""
    mask = sm.candidate_mask()
    rows, cols = set(), set()
    for i, j in pairs:
        assert mask[i, j], f"{name} matched an absent hole ({i}, {j})"
        assert i not in rows, f"{name} matched row {i} twice"
        assert j not in cols, f"{name} matched col {j} twice"
        rows.add(i)
        cols.add(j)


# ASYMMETRIC candidate structure on purpose: a cyclic permutation is NOT transpose-symmetric,
# so a matcher that returns transposed (col, row) pairs — an orientation bug — is caught here
# (a symmetric anti-diagonal fixture would silently pass it). Values are chosen so a hole,
# if densified to 0, would win: negative for maximize, positive for minimize.
_CYCLE = ([0, 1, 2], [1, 2, 0])  # candidates (0,1), (1,2), (2,0); diagonal + rest absent
_RECT = ([0, 1], [1, 2])  # a 2x3 blocked matrix: candidates (0,1), (1,2)
# FORCED-PARTIAL: all three rows' only candidate is column 0, so a full-cardinality solver
# MUST assign two rows onto holes — this is the fixture that actually exercises drop_holes.
# (_CYCLE/_RECT admit a complete real matching, so their raw LAP never lands on a hole and
# a drop_holes regression would be invisible to them.)
_FORCED = ([0, 1, 2], [0, 0, 0])


@pytest.mark.parametrize("name", matchers.names())
@pytest.mark.parametrize(
    "sense,data,rowcol,shape",
    [
        ("maximize", [-0.9, -0.8, -0.7], _CYCLE, (3, 3)),
        ("minimize", [5.0, 6.0, 7.0], _CYCLE, (3, 3)),
        ("maximize", [-0.9, -0.8], _RECT, (2, 3)),  # rectangular blocked input
        ("maximize", [-0.9, -0.8, -0.7], _FORCED, (3, 3)),  # forces hole assignments
        ("minimize", [5.0, 6.0, 7.0], _FORCED, (3, 3)),
    ],
)
def test_matcher_valid_partial_matching_on_blocked_input(name, sense, data, rowcol, shape):
    sm = ScoreMatrix(csr_matrix((data, rowcol), shape=shape), sense=sense)
    _assert_valid_partial_matching(name, _run_matcher_or_skip(name, sm), sm)


@pytest.mark.parametrize("name", matchers.names())
def test_matcher_leaves_row_without_candidate_unmatched(name):
    # row 2 has NO candidate at all -> it must be left unmatched, never assigned a hole
    sm = ScoreMatrix(csr_matrix(([0.9, 0.8], ([0, 1], [0, 1])), shape=(3, 3)), sense="maximize")
    pairs = _run_matcher_or_skip(name, sm)
    _assert_valid_partial_matching(name, pairs, sm)
    assert dict(pairs) == {0: 0, 1: 1}


def test_legacy_matcher_output_is_hole_filtered_at_the_boundary():
    # D11 must not depend on a third-party matcher's discipline: resolve_matcher drops the
    # holes a legacy raw-array matcher assigns (here a per-row argmax over a row of holes).
    def per_row_argmax(scores, **kw):
        arr = np.asarray(scores, dtype=float)
        return [(i, int(arr[i].argmax())) for i in range(arr.shape[0])]

    sm = ScoreMatrix(csr_matrix(([0.9, 0.8], ([0, 1], [0, 1])), shape=(3, 3)), sense="maximize")
    pairs = list(resolve_matcher(per_row_argmax)(sm))
    assert pairs == [(0, 0), (1, 1)]  # the (2, x) hole row 2 argmaxed into is dropped

    # and end-to-end through the facade: no fabricated 0.0 score for a non-candidate
    m = match(["aa", "bb", "zz"], ["aa", "bb", "qq"], compare="ratio",
              block=lambda A, B: [(0, 0), (1, 1)], how=per_row_argmax)
    assert list(m) == [(0, 0), (1, 1)] and 0.0 not in m.scores


@pytest.mark.parametrize("sense", ["maximize", "minimize"])
def test_soft_never_selects_a_hole(sense):
    pytest.importorskip("ot")
    # a shared-column block forces the transport plan to put mass on a hole (the case the
    # anti-diagonal fixture missed): both rows' only candidate is column 0.
    data = [-0.9, -0.8] if sense == "maximize" else [0.1, 0.2]
    S = csr_matrix((data, ([0, 1], [0, 0])), shape=(2, 2))
    sm = ScoreMatrix(S, sense=sense)
    pairs = sm.drop_holes(harden(soft_match(sm)))
    _assert_valid_partial_matching("soft", pairs, sm)
    assert pairs == [(0, 0)] or pairs == [(1, 0)]  # only column 0 is a real candidate


# --- ScoreMatrix as the native matcher input (the deep contract) ------------------

def test_matchers_accept_scorematrix_natively():
    S = csr_matrix(([0.9, 0.8, 0.7], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    sm = ScoreMatrix(S, sense="maximize")
    assert dict(optimal_matching(sm)) == IDENTITY
    assert dict(greedy_matching(sm)) == IDENTITY
    assert dict(stable_matching(sm)) == IDENTITY


def test_resolve_matcher_passes_scorematrix_to_native_and_dense_to_legacy():
    # a NATIVE matcher (marked) receives the ScoreMatrix; a LEGACY raw-array callable
    # (declares `sense`) receives a pre-worst-cased dense array, never the sparse object.
    S = csr_matrix(([-0.9, -0.8], ([0, 1], [1, 0])), shape=(2, 2))
    sm = ScoreMatrix(S, sense="maximize")
    seen = {}

    def legacy(scores, *, sense="maximize"):
        seen["type"] = type(scores).__name__
        seen["sense"] = sense
        return []

    resolve_matcher(legacy)(sm)
    assert seen["type"] == "ndarray"  # got a dense view, not the csr_matrix
    assert seen["sense"] == "maximize"


def test_kuhn_munkres_and_max_weight_blocked_negative_scores():
    pytest.importorskip("networkx")
    # the exact regression: blocked matrix, negative real scores -> must pick the real
    # anti-diagonal candidates, not the absent diagonal holes.
    S = csr_matrix(([-0.9, -0.8], ([0, 1], [1, 0])), shape=(2, 2))
    assert dict(resolve_matcher("kuhn_munkres")(S)) == {0: 1, 1: 0}
    assert dict(resolve_matcher("max_weight")(S)) == {0: 1, 1: 0}


def test_max_weight_asymmetric_and_rectangular_blocked():
    # review regressions: max_weight used to return transposed pairs from networkx's
    # unordered edges -> silent empty (asymmetric square) or IndexError (rectangular).
    pytest.importorskip("networkx")
    cyc = csr_matrix(([0.9, 0.8, 0.7], ([0, 1, 2], [1, 2, 0])), shape=(3, 3))
    assert dict(resolve_matcher("max_weight")(cyc)) == {0: 1, 1: 2, 2: 0}
    rect = csr_matrix(([0.9, 0.8], ([0, 1], [1, 2])), shape=(2, 3))
    assert dict(resolve_matcher("max_weight")(rect)) == {0: 1, 1: 2}


def test_maximal_matching_orients_edges():
    # maximal_matching must return (row, col), never a transposed (col, row) pair
    pytest.importorskip("networkx")
    from equate.util import maximal_matching

    # cyclic candidates on a dense matrix; the only max-weight matching is the cycle
    S = np.array([[0.0, 0.9, 0.0], [0.0, 0.0, 0.8], [0.7, 0.0, 0.0]])
    assert dict(maximal_matching(S)) == {0: 1, 1: 2, 2: 0}


def test_resolve_matcher_legacy_contract_variants_get_raw_array():
    # every legal shape of the legacy raw-array contract must still receive a raw array,
    # not a ScoreMatrix (review: signature-sniffing for a literal `sense` misclassified these)
    seen = {}

    def kwargs_matcher(scores, **kw):
        seen["kwargs"] = type(scores).__name__
        return []

    def named_dir_matcher(scores, *, direction="maximize"):
        seen["named"] = type(scores).__name__
        return []

    def no_dir_matcher(scores):
        seen["nodir"] = type(scores).__name__
        return []

    S = csr_matrix(([-0.9, -0.8], ([0, 1], [1, 0])), shape=(2, 2))
    for key, m in [("kwargs", kwargs_matcher), ("named", named_dir_matcher), ("nodir", no_dir_matcher)]:
        resolve_matcher(m)(S)
        assert seen[key] == "ndarray", f"{key} matcher got {seen[key]}, not a raw array"


def test_match_facade_accepts_legacy_kwargs_matcher():
    # end-to-end back-compat: a custom raw-array matcher passed as how= still works
    import numpy as np

    def per_row_argmax(scores, **kw):
        arr = np.asarray(scores, dtype=float)
        return [(i, int(arr[i].argmax())) for i in range(arr.shape[0])]

    m = match(KEYS, VALS, compare="ratio", how=per_row_argmax)
    assert dict(m.labeled_pairs()) == EXPECTED


# --- randomized conformance: the property battery + the optimality differential ---------
# The fixtures above all use [0, 1] scores, and THAT is why they missed the big-M defect:
# a hole was worst-cased to `max_real_cost + 1`, which only beats one real CELL, while a LAP
# compares TOTALS — so the solver bought a hole to save >1 elsewhere and drop_holes then
# deleted the pair, returning a matching strictly dominated by an available all-real one.
# Unbounded scores (dot products, BM25, counts — equate ships `dot` with bounded=False) are
# the regime that exposes it, so these sweeps generate scores with a floor above 1.0.

_OPTIMAL_FAMILY = ("optimal", "hungarian", "max_weight", "kuhn_munkres")


def _random_blocked(rng, *, shape, density, low=1.5, high=100.0):
    """A blocked (sparse) score matrix with unbounded scores; may leave rows/cols empty."""
    n, m = shape
    mask = rng.random((n, m)) < density
    rows, cols = np.nonzero(mask)
    data = rng.uniform(low, high, size=len(rows))
    return csr_matrix((data, (rows, cols)), shape=shape)


def _best_real_matching(sm):
    """Brute-force ``(cardinality, total_score)`` of the best ALL-REAL matching.

    The semantics blocking implies, and that the big-M makes the solvers obey: use as many
    real candidate pairs as possible, then optimize the score among those. Exponential — for
    tiny fixtures only.
    """
    from itertools import combinations, permutations

    n, m = sm.shape
    stored = {(i, j): v for i, j, v in sm.stored_entries()}
    pick_best = max if sm.sense == "maximize" else min
    for k in range(min(n, m), 0, -1):
        totals = [
            sum(stored[p] for p in pairs)
            for rows in combinations(range(n), k)
            for cols in permutations(range(m), k)
            for pairs in [list(zip(rows, cols))]
            if all(p in stored for p in pairs)
        ]
        if totals:
            return k, pick_best(totals)
    return 0, 0.0


@pytest.mark.parametrize("sense", ["maximize", "minimize"])
def test_every_matcher_returns_a_valid_partial_matching_under_fuzz(sense):
    """Registry-wide: injective, in-bounds, hole-free, cardinality-sane — over random input."""
    rng = np.random.default_rng(0xD11)
    for name in matchers.names():
        for trial in range(40):
            shape = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
            sm = ScoreMatrix(
                _random_blocked(rng, shape=shape, density=float(rng.uniform(0.2, 1.0))),
                sense=sense,
            )
            pairs = _run_matcher_or_skip(name, sm)
            # in-bounds is what catches a transposed pair on a rectangular input
            for i, j in pairs:
                assert 0 <= i < shape[0] and 0 <= j < shape[1], (
                    f"{name} returned out-of-bounds pair ({i}, {j}) for shape {shape} "
                    f"[trial {trial}]"
                )
            _assert_valid_partial_matching(name, pairs, sm)
            assert len(pairs) <= min(shape), f"{name} over-matched (trial {trial})"


@pytest.mark.parametrize("sense", ["maximize", "minimize"])
def test_optimal_family_is_never_dominated_by_an_available_all_real_matching(sense):
    """An 'optimal' matcher must not lose to a matching that was there for the taking.

    With the too-weak `max_real_cost + 1` hole penalty this failed on ~3% of random unbounded
    inputs: `optimal` (the DEFAULT matcher) returned fewer pairs AND a worse total than a
    feasible all-real matching — it was beaten by the `greedy` heuristic.
    """
    rng = np.random.default_rng(0xD11)
    better = max if sense == "maximize" else min
    for name in _OPTIMAL_FAMILY:
        for trial in range(60):
            shape = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
            sm = ScoreMatrix(
                _random_blocked(rng, shape=shape, density=float(rng.uniform(0.3, 1.0))),
                sense=sense,
            )
            pairs = _run_matcher_or_skip(name, sm)
            got_total = sum(sm.score_at(i, j) for i, j in pairs)
            best_k, best_total = _best_real_matching(sm)
            assert len(pairs) == best_k, (
                f"{name} matched {len(pairs)} pairs but {best_k} real pairs were available "
                f"(trial {trial}, sense={sense}, shape={shape})\n{sm.data.toarray()}"
            )
            assert better(got_total, best_total) == pytest.approx(got_total), (
                f"{name} scored {got_total} but {best_total} was available "
                f"(trial {trial}, sense={sense}, shape={shape})\n{sm.data.toarray()}"
            )
