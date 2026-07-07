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
