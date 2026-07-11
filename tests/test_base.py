"""Tests for equate.base: the to_cost SSOT, the score/sense contract, and the
structured result types — plus that the cost-based matchers now agree."""

import numpy as np
import pytest

from equate.base import to_cost, ScoreMatrix, Matching, Partition, Candidate, Explanation
from equate.util import (
    hungarian_matching,
    stable_marriage_matching,
    kuhn_munkres_matching,
)


# --- to_cost SSOT -----------------------------------------------------------------

def test_to_cost_maximize_is_nonnegative_complement():
    S = np.array([[0.9, 0.1], [0.2, 0.8]])
    C = to_cost(S, sense="maximize")
    assert (C >= 0).all()
    # complement to the global max
    np.testing.assert_allclose(C, S.max() - S)
    # the best similarity becomes the lowest cost (argmax of S == argmin of C per row)
    assert C.argmin(axis=1).tolist() == S.argmax(axis=1).tolist()


def test_to_cost_minimize_is_passthrough_same_object():
    D = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert to_cost(D, sense="minimize") is D


def test_to_cost_rejects_bad_sense():
    with pytest.raises(ValueError):
        to_cost(np.eye(2), sense="whatever")


def test_to_cost_handles_sparse_maximize():
    sparse = pytest.importorskip("scipy.sparse")
    S = sparse.csr_matrix(np.array([[0.9, 0.1], [0.2, 0.8]]))
    C = to_cost(S, sense="maximize")
    assert not sparse.issparse(C)  # densified for the solver
    np.testing.assert_allclose(C, np.array([[0.0, 0.8], [0.7, 0.1]]))


# --- matchers now agree (the fixed latent inconsistency) --------------------------

DIAG_DOMINANT = np.array(
    [[0.9, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.2, 0.7]]
)
IDENTITY = {0: 0, 1: 1, 2: 2}


def test_hungarian_optimal_identity():
    assert dict(hungarian_matching(DIAG_DOMINANT)) == IDENTITY


def test_stable_marriage_identity():
    assert dict(stable_marriage_matching(DIAG_DOMINANT)) == IDENTITY


def test_kuhn_munkres_identity():
    pytest.importorskip("networkx")  # optional equate[graph] extra
    assert dict(kuhn_munkres_matching(DIAG_DOMINANT)) == IDENTITY


def test_hungarian_and_kuhn_munkres_agree():
    pytest.importorskip("networkx")
    assert dict(hungarian_matching(DIAG_DOMINANT)) == dict(
        kuhn_munkres_matching(DIAG_DOMINANT)
    )


def test_hungarian_minimize_sense_matches_maximize_on_negated_scores():
    # A distance matrix whose smallest entries lie on the diagonal.
    D = np.array([[0.1, 0.9, 1.0], [0.9, 0.2, 0.8], [1.0, 0.8, 0.3]])
    assert dict(hungarian_matching(D, sense="minimize")) == IDENTITY


# --- ScoreMatrix ------------------------------------------------------------------

def test_score_matrix_to_cost_honors_sense():
    S = np.array([[0.9, 0.1], [0.2, 0.8]])
    sm = ScoreMatrix(S, sense="maximize", row_labels=["a", "b"], col_labels=["x", "y"])
    assert sm.shape == (2, 2)
    np.testing.assert_allclose(sm.to_cost(), to_cost(S))


# --- structured result types ------------------------------------------------------

def test_matching_iterates_as_pairs():
    m = Matching(pairs=[(0, 2), (1, 0)], scores=[0.9, 0.7])
    assert dict(m) == {0: 2, 1: 0}
    assert list(m) == [(0, 2), (1, 0)]
    assert len(m) == 2


def test_matching_labeled_pairs_uses_labels():
    m = Matching(
        pairs=[(0, 1), (1, 0)],
        row_labels=["apple", "banana"],
        col_labels=["nana", "app"],
    )
    assert list(m.labeled_pairs()) == [("apple", "app"), ("banana", "nana")]


def test_matching_labeled_pairs_falls_back_to_indices():
    m = Matching(pairs=[(0, 1)])
    assert list(m.labeled_pairs()) == [(0, 1)]


def test_partition_groups_and_iterates_as_within_cluster_pairs():
    # items 0,2 are one entity; 1,3 another; 4 alone
    p = Partition(labels=[0, 1, 0, 1, 2])
    assert p.groups() == {0: [0, 2], 1: [1, 3], 2: [4]}
    assert set(p) == {(0, 2), (1, 3)}


def test_candidate_and_explanation_are_structured():
    c = Candidate(left="apple", right="app", score=0.8, explanation=Explanation("edit-close"))
    assert c.left == "apple" and c.score == 0.8
    assert c.explanation.summary == "edit-close"


def test_candidate_iterates_as_pair():
    cands = [Candidate("a", "x", 0.9), Candidate("b", "y", 0.8)]
    assert dict(cands) == {"a": "x", "b": "y"}


# --- review-driven fixes: empty inputs, sparse edge cases -------------------------

@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0)])
def test_to_cost_empty_does_not_crash(shape):
    C = to_cost(np.empty(shape))
    assert C.shape == shape


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (2, 0)])
def test_matchers_handle_empty_input(shape):
    assert dict(hungarian_matching(np.empty(shape))) == {}
    assert stable_marriage_matching(np.empty(shape)) == []


def test_to_cost_sparse_maximize_avoids_absent_cells_with_negative_scores():
    # only the anti-diagonal holds real (negative) candidates; the diagonal is absent
    csr = pytest.importorskip("scipy.sparse").csr_matrix
    S = csr(([-0.3, -0.4], ([0, 1], [1, 0])), shape=(2, 2))
    # must pick the two real (anti-diagonal) candidates, not the never-scored diagonal
    assert dict(hungarian_matching(S)) == {0: 1, 1: 0}


def test_to_cost_sparse_minimize_densifies_and_avoids_absent():
    sparse = pytest.importorskip("scipy.sparse")
    # a full sparse cost matrix (no absent cells): minimize should not crash
    D = sparse.csr_matrix(np.array([[0.1, 0.9], [0.8, 0.2]]))
    assert dict(hungarian_matching(D, sense="minimize")) == {0: 0, 1: 1}
    # absent cells in a minimize matrix must be worst-case, not cheapest-at-0
    absent_ok = sparse.csr_matrix(([0.1, 0.2], ([0, 1], [1, 0])), shape=(2, 2))
    assert dict(hungarian_matching(absent_ok, sense="minimize")) == {0: 1, 1: 0}


def test_comparator_meta_defaults_and_directionality():
    from equate.base import ComparatorMeta

    default = ComparatorMeta()
    assert default.polarity == "similarity"
    assert default.is_symmetric is True and default.is_metric is False
    directional = ComparatorMeta(polarity="distance", is_symmetric=False, bounded=True)
    assert directional.is_symmetric is False and directional.bounded is True


# --- ScoreMatrix sanctioned densify views (the deep hole-worst-casing contract) ----

def test_scorematrix_coerce_passthrough_wrap_and_conflict():
    sm = ScoreMatrix(np.eye(2), sense="minimize")
    assert ScoreMatrix.coerce(sm) is sm  # a ScoreMatrix passes through unchanged
    assert ScoreMatrix.coerce(sm, sense="minimize") is sm  # matching sense is fine
    wrapped = ScoreMatrix.coerce(np.eye(2))  # a raw array is wrapped, default maximize
    assert isinstance(wrapped, ScoreMatrix) and wrapped.sense == "maximize"
    with pytest.raises(ValueError):  # a conflicting sense is an error, never silent
        ScoreMatrix.coerce(sm, sense="maximize")


def test_scorematrix_dense_views_dense_input():
    sm = ScoreMatrix(np.array([[0.9, 0.1], [0.2, 0.8]]), sense="maximize")
    assert sm.is_sparse is False
    np.testing.assert_allclose(sm.dense_cost(), sm.data.max() - sm.data)
    np.testing.assert_allclose(sm.dense_similarity(), -sm.dense_cost())
    assert sm.candidate_mask().all()  # a dense matrix has no holes
    assert sm.score_at(0, 0) == 0.9


def test_scorematrix_sparse_views_worst_case_holes():
    csr = pytest.importorskip("scipy.sparse").csr_matrix
    S = csr(([-0.9, -0.8], ([0, 1], [1, 0])), shape=(2, 2))
    sm = ScoreMatrix(S, sense="maximize")
    assert sm.is_sparse is True
    mask = sm.candidate_mask()
    assert mask.tolist() == [[False, True], [True, False]]  # only stored cells
    cost = sm.dense_cost()
    # every hole (absent cell) is strictly worse (higher cost) than every real cell
    assert cost[~mask].min() > cost[mask].max()
    # dense_similarity mirrors it: holes strictly lowest
    sim = sm.dense_similarity()
    assert sim[~mask].max() < sim[mask].min()
    assert sorted(sm.stored_entries()) == [(0, 1, -0.9), (1, 0, -0.8)]
    assert sm.score_at(0, 1) == -0.9  # retained raw score of a real candidate


def test_scorematrix_legacy_view_orientation():
    csr = pytest.importorskip("scipy.sparse").csr_matrix
    S = csr(([-0.9, -0.8], ([0, 1], [1, 0])), shape=(2, 2))
    # maximize -> similarity orientation; minimize -> cost orientation
    assert np.allclose(ScoreMatrix(S, sense="maximize").legacy_view(),
                       ScoreMatrix(S, sense="maximize").dense_similarity())
    assert np.allclose(ScoreMatrix(S, sense="minimize").legacy_view(),
                       ScoreMatrix(S, sense="minimize").dense_cost())


def test_scorematrix_matcher_marker():
    from equate.base import scorematrix_matcher

    @scorematrix_matcher
    def m(sm):
        return []

    assert m._scorematrix_native is True
