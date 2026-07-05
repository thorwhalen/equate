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
