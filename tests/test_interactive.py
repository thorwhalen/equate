"""Tests for the interactive stage: Murty k-best, CandidateStore, ConstraintSet +
reoptimize (edit-and-re-solve), and review/active-learning queues."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from equate.interactive import (
    CandidateStore,
    k_best_assignments,
    solve_constrained,
    ConstraintSet,
    reoptimize,
    review_queue,
    uncertainty_sampling,
)
from equate.base import to_cost


# --- Murty k-best -----------------------------------------------------------------

def test_k_best_two_by_two():
    result = k_best_assignments([[1, 2], [2, 1]], 2)
    assert [a for a, _ in result] == [{0: 0, 1: 1}, {0: 1, 1: 0}]
    assert [c for _, c in result] == [2.0, 4.0]  # increasing cost


def test_k_best_increasing_and_distinct():
    rng = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    result = k_best_assignments(rng, 6)  # 3! = 6 assignments
    costs = [c for _, c in result]
    assert costs == sorted(costs)  # non-decreasing
    keys = {tuple(sorted(a.items())) for a, _ in result}
    assert len(keys) == len(result)  # all distinct
    # the best matches scipy's LAP
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(np.array(rng))
    assert result[0][0] == dict(zip(r.tolist(), c.tolist()))


def test_k_best_more_than_available():
    assert len(k_best_assignments([[1, 2], [2, 1]], 99)) == 2


# --- constrained solve ------------------------------------------------------------

def test_solve_constrained_forces_and_forbids():
    cost = np.array([[1.0, 2.0], [2.0, 1.0]])
    forced = solve_constrained(cost, forced=[(0, 1)])
    assert forced[0] == {0: 1, 1: 0}  # forcing 0->1 forces 1->0
    forbidden = solve_constrained(cost, forbidden=[(0, 0)])
    assert forbidden[0] == {0: 1, 1: 0}  # can't use the cheap diagonal


def test_solve_constrained_infeasible_returns_none():
    cost = np.array([[1.0, 2.0], [1.0, 2.0]])
    # forbid both columns for row 0 -> no assignment
    assert solve_constrained(cost, forbidden=[(0, 0), (0, 1)]) is None


# --- CandidateStore ---------------------------------------------------------------

SCORES = np.array([[0.9, 0.8, 0.1], [0.2, 0.85, 0.3], [0.1, 0.2, 0.95]])


def test_candidate_store_top_k_and_iter():
    store = CandidateStore.from_scores(SCORES, k=2)
    assert store.top(0, 1) == [(0, pytest.approx(0.9))]
    assert [j for j, _ in store.top(0, 2)] == [0, 1]  # top-2 by score
    assert dict(store) == {0: 0, 1: 1, 2: 2}  # iterate = top-1 pairs


def test_candidate_store_margin_and_sparse():
    store = CandidateStore.from_scores(SCORES, k=3)
    # row 0: top1=0.9, top2=0.8 -> margin 0.1
    assert store.margin(0) == pytest.approx(0.1)
    # sparse store considers only stored candidates
    S = csr_matrix(([0.9, 0.7], ([0, 0], [0, 1])), shape=(1, 3))
    ssp = CandidateStore.from_scores(S, k=2)
    assert [j for j, _ in ssp.top(0, 2)] == [0, 1]


# --- ConstraintSet + reoptimize (edit-and-re-solve) -------------------------------

def test_reoptimize_baseline_is_optimal():
    assert reoptimize(SCORES) == [(0, 0), (1, 1), (2, 2)]


def test_reoptimize_respects_a_rejected_edge():
    # user rejects 0->0; the system re-solves the rest optimally around it
    cs = ConstraintSet().reject(0, 0)
    result = dict(reoptimize(SCORES, cs))
    assert result[0] != 0  # 0 is no longer matched to 0


def test_reoptimize_respects_a_confirmed_edge():
    cs = ConstraintSet().confirm(0, 1)  # force apple->1
    result = dict(reoptimize(SCORES, cs))
    assert result[0] == 1


def test_reoptimize_infeasible_raises():
    cs = ConstraintSet()
    cs.reject(0, 0).reject(0, 1).reject(0, 2)  # row 0 has no allowed column
    with pytest.raises(ValueError):
        reoptimize(SCORES, cs)


# --- review / active learning -----------------------------------------------------

def test_review_queue_surfaces_most_uncertain_first():
    # row 1 has the smallest top1-top2 margin (0.85 vs 0.3 = 0.55; row 0: 0.1; row 2: 0.75)
    store = CandidateStore.from_scores(SCORES, k=3)
    queue = review_queue(store)
    assert queue[0] == 0  # row 0 is the most uncertain (margin 0.1)


def test_uncertainty_sampling_returns_n_most_uncertain():
    store = CandidateStore.from_scores(SCORES, k=3)
    assert uncertainty_sampling(store, n=1) == [0]


# --- review-driven fixes ----------------------------------------------------------

def test_k_best_tall_matrix_enumerates_all():
    # n > m: the old per-row feasibility pre-check pruned these; must enumerate all 4
    result = k_best_assignments([[3], [2], [5], [1]], 4)
    assert [c for _, c in result] == [1.0, 2.0, 3.0, 5.0]
    assert [a for a, _ in result] == [{3: 0}, {1: 0}, {0: 0}, {2: 0}]


def test_k_best_rectangular_two_by_one():
    assert k_best_assignments([[3], [2]], 2) == [({1: 0}, 2.0), ({0: 0}, 3.0)]


def test_solve_constrained_rejects_invalid_forced_edges():
    inf_cost = np.array([[np.inf, 5.0], [5.0, 1.0]])
    assert solve_constrained(inf_cost, forced=[(0, 0)]) is None  # impossible forced edge
    cost = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert solve_constrained(cost, forced=[(0, 0)], forbidden=[(0, 0)]) is None  # contradiction
    assert solve_constrained(cost, forced=[(0, 0), (1, 0)]) is None  # two forced share col 0


def test_reoptimize_sparse_worst_cases_absent_cells():
    # only the anti-diagonal is a candidate (negative scores); must pick real candidates,
    # not the never-scored holes (the #7-class sparse regression)
    S = csr_matrix(([-0.3, -0.4], ([0, 1], [1, 0])), shape=(2, 2))
    assert dict(reoptimize(S)) == {0: 1, 1: 0}


def test_reoptimize_drops_hole_assignments():
    # forced-partial: both rows' only candidate is column 0, so the LAP must leave one row
    # unmatched rather than assign the absent (1,1) hole (D11 — the review-flagged leak)
    S = csr_matrix(([0.9, 0.8], ([0, 1], [0, 0])), shape=(2, 2))
    assert dict(reoptimize(S, sense="maximize")) == {0: 0}
    assert reoptimize(csr_matrix((2, 2)), sense="maximize") == []  # all-absent -> no matches


def test_reoptimize_keeps_user_forced_edge_even_if_absent():
    # a user-confirmed edge is an assertion that overrides blocking, so it is NOT dropped
    S = csr_matrix(([0.9, 0.8], ([0, 1], [0, 0])), shape=(2, 2))
    cs = ConstraintSet().confirm(1, 1)  # force the (1,1) hole
    assert dict(reoptimize(S, cs, sense="maximize")) == {0: 0, 1: 1}


def test_candidate_store_ranks_nan_last_and_ties_by_lower_index():
    store = CandidateStore.from_scores(np.array([[0.5, 0.5, float("nan")]]), k=3)
    cols = [j for j, _ in store.top(0, 3)]
    assert cols[:2] == [0, 1]  # tie broken by lower index
    assert cols[2] == 2  # NaN sorts last, not first


def test_margin_infinite_for_single_or_no_candidates():
    store = CandidateStore(candidates={0: [(0, 0.9)], 1: []})
    assert store.margin(0) == float("inf")  # a single candidate is unambiguous
    assert store.margin(1) == float("inf")  # no candidates -> nothing uncertain
