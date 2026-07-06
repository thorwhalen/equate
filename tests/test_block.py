"""Tests for the block stage: all_pairs, keyed/sorted-neighbourhood/brute-kNN blockers,
blocking metrics (PC/RR/PQ), the sparse score_candidates seam, and meta-blocking."""

import numpy as np
import pytest
from scipy.sparse import issparse

from equate.block import (
    blockers,
    resolve_blocker,
    all_pairs,
    key_blocking,
    sorted_neighborhood,
    first_chars,
    qgram_keys,
    whole_value,
    blocking_metrics,
    score_candidates,
    metablock,
    brute_knn_blocking,
)
from equate.base import ScoreMatrix


# --- all_pairs & resolve ----------------------------------------------------------

def test_all_pairs_cross_and_self_join():
    assert list(all_pairs(["a", "b"], ["x", "y"])) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert list(all_pairs(["a", "b", "c"])) == [(0, 1), (0, 2), (1, 2)]


def test_resolve_blocker_none_name_callable():
    assert resolve_blocker(None) is all_pairs
    assert callable(resolve_blocker("all_pairs"))
    assert resolve_blocker(all_pairs) is all_pairs


def test_registry_has_blockers():
    for name in ("all_pairs", "key", "sorted_neighborhood", "brute_knn", "ann", "minhash_lsh"):
        assert name in blockers


# --- keyed blocking ---------------------------------------------------------------

def test_key_blocking_first_char():
    A = ["apple", "banana", "avocado"]
    B = ["apricot", "berry", "cherry"]
    pairs = set(key_blocking(first_chars(1))(A, B))
    # 'apple'/'avocado' (a) block with 'apricot' (a); 'banana' (b) with 'berry' (b)
    assert pairs == {(0, 0), (2, 0), (1, 1)}


def test_key_blocking_self_join_yields_i_lt_j():
    A = ["apple", "avocado", "banana"]
    pairs = list(key_blocking(first_chars(1))(A))
    assert pairs == [(0, 1)]  # only the two 'a' items, i<j


def test_qgram_blocking_finds_overlap():
    A = ["hello", "world"]
    B = ["hells", "word"]
    pairs = set(key_blocking(qgram_keys(2))(A, B))
    assert (0, 0) in pairs  # hello/hells share q-grams
    assert (1, 1) in pairs  # world/word share q-grams


def test_whole_value_exact_key():
    pairs = set(key_blocking(whole_value())(["Cat", "dog"], ["cat", "DOG"]))
    assert pairs == {(0, 0), (1, 1)}


def test_sorted_neighborhood_self_join_window():
    pairs = set(sorted_neighborhood(window=2)(["c", "a", "b"]))
    # sorted order a,b,c -> window-2 neighbours (a,b),(b,c) -> original indices
    assert pairs == {(1, 2), (0, 2)}


# --- blocking metrics -------------------------------------------------------------

def test_blocking_metrics():
    cand = [(0, 0), (1, 1), (0, 1)]
    true = [(0, 0), (1, 1), (2, 2)]
    m = blocking_metrics(cand, true, n_a=3, n_b=3)
    assert m["pair_completeness"] == pytest.approx(2 / 3)  # 2 of 3 true kept
    assert m["pairs_quality"] == pytest.approx(2 / 3)  # 2 of 3 candidates true
    assert m["reduction_ratio"] == pytest.approx(1 - 3 / 9)
    assert m["n_candidates"] == 3


# --- score_candidates (sparse ScoreMatrix seam) -----------------------------------

def test_score_candidates_builds_sparse_score_matrix():
    A = ["apple", "banana"]
    B = ["appl", "banan", "cherry"]
    cands = [(0, 0), (1, 1)]  # only score the diagonal-ish candidates
    sm = score_candidates(A, B, cands, comparator="ratio")
    assert isinstance(sm, ScoreMatrix) and issparse(sm.data)
    assert sm.shape == (2, 3)
    dense = sm.data.toarray()
    assert dense[0, 0] > 0.8 and dense[1, 1] > 0.8
    assert dense[0, 2] == 0.0  # (0,2) was not a candidate -> structurally absent


def test_score_candidates_dedupes_duplicate_pairs():
    sm = score_candidates(["a"], ["a"], [(0, 0), (0, 0)], comparator="ratio")
    # duplicate candidate must be scored once, not summed to 2.0
    assert sm.data.toarray()[0, 0] == pytest.approx(1.0)


def test_blocked_scoring_feeds_a_matcher():
    from equate.util import hungarian_matching

    A = ["apple", "banana", "cherry"]
    B = ["banan", "appl", "chery"]
    cands = list(all_pairs(A, B))  # dense here, but via the blocker seam
    sm = score_candidates(A, B, cands, comparator="ratio")
    result = dict(hungarian_matching(sm.data, sense=sm.sense))
    assert result == {0: 1, 1: 0, 2: 2}  # apple->appl, banana->banan, cherry->chery


# --- meta-blocking ----------------------------------------------------------------

def test_metablock_dedupe_and_prune():
    assert list(metablock.dedupe_pairs([(0, 0), (0, 0), (1, 1)])) == [(0, 0), (1, 1)]
    # item a0 appears in 3 candidates; prune with max_degree=2 removes its pairs
    cands = [(0, 0), (0, 1), (0, 2), (1, 3)]
    assert list(metablock.prune_frequent(cands, max_degree=2)) == [(1, 3)]


# --- brute-force kNN blocker (pure numpy, core) -----------------------------------

def test_brute_knn_blocking_returns_top_k():
    A = ["apple pie", "banana bread"]
    B = ["apple tart", "banana cake", "cherry cola"]
    pairs = set(brute_knn_blocking("tfidf", k=1)(A, B))
    # each A item's single nearest B item: apple->apple tart, banana->banana cake
    assert (0, 0) in pairs and (1, 1) in pairs
    # k=1 -> exactly one candidate per A row
    assert len([p for p in pairs if p[0] == 0]) == 1


# --- optional-dep blockers are registered but lazy --------------------------------

def test_ann_and_lsh_blockers_registered():
    assert "ann" in blockers and "minhash_lsh" in blockers


def test_minhash_lsh_if_available():
    pytest.importorskip("datasketch")
    from equate.block import minhash_lsh_blocking

    A = ["the quick brown fox", "a totally different sentence"]
    B = ["the quick brown dog", "yet another unrelated line"]
    pairs = set(minhash_lsh_blocking(threshold=0.2)(A, B))
    assert (0, 0) in pairs  # the two "quick brown" sentences share tokens


# --- review-driven fixes ----------------------------------------------------------

def test_brute_knn_self_join_emits_exact_k_no_overproduce():
    # three identical items (all tie at max similarity): each row must emit exactly k=1,
    # never k+1, and no self-pair
    pairs = list(brute_knn_blocking("tfidf", k=1)(["x", "x", "x"]))
    assert len(pairs) == 3
    assert all(i < j for i, j in pairs)  # normalized i<j, no (i,i)


def test_blocking_metrics_self_join_canonicalizes_orientation():
    m = blocking_metrics([(1, 0)], [(0, 1)], n_a=3)  # self-join (n_b=None)
    assert m["pair_completeness"] == 1.0  # (1,0) matches true (0,1)


def test_blocking_metrics_reduction_ratio_clamped_to_unit_interval():
    m = blocking_metrics([(0, 1), (0, 2), (1, 2), (0, 1)], [], n_a=3)
    assert 0.0 <= m["reduction_ratio"] <= 1.0


def test_prune_frequent_self_join_combines_slot_degrees():
    cands = [(0, 1), (0, 2), (1, 2)]  # each item touches 2 candidates
    assert list(metablock.prune_frequent(cands, max_degree=1, self_join=True)) == []


def test_score_candidates_rejects_out_of_range_index():
    with pytest.raises(IndexError):
        score_candidates(["a"], ["b"], [(0, 5)], comparator="ratio")


def test_all_pairs_via_registry_rejects_config():
    with pytest.raises(TypeError):
        blockers.create("all_pairs", foo=1)
