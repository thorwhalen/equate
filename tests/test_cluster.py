"""Tests for the resolve/cluster stage (connected-components, correlation clustering,
canonicalization), Fellegi-Sunter classify + EM, and the dedupe()/resolve() facades."""

import pytest

from equate.cluster import (
    clusterers,
    resolve_clusterer,
    connected_components,
    correlation_clustering,
    canonicalize,
)
from equate.cluster.classify import classify, estimate_mu_em
from equate.base import Partition
from equate import dedupe, resolve, match


# --- clustering -------------------------------------------------------------------

def test_connected_components_transitive_closure():
    # 0-1 and 1-2 chain into one component (transitivity)
    p = connected_components([(0, 1), (1, 2), (3, 4)], 6)
    assert p.groups() == {0: [0, 1, 2], 1: [3, 4], 2: [5]}


def test_correlation_clustering_does_not_chain_weak_edges():
    # a-b strong, b-c strong, a-c ABSENT: connected-components would merge all three;
    # pivot clusters a with its positive neighbours only
    scored = [(0, 1, 0.9), (1, 2, 0.9)]
    p = correlation_clustering(scored, 3, threshold=0.5)
    # pivot 0 pulls in 1; 2 is left for its own cluster (0's cluster already took 1)
    assert p.groups() == {0: [0, 1], 1: [2]}


def test_resolve_clusterer_uniform_signature():
    scored = [(0, 1, 0.9), (2, 3, 0.2)]
    cc = resolve_clusterer("connected_components")(scored, 4, threshold=0.5)
    assert cc.groups() == {0: [0, 1], 1: [2], 2: [3]}  # (2,3) below threshold
    assert "connected_components" in clusterers and "correlation" in clusterers


# --- canonicalization -------------------------------------------------------------

def test_canonicalize_policies():
    records = [
        {"name": "Jon", "age": None},
        {"name": "Jon", "age": 30},
        {"name": "Jonathan", "age": 30},
    ]
    part = Partition([0, 0, 0])  # all one cluster
    first = canonicalize(part, records, policy="first")[0]
    complete = canonicalize(part, records, policy="most_complete")[0]
    majority = canonicalize(part, records, policy="majority")[0]
    assert first == records[0]
    assert complete["age"] == 30  # most_complete prefers the record with fewer Nones
    assert majority["name"] == "Jon" and majority["age"] == 30  # per-field vote


# --- Fellegi-Sunter classify + EM -------------------------------------------------

def test_classify_three_way_decision():
    assert [classify(w, lower=-1.0, upper=1.0) for w in (2.0, 0.0, -3.0)] == [
        "match",
        "possible_match",
        "non_match",
    ]


def test_classify_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        classify(0.0, lower=1.0, upper=-1.0)


def test_estimate_mu_em_separates_matches():
    # matches agree on both fields; non-matches disagree -> m should exceed u
    matches = [{"name": 1.0, "city": 1.0}] * 20
    nonmatches = [{"name": 0.0, "city": 0.0}] * 20
    m, u = estimate_mu_em(matches + nonmatches)
    for field in ("name", "city"):
        assert m[field] > u[field]


# --- dedupe / resolve facades -----------------------------------------------------

def test_dedupe_groups_near_duplicates():
    p = dedupe(["Jon Smith", "Jon Smyth", "Kate Doe"], threshold=0.8)
    assert isinstance(p, Partition)
    assert p.groups() == {0: [0, 1], 1: [2]}


def test_dedupe_iterates_as_duplicate_pairs():
    p = dedupe(["abc", "abd", "xyz"], threshold=0.6)
    assert list(p) == [(0, 1)]  # abc~abd are duplicates; xyz alone


def test_resolve_pools_collections():
    p = resolve(["apple", "banana"], ["appl", "cherry"], threshold=0.7)
    # apple(0) ~ appl(2) group; banana(1) and cherry(3) alone
    assert p.groups()[0] == [0, 2]


def test_match_how_cluster_returns_partition():
    A = ["apple", "banana"]
    B = ["appl", "cherry"]
    p = match(A, B, compare="ratio", how="cluster", threshold=0.7)
    assert isinstance(p, Partition)
    # A[0]=apple (idx 0) clusters with B[0]=appl (idx len(A)+0 = 2)
    assert 0 in p.groups().get(p.labels[0]) and 2 in p.groups().get(p.labels[0])


# --- review-driven fixes ----------------------------------------------------------

def test_clusterers_reject_out_of_range_and_negative_indices():
    with pytest.raises(ValueError):
        connected_components([(-1, 0)], 5)  # negative would silently alias to item 4
    with pytest.raises(ValueError):
        connected_components([(0, 9)], 5)  # out of range
    with pytest.raises(ValueError):
        correlation_clustering([(0, -1, 1.0)], 5, threshold=0.5)


def test_connected_components_usable_as_clusterer_callable():
    # passing the public function to resolve_clusterer must honor the (scored_pairs, n,
    # *, threshold, sense) contract, not crash on the name/adapter mismatch
    cl = resolve_clusterer(connected_components)
    assert cl([(0, 1, 0.9), (2, 3, 0.2)], 4, threshold=0.5).groups() == {
        0: [0, 1],
        1: [2],
        2: [3],
    }


def test_dedupe_with_distance_comparator_uses_minimize_sense():
    from equate.compare.string import levenshtein_distance

    # distance: lower = more similar; threshold=1 links pairs within edit distance 1
    p = dedupe(["cat", "car", "dog"], compare=levenshtein_distance, threshold=1)
    assert p.groups() == {0: [0, 1], 1: [2]}  # cat~car (dist 1); dog alone


def test_canonicalize_majority_scalar_and_unhashable():
    part = Partition([0, 0, 0])
    # scalar records: value vote (was silently returning {})
    assert canonicalize(part, ["Jon", "Jon", "Jonathan"], policy="majority")[0] == "Jon"
    # unhashable field values fall back to the first, not crash
    recs = [{"tags": ["a"]}, {"tags": ["b"]}, {"tags": ["a"]}]
    assert canonicalize(part, recs, policy="majority")[0]["tags"] == ["a"]


def test_canonicalize_unknown_policy_raises_even_when_empty():
    with pytest.raises(ValueError):
        canonicalize(Partition([]), [], policy="bogus")


def test_classify_rejects_nan():
    with pytest.raises(ValueError):
        classify(float("nan"), lower=-1.0, upper=1.0)


def test_estimate_mu_em_skips_missing_fields():
    # 'name' is present in both classes and separates them; 'city' appears only in the
    # matches, so with missing!=disagreement it has NO non-match evidence and must NOT
    # spuriously discriminate (the old missing==disagreement bug would make m>>u for it)
    matches = [{"name": 1.0, "city": 1.0}] * 15
    nonmatches = [{"name": 0.0}] * 15  # 'city' absent, not a disagreement
    m, u = estimate_mu_em(matches + nonmatches)
    assert m["name"] > u["name"]  # name genuinely discriminates
    assert abs(m["city"] - u["city"]) < 0.05  # city cannot be estimated -> ~equal
