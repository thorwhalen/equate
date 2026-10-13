"""Tests for equate.evaluate: metrics, granularity helpers, splits, and the harness."""

import math

import numpy as np
import pytest

from equate.evaluate import (
    Benchmark,
    Scores,
    assert_no_leakage,
    bcubed,
    benchmark_matcher,
    ceaf,
    cluster_metrics,
    cross_pairs_to_labels,
    entity_split,
    labels_to_pairs,
    list_benchmarks,
    load_benchmark,
    pairs_to_labels,
    pairwise_prf,
    pr_auc,
    ranking_metrics,
    select_threshold,
)


def test_scores_never_disagree_with_their_f1():
    s = Scores.from_pr(0.5, 1.0)
    assert math.isclose(s.f1, 2 / 3)
    with pytest.raises(Exception):
        s.f1 = 1.0  # frozen


def test_empty_prediction_and_empty_gold():
    assert pairwise_prf([], []).f1 == 1.0
    s = pairwise_prf([], [(0, 0)])
    assert (s.precision, s.recall) == (1.0, 0.0)


def test_pairwise_self_join_ignores_orientation_and_self_pairs():
    s = pairwise_prf([(1, 0), (2, 2)], [(0, 1)], self_join=True)
    assert (s.precision, s.recall) == (1.0, 1.0)
    assert pairwise_prf([(1, 0)], [(0, 1)]).f1 == 0.0  # two tables: orientation matters


def test_pairs_labels_round_trip():
    labels = pairs_to_labels([(3, 1), (1, 0)], n=5)
    assert labels == [0, 0, 1, 0, 2]
    assert pairs_to_labels(labels_to_pairs(labels), n=5) == labels
    with pytest.raises(IndexError):
        pairs_to_labels([(0, 9)], n=3)


def test_perfect_and_degenerate_clusterings():
    true = [0, 0, 1, 1, 2]
    assert bcubed(true, true).f1 == 1.0
    assert ceaf(true, true).f1 == 1.0
    assert cluster_metrics(true, true, backend="numpy")["ari"] == 1.0
    # everything in one cluster: perfect B3 recall, poor precision
    one = bcubed([0] * 5, true)
    assert one.recall == 1.0 and one.precision < 0.5


def test_label_shape_mismatch_is_an_informative_error():
    with pytest.raises(ValueError, match="pairs_to_labels"):
        bcubed([0, 1], [0, 1, 2])


def test_ceaf_mention_and_bad_variant():
    assert 0 < ceaf([0, 0, 1, 1], [0, 0, 0, 1], variant="mention").f1 < 1
    with pytest.raises(ValueError):
        ceaf([0], [0], variant="nope")


@pytest.mark.parametrize("seed", range(5))
def test_numpy_cluster_metrics_agree_with_sklearn(seed):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(seed)
    true = rng.integers(0, 6, size=40)
    pred = rng.integers(0, 4, size=40)
    a = cluster_metrics(pred, true, backend="numpy")
    b = cluster_metrics(pred, true, backend="sklearn")
    for k in a:
        assert math.isclose(a[k], b[k], abs_tol=1e-9), k


def test_cluster_metrics_backend_is_validated():
    with pytest.raises(ValueError):
        cluster_metrics([0], [0], backend="torch")


def test_pr_auc_matches_sklearn_and_handles_ties():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = np.round(rng.random(200), 1)  # many ties
    assert math.isclose(pr_auc(s, y), skm.average_precision_score(y, s), abs_tol=1e-12)
    assert math.isnan(pr_auc([0.1, 0.2], [0, 0]))


def test_select_threshold_never_needs_test_data():
    t, s = select_threshold([0.9, 0.2, 0.8, 0.1], [1, 0, 1, 0])
    assert t == 0.8 and s.f1 == 1.0
    with pytest.raises(ValueError):
        select_threshold([], [])
    with pytest.raises(ValueError):
        select_threshold([0.5], [1], optimize="accuracy")


def test_ranking_metrics_multi_gold_and_missing():
    m = ranking_metrics({"q": ["a", "x", "b"]}, {"q": {"a", "b"}, "r": "z"}, ks=(1, 3))
    assert m["recall_at_1"] == 0.25  # q: 1/2, r: 0
    assert m["recall_at_3"] == 0.5
    assert m["n_queries"] == 2
    with pytest.raises(ValueError):
        ranking_metrics({}, {})


def test_entity_split_is_disjoint_in_entities_and_covers_every_record():
    ids = np.random.default_rng(0).integers(0, 30, size=200)
    s = entity_split(ids, seed=3)
    assert_no_leakage(ids, s.train, s.val, s.test)
    assert sorted(s.train + s.val + s.test) == list(range(200))
    assert entity_split(ids, seed=3) == s  # deterministic
    with pytest.raises(ValueError):
        entity_split(ids, ratios=(1, 1))


def test_cross_pairs_bridge_two_tables_to_entities():
    labels = cross_pairs_to_labels([(0, 1), (1, 0)], n_a=2, n_b=3)
    assert labels[0] == labels[2 + 1] and labels[1] == labels[2 + 0]
    assert labels[2 + 2] not in (labels[0], labels[1])


def test_registry_lists_and_loads_the_toy_benchmark():
    assert "toy_names" in list_benchmarks()
    ds = load_benchmark("toy_names", n_entities=10, n_distractors=5)
    assert len(ds.a) == 10 and len(ds.b) == 15 and len(ds.gold) == 10
    assert 0 < ds.base_rate < 1


def test_harness_scores_only_unseen_test_entities_and_reports_cost():
    ds = load_benchmark("toy_names", n_entities=40)
    seen = []

    def spy(a, b):
        seen.append((list(a), list(b)))
        return []  # predicts nothing

    report = benchmark_matcher(spy, on=[ds], n_seeds=2)
    assert len(report.rows) == 2
    for row in report.rows:
        assert row.precision == 1.0 and row.recall == 0.0  # empty prediction
        assert row.seconds >= 0 and 0 < row.base_rate < 1
    # only test-split records reach the matcher
    assert all(len(a) < len(ds.a) for a, _ in seen)
    md = report.to_markdown()
    assert "toy_names" in md and "base rate" in md and "easy" in md


def test_harness_fit_gets_train_and_val_only():
    ds = load_benchmark("toy_names", n_entities=40)
    calls = []

    def fit(tr_a, tr_b, tr_pairs, va_a, va_b, va_pairs):
        calls.append((set(tr_a), set(va_a)))
        from equate import match

        return lambda a, b: match(a, b, compare="ratio")

    report = benchmark_matcher(on=[ds], n_seeds=2, fit=fit)
    assert len(calls) == 2
    s = report.summary()["toy_names"]
    assert s["f1"][0] > 0.8
    tr, va = calls[0]
    assert not (tr & va)


def test_benchmark_is_a_plain_container():
    ds = Benchmark("x", ["a"], ["a"], frozenset({(0, 0)}))
    assert ds.base_rate == 1.0
    r = benchmark_matcher(lambda a, b: [(0, 0)] if a and b else [], on=[ds], n_seeds=1, ratios=(0, 0, 1))
    assert r.rows[0].f1 == 1.0
