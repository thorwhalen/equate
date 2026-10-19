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


# --- regressions from the adversarial review -----------------------------------


def test_harness_skips_a_test_part_with_no_gold_instead_of_scoring_it_perfect():
    ds = Benchmark("tiny", ["a", "b"], ["a", "b"], frozenset({(0, 0), (1, 1)}))
    with pytest.warns(UserWarning, match="no gold pair"):
        with pytest.raises(ValueError, match="too small"):
            benchmark_matcher(lambda a, b: [], on=[ds], n_seeds=3, ratios=(1, 1, 0))


def test_harness_rejects_matcher_and_fit_together_and_accepts_a_single_name():
    with pytest.raises(ValueError, match="either matcher or fit"):
        benchmark_matcher(lambda a, b: [], fit=lambda *a: None)
    assert benchmark_matcher(on="toy_names", n_seeds=1).rows[0].benchmark == "toy_names"


def test_ranking_counts_a_repeated_candidate_once():
    m = ranking_metrics({"q": ["a", "a", "b"]}, {"q": "a"}, ks=(2,))
    assert m["recall_at_2"] == 1.0 and m["map"] == 1.0


def test_pm_one_labels_are_refused():
    with pytest.raises(ValueError, match="0/1"):
        pr_auc([0.1, 0.9], [1, -1])


def test_empty_cluster_metrics_agree_across_backends():
    assert cluster_metrics([], [], backend="numpy")["b3_f1"] == 1.0


def test_nan_ratio_is_refused():
    with pytest.raises(ValueError):
        entity_split([0, 1, 2], ratios=(float("nan"), 0.2, 0.2))


@pytest.mark.parametrize("seed", range(10))
def test_toy_gold_is_unambiguous(seed):
    ds = load_benchmark("toy_names", n_entities=200, seed=seed)
    assert len(set(ds.a)) == len(ds.a) and len(set(ds.b)) == len(ds.b)
    assert not set(ds.a) & set(ds.b)


# --- bootstrap CIs and GMD (equate#25) -----------------------------------------------

from equate.evaluate import bootstrap_ci, gmd, labels_to_pairs, pair_cost  # noqa: E402


def test_gmd_counts_operations_and_equals_pairwise_errors_with_pair_cost():
    import numpy as np

    assert gmd([0, 0, 1, 1], [0, 0, 0, 1]) == 2.0  # one split, one merge
    assert gmd([0, 1, 2], [0, 1, 2]) == 0.0
    assert gmd([0, 0, 0], [0, 1, 2]) == 2.0  # split twice
    assert gmd([0, 1, 2], [5, 5, 5]) == 2.0  # merge twice (labels need not agree)
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(1, 10))
        p, t = rng.integers(0, 4, n), rng.integers(0, 4, n)
        errors = set(labels_to_pairs(p)) ^ set(labels_to_pairs(t))
        assert gmd(p, t, merge_cost=pair_cost, split_cost=pair_cost) == len(errors)


def test_gmd_costs_are_asymmetric_on_purpose():
    over, under = [0, 0, 0, 0], [0, 1, 2, 3]  # one big cluster vs singletons
    truth = [0, 0, 1, 1]
    cheap_merge = dict(merge_cost=lambda x, y: 0.1, split_cost=lambda x, y: 1.0)
    assert gmd(over, truth, **cheap_merge) == 1.0  # one split needed
    assert gmd(under, truth, **cheap_merge) == pytest.approx(0.2)  # two cheap merges


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible():
    gold = {(i, i) for i in range(40)}
    pred = {(i, i) for i in range(30)} | {(i, i + 1) for i in range(30, 40)}
    ents = [f"e{i}" for i in range(41)]
    a = bootstrap_ci(pred, gold, ents, n_boot=500)
    assert a == bootstrap_ci(pred, gold, ents, n_boot=500)  # seeded
    lo, hi = a["f1"]
    assert lo < 0.75 < hi and hi - lo < 0.5
    narrow = bootstrap_ci(pred, gold, ents, level=0.5, n_boot=500)
    assert narrow["f1"][1] - narrow["f1"][0] < hi - lo
    with pytest.raises(ValueError):
        bootstrap_ci(pred, gold, ents, level=1.5)


def test_bootstrap_resamples_entities_not_pairs():
    # one entity owns all the errors: resampling entities must swing F1 widely
    gold = {(0, 0), (1, 1), (2, 2), (3, 3)}
    pred = {(0, 0), (1, 1), (2, 2), (3, 9)}
    ci = bootstrap_ci(pred, gold, ["a", "b", "c", "d"], n_boot=2000)
    assert ci["f1"][0] < 0.6 and ci["f1"][1] == 1.0


def test_harness_ci_auto_only_for_single_runs():
    ds = load_benchmark("toy_names", n_entities=40)
    multi = benchmark_matcher(on=[ds], n_seeds=2)
    assert all(r.ci is None for r in multi.rows)
    single = benchmark_matcher(on=[ds], n_seeds=1)
    assert set(single.rows[0].ci) == {"precision", "recall", "f1"}
    assert "[" in single.to_markdown().splitlines()[-1]
    always = benchmark_matcher(on=[ds], n_seeds=2, ci=0.9)
    assert all(r.ci for r in always.rows)
    assert benchmark_matcher(on=[ds], n_seeds=1, ci=None).rows[0].ci is None
    with pytest.raises(ValueError, match="ci must be"):
        benchmark_matcher(on=[ds], ci=95)


# --- regressions from the adversarial review of #25 --------------------------------


def test_degenerate_resamples_do_not_inflate_the_interval():
    # 20 entities, 1 gold pair, 1 wrong prediction: every score is 0
    gold, pred = {(0, 0)}, {(5, 5)}
    ci = bootstrap_ci(pred, gold, [f"e{i}" for i in range(20)], n_boot=500)
    assert ci["f1"] == (0.0, 0.0)  # was (0.0, 1.0): empty resamples counted as perfect


def test_auto_ci_follows_the_rows_actually_scored():
    import warnings

    # 3 seeds requested, 2 skipped (no gold pair in their test part): one run is left
    ds = load_benchmark("toy_names", n_entities=2, n_distractors=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = benchmark_matcher(on=[ds], n_seeds=3, ratios=(0.5, 0.25, 0.25))
    assert len(report.rows) == 1 and report.rows[0].ci is not None


def test_bad_ci_arguments_fail_before_the_matcher_runs():
    calls = []

    def matcher(a, b):
        calls.append(1)
        return []

    for kw in ({"ci": 1.5}, {"ci": "always"}, {"n_boot": 0}, {"n_boot": True}):
        with pytest.raises(ValueError):
            benchmark_matcher(matcher, on=["toy_names"], n_seeds=1, **kw)
    assert calls == []


def test_rows_with_an_interval_stay_hashable():
    row = benchmark_matcher(on=["toy_names"], n_seeds=1).rows[0]
    assert row.ci and isinstance(hash(row), int)


def test_bootstrap_memory_is_chunked():
    ents = [f"e{i}" for i in range(20_000)]
    gold = {(i, i) for i in range(20_000)}
    ci = bootstrap_ci(gold, gold, ents, n_boot=200)  # 200 × 20k at once would be 32 MB+
    assert ci["f1"] == (1.0, 1.0)


def test_linear_cost_is_the_order_independent_family():
    from equate.evaluate import linear_cost

    assert gmd([0, 0, 1, 1], [0, 0, 0, 1], merge_cost=linear_cost(0, 1), split_cost=linear_cost(0, 1)) == 3.0
    # label names do not matter for this family
    p = [0] * 6
    assert gmd(p, [0, 1, 1, 2, 2, 2], split_cost=linear_cost(1, 2)) == gmd(
        p, [2, 1, 1, 0, 0, 0], split_cost=linear_cost(1, 2)
    )
