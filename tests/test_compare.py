"""Tests for the compare stage: string/numeric/geo/vector comparators, the registry,
the direct()/featurized() constructors, and comparison-vector combiners."""

import math

import numpy as np
import pytest

from equate.compare import (
    comparators,
    resolve_comparator,
    direct,
    featurized,
    comparison_vector,
    weighted_sum,
    mean,
    max_combiner,
    fellegi_sunter,
    threshold,
)
from equate.compare import string as S
from equate.compare import numeric_geo as NG
from equate.compare import vector as V


# --- string comparators -----------------------------------------------------------

def test_ratio_and_levenshtein():
    assert S.ratio("abc", "abc") == 1.0
    assert S.levenshtein_distance("kitten", "sitting") == 3
    assert S.levenshtein("abc", "abc") == 1.0
    assert 0.0 < S.levenshtein("jon", "john") < 1.0


def test_monge_elkan_is_directional():
    a, b = "john smith", "smith"
    # a has 2 tokens (one unmatched), b has 1 -> asymmetric
    assert S.monge_elkan(a, b) != S.monge_elkan(b, a)
    assert S.monge_elkan.meta.is_symmetric is False


def test_comparator_metas_declare_metricness():
    assert S.ratio.meta.polarity == "similarity" and S.ratio.meta.is_metric is False
    assert S.levenshtein_distance.meta.is_metric is True
    assert V.cosine.meta.is_metric is False
    assert V.angular_distance.meta.is_metric is True


# --- numeric / geo ----------------------------------------------------------------

def test_decay_functions():
    assert NG.exp_decay(1.0)(0, 0) == 1.0
    assert NG.linear_decay(10.0)(0, 5) == 0.5
    assert NG.linear_decay(10.0)(0, 100) == 0.0  # clamped at 0
    assert NG.gaussian_decay(1.0)(0, 0) == 1.0


def test_haversine_and_geo_similarity():
    assert round(NG.haversine((0.0, 0.0), (0.0, 1.0)), 0) == 111.0
    geo = NG.geo_similarity(scale_km=100.0)
    assert geo((0.0, 0.0), (0.0, 0.0)) == 1.0
    assert 0.0 < geo((0.0, 0.0), (0.0, 1.0)) < 1.0


# --- vector -----------------------------------------------------------------------

def test_vector_comparators():
    u, v = [1.0, 0.0], [1.0, 1.0]
    assert V.cosine(u, u) == 1.0
    assert V.cosine([0.0, 0.0], u) == 0.0  # zero vector -> 0, no div error
    assert V.dot(u, v) == 1.0
    np.testing.assert_allclose(V.angular_distance(u, u), 0.0, atol=1e-9)


# --- registry ---------------------------------------------------------------------

def test_registry_has_core_comparators():
    for name in ("ratio", "levenshtein", "monge_elkan", "cosine", "haversine", "exp_decay"):
        assert name in comparators


def test_resolve_comparator_name_callable_and_config():
    assert resolve_comparator("ratio")("ab", "ab") == 1.0
    assert resolve_comparator(str.upper)("hi") == "HI"  # callable passthrough
    decay = resolve_comparator("exp_decay", scale=2.0)  # configurable factory
    assert decay(0, 0) == 1.0


def test_resolve_comparator_unknown_option_raises():
    with pytest.raises(TypeError):
        resolve_comparator("ratio", nonsense=1)  # plain comparator takes no config


# --- direct() / featurized() constructors -----------------------------------------

def test_direct_builds_matrix_not_indexable():
    build = direct(S.ratio)
    M = build(["abc", "xyz"], ["abc", "abd"])
    assert M.shape == (2, 2)
    assert M[0, 0] == 1.0  # 'abc' vs 'abc'
    assert build.indexable is False


def test_featurized_builds_indexable_matrix_like_similarity_matrix():
    from equate.util import similarity_matrix

    build = featurized("tfidf")
    keys = ["apple pie", "banana split"]
    values = ["american pie", "banana republic"]
    M = build(keys, values)
    assert build.indexable is True
    np.testing.assert_allclose(M, similarity_matrix(keys, values), atol=1e-9)


# --- comparison vectors & combiners -----------------------------------------------

def test_comparison_vector_and_combiners():
    fields = {"name": S.levenshtein, "city": S.ratio}
    a = {"name": "jon", "city": "nyc"}
    b = {"name": "john", "city": "nyc"}
    cv = comparison_vector(a, b, fields)
    assert cv["city"] == 1.0 and 0 < cv["name"] < 1
    assert mean(cv) == (cv["name"] + cv["city"]) / 2
    assert max_combiner(cv) == 1.0
    ws = weighted_sum({"name": 2.0})(cv)
    assert ws == 2.0 * cv["name"] + 1.0 * cv["city"]


def test_fellegi_sunter_combiner():
    m = {"name": 0.9, "city": 0.8}
    u = {"name": 0.1, "city": 0.2}
    combine = fellegi_sunter(m, u)
    agree = {"name": 1.0, "city": 1.0}
    disagree = {"name": 0.0, "city": 0.0}
    # full agreement -> positive match weight; full disagreement -> negative
    assert combine(agree) > 0 > combine(disagree)
    expected_agree = math.log(0.9 / 0.1) + math.log(0.8 / 0.2)
    assert combine(agree) == pytest.approx(expected_agree)


def test_threshold_calibrator():
    hard = threshold(0.5)
    assert hard(0.6) is True and hard(0.4) is False
    soft = threshold(0.5, hard=False)
    assert soft(0.6) == 0.6


# --- optional-dep comparators are registered but lazy -----------------------------

def test_optional_string_comparators_registered():
    assert "jaro_winkler" in comparators and "phonetic" in comparators


def test_jaro_winkler_if_available():
    pytest.importorskip("rapidfuzz")
    jw = resolve_comparator("jaro_winkler")
    assert jw("martha", "marhta") > 0.9


# --- review-driven fixes ----------------------------------------------------------

def test_cosine_numerically_stable_for_huge_vectors():
    # identical huge vectors must be similarity 1.0 (no overflow -> NaN)
    huge = [1e200, 1e200]
    assert V.cosine(huge, huge) == pytest.approx(1.0)
    np.testing.assert_allclose(V.angular_distance(huge, huge), 0.0, atol=1e-6)


def test_cosine_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        V.cosine([1.0, 0.0], [1.0, 0.0, 0.0])


def test_decay_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        NG.exp_decay(0)
    with pytest.raises(ValueError):
        NG.gaussian_decay(-1.0)
    with pytest.raises(ValueError):
        NG.linear_decay(0)


def test_direct_materializes_one_shot_iterables():
    build = direct(S.ratio)
    M = build((k for k in ["ab", "cd"]), (v for v in ["ab", "ce"]))
    assert M.shape == (2, 2) and M[0, 0] == 1.0


def test_direct_empty_a_has_2d_shape():
    build = direct(S.ratio)
    assert build([], ["a", "b"]).shape == (0, 2)


def test_monge_elkan_configurable_via_registry():
    cmp = resolve_comparator("monge_elkan", sim=S.ratio)
    assert callable(cmp) and 0.0 <= cmp("john smith", "smith john") <= 1.0


def test_comparison_vector_missing_field_scores_missing_not_crash():
    fields = {"name": S.levenshtein}
    cv = comparison_vector({"name": "jon"}, {}, fields)  # 'name' missing in b
    assert cv["name"] == 0.0  # default missing score; no TypeError from None


def test_fellegi_sunter_handles_extreme_probabilities():
    combine = fellegi_sunter({"f": 1.0}, {"f": 0.0})  # would be log(0)/div0 unclamped
    assert combine({"f": 1.0}) > 0  # clamped, finite


def test_fellegi_sunter_missing_field_gives_clear_keyerror():
    combine = fellegi_sunter({"name": 0.9}, {"name": 0.1})
    with pytest.raises(KeyError, match="city"):
        combine({"city": 1.0})
