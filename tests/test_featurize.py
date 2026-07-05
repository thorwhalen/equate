"""Tests for the featurize stage: char-n-gram TF-IDF (numpy/scipy), the identity and
key-function featurizers, the registry, and the rewired default match path."""

import numpy as np
import pytest
from scipy.sparse import issparse

from equate.featurize import (
    featurizers,
    resolve_featurizer,
    identity,
    mk_tfidf,
    TfidfFeaturizer,
    char_ngrams,
)
from equate._vector import cosine_similarity, l2_normalize


def test_char_ngrams_boundary_padded():
    assert char_ngrams("cat", (2, 3)) == [" c", "ca", "at", "t ", " ca", "cat", "at "]


def test_tfidf_fit_transform_l2_normalized():
    feat = TfidfFeaturizer().fit(["apple pie", "apple crumble", "banana split"])
    X = feat.transform(["apple pie", "banana split"])
    assert issparse(X)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-9)


def test_tfidf_transform_requires_fit():
    with pytest.raises(RuntimeError):
        TfidfFeaturizer().transform(["x"])


def test_tfidf_is_deterministic():
    a = mk_tfidf(["cat", "hat", "bat"])(["cat"]).toarray()
    b = mk_tfidf(["cat", "hat", "bat"])(["cat"]).toarray()
    np.testing.assert_allclose(a, b)


def test_identity_featurizer():
    assert identity(["a", "b"]) == ["a", "b"]


def test_registry_has_expected_featurizers():
    for name in ("identity", "tfidf", "sbert", "openai"):
        assert name in featurizers


def test_resolve_featurizer_default_is_tfidf():
    feat = resolve_featurizer(None, corpus=["apple", "banana"])
    assert isinstance(feat, TfidfFeaturizer)


def test_resolve_featurizer_by_name_and_callable():
    feat = resolve_featurizer("tfidf", corpus=["apple", "banana"])
    assert isinstance(feat, TfidfFeaturizer)
    assert resolve_featurizer(str.upper)("hi") == "HI"  # callable passes through


def test_embedder_facades_are_lazy_not_imported():
    """sbert/openai are registered as factories but must not import heavy libs."""
    import sys

    # retrieving the factory must not import the underlying library
    assert callable(featurizers.factory("sbert"))
    assert "sentence_transformers" not in sys.modules
    assert "openai" not in sys.modules


def test_cosine_similarity_matches_hand_computation():
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    Y = np.array([[1.0, 0.0]])
    sims = cosine_similarity(X, Y)
    np.testing.assert_allclose(sims.ravel(), [1.0, 1.0 / np.sqrt(2)], atol=1e-9)


def test_default_similarity_matrix_is_pure_numpy_and_sensible():
    from equate.util import similarity_matrix

    keys = ["apple pie", "apple crumble", "banana split"]
    values = ["american pie", "big apple", "american girl", "banana republic"]
    m = similarity_matrix(keys, values)
    assert m.shape == (3, 4)
    # 'banana split' should be most similar to 'banana republic' (col 3)
    assert int(np.argmax(m[2])) == 3


def test_default_match_path_still_matches_bananas():
    from equate import match_keys_to_values

    keys = ["apple pie", "banana split"]
    values = ["american pie", "banana republic"]
    result = dict(match_keys_to_values(keys, values))
    assert result["banana split"] == "banana republic"
