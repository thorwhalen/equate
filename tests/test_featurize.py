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


# --- review-driven fixes ----------------------------------------------------------

def test_resolve_featurizer_without_corpus_fails_fast():
    # a fittable featurizer resolved without a corpus must raise at the call site,
    # not hand back a half-built object that only explodes later inside transform
    with pytest.raises(ValueError):
        resolve_featurizer("tfidf")
    with pytest.raises(ValueError):
        resolve_featurizer()  # default is 'tfidf'


def test_resolve_featurizer_unknown_option_raises_not_swallowed():
    with pytest.raises(TypeError):
        resolve_featurizer("tfidf", corpus=["a", "b"], not_a_real_option=1)


def test_char_ngrams_short_text_falls_back_to_padding():
    assert char_ngrams("", (3, 4)) == ["  "]
    # empty strings then get a (single) feature and can match each other
    feat = mk_tfidf(["", "x"], ngram_range=(3, 4))
    sim = cosine_similarity(feat([""]), feat([""]))
    np.testing.assert_allclose(sim, [[1.0]], atol=1e-9)


def test_match_keys_to_values_accepts_generators():
    from equate import match_keys_to_values

    keys = (k for k in ["apple pie", "banana split"])
    values = (v for v in ["american pie", "banana republic"])
    result = dict(match_keys_to_values(keys, values))
    assert result["banana split"] == "banana republic"


def test_resolve_featurizer_does_not_refit_passed_featurizer():
    # a caller-supplied fitted featurizer must be returned as-is, never re-fit/mutated
    feat = TfidfFeaturizer().fit(["hello world", "foo bar"])
    vocab_before = dict(feat.vocabulary_)
    resolved = resolve_featurizer(feat, corpus=["completely", "different", "corpus"])
    assert resolved is feat
    assert feat.vocabulary_ == vocab_before  # unchanged
