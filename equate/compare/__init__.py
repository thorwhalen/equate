"""The compare stage (②): score how alike two (featurized) items are.

There are two routes (decision register / doc 00-taxonomy):

- **featurize-then-compare** (:func:`featurized`) — an *indexable* vector metric over
  independent representations (what the legacy ``similarity_matrix`` does); only this
  route can drive an ANN/LSH index;
- **direct pairwise compare** (:func:`direct`) — an *opaque* scorer ``h(a, b) -> float``
  (edit distance, a cross-encoder/LLM); it can only re-rank a candidate set.

Pairwise comparators are ``(a, b) -> float`` callables; select one by name via the
:data:`comparators` registry or pass a callable. Each carries a
:class:`equate.base.ComparatorMeta` (polarity / bounded / is_metric / is_symmetric) so
the framework adapts — never treating a non-metric as a metric, keeping directional
comparators (Monge-Elkan, containment) directional until the matcher boundary (D3).
"""

from functools import partial

import numpy as np

from equate.registry import Registry
from equate.base import ComparatorMeta
from equate._vector import cosine_similarity
from equate.featurize import resolve_featurizer
from equate.compare import string as _string
from equate.compare import numeric_geo as _ng
from equate.compare import vector as _vec
from equate.compare.vectorize import (
    comparison_vector,
    weighted_sum,
    mean,
    max_combiner,
    fellegi_sunter,
)
from equate.compare.calibrate import threshold, platt, isotonic

__all__ = [
    'comparators',
    'resolve_comparator',
    'direct',
    'featurized',
    'comparison_vector',
    'weighted_sum',
    'mean',
    'max_combiner',
    'fellegi_sunter',
    'threshold',
    'platt',
    'isotonic',
]

#: the compare-stage strategy registry (name -> lazy factory)
comparators = Registry('comparator')


def _const(obj, name):
    """A factory returning a fixed comparator; passing config is a clear error."""

    def factory(**config):
        if config:
            raise TypeError(
                f"comparator {name!r} takes no configuration; got {list(config)}"
            )
        return obj

    return factory


def _configurable(fn):
    """A factory returning ``fn``, partially applied with any keyword config."""

    def factory(**config):
        return partial(fn, **config) if config else fn

    return factory


# Fixed pairwise comparators (core, or lazy-on-first-call for optional-dep ones).
for _name, _fn in [
    ('ratio', _string.ratio),
    ('levenshtein', _string.levenshtein),
    ('levenshtein_distance', _string.levenshtein_distance),
    ('jaro_winkler', _string.jaro_winkler),
    ('cosine', _vec.cosine),
    ('dot', _vec.dot),
    ('angular', _vec.angular_distance),
    ('haversine', _ng.haversine),
]:
    comparators.register(_name, _const(_fn, _name), meta=getattr(_fn, 'meta', None))

# Configurable pairwise comparators (accept keyword config, applied via functools.partial).
comparators.register(
    'monge_elkan', _configurable(_string.monge_elkan), meta=_string.monge_elkan.meta
)
comparators.register(
    'phonetic', _configurable(_string.phonetic_match), meta=_string.phonetic_match.meta
)

# Comparator factories (build a configured comparator from config).
comparators.register('exp_decay', _ng.exp_decay)
comparators.register('linear_decay', _ng.linear_decay)
comparators.register('gaussian_decay', _ng.gaussian_decay)
comparators.register('geo', _ng.geo_similarity)


def resolve_comparator(spec, **config):
    """Resolve ``spec`` to a pairwise comparator.

    A callable passes through; a registered name (``str``) is built via its factory
    (``**config`` forwarded, so an unknown option raises instead of being ignored).
    """
    if callable(spec):
        return spec
    if isinstance(spec, str):
        return comparators.create(spec, **config)
    raise TypeError(
        f"comparator spec must be a registered name or a callable, "
        f"got {type(spec).__name__}"
    )


def direct(h, *, meta=None):
    """Wrap an opaque pairwise scorer ``h(a, b) -> float`` into a score-matrix builder
    ``build(A, B) -> ndarray``. Marked **not indexable** — usable only to re-rank a
    candidate set, never to drive an ANN/LSH index.
    """

    def build(A, B):
        A, B = list(A), list(B)  # materialize: B is iterated once per row of A
        if not A:
            return np.empty((0, len(B)), dtype=float)
        return np.array([[h(a, b) for b in B] for a in A], dtype=float)

    build.comparator = h
    build.meta = meta or getattr(h, 'meta', None) or ComparatorMeta()
    build.indexable = False
    return build


def featurized(featurizer='tfidf', metric=None, *, meta=None):
    """Build a score-matrix builder that featurizes both collections then applies a
    vector ``metric`` (default cosine over the two matrices). Marked **indexable** — the
    featurize-then-compare route; this is what the legacy ``similarity_matrix`` does.
    """
    metric = metric if metric is not None else cosine_similarity

    def build(A, B):
        A, B = list(A), list(B)
        feat = resolve_featurizer(featurizer, corpus=A + B)
        return metric(feat(A), feat(B))

    build.indexable = True
    build.meta = meta or ComparatorMeta(
        polarity='similarity', bounded=True, is_metric=False, is_symmetric=True
    )
    return build
