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
from equate.compare import sequence as _seq
from equate.compare import sets as _sets
from equate.compare.vectorize import (
    comparison_vector,
    weighted_sum,
    mean,
    max_combiner,
    fellegi_sunter,
)
from equate.compare.calibrate import threshold, platt, isotonic

__all__ = [
    "comparators",
    "resolve_comparator",
    "direct",
    "polarity_sense",
    "featurized",
    "comparison_vector",
    "weighted_sum",
    "mean",
    "max_combiner",
    "fellegi_sunter",
    "threshold",
    "platt",
    "isotonic",
]

#: the compare-stage strategy registry (name -> lazy factory)
comparators = Registry("comparator")


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
        if not config:
            return fn
        configured = partial(fn, **config)
        # keep the declared metadata (polarity drives the facade's default ``sense``) and
        # an alignment twin, configured the same way
        configured.meta = getattr(fn, "meta", None)
        if hasattr(fn, "align"):
            configured.align = partial(fn.align, **config)
        return configured

    return factory


# Fixed pairwise comparators (core, or lazy-on-first-call for optional-dep ones).
for _name, _fn in [
    ("ratio", _string.ratio),
    ("levenshtein", _string.levenshtein),
    ("levenshtein_distance", _string.levenshtein_distance),
    ("jaro_winkler", _string.jaro_winkler),
    ("cosine", _vec.cosine),
    ("dot", _vec.dot),
    ("angular", _vec.angular_distance),
    ("haversine", _ng.haversine),
]:
    comparators.register(_name, _const(_fn, _name), meta=getattr(_fn, "meta", None))

# Set comparators (doc 07): value-set overlap, for columns, token bags and other sets.
for _name, _fn in [
    ("jaccard", _sets.jaccard),
    ("overlap", _sets.overlap),
    ("containment", _sets.containment),
]:
    comparators.register(_name, _const(_fn, _name), meta=_fn.meta)

# Configurable pairwise comparators (accept keyword config, applied via functools.partial).
comparators.register(
    "monge_elkan", _configurable(_string.monge_elkan), meta=_string.monge_elkan.meta
)
comparators.register(
    "phonetic", _configurable(_string.phonetic_match), meta=_string.phonetic_match.meta
)

# Sequence-alignment comparators (doc 12, D8): configurable, numpy core with optional
# faster backends (``backend=`` -> equate[timeseries] / equate[bio]).
for _name, _fn in [
    ("dtw", _seq.dtw),
    ("subsequence_dtw", _seq.subsequence_dtw),
    ("soft_dtw", _seq.soft_dtw),
    ("needleman_wunsch", _seq.needleman_wunsch),
    ("smith_waterman", _seq.smith_waterman),
]:
    comparators.register(_name, _configurable(_fn), meta=_fn.meta)

# Comparator factories (build a configured comparator from config).
comparators.register("exp_decay", _ng.exp_decay)
comparators.register("linear_decay", _ng.linear_decay)
comparators.register("gaussian_decay", _ng.gaussian_decay)
comparators.register("geo", _ng.geo_similarity)


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


def polarity_sense(comparator, *, default="maximize"):
    """The score ``sense`` a comparator's declared polarity implies.

    A ``'distance'`` comparator (DTW, edit distance) is minimized, a ``'similarity'``
    maximized; a comparator with no declared metadata gets ``default``.

    >>> polarity_sense(resolve_comparator('dtw'))
    'minimize'
    >>> polarity_sense(resolve_comparator('ratio'))
    'maximize'
    """
    meta = getattr(comparator, "meta", None)
    while meta is None and isinstance(
        comparator, partial
    ):  # a user's partial(dtw, ...)
        comparator = comparator.func
        meta = getattr(comparator, "meta", None)
    polarity = getattr(meta, "polarity", None)
    if polarity == "distance":
        return "minimize"
    if polarity == "similarity":
        return "maximize"
    return default


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
    build.meta = meta or getattr(h, "meta", None) or ComparatorMeta()
    build.indexable = False
    return build


def featurized(featurizer="tfidf", metric=None, *, meta=None):
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
        polarity="similarity", bounded=True, is_metric=False, is_symmetric=True
    )
    return build
