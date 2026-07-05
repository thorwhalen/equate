"""The featurize stage (①): map objects to comparable representations.

A *featurizer* is a batch callable ``Iterable[object] -> representations``. The default
is a pure numpy/scipy char-n-gram TF-IDF (:mod:`equate.featurize.tfidf`) — no heavy
dependencies; dense embedders (SBERT, OpenAI, ...) are optional extras
(:mod:`equate.featurize.text`). Select a featurizer by name via the :data:`featurizers`
registry, or pass a callable. Corpus-dependent featurizers (TF-IDF) are built by
fitting on a corpus, which :func:`resolve_featurizer` handles.
"""

from equate.base import FeaturizerMeta
from equate.registry import Registry
from equate.featurize.tfidf import (
    TfidfFeaturizer,
    mk_tfidf,
    char_ngrams,
    DFLT_NGRAM_RANGE,
)
from equate.featurize import text as _text

__all__ = [
    'featurizers',
    'identity',
    'resolve_featurizer',
    'TfidfFeaturizer',
    'mk_tfidf',
    'char_ngrams',
    'FeaturizerMeta',
]

#: the featurize-stage strategy registry (name -> lazy factory)
featurizers = Registry('featurizer')


def identity(objects):
    """The identity featurizer: use the objects themselves as their representation.

    This is the degenerate case that recovers exact matching (φ = id).
    """
    return list(objects)


identity.meta = FeaturizerMeta(output_kinds=('scalar', 'structured'))


@featurizers.register('identity', meta=identity.meta)
def _identity_factory():
    return identity


@featurizers.register('tfidf', meta=TfidfFeaturizer.meta)
def _tfidf_factory(*, ngram_range=DFLT_NGRAM_RANGE, lowercase=True):
    """Build a char-n-gram TF-IDF featurizer (unfit; ``resolve_featurizer`` fits it)."""
    return TfidfFeaturizer(ngram_range=ngram_range, lowercase=lowercase)


# Dense embedders (optional extras) — registered as lazy factories. The metadata is
# defined once (in text.py) and mirrored here, so the two copies cannot diverge.
featurizers.register('sbert', _text.sbert_featurizer, meta=_text.SBERT_META)
featurizers.register('openai', _text.openai_featurizer, meta=_text.OPENAI_META)


def resolve_featurizer(spec=None, *, corpus=None, **config):
    """Resolve ``spec`` to a ready-to-use batch featurizer.

    - ``None`` -> the default ``'tfidf'``;
    - a registered name (``str``) -> built via its factory (``**config`` is forwarded, so
      an unknown option raises rather than being silently ignored);
    - a callable -> returned as-is (assumed already a batch featurizer).

    A *fittable* featurizer (one exposing ``fit``, e.g. TF-IDF) is fit on ``corpus``
    here — and a corpus is then required: resolving a fittable featurizer without one
    raises immediately (a clear error at the call site) instead of returning a
    half-built object that only fails later inside ``transform``.
    """
    if spec is None:
        spec = 'tfidf'
    if isinstance(spec, str):
        feat = featurizers.create(spec, **config)
    elif callable(spec):
        feat = spec
    else:
        raise TypeError(
            f"featurizer spec must be a registered name or a callable, "
            f"got {type(spec).__name__}"
        )
    fit = getattr(feat, 'fit', None)
    if callable(fit):
        if corpus is None:
            raise ValueError(
                f"featurizer {spec!r} must be fit on a corpus; pass corpus=... "
                f"(e.g. resolve_featurizer({spec!r}, corpus=your_texts))"
            )
        feat = fit(corpus) or feat
    # make the declared metadata reachable off the resolved featurizer
    if isinstance(spec, str) and not hasattr(feat, 'meta'):
        try:
            feat.meta = featurizers.meta(spec)
        except (AttributeError, TypeError):
            pass
    return feat
