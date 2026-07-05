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
def _identity_factory(corpus=None, **_):
    return identity


@featurizers.register('tfidf', meta=TfidfFeaturizer.meta)
def _tfidf_factory(corpus=None, *, ngram_range=DFLT_NGRAM_RANGE, lowercase=True):
    """Build a char-n-gram TF-IDF featurizer, fit on ``corpus`` when one is given."""
    feat = TfidfFeaturizer(ngram_range=ngram_range, lowercase=lowercase)
    if corpus is not None:
        feat.fit(corpus)
    return feat


# Dense embedders (optional extras) — registered as lazy factories.
featurizers.register(
    'sbert',
    _text.sbert_featurizer,
    meta=FeaturizerMeta(output_kinds=('vector',), normalize=True, license='Apache-2.0'),
)
featurizers.register(
    'openai',
    _text.openai_featurizer,
    meta=FeaturizerMeta(output_kinds=('vector',), license='proprietary'),
)


def resolve_featurizer(spec=None, *, corpus=None):
    """Resolve ``spec`` to a ready-to-use batch featurizer.

    - ``None`` -> the default ``'tfidf'`` (fit on ``corpus``);
    - a registered name (``str``) -> built via its factory (fit on ``corpus`` if it is a
      corpus-dependent featurizer);
    - a callable -> returned as-is (assumed already a batch featurizer).
    """
    if spec is None:
        spec = 'tfidf'
    if isinstance(spec, str):
        return featurizers.create(spec, corpus=corpus)
    return featurizers.resolve(spec)
