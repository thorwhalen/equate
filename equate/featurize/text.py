"""Dense text-embedding featurizers, behind optional extras.

Each is a *lazily-built* strategy: the heavy library is imported only when the
strategy is actually constructed, via :func:`equate._dependencies.require`, so a
missing extra raises a friendly ``MissingDependencyError`` instead of breaking
``import equate``. A representative set is provided here; more embedders (bge-m3, e5,
nomic, cohere, voyage, and image/audio modalities) are added the same way as their
stages land — see the ``equate-dev-add-strategy`` skill.
"""

from equate.base import FeaturizerMeta
from equate._dependencies import require

__all__ = ['sbert_featurizer', 'openai_featurizer']


def sbert_featurizer(*, model='sentence-transformers/all-MiniLM-L6-v2', **_):
    """Sentence-Transformers (SBERT) embedder — requires ``equate[embeddings]``.

    Returns a batch featurizer ``texts -> L2-normalized embedding matrix``.
    """
    st = require(
        'sentence_transformers', extra='embeddings', purpose='SBERT text embeddings'
    )
    model_obj = st.SentenceTransformer(model)

    def featurize(texts):
        return model_obj.encode(list(texts), normalize_embeddings=True)

    featurize.meta = FeaturizerMeta(
        output_kinds=('vector',), normalize=True, license='Apache-2.0'
    )
    return featurize


def openai_featurizer(*, model='text-embedding-3-small', **_):
    """OpenAI embeddings — requires ``equate[api]`` and an ``OPENAI_API_KEY``.

    Returns a batch featurizer ``texts -> embedding matrix``.
    """
    openai = require('openai', extra='api', purpose='OpenAI text embeddings')
    client = openai.OpenAI()

    def featurize(texts):
        import numpy as np

        resp = client.embeddings.create(model=model, input=list(texts))
        return np.array([d.embedding for d in resp.data])

    featurize.meta = FeaturizerMeta(
        output_kinds=('vector',), normalize=False, license='proprietary'
    )
    return featurize
