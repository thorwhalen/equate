"""Pure-numpy/scipy character-n-gram TF-IDF — equate's zero-heavy-dependency default
text featurizer (decision register D9). No scikit-learn or grub required.

Character n-grams make it robust to typos and morphology, which suits fuzzy name/SKU
matching; the output vectors are L2-normalized, so cosine similarity is a dot product.
"""

import math
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix

from equate.base import FeaturizerMeta
from equate._vector import l2_normalize

__all__ = ["TfidfFeaturizer", "mk_tfidf", "char_ngrams", "DFLT_NGRAM_RANGE"]

DFLT_NGRAM_RANGE = (2, 4)


def char_ngrams(text, ngram_range=DFLT_NGRAM_RANGE, *, lowercase=True):
    """Character n-grams of ``text`` over the inclusive ``ngram_range``.

    The whole string is padded with one leading and one trailing space (so prefixes and
    suffixes are captured), and n-grams slide over the entire padded string — they *may
    therefore span the spaces between words*. This is deliberate (it helps fuzzy
    name/SKU matching) and differs from scikit-learn's per-word ``char_wb`` analyzer. A
    text too short for any ``n`` in the range falls back to the padded string itself, so
    tiny or empty inputs still yield a (single) feature rather than an all-zero vector.

    >>> char_ngrams('cat', (2, 3))
    [' c', 'ca', 'at', 't ', ' ca', 'cat', 'at ']
    >>> char_ngrams('', (3, 4))          # too short for any n: falls back to the padding
    ['  ']
    """
    if lowercase:
        text = text.lower()
    padded = f" {text} "
    lo, hi = ngram_range
    grams = []
    for n in range(lo, hi + 1):
        if len(padded) >= n:
            grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams or [padded]


class TfidfFeaturizer:
    """Fittable char-n-gram TF-IDF featurizer producing L2-normalized sparse vectors.

    Fit on a corpus to learn the vocabulary and inverse document frequencies, then
    transform texts to vectors. A fitted instance is itself a batch featurizer, so
    ``vecs = featurizer(texts)`` works.
    """

    #: declared metadata (see equate.base.FeaturizerMeta)
    meta = FeaturizerMeta(output_kinds=("vector",), normalize=True, license="core")

    def __init__(self, *, ngram_range=DFLT_NGRAM_RANGE, lowercase=True):
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.vocabulary_ = None
        self.idf_ = None

    def _grams(self, text):
        return char_ngrams(text, self.ngram_range, lowercase=self.lowercase)

    def fit(self, texts):
        texts = list(texts)
        vocab: dict = {}
        df: dict = {}
        for t in texts:
            for g in set(self._grams(t)):  # document frequency counts each gram once
                if g not in vocab:
                    vocab[g] = len(vocab)
                df[g] = df.get(g, 0) + 1
        n_docs = len(texts)
        idf = np.ones(len(vocab), dtype=float)
        for g, j in vocab.items():
            idf[j] = math.log((1 + n_docs) / (1 + df[g])) + 1.0  # smoothed idf
        self.vocabulary_ = vocab
        self.idf_ = idf
        return self

    def transform(self, texts):
        if self.vocabulary_ is None:
            raise RuntimeError("TfidfFeaturizer must be fit before transform")
        texts = list(texts)
        vocab = self.vocabulary_
        rows, cols, data = [], [], []
        for r, t in enumerate(texts):
            counts: dict = {}
            for g in self._grams(t):
                j = vocab.get(g)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(r)
                cols.append(j)
                data.append(c * self.idf_[j])
        X = csr_matrix(
            (data, (rows, cols)), shape=(len(texts), len(vocab)), dtype=float
        )
        return l2_normalize(X)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    def __call__(self, texts):
        return self.transform(texts)


def mk_tfidf(*learn_texts, ngram_range=DFLT_NGRAM_RANGE, lowercase=True):
    """Fit a char-n-gram TF-IDF on the given collection(s) and return the fitted,
    callable batch featurizer — the pure numpy/scipy replacement for the legacy
    grub-backed ``equate.util.mk_text_to_vect``.
    """
    corpus = list(chain.from_iterable(learn_texts))
    return TfidfFeaturizer(ngram_range=ngram_range, lowercase=lowercase).fit(corpus)
