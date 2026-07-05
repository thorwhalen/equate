"""Vector normalization and cosine similarity — pure numpy/scipy.

These back equate's default (vector) compare path so it needs no scikit-learn:
``cosine_similarity`` is a drop-in for ``sklearn.metrics.pairwise.cosine_similarity``.
"""

import numpy as np
from scipy.sparse import issparse, diags

__all__ = ['l2_normalize', 'cosine_similarity']


def l2_normalize(X):
    """Row-wise L2-normalize a dense ndarray or scipy sparse matrix (zero rows stay zero)."""
    if issparse(X):
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return diags(1.0 / norms) @ X
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def cosine_similarity(X, Y=None):
    """Cosine similarity between the rows of ``X`` and the rows of ``Y`` (default ``Y=X``).

    Pure numpy/scipy replacement for ``sklearn.metrics.pairwise.cosine_similarity``;
    accepts dense or sparse inputs and returns a dense ``ndarray``.

    >>> import numpy as np
    >>> cosine_similarity(np.array([[1.0, 0.0], [0.0, 1.0]])).tolist()
    [[1.0, 0.0], [0.0, 1.0]]
    """
    if Y is None:
        Y = X
    prod = l2_normalize(X) @ l2_normalize(Y).T
    if issparse(prod):
        prod = prod.toarray()
    return np.asarray(prod)
