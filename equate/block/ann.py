"""ANN / dense-vector blocking: candidate generation by nearest-neighbour search.

A brute-force k-NN (pure numpy, core) is the exact correctness baseline; hnswlib gives a
fast approximate index behind ``equate[ann]``. Both featurize the objects and return each
A-item's nearest B-items. Reusing the *same* embedder for blocking and scoring is the
cheap default; measure PC/RR/PQ to check it isn't capping recall (decision register D6).
"""

import numpy as np

from equate._dependencies import require
from equate._vector import l2_normalize

__all__ = ['brute_knn_blocking', 'ann_blocking']


def _dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X, dtype=float)


def brute_knn_blocking(featurizer='tfidf', *, k=5):
    """Blocker: featurize both sides and emit each A-item's top-``k`` nearest B-items by
    cosine (pure numpy; exact; fine for small/medium collections)."""
    from equate.featurize import resolve_featurizer

    def blocker(A, B=None):
        self_join = B is None or B is A
        A = list(A)
        B = A if self_join else list(B)
        feat = resolve_featurizer(featurizer, corpus=A + B)
        XA = _dense(l2_normalize(feat(A)))
        XB = _dense(l2_normalize(feat(B)))
        sims = XA @ XB.T
        kk = min(k + (1 if self_join else 0), XB.shape[0])
        for i in range(sims.shape[0]):
            for j in np.argsort(-sims[i])[:kk]:
                j = int(j)
                if self_join and j == i:
                    continue  # skip the trivial self-match
                yield (i, j)

    return blocker


def ann_blocking(featurizer='tfidf', *, k=5, ef_construction=200, m=16):
    """Blocker using an hnswlib approximate-NN cosine index — requires ``equate[ann]``."""
    hnswlib = require('hnswlib', extra='ann', purpose='ANN blocking')
    from equate.featurize import resolve_featurizer

    def blocker(A, B=None):
        self_join = B is None or B is A
        A = list(A)
        B = A if self_join else list(B)
        feat = resolve_featurizer(featurizer, corpus=A + B)
        XA = _dense(feat(A)).astype('float32')
        XB = _dense(feat(B)).astype('float32')
        index = hnswlib.Index(space='cosine', dim=XB.shape[1])
        index.init_index(max_elements=len(B), ef_construction=ef_construction, M=m)
        index.add_items(XB, list(range(len(B))))
        kk = min(k + (1 if self_join else 0), len(B))
        labels, _ = index.knn_query(XA, k=kk)
        for i, row in enumerate(labels):
            for j in row:
                j = int(j)
                if self_join and j == i:
                    continue
                yield (i, j)

    return blocker
