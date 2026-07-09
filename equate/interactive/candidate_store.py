"""The CandidateStore: per left-item top-k candidates with scores — the SSOT for review
and interactive re-optimization.

Retaining the top-k candidates (not just the single best) is what lets a user edit a match
and have the system re-solve from the remembered alternatives (doc 08-interactive). It is a
plain dataclass so it serializes (e.g. persist with ``dol.cache_this`` to resume a session).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.sparse import issparse

__all__ = ['CandidateStore']


@dataclass
class CandidateStore:
    """Per left index ``i``, its top-k ``(right_index, score)`` candidates, best-first.

    Iterating yields the top-1 ``(i, j)`` pairs, so ``dict(store)`` gives the greedy 1:n
    match. ``row_labels`` / ``col_labels`` name the two collections.
    """

    candidates: dict  # i -> [(j, score), ...] best-first
    sense: str = 'maximize'
    row_labels: Optional[list] = None
    col_labels: Optional[list] = None

    @classmethod
    def from_scores(cls, scores, *, k=5, sense='maximize', row_labels=None, col_labels=None):
        """Build a store of each row's top-``k`` candidates from a (dense or sparse) score
        matrix. A sparse matrix considers only its stored (candidate) cells."""
        candidates: dict = {}
        if issparse(scores):
            csr = scores.tocsr()
            for i in range(csr.shape[0]):
                lo, hi = csr.indptr[i], csr.indptr[i + 1]
                pairs = list(zip(csr.indices[lo:hi].tolist(), csr.data[lo:hi].tolist()))
                pairs.sort(key=lambda p: p[1], reverse=(sense == 'maximize'))
                candidates[i] = [(int(j), float(s)) for j, s in pairs[:k]]
        else:
            dense = np.asarray(scores, dtype=float)
            for i in range(dense.shape[0]):
                row = dense[i]
                order = np.argsort(row)
                if sense == 'maximize':
                    order = order[::-1]
                candidates[i] = [(int(j), float(row[j])) for j in order[:k]]
        return cls(candidates, sense, row_labels, col_labels)

    def top(self, i, n=1):
        """The top-``n`` ``(right_index, score)`` candidates for left item ``i``."""
        return self.candidates.get(i, [])[:n]

    def margin(self, i):
        """Confidence margin between item ``i``'s top-1 and top-2 candidate scores.

        A small margin means the top choice is uncertain (two close candidates) — the
        signal for review triage and active-learning query selection. Items with a single
        candidate return that candidate's score; items with none return 0.0.
        """
        c = self.candidates.get(i, [])
        if len(c) >= 2:
            return abs(c[0][1] - c[1][1])
        return c[0][1] if c else 0.0

    def __iter__(self):
        for i, c in self.candidates.items():
            if c:
                yield (i, c[0][0])
