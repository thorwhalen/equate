"""The CandidateStore: per left-item top-k candidates with scores — the SSOT for review
and interactive re-optimization.

Retaining the top-k candidates (not just the single best) is what lets a user edit a match
and have the system re-solve from the remembered alternatives (doc 08-interactive). It is a
plain dataclass so it serializes (e.g. persist with ``dol.cache_this`` to resume a session).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.sparse import issparse

__all__ = ["CandidateStore"]


def _ranked(pairs, sense):
    """Sort ``(index, score)`` pairs best-first, deterministically.

    Best score first (per ``sense``); NaN scores sort *last*; ties break by lower index —
    the same ordering for the dense and sparse paths.
    """

    def key(p):
        j, s = p
        if math.isnan(s):
            return (1, 0.0, j)  # NaN candidates go last
        return (
            0,
            -s if sense == "maximize" else s,
            j,
        )  # best first; ties -> lower index

    return [(int(j), float(s)) for j, s in sorted(pairs, key=key)]


@dataclass
class CandidateStore:
    """Per left index ``i``, its top-k ``(right_index, score)`` candidates, best-first.

    Iterating yields the top-1 ``(i, j)`` pairs, so ``dict(store)`` gives the greedy 1:n
    match. ``row_labels`` / ``col_labels`` name the two collections.
    """

    candidates: dict  # i -> [(j, score), ...] best-first
    sense: str = "maximize"
    row_labels: Optional[list] = None
    col_labels: Optional[list] = None

    @classmethod
    def from_scores(
        cls, scores, *, k=5, sense="maximize", row_labels=None, col_labels=None
    ):
        """Build a store of each row's top-``k`` candidates from a (dense or sparse) score
        matrix.

        A **sparse** matrix considers only its stored (candidate) cells — which is exactly
        the blocking semantics (an absent cell is not a candidate), so for a blocked matrix
        this intentionally differs from the dense all-cells behavior regardless of ``sense``.
        """
        candidates: dict = {}
        if issparse(scores):
            csr = scores.tocsr()
            for i in range(csr.shape[0]):
                lo, hi = csr.indptr[i], csr.indptr[i + 1]
                pairs = list(zip(csr.indices[lo:hi].tolist(), csr.data[lo:hi].tolist()))
                candidates[i] = _ranked(pairs, sense)[:k]
        else:
            dense = np.asarray(scores, dtype=float)
            for i in range(dense.shape[0]):
                row = dense[i]
                pairs = [(j, row[j]) for j in range(dense.shape[1])]
                candidates[i] = _ranked(pairs, sense)[:k]
        return cls(candidates, sense, row_labels, col_labels)

    def top(self, i, n=1):
        """The top-``n`` ``(right_index, score)`` candidates for left item ``i``."""
        return self.candidates.get(i, [])[:n]

    def margin(self, i):
        """Confidence gap between item ``i``'s top-1 and top-2 candidate scores.

        A *small* margin means the top choice is uncertain (two close candidates) — the
        signal for review triage and active-learning query selection. An item with 0 or 1
        candidates is unambiguous (no rival to be confused with), so its margin is ``+inf``
        — it sorts LAST in a most-uncertain-first queue, never spuriously to the front.
        """
        c = self.candidates.get(i, [])
        if len(c) >= 2:
            return abs(c[0][1] - c[1][1])
        return float("inf")

    def __iter__(self):
        # top-1 pair per left item -> a greedy 1:n mapping. NOTE: not injective — several
        # left items can share the same best right item (use the match stage for a 1:1
        # assignment).
        for i, c in self.candidates.items():
            if c:
                yield (i, c[0][0])
