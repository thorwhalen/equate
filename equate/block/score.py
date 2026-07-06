"""Score only the blocker's candidate pairs into a sparse ScoreMatrix (decision D1).

This is the seam where block → compare → match meet: the blocker's lazy candidate stream
is scored pairwise into a sparse matrix (the compare→match SSOT), with the cells the
blocker skipped left structurally absent (a non-candidate, which the matcher treats as
worst-case — see :func:`equate.base.to_cost`).
"""

from scipy.sparse import csr_matrix

from equate.base import ScoreMatrix
from equate.compare import resolve_comparator

__all__ = ['score_candidates']


def score_candidates(A, B, candidate_pairs, comparator='ratio', *, sense='maximize'):
    """Score each candidate ``(i, j)`` pair with ``comparator`` -> a sparse ScoreMatrix.

    ``comparator`` is a pairwise comparator (a registered name or a callable). Duplicate
    candidate pairs are collapsed (scored once) so a sparse matrix never sums them.
    """
    A, B = list(A), list(B)
    comp = resolve_comparator(comparator)
    rows, cols, data = [], [], []
    seen = set()
    for i, j in candidate_pairs:
        if (i, j) in seen:
            continue
        seen.add((i, j))
        rows.append(i)
        cols.append(j)
        data.append(float(comp(A[i], B[j])))
    matrix = csr_matrix((data, (rows, cols)), shape=(len(A), len(B)), dtype=float)
    return ScoreMatrix(matrix, sense=sense, row_labels=A, col_labels=B)
