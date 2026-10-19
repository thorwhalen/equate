"""How hard is a benchmark? A measured difficulty profile to print next to a score.

The popular matching benchmarks are mostly *easy*: a single similarity threshold already
separates their matches from their non-matches (Papadakis et al. 2023, "degree of
linearity"; Primpeli & Bizer 2020 — research doc 14, §7.6). So a 98% F1 means little
until you know what a trivial matcher gets. :func:`linearity` is that trivial matcher's
best case: the highest pairwise F1 any **one** threshold on TF-IDF cosine similarity
reaches over **all** ``a × b`` pairs. Near 1 the benchmark cannot tell matchers apart;
well below it, it can.

It is adapted from Papadakis et al., who threshold the *labelled* pairs of a split; over
the full cross product it also charges the threshold for every non-candidate pair, which
is the setting :func:`~equate.evaluate.benchmark_matcher` scores.
"""

from collections.abc import Sequence

import numpy as np

__all__ = ["linearity", "difficulty_profile", "difficulty_label"]

#: linearity at or above which a benchmark is "easy" / "moderate" (else "hard")
DFLT_EASY_AT = 0.9
DFLT_MODERATE_AT = 0.7


def linearity(a: Sequence, b: Sequence, gold) -> float:
    """Best pairwise F1 of a single threshold on TF-IDF cosine similarity over ``a × b``.

    Records (mappings) are compared through their joined values. Ties are cut together
    (a threshold cannot split equal scores). ``0.0`` when there is nothing to find.

    >>> round(linearity(['apple pie', 'kiwi'], ['kiwi', 'apple pie'], {(0, 1), (1, 0)}), 2)
    1.0
    """
    gold = {(int(i), int(j)) for i, j in gold}
    if not gold or not len(a) or not len(b):
        return 0.0
    from equate.facade import score_matrix
    from equate.featurize.records import record_text

    S = np.asarray(
        score_matrix([record_text(x) for x in a], [record_text(x) for x in b]).data,
        dtype=float,
    )
    m = S.shape[1]
    flat = S.ravel()
    is_gold = np.zeros(flat.shape, dtype=bool)
    is_gold[[i * m + j for i, j in gold]] = True
    order = np.argsort(-flat, kind="stable")
    tp = np.cumsum(is_gold[order])
    k = np.arange(1, flat.size + 1)
    f1 = 2 * tp / (k + len(gold))
    ranked = flat[order]
    cut = np.r_[ranked[1:] != ranked[:-1], True]  # last position of each tie group
    return float(f1[cut].max())


def difficulty_label(
    lin: float, *, easy_at: float = DFLT_EASY_AT, moderate_at: float = DFLT_MODERATE_AT
) -> str:
    """``'easy'`` / ``'moderate'`` / ``'hard'`` from a :func:`linearity` value."""
    if lin >= easy_at:
        return "easy"
    if lin >= moderate_at:
        return "moderate"
    return "hard"


def difficulty_profile(a: Sequence, b: Sequence, gold) -> dict:
    """``{'linearity', 'level', 'base_rate', 'n_a', 'n_b', 'n_gold'}`` for a matching task."""
    lin = linearity(a, b, gold)
    n_a, n_b, n_gold = len(a), len(b), len(set(gold))
    return {
        "linearity": round(lin, 4),
        "level": difficulty_label(lin),
        "base_rate": n_gold / (n_a * n_b) if n_a and n_b else 0.0,
        "n_a": n_a,
        "n_b": n_b,
        "n_gold": n_gold,
    }
