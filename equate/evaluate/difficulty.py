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
#: the largest a × b that :func:`linearity` scores densely: peak memory is ~130 bytes per
#: pair (the TF-IDF scoring dominates), so 10M pairs is ~1.3 GB; Amazon-Google is 4.4M
DFLT_MAX_PAIRS = 10_000_000


def linearity(
    a: Sequence, b: Sequence, gold, *, max_pairs: int = DFLT_MAX_PAIRS
) -> float:
    """Best pairwise F1 of a single threshold on TF-IDF cosine similarity over ``a × b``.

    Records (mappings) are compared through their joined values. Ties are cut together
    (a threshold cannot split equal scores). ``0.0`` when there is nothing to find.
    Every pair is scored densely (peak ~130 bytes per pair), so inputs over ``max_pairs``
    pairs raise ``ValueError`` rather than exhaust memory; raise the limit knowingly.

    >>> round(linearity(['apple pie', 'kiwi'], ['kiwi', 'apple pie'], {(0, 1), (1, 0)}), 2)
    1.0
    """
    gold = {(int(i), int(j)) for i, j in gold}
    n, m = len(a), len(b)
    if not gold or not n or not m:
        return 0.0
    if n * m > max_pairs:
        raise ValueError(
            f"{n} x {m} = {n * m} pairs is over max_pairs={max_pairs}: linearity "
            f"scores every pair densely"
        )
    from equate.facade import score_matrix
    from equate.featurize.records import record_text

    flat = np.asarray(
        score_matrix([record_text(x) for x in a], [record_text(x) for x in b]).data,
        dtype=float,
    ).ravel()
    gold_scores = np.sort(flat[[i * m + j for i, j in gold]])
    ranked = np.sort(flat)[::-1]  # descending
    # cut after the last element of each tie group: k pairs at or above that score
    k = np.flatnonzero(np.r_[ranked[1:] != ranked[:-1], True]) + 1
    thresholds = ranked[k - 1]
    tp = len(gold_scores) - np.searchsorted(gold_scores, thresholds, side="left")
    return float((2 * tp / (k + len(gold_scores))).max())


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
