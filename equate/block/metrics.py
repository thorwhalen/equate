"""Blocking evaluation metrics (stage ②): quantify the recall/efficiency trade-off of a
candidate set. Pair completeness (blocking recall) upper-bounds the whole system's
recall, so it is the metric to protect (doc 02-blocking).
"""

__all__ = ['blocking_metrics']


def blocking_metrics(candidate_pairs, true_pairs, *, n_a, n_b=None):
    """Return ``{pair_completeness, reduction_ratio, pairs_quality, n_candidates}``.

    - **pair_completeness (PC)** = ``|candidates ∩ true| / |true|`` — blocking *recall*
      (upper-bounds end-to-end recall);
    - **reduction_ratio (RR)** = ``1 - |candidates| / |all pairs|`` — how much blocking pruned;
    - **pairs_quality (PQ)** = ``|candidates ∩ true| / |candidates|`` — blocking *precision*.

    ``n_b=None`` means a self-join (dedup) over ``n_a`` items, so
    ``|all pairs| = n_a*(n_a-1)/2``; otherwise ``|all pairs| = n_a*n_b``.

    >>> m = blocking_metrics([(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)], n_a=3, n_b=3)
    >>> round(m['pair_completeness'], 2), round(m['pairs_quality'], 2)
    (0.67, 1.0)
    """
    cand = {tuple(p) for p in candidate_pairs}
    true = {tuple(p) for p in true_pairs}
    total = n_a * (n_a - 1) // 2 if n_b is None else n_a * n_b
    tp = len(cand & true)
    return {
        'pair_completeness': tp / len(true) if true else 1.0,
        'reduction_ratio': 1.0 - len(cand) / total if total else 0.0,
        'pairs_quality': tp / len(cand) if cand else 0.0,
        'n_candidates': len(cand),
    }
