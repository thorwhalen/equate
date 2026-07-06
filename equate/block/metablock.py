"""Meta-blocking (stage ②): refine a candidate set to raise precision without losing
much recall — deduplicate pairs and prune those touching over-connected (uninformative)
items. A modular ``refine(candidates) -> candidates`` post-processor keeps recall-first
blocking cheap and precision recovery optional (doc 02-blocking).
"""

from collections import Counter

__all__ = ['dedupe_pairs', 'prune_frequent']


def dedupe_pairs(candidate_pairs):
    """Yield unique candidate pairs, order-preserving."""
    seen = set()
    for p in candidate_pairs:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            yield t


def prune_frequent(candidate_pairs, *, max_degree):
    """Drop candidate pairs touching an item that appears in more than ``max_degree``
    candidates (block-filtering: over-connected items are usually uninformative).

    Materializes the candidate set (a degree count needs a full pass).
    """
    pairs = [tuple(p) for p in candidate_pairs]
    degree = Counter()
    for i, j in pairs:
        degree[('a', i)] += 1
        degree[('b', j)] += 1
    for i, j in pairs:
        if degree[('a', i)] <= max_degree and degree[('b', j)] <= max_degree:
            yield (i, j)
