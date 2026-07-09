"""Human review triage and active-learning query selection.

Both rank items by *uncertainty* — the confidence margin between an item's top-1 and top-2
candidates (small margin = an ambiguous, review-worthy decision). :func:`review_queue`
orders items for a verification UI; :func:`uncertainty_sampling` picks the most-informative
items for an oracle to label (active learning). An ``Oracle`` is any callable
``(i, j) -> bool`` (a human, an LLM, or a crowd) — kept as a plain protocol, not a class.
"""

__all__ = ['review_queue', 'uncertainty_sampling']


def review_queue(candidate_store, *, most_uncertain_first=True):
    """Order the left items of a :class:`~equate.interactive.CandidateStore` for review.

    By default the most-uncertain items (smallest top-1/top-2 margin) come first — the
    decisions a human should look at soonest.
    """
    items = list(candidate_store.candidates)
    return sorted(items, key=candidate_store.margin, reverse=not most_uncertain_first)


def uncertainty_sampling(candidate_store, *, n=10):
    """Active-learning query: the ``n`` most uncertain items to label next.

    Uncertainty sampling — label the items whose top choice is least certain (smallest
    margin), since those labels are the most informative for training a matcher.
    """
    return review_queue(candidate_store, most_uncertain_first=True)[:n]
