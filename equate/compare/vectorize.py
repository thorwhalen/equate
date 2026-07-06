"""Multi-field record comparison (stage ②): build a per-pair *comparison vector* from
per-field comparators, then reduce it to a single score with a combiner.

This is the record-matching bridge: the same field comparators can feed a raw score, a
probabilistic Fellegi-Sunter weight, or (later) a learned classifier interchangeably —
the "key" architectural move of decision register D5.
"""

import math
from collections.abc import Mapping

__all__ = [
    'comparison_vector',
    'weighted_sum',
    'mean',
    'max_combiner',
    'fellegi_sunter',
]


def _get(record, field):
    if isinstance(record, Mapping):
        return record.get(field)
    return getattr(record, field, None)


def comparison_vector(a, b, field_comparators):
    """Return ``{field: comparator(a[field], b[field])}`` for each field comparator.

    ``field_comparators`` maps a field name to a pairwise comparator; records may be
    mappings or objects (attribute access).

    >>> from equate.compare.string import levenshtein
    >>> cv = comparison_vector({'name': 'jon'}, {'name': 'john'}, {'name': levenshtein})
    >>> round(cv['name'], 2)
    0.75
    """
    return {f: cmp(_get(a, f), _get(b, f)) for f, cmp in field_comparators.items()}


def weighted_sum(weights=None):
    """Combiner: weighted sum of a comparison vector (default weight 1.0 per field)."""
    weights = weights or {}

    def combine(cv):
        return sum(weights.get(f, 1.0) * s for f, s in cv.items())

    return combine


def mean(cv):
    """Combiner: arithmetic mean of a comparison vector."""
    return sum(cv.values()) / len(cv) if cv else 0.0


def max_combiner(cv):
    """Combiner: maximum field score in a comparison vector."""
    return max(cv.values(), default=0.0)


def fellegi_sunter(m_probs, u_probs):
    """Fellegi-Sunter combiner (decision register D5): sum of per-field match weights.

    For each field with agreement ``s`` in [0, 1], contributes
    ``s*log(m/u) + (1-s)*log((1-m)/(1-u))`` — the log-likelihood-ratio "match weight".
    ``m_probs`` / ``u_probs`` are the m- and u-probabilities per field (from labels or
    EM; the EM estimation lives in the classify stage, roadmap #8). Returns a log-odds
    score (higher = more likely a match).
    """

    def combine(cv):
        total = 0.0
        for f, s in cv.items():
            m = m_probs[f]
            u = u_probs[f]
            total += s * math.log(m / u) + (1.0 - s) * math.log((1.0 - m) / (1.0 - u))
        return total

    return combine
