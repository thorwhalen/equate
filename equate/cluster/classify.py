"""Classification (the optional decide stage): map a comparison score to a decision.

The Fellegi-Sunter model turns per-field agreement into a match *weight* (log-odds; the
combiner lives in :mod:`equate.compare.vectorize`, decision register D5) and then a
*three-way* decision — match / possible-match / non-match — via two thresholds. The
possible-match band is the abstain region that feeds human review (roadmap #9). The m- and
u-probabilities can come from labels or be estimated unsupervised by EM here. The default
matching path stays score-only; classification is opt-in.
"""

import math

__all__ = ['classify', 'estimate_mu_em']


def classify(weight, *, lower, upper):
    """Three-way Fellegi-Sunter decision from a match ``weight`` and two thresholds.

    ``weight >= upper`` -> ``'match'``; ``weight <= lower`` -> ``'non_match'``; between ->
    ``'possible_match'`` (the clerical-review / abstain band). Requires ``lower <= upper``.

    >>> [classify(w, lower=-1.0, upper=1.0) for w in (2.0, 0.0, -3.0)]
    ['match', 'possible_match', 'non_match']
    """
    if math.isnan(weight) or math.isnan(lower) or math.isnan(upper):
        raise ValueError("classify received a NaN weight or threshold")
    if lower > upper:
        raise ValueError(f"lower ({lower}) must be <= upper ({upper})")
    if weight >= upper:
        return 'match'
    if weight <= lower:
        return 'non_match'
    return 'possible_match'


def estimate_mu_em(comparison_vectors, *, iters=50, agree_threshold=0.5, eps=1e-6):
    """Unsupervised EM estimate of per-field m- and u-probabilities (a 2-class mixture).

    ``comparison_vectors`` is an iterable of ``{field: agreement}`` (agreement in [0, 1]);
    each is binarized at ``agree_threshold``. A field *missing* from a record contributes
    no evidence for that record (it is skipped, not treated as a disagreement). Returns
    ``(m_probs, u_probs)`` dicts suitable for :func:`equate.compare.vectorize.fellegi_sunter`;
    m > u for a discriminating field when the data has genuine match/non-match structure.
    """
    cvs = [
        {f: (1.0 if s >= agree_threshold else 0.0) for f, s in cv.items()}
        for cv in comparison_vectors
    ]
    fields = sorted({f for cv in cvs for f in cv})
    if not cvs or not fields:
        return {}, {}

    m = {f: 0.9 for f in fields}
    u = {f: 0.1 for f in fields}
    p = 0.1  # prior probability of a match

    def _clamp(x):
        return min(1.0 - eps, max(eps, x))

    for _ in range(iters):
        posteriors = []
        for cv in cvs:
            log_m = math.log(_clamp(p))
            log_u = math.log(_clamp(1.0 - p))
            for f, a in cv.items():  # only the fields actually present in this record
                log_m += a * math.log(m[f]) + (1.0 - a) * math.log(1.0 - m[f])
                log_u += a * math.log(u[f]) + (1.0 - a) * math.log(1.0 - u[f])
            hi = max(log_m, log_u)
            pm, pu = math.exp(log_m - hi), math.exp(log_u - hi)
            posteriors.append(pm / (pm + pu))

        p = sum(posteriors) / len(cvs)
        for f in fields:
            # only records where the field is present count toward its m/u
            present = [(w, cv[f]) for w, cv in zip(posteriors, cvs) if f in cv]
            wm = sum(w for w, _ in present) or eps
            wu = sum(1.0 - w for w, _ in present) or eps
            m[f] = _clamp(sum(w * a for w, a in present) / wm)
            u[f] = _clamp(sum((1.0 - w) * a for w, a in present) / wu)
    return m, u
