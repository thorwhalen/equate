"""Score calibration (stage ②): map raw comparator scores to a decision or a
probability. Calibration is *per comparator* and opt-in — never auto-applied across
different comparators (decision register D2), because their score spaces differ.
"""

from equate._dependencies import require

__all__ = ['threshold', 'platt', 'isotonic']


def threshold(t, *, hard=True):
    """Return a calibrator: ``hard`` -> boolean ``score >= t``; else pass the score through."""

    def cal(score):
        return bool(score >= t) if hard else score

    return cal


def platt(scores, labels):
    """Fit Platt scaling (logistic regression on 1-D scores) -> ``P(match)``.

    Requires ``equate[sklearn]``. Returns a calibrator mapping a raw score to a
    probability.
    """
    import numpy as np

    lm = require('sklearn.linear_model', extra='sklearn', purpose='Platt calibration')
    model = lm.LogisticRegression().fit(np.asarray(scores).reshape(-1, 1), labels)

    def cal(score):
        return float(model.predict_proba([[score]])[0, 1])

    return cal


def isotonic(scores, labels):
    """Fit isotonic regression -> a monotone calibrated probability.

    Requires ``equate[sklearn]``. Returns a calibrator mapping a raw score to a
    probability.
    """
    iso = require('sklearn.isotonic', extra='sklearn', purpose='isotonic calibration')
    model = iso.IsotonicRegression(out_of_bounds='clip').fit(scores, labels)

    def cal(score):
        return float(model.predict([score])[0])

    return cal
