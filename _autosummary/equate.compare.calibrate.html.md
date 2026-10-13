# equate.compare.calibrate

Score calibration (stage ②): map raw comparator scores to a decision or a
probability. Calibration is *per comparator* and opt-in — never auto-applied across
different comparators (decision register D2), because their score spaces differ.

### Functions

| [`threshold`](#equate.compare.calibrate.threshold)(t, \*[, hard])   | Return a calibrator: `hard` -> boolean `score >= t`; else pass the score through.   |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`platt`](#equate.compare.calibrate.platt)(scores, labels)      | Fit Platt scaling (logistic regression on 1-D scores) -> `P(match)`.                |
| [`isotonic`](#equate.compare.calibrate.isotonic)(scores, labels)   | Fit isotonic regression -> a monotone calibrated probability.                       |

### equate.compare.calibrate.isotonic(scores, labels)

Fit isotonic regression -> a monotone calibrated probability.

Requires `equate[sklearn]`. Returns a calibrator mapping a raw score to a
probability.

### equate.compare.calibrate.platt(scores, labels)

Fit Platt scaling (logistic regression on 1-D scores) -> `P(match)`.

Requires `equate[sklearn]`. Returns a calibrator mapping a raw score to a
probability.

### equate.compare.calibrate.threshold(t, , hard=True)

Return a calibrator: `hard` -> boolean `score >= t`; else pass the score through.
