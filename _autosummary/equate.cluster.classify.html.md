# equate.cluster.classify

Classification (the optional decide stage): map a comparison score to a decision.

The Fellegi-Sunter model turns per-field agreement into a match *weight* (log-odds; the
combiner lives in [`equate.compare.vectorize`](equate.compare.vectorize.html.md#module-equate.compare.vectorize), decision register D5) and then a
*three-way* decision — match / possible-match / non-match — via two thresholds. The
possible-match band is the abstain region that feeds human review (roadmap #9). The m- and
u-probabilities can come from labels or be estimated unsupervised by EM here. The default
matching path stays score-only; classification is opt-in.

### Functions

| [`classify`](#equate.cluster.classify.classify)(weight, \*, lower, upper)            | Three-way Fellegi-Sunter decision from a match `weight` and two thresholds.       |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`estimate_mu_em`](#equate.cluster.classify.estimate_mu_em)(comparison_vectors, \*[, ...]) | Unsupervised EM estimate of per-field m- and u-probabilities (a 2-class mixture). |

### equate.cluster.classify.classify(weight, , lower, upper)

Three-way Fellegi-Sunter decision from a match `weight` and two thresholds.

`weight >= upper` -> `'match'`; `weight <= lower` -> `'non_match'`; between ->
`'possible_match'` (the clerical-review / abstain band). Requires `lower <= upper`.

```pycon
>>> [classify(w, lower=-1.0, upper=1.0) for w in (2.0, 0.0, -3.0)]
['match', 'possible_match', 'non_match']
```

### equate.cluster.classify.estimate_mu_em(comparison_vectors, , iters=50, agree_threshold=0.5, eps=1e-06)

Unsupervised EM estimate of per-field m- and u-probabilities (a 2-class mixture).

`comparison_vectors` is an iterable of `{field: agreement}` (agreement in [0, 1]);
each is binarized at `agree_threshold`. A field *missing* from a record contributes
no evidence for that record (it is skipped, not treated as a disagreement). Returns
`(m_probs, u_probs)` dicts suitable for [`equate.compare.vectorize.fellegi_sunter()`](equate.compare.vectorize.html.md#equate.compare.vectorize.fellegi_sunter);
m > u for a discriminating field when the data has genuine match/non-match structure.
