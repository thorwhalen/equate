"""The interactive stage: human-in-the-loop review and re-optimization.

The payoff of retaining **top-k candidates with scores** (doc 08-interactive): a user can
edit a match and the system re-solves the rest from the remembered alternatives. The pieces:

- :class:`CandidateStore` — the per-item top-k SSOT (built from a score matrix; serializable);
- :func:`k_best_assignments` — Murty's k-best global matchings, in order (in-house, pure scipy);
- :class:`ConstraintSet` + :func:`reoptimize` — every edit is a force/forbid constraint,
  re-solved with one constrained LAP, not a from-scratch recompute;
- :func:`review_queue` / :func:`uncertainty_sampling` — rank items by confidence margin for
  human review and active-learning query selection.

Structured explanations use :class:`equate.base.Explanation`. Heavier active-learning
backends (modAL / scikit-activeml) are optional (``equate[active]``); the margin-based
query strategy here needs no extra.
"""

from equate.interactive.kbest import solve_constrained, k_best_assignments
from equate.interactive.candidate_store import CandidateStore
from equate.interactive.constraints import ConstraintSet, reoptimize
from equate.interactive.review import review_queue, uncertainty_sampling

__all__ = [
    'CandidateStore',
    'k_best_assignments',
    'solve_constrained',
    'ConstraintSet',
    'reoptimize',
    'review_queue',
    'uncertainty_sampling',
]
