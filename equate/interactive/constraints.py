"""Interactive constraints and re-optimization: every user edit is a constraint, not a
recompute.

A *confirm* forces an edge into the assignment; a *reject* forbids one. :func:`reoptimize`
re-solves the optimal assignment honoring the accumulated constraints — the primitive
behind "edit a match and the system updates the rest from the remembered alternatives"
(doc 08-interactive). The constraint set is append-only so a session's edit history is a
replayable log.
"""

from dataclasses import dataclass, field

from equate.base import ScoreMatrix
from equate.interactive.kbest import solve_constrained

__all__ = ["ConstraintSet", "reoptimize"]


@dataclass
class ConstraintSet:
    """Append-only record of user edits: ``forced`` (confirmed) and ``forbidden``
    (rejected) ``(i, j)`` edges."""

    forced: list = field(default_factory=list)
    forbidden: list = field(default_factory=list)

    def confirm(self, i, j):
        """Force edge ``(i, j)`` into the assignment (a user-confirmed match)."""
        self.forced.append((i, j))
        return self

    def reject(self, i, j):
        """Forbid edge ``(i, j)`` from the assignment (a user-rejected match)."""
        self.forbidden.append((i, j))
        return self


def reoptimize(scores, constraints=None, *, sense="maximize"):
    """Re-solve the optimal 1:1 assignment honoring ``constraints`` (confirmed/rejected).

    Returns the matched ``(i, j)`` pairs. A rejected pair is a *hole* (a non-candidate, the
    blocked-matching semantics of D11), not an ``inf`` cost: the re-solve matches as many
    non-rejected pairs as possible, then the best score, so rejecting every candidate of an
    item leaves it unmatched. Raises ``ValueError`` only if the constraints contradict each
    other (the same pair both confirmed and rejected, or two confirms sharing a row or
    column). This is O(one LAP): the whole point is that an edit is a cheap constrained
    re-solve, not a from-scratch recompute.
    """
    constraints = constraints or ConstraintSet()
    forced = list(constraints.forced)
    if set(forced) & set(constraints.forbidden):
        raise ValueError("the constraints are infeasible — a pair is both confirmed and rejected")
    # dense_cost worst-cases a sparse (blocked) matrix's absent cells (never densify first
    # — that loses the worst-casing, a #7-class bug); D11
    sm = ScoreMatrix.coerce(scores, sense=sense)
    if constraints.forbidden:
        mask = sm.candidate_mask()
        for i, j in constraints.forbidden:
            mask[i, j] = False
        sm = sm.restricted(mask)
    result = solve_constrained(sm.dense_cost(), forced, [])
    if result is None:
        raise ValueError("the constraints are infeasible — no valid assignment exists")
    assignment, _ = result
    # a full LAP can land on a non-candidate hole (worst-cased, never *preferred*, but still
    # assignable in forced-partial / all-absent cases); drop those, but keep user-forced edges
    return sorted(sm.drop_holes(assignment.items(), keep=forced))
