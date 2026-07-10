"""Interactive constraints and re-optimization: every user edit is a constraint, not a
recompute.

A *confirm* forces an edge into the assignment; a *reject* forbids one. :func:`reoptimize`
re-solves the optimal assignment honoring the accumulated constraints — the primitive
behind "edit a match and the system updates the rest from the remembered alternatives"
(doc 08-interactive). The constraint set is append-only so a session's edit history is a
replayable log.
"""

from dataclasses import dataclass, field

from equate.base import to_cost
from equate.interactive.kbest import solve_constrained

__all__ = ['ConstraintSet', 'reoptimize']


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


def reoptimize(scores, constraints=None, *, sense='maximize'):
    """Re-solve the optimal 1:1 assignment honoring ``constraints`` (confirmed/rejected).

    Returns the matched ``(i, j)`` pairs. Raises ``ValueError`` if the constraints are
    infeasible (no valid assignment). This is O(one LAP): the whole point is that an edit
    is a cheap constrained re-solve, not a from-scratch recompute.
    """
    constraints = constraints or ConstraintSet()
    # to_cost handles a sparse (blocked) matrix via _sparse_to_cost, worst-casing absent
    # cells; densifying first would take the dense branch and lose that (a #7-class bug)
    cost = to_cost(scores, sense=sense)
    result = solve_constrained(cost, constraints.forced, constraints.forbidden)
    if result is None:
        raise ValueError("the constraints are infeasible — no valid assignment exists")
    assignment, _ = result
    return sorted(assignment.items())
