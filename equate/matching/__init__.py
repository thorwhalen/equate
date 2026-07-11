"""The match stage (③): turn a score matrix into a correspondence.

A *Matcher* is a strategy over a score structure — the choice is the **objective**, not
the algorithm (decision register D4). Select by name via the :data:`matchers` registry or
pass a callable ``(scores, *, sense) -> pairs``. The optimal matcher is sparse-aware
(routes a blocked matrix to the sparse LAP); soft matching (optimal transport) lives in a
parallel :func:`soft_match` / :func:`harden` seam.
"""

import inspect

from equate.base import ScoreMatrix, scorematrix_matcher
from equate.registry import Registry
from equate.matching.assign import optimal_matching, greedy_matching, stable_matching
from equate.matching.soft import soft_match, harden

__all__ = [
    "matchers",
    "resolve_matcher",
    "optimal_matching",
    "greedy_matching",
    "stable_matching",
    "soft_match",
    "harden",
]

#: the match-stage strategy registry (name -> lazy factory)
matchers = Registry("matcher")


def _const(fn, name):
    """Factory returning a fixed matcher; passing config is a clear error."""

    def factory(**config):
        if config:
            raise TypeError(
                f"matcher {name!r} takes no configuration; got {list(config)}"
            )
        return fn

    return factory


@scorematrix_matcher
def _max_weight_matcher(scores, *, sense=None):
    """Maximum-weight bipartite matching via networkx — requires ``equate[graph]``.

    Feeds :meth:`~equate.base.ScoreMatrix.dense_similarity` (holes strictly worst, so an
    absent cell is never a preferred edge) and drops any hole assignment forced by
    max-cardinality via :meth:`~equate.base.ScoreMatrix.candidate_mask`.
    """
    from equate.util import maximal_matching

    sm = ScoreMatrix.coerce(scores, sense=sense)
    return sm.drop_holes(list(maximal_matching(sm.dense_similarity())))


@scorematrix_matcher
def _kuhn_munkres_matcher(scores, *, sense=None):
    """Hungarian assignment via networkx — requires ``equate[graph]``.

    Feeds :meth:`~equate.base.ScoreMatrix.dense_cost` (absent cells worst-cased) as an
    already-minimized cost and drops any hole assignment via
    :meth:`~equate.base.ScoreMatrix.candidate_mask` — the sparse-safe counterpart to the
    old densify-then-``to_cost`` path (which silently lost the worst-casing; D11).
    """
    from equate.util import kuhn_munkres_matching

    sm = ScoreMatrix.coerce(scores, sense=sense)
    return sm.drop_holes(list(kuhn_munkres_matching(sm.dense_cost(), sense="minimize")))


matchers.register("optimal", _const(optimal_matching, "optimal"))
matchers.register("hungarian", _const(optimal_matching, "hungarian"))  # alias
matchers.register("greedy", _const(greedy_matching, "greedy"))
matchers.register("stable", _const(stable_matching, "stable"))
matchers.register("max_weight", _const(_max_weight_matcher, "max_weight"))
matchers.register("kuhn_munkres", _const(_kuhn_munkres_matcher, "kuhn_munkres"))

#: facade ``how=`` objective aliases -> registered matcher names
_HOW_ALIASES = {"assign": "optimal"}


def _is_legacy_matcher(matcher) -> bool:
    """Whether ``matcher`` uses the legacy raw-array ``(scores, *, sense)`` contract.

    **Native means explicitly opted in** via :func:`~equate.base.scorematrix_matcher`;
    everything else is legacy and is handed a pre-worst-cased dense array. This is the
    safe default: every pre-existing user matcher followed the raw-array contract (in any
    of its legal shapes — ``sense=`` keyword, ``**kwargs``, a differently-named direction
    arg, or no direction at all), so signature-sniffing for a literal ``sense`` parameter
    would misclassify those as native and hand them a ``ScoreMatrix`` they cannot read. A
    new ScoreMatrix-consuming matcher marks itself; nothing else changes contract.
    """
    return not getattr(matcher, "_scorematrix_native", False)


def _accepts_sense(matcher) -> bool:
    """Whether a legacy matcher can be handed ``sense=`` (has the param or ``**kwargs``).

    Only decides *how* to call an already-classified legacy matcher — never native-vs-legacy.
    A legacy matcher that omits ``sense`` (e.g. a plain ``(scores)`` argmax) is called with
    just the array, so every legal shape of the raw-array contract keeps working.
    """
    try:
        params = inspect.signature(matcher).parameters
    except (ValueError, TypeError):
        return False
    return "sense" in params or any(
        p.kind == p.VAR_KEYWORD for p in params.values()
    )


def resolve_matcher(spec="optimal", **config):
    """Resolve ``spec`` to a runner ``(scores_or_scorematrix, *, sense=None) -> pairs``.

    A callable passes through; a registered name (or a facade ``how=`` alias like
    ``'assign' -> 'optimal'``) is built via the registry. The returned runner coerces its
    argument to a :class:`~equate.base.ScoreMatrix` and then either hands that matrix to a
    **native** matcher (so it worst-cases holes through the sanctioned views) or hands a
    **legacy** raw-array matcher a pre-worst-cased dense array — so the sparse
    hole-worst-casing holds no matter which contract the matcher follows (D11).
    """
    if isinstance(spec, str):
        matcher = matchers.create(_HOW_ALIASES.get(spec, spec), **config)
    elif callable(spec):
        matcher = spec
    else:
        raise TypeError(
            f"matcher spec must be a registered name or a callable, "
            f"got {type(spec).__name__}"
        )
    legacy = _is_legacy_matcher(matcher)
    pass_sense = legacy and _accepts_sense(matcher)

    def run(scores, *, sense=None):
        sm = ScoreMatrix.coerce(scores, sense=sense)
        if not legacy:
            pairs = matcher(sm)
        else:
            view = sm.legacy_view()
            pairs = matcher(view, sense=sm.sense) if pass_sense else matcher(view)
        # D11 is enforced HERE, at the boundary — so the invariant never depends on an
        # individual matcher remembering to drop holes. A native matcher has already
        # dropped them (this is idempotent); a legacy/third-party matcher that assigns a
        # row with no candidate (worst-cased, but still assignable) is corrected here.
        return sm.drop_holes(pairs)

    return run
