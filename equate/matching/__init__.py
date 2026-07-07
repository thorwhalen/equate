"""The match stage (③): turn a score matrix into a correspondence.

A *Matcher* is a strategy over a score structure — the choice is the **objective**, not
the algorithm (decision register D4). Select by name via the :data:`matchers` registry or
pass a callable ``(scores, *, sense) -> pairs``. The optimal matcher is sparse-aware
(routes a blocked matrix to the sparse LAP); soft matching (optimal transport) lives in a
parallel :func:`soft_match` / :func:`harden` seam.
"""

import numpy as np
from scipy.sparse import issparse

from equate.registry import Registry
from equate.matching.assign import optimal_matching, greedy_matching, stable_matching
from equate.matching.soft import soft_match, harden

__all__ = [
    'matchers',
    'resolve_matcher',
    'optimal_matching',
    'greedy_matching',
    'stable_matching',
    'soft_match',
    'harden',
]

#: the match-stage strategy registry (name -> lazy factory)
matchers = Registry('matcher')


def _const(fn, name):
    """Factory returning a fixed matcher; passing config is a clear error."""

    def factory(**config):
        if config:
            raise TypeError(
                f"matcher {name!r} takes no configuration; got {list(config)}"
            )
        return fn

    return factory


def _max_weight_matcher(scores, *, sense='maximize'):
    """Maximum-weight bipartite matching via networkx — requires ``equate[graph]``."""
    from equate.util import maximal_matching

    S = scores.toarray() if issparse(scores) else np.asarray(scores, dtype=float)
    return list(maximal_matching(-S if sense == 'minimize' else S))


def _kuhn_munkres_matcher(scores, *, sense='maximize'):
    """Hungarian assignment via networkx — requires ``equate[graph]``."""
    from equate.util import kuhn_munkres_matching

    S = scores.toarray() if issparse(scores) else scores
    return list(kuhn_munkres_matching(S, sense=sense))


matchers.register('optimal', _const(optimal_matching, 'optimal'))
matchers.register('hungarian', _const(optimal_matching, 'hungarian'))  # alias
matchers.register('greedy', _const(greedy_matching, 'greedy'))
matchers.register('stable', _const(stable_matching, 'stable'))
matchers.register('max_weight', _const(_max_weight_matcher, 'max_weight'))
matchers.register('kuhn_munkres', _const(_kuhn_munkres_matcher, 'kuhn_munkres'))

#: facade ``how=`` objective aliases -> registered matcher names
_HOW_ALIASES = {'assign': 'optimal'}


def resolve_matcher(spec='optimal', **config):
    """Resolve ``spec`` to a matcher ``(scores, *, sense) -> pairs``.

    A callable passes through; a registered name (or a facade ``how=`` alias like
    ``'assign' -> 'optimal'``) is built via the registry.
    """
    if callable(spec):
        return spec
    if isinstance(spec, str):
        return matchers.create(_HOW_ALIASES.get(spec, spec), **config)
    raise TypeError(
        f"matcher spec must be a registered name or a callable, "
        f"got {type(spec).__name__}"
    )
