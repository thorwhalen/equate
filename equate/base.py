"""Foundational contracts and data structures for equate.

This module is the stable spine of the matching framework. It defines:

- the four **stage protocols** (``Featurizer``, ``Comparator``, ``Blocker``,
  ``Matcher``) that every strategy satisfies — each is just a callable with a
  documented contract, so a plain lambda qualifies;
- the ``ScoreMatrix`` that flows from the compare stage to the match stage (the
  single source of truth carrying the scores, their ``sense``, and row/col labels);
- the structured result types (``Candidate``, ``Matching``, ``Explanation``,
  ``Partition``) returned at every public boundary — never printouts — so the same
  core dispatches cleanly to a CLI, an HTTP service, and a declarative UI;
- ``to_cost`` — the single similarity->cost conversion that every cost-minimizing
  matcher routes through, so they can never disagree.

The three-stage decomposition (featurize -> compare -> match/resolve) and the design
decisions realized here are documented in ``docs/research/`` — start with
``00-taxonomy-and-terminology.md`` and the decision register
``11-design-decisions-and-open-questions.md``.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
from scipy.sparse import issparse

__all__ = [
    'Sense',
    'Featurizer',
    'Comparator',
    'Blocker',
    'Matcher',
    'to_cost',
    'ScoreMatrix',
    'Explanation',
    'Candidate',
    'Matching',
    'Partition',
]

# --- Stage protocols (structural: a plain callable/lambda satisfies each) ---------
# These are documented type aliases, not ABCs — favouring functions over classes.

#: Whether higher scores are better (``'maximize'``, a similarity) or lower scores
#: are better (``'minimize'``, a distance/cost).
Sense = Literal['maximize', 'minimize']

#: ① Featurize — map each object to a comparable representation (batched). ``id`` and
#: any ``key=`` function (``str.lower``, ``operator.attrgetter('name')``) are valid.
Featurizer = Callable[[Iterable[Any]], Sequence[Any]]

#: ② Compare — score how alike two representations are (higher = more alike by the
#: canonical ``'maximize'`` sense).
Comparator = Callable[[Any, Any], float]

#: ② Block — candidate generation: yield ``(i, j)`` index pairs lazily, never a
#: materialized n×m matrix. The default ``all_pairs`` blocker is the correctness base.
Blocker = Callable[..., Iterable[tuple]]

#: ③ Match — turn a score structure into matched ``(i, j)`` pairs; accepts ``*, sense``.
Matcher = Callable[..., Iterable[tuple]]


def to_cost(scores, *, sense: Sense = 'maximize'):
    """Convert a score array to a **cost** array for minimization-based matchers.

    This is the single source of truth for the similarity->cost conversion, so every
    matcher agrees. Historically the matchers each converted differently
    (``hungarian`` used ``S.max() - S``, ``kuhn_munkres`` used ``-S``, ``stable`` used
    ``1 - S``) — equivalent only by luck, and wrong for unbounded or non-``[0, 1]``
    scores. Route every cost-minimizing matcher through this instead.

    - ``sense='maximize'`` (a *similarity*, higher = better): return the complement to
      the maximum, ``S.max() - S``. It is non-negative (safe for min-weight solvers)
      and, differing from ``-S`` only by a global constant, yields the same optimal
      assignment.
    - ``sense='minimize'`` (already a *distance*/cost, lower = better): returned
      unchanged. An edit/DTW/geographic distance is never forced through ``1 - S``.

    >>> import numpy as np
    >>> S = np.array([[0.9, 0.1], [0.2, 0.8]])
    >>> to_cost(S).round(2).tolist()
    [[0.0, 0.8], [0.7, 0.1]]
    >>> to_cost(S, sense='minimize') is S
    True
    """
    if sense == 'minimize':
        return scores
    if sense != 'maximize':
        raise ValueError(f"sense must be 'maximize' or 'minimize', got {sense!r}")
    arr = scores.toarray() if issparse(scores) else np.asarray(scores, dtype=float)
    return arr.max() - arr


@dataclass
class ScoreMatrix:
    """Scored candidate structure flowing compare -> match (the SSOT).

    ``data`` is a dense ``numpy`` array or a ``scipy.sparse`` matrix of pairwise
    scores; ``sense`` records the polarity (see :data:`Sense`); ``row_labels`` /
    ``col_labels`` name the two collections so a downstream matcher never guesses
    orientation or polarity. A sparse ``data`` is exactly what a blocker produces:
    the cells it chose not to score are simply absent.
    """

    data: Any
    sense: Sense = 'maximize'
    row_labels: Optional[Sequence] = None
    col_labels: Optional[Sequence] = None

    @property
    def shape(self):
        return self.data.shape

    def to_cost(self):
        """Cost view of the scores, honouring ``sense`` (see :func:`to_cost`)."""
        return to_cost(self.data, sense=self.sense)


@dataclass
class Explanation:
    """Render-agnostic justification for a decision (rules / attributions / diffs).

    A structured payload, never a printed string, so any UI can render it declaratively.
    """

    summary: str = ''
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    """One scored correspondence between a ``left`` item and a ``right`` item."""

    left: Any
    right: Any
    score: float
    explanation: Optional[Explanation] = None


@dataclass
class Matching:
    """Default (hard) matcher output: matched index pairs, with retained scores.

    Iterating a ``Matching`` yields ``(i, j)`` index tuples, so ``dict(matching)`` and
    ``list(matching)`` work. Richer tiers are *optional attributes* rather than
    separate return types: ``plan`` holds a soft transport plan, ``probability`` holds
    per-pair calibrated match probabilities. ``Partition`` (clustering) and, later,
    ``PartitionPosterior`` (Bayesian) also iterate as their point-estimate pairs, so a
    caller can always ``dict(...)`` a result regardless of the ``how=`` strategy.
    """

    pairs: Sequence[tuple]
    scores: Optional[Sequence[float]] = None
    plan: Optional[Any] = None
    probability: Optional[Sequence[float]] = None
    row_labels: Optional[Sequence] = None
    col_labels: Optional[Sequence] = None

    def __iter__(self):
        return iter(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def labeled_pairs(self):
        """Yield ``(row_label, col_label)``, using labels when present, else indices."""
        for i, j in self.pairs:
            r = self.row_labels[i] if self.row_labels is not None else i
            c = self.col_labels[j] if self.col_labels is not None else j
            yield r, c


@dataclass
class Partition:
    """Clustering / collective output: a map from item index to entity (cluster) id.

    Iterating yields the induced within-cluster ``(i, j)`` pairs (the point estimate),
    so a ``Partition`` is drop-in where matched pairs are expected. This is the
    canonical *clustering* output — it subsumes pairwise links, 1:1 assignment and
    dedup, and generalizes ``match(A, B)`` to ``resolve(*collections)``.
    """

    labels: Sequence[int]

    def groups(self) -> Mapping[int, list]:
        """Return ``{cluster_id: [item_index, ...]}``."""
        out: dict = {}
        for item, cluster in enumerate(self.labels):
            out.setdefault(cluster, []).append(item)
        return out

    def __iter__(self):
        for members in self.groups().values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    yield members[a], members[b]
