"""Align several sources of the same items: one group per item, at most one record per source.

Several sources often describe the same items, each partially and each with its own kind
of error, and no shared identifier: a recording split into utterances (good timing, noisy
transcript), a printed sheet (exact text, no timing), an OCR or PDF extraction (exact
text, broken layout). :func:`align_sources` decides which records of the sources are the
same item; :func:`equate.fusion.fuse` then merges each item field by field and reports
where the sources disagree.

The alignment is **progressive** (the classic multiple-sequence-alignment strategy): the
first source is the backbone, and each further source is aligned against the groups built
so far. One step is a two-collection matching problem over a *gain* matrix,
``gain[g, j] = similarity(group g, record j) - threshold``, solved by the ``order``
strategy:

- ``'monotone'`` (default): the sources list the items in the same order, gaps allowed.
  Solved by equate's :func:`~equate.compare.sequence.needleman_wunsch_alignment` with the
  gain as substitution score and free gaps, so a record missing from one source becomes a
  gap instead of shifting everything after it. Records left over on both sides within
  ``window`` positions of each other are then paired by assignment when they match, which
  is how a **local order swap** (two items read in the opposite order) is recovered rather
  than cascading; such groups are marked ``reordered``.
- ``'free'``: no order; an optimal assignment (equate's ``'assign'`` matcher).
- any callable ``gain -> iterable of (g, j)``: your own strategy.

Records are compared on the fields they share (``compare``), with equate's comparators or
any callable, so domain knowledge (a phonetic reading, a unit normaliser) is injected, not
built in. A group's ``margin`` is its score minus the best score a nearby rival record
would get in its place: low (or negative) where the alignment is unsure.

>>> sheet = [{'text': 'bonjour'}, {'text': 'merci'}, {'text': 'au revoir'}]
>>> audio = [{'text': 'bonjou', 'start': 0.0}, {'text': 'au revoar', 'start': 2.1}]
>>> al = align_sources({'sheet': sheet, 'audio': audio})
>>> [g.members for g in al]
[{'sheet': 0, 'audio': 0}, {'sheet': 1}, {'sheet': 2, 'audio': 1}]
"""

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

import numpy as np

from equate.base import ComparatorMeta, ScoreMatrix
from equate.compare import polarity_sense, resolve_comparator
from equate.compare.sequence import needleman_wunsch_alignment
from equate.compare.vectorize import _get
from equate.matching import resolve_matcher

__all__ = [
    "align_sources",
    "record_similarity",
    "SourceGroup",
    "SourceAlignment",
    "monotone_order",
    "free_order",
    "DFLT_ALIGN_THRESHOLD",
    "DFLT_ORDER_WINDOW",
    "DFLT_FIELD_COMPARATOR",
]

#: minimum record similarity for two records to be the same item (strictly above it)
DFLT_ALIGN_THRESHOLD = 0.5
#: how far (in positions) a ``'monotone'`` alignment looks for swapped items and rivals
DFLT_ORDER_WINDOW = 2
#: the comparator used for a shared string field when ``compare`` names no comparator
DFLT_FIELD_COMPARATOR = "ratio"

_SIM_META = ComparatorMeta(
    polarity="similarity", bounded=True, is_metric=False, is_symmetric=True
)


# --- record comparison ----------------------------------------------------------------


def _fields(record):
    if isinstance(record, Mapping):
        return list(record)
    if hasattr(record, "_fields"):  # a namedtuple
        return list(record._fields)
    return list(vars(record)) if hasattr(record, "__dict__") else []


def _field_comparator(spec, field_name):
    cmp = resolve_comparator(spec)
    if polarity_sense(cmp) == "minimize":
        raise ValueError(
            f"the comparator for field {field_name!r} is a distance; align_sources needs "
            f"similarities (higher = more alike). Use e.g. 'levenshtein' instead of "
            f"'levenshtein_distance', or wrap your distance d as lambda a, b: 1 / (1 + d(a, b))."
        )
    return cmp


def record_similarity(compare=None, *, missing=0.0):
    """Build a record comparator ``(a, b) -> float`` from per-field comparators.

    ``compare`` is:

    - ``None`` (default): every field both records hold as a string, scored with
      :data:`DFLT_FIELD_COMPARATOR`;
    - a mapping ``{field: comparator}`` (a registered name or a callable): those fields;
    - a callable ``(a, b) -> float``: returned as it is.

    The score is the **mean over the fields both records actually have** (a field missing
    or ``None`` on either side is skipped, not scored zero), so a source with extra fields
    is compared on what it shares. Records sharing no compared field score ``missing``.

    >>> sim = record_similarity({'text': 'ratio'})
    >>> sim({'text': 'merci'}, {'text': 'merci', 'pinyin': 'xiexie'})
    1.0
    >>> sim({'start': 1.0}, {'text': 'merci'})
    0.0
    """
    if callable(compare):
        return compare
    if compare is not None and not isinstance(compare, Mapping):
        raise TypeError(
            f"compare must be None, a mapping {{field: comparator}} or a callable, "
            f"got {type(compare).__name__}"
        )
    named = (
        None
        if compare is None
        else {f: _field_comparator(c, f) for f, c in compare.items()}
    )
    default = None if named is not None else resolve_comparator(DFLT_FIELD_COMPARATOR)

    def similarity(a, b):
        if named is not None:
            pairs = [(f, cmp) for f, cmp in named.items()]
        else:
            b_fields = set(_fields(b))
            shared = [f for f in _fields(a) if f in b_fields]
            pairs = [
                (f, default)
                for f in shared
                if isinstance(_get(a, f), str) and isinstance(_get(b, f), str)
            ]
        scores = []
        for f, cmp in pairs:
            va, vb = _get(a, f), _get(b, f)
            if va is not None and vb is not None:
                scores.append(float(cmp(va, vb)))
        return sum(scores) / len(scores) if scores else missing

    similarity.meta = _SIM_META
    return similarity


# --- ordering strategies (gain matrix -> pairs) -----------------------------------------


def free_order(gain):
    """Pair rows and columns of ``gain`` by optimal assignment, keeping positive gains."""
    gain = np.asarray(gain, dtype=float)
    if not gain.size:
        return []
    sm = ScoreMatrix(np.maximum(gain, 0.0), sense="maximize")
    return [(i, j) for i, j in resolve_matcher("assign")(sm) if gain[i, j] > 0]


def _window_bounds(rows, matched, n_cols, window):
    """For each gap row, the column range within ``window`` of the gap it sits in."""
    anchors = sorted(matched)
    keys = [i for i, _ in anchors]
    bounds = {}
    for i in rows:
        k = bisect_left(keys, i)
        before = max((j for _, j in anchors[:k]), default=-1)
        after = min((j for _, j in anchors[k:]), default=n_cols)
        bounds[i] = (before + 1 - window, after - 1 + window)
    return bounds


def monotone_order(gain, *, window=DFLT_ORDER_WINDOW):
    """Order-preserving pairing of ``gain``'s rows and columns, gaps free.

    A Needleman-Wunsch global alignment with ``gain`` as substitution score and zero gap
    cost: the maximum-gain set of non-crossing pairs. Rows and columns it leaves unpaired
    are then paired by assignment when they have positive gain and lie within ``window``
    positions of each other (a local order swap); those pairs cross the others.

    >>> monotone_order([[1, -1], [-1, 1]])
    [(0, 0), (1, 1)]
    >>> sorted(monotone_order([[-1, 1], [1, -1]]))  # a swap, recovered
    [(0, 1), (1, 0)]
    """
    gain = np.asarray(gain, dtype=float)
    n, m = gain.shape
    if not (n and m):
        return []
    path = needleman_wunsch_alignment(
        range(n), range(m), substitution=lambda i, j: gain[i, j], gap=0.0
    ).path
    matched = [(i, j) for i, j in path if i is not None and j is not None]
    matched = [(i, j) for i, j in matched if gain[i, j] > 0]
    if window is None or window < 0:
        return matched
    rows = sorted(set(range(n)) - {i for i, _ in matched})
    cols = sorted(set(range(m)) - {j for _, j in matched})
    if not (rows and cols):
        return matched
    bounds = _window_bounds(rows, matched, m, window)
    sub = np.array(
        [
            [gain[i, j] if bounds[i][0] <= j <= bounds[i][1] else 0.0 for j in cols]
            for i in rows
        ]
    ).reshape(len(rows), len(cols))
    extra = [(rows[a], cols[b]) for a, b in free_order(sub)]
    return matched + extra


_ORDERS = {"monotone": monotone_order, "free": free_order}


def _resolve_order(order, window):
    if callable(order):
        return order
    if order == "monotone":
        return partial(monotone_order, window=window)
    if order in _ORDERS:
        return _ORDERS[order]
    raise ValueError(
        f"unknown order {order!r}; use 'monotone', 'free', or a callable gain -> pairs"
    )


# --- results ------------------------------------------------------------------------------


@dataclass
class SourceGroup:
    """One item: the records the sources hold for it, at most one per source.

    - ``members``: ``{source: index}`` into that source's sequence, in source order;
    - ``records``: ``{source: record}``, the records themselves;
    - ``scores``: the pairwise record similarities, as ``(source_a, source_b, score)``;
    - ``score``: their mean (``None`` for a single-source group);
    - ``margin``: ``score`` minus the best a nearby rival record would score in a member's
      place, over all members (low or negative = unsure; ``None`` for a single record);
    - ``reordered``: the group crosses another one, i.e. some source lists this item out of
      the order the others do (a local swap).

    JSON-able when the records are.
    """

    members: dict
    records: dict
    scores: list = field(default_factory=list)
    score: Optional[float] = None
    margin: Optional[float] = None
    reordered: bool = False

    @property
    def sources(self):
        """The sources holding this item."""
        return list(self.members)


@dataclass
class SourceAlignment:
    """The result of :func:`align_sources`: groups in alignment order.

    Iterates as its :class:`SourceGroup` items; ``sources`` names every input source, in
    order (so a consumer can tell which sources a group lacks).
    """

    groups: list
    sources: list
    order: Any = "monotone"
    threshold: float = DFLT_ALIGN_THRESHOLD

    def __iter__(self):
        return iter(self.groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, k):
        return self.groups[k]

    def missing(self, group):
        """The sources that have no record for ``group``."""
        return [s for s in self.sources if s not in group.members]


# --- the progressive alignment ---------------------------------------------------------


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _crossing(pairs):
    """The pairs of ``(i, j)`` that cross another pair (an order inversion)."""
    out = set()
    for a, (i, j) in enumerate(pairs):
        for i2, j2 in pairs[a + 1 :]:
            if (i - i2) * (j - j2) < 0:
                out.update({(i, j), (i2, j2)})
    return out


def _insert_unmatched(groups, new, pairs, *, source):
    """New single-record groups for the columns no pair took, placed after the group
    holding the nearest preceding column (so a 'monotone' run keeps its order)."""
    row_of = {j: i for i, j in pairs}
    head = []  # unmatched columns before every matched one
    after = {i: [] for i in range(len(groups))}  # per group: columns to insert after it
    for j in range(len(new)):
        if j in row_of:
            continue
        preceding = [jj for jj in row_of if jj < j]
        if preceding:
            after[row_of[max(preceding)]].append(j)
        else:
            head.append(j)
    ordered = [[(source, j)] for j in head]
    for i, g in enumerate(groups):
        ordered.append(g)
        ordered.extend([(source, j)] for j in after[i])
    return ordered


def align_sources(
    sources,
    *,
    compare=None,
    order="monotone",
    threshold=DFLT_ALIGN_THRESHOLD,
    window=DFLT_ORDER_WINDOW,
):
    """Align N sources of records into groups, one per item, at most one record per source.

    ``sources`` maps a source name to a sequence of records (dicts, or objects with
    attributes). The first source is the backbone of the progressive alignment: put the
    most complete one first.

    - ``compare``: how two records are scored (see :func:`record_similarity`): ``None``
      (shared string fields, ``ratio``), ``{field: comparator}`` with equate's comparator
      names or any callables, or one record callable ``(a, b) -> float``. Similarities,
      higher = more alike, roughly in ``[0, 1]``.
    - ``order``: ``'monotone'`` (same order, gaps allowed; default), ``'free'``
      (assignment), or a callable ``gain -> pairs`` (see the module docstring).
    - ``threshold``: two records are aligned only if their similarity is strictly above it.
      A record is compared with a group by its mean similarity to the group's members.
    - ``window``: for ``'monotone'``, how many positions away a swapped item is looked for;
      for every order, how many positions away rival records are looked for when computing
      ``margin`` (``None``: everywhere).

    Returns a :class:`SourceAlignment` (iterate it for :class:`SourceGroup` items).

    With two sources and the items in the same order, this is :func:`equate.match` on the
    shared fields, keeping pairs above ``threshold``.
    """
    if not isinstance(sources, Mapping) or not sources:
        raise TypeError(
            "sources must be a non-empty mapping {name: sequence of records}"
        )
    names = list(sources)
    data = {s: list(sources[s]) for s in names}
    sim = record_similarity(compare, missing=None)
    solve = _resolve_order(order, window)

    cache = {}

    def pair_sim(s, i, t, j):
        key = (s, i, t, j) if (s, i) <= (t, j) else (t, j, s, i)
        if key not in cache:
            v = sim(data[s][i], data[t][j])
            cache[key] = None if v is None else float(v)
        return cache[key]

    def group_sim(group, t, j):
        return _mean(pair_sim(s, i, t, j) for s, i in group)

    groups = [[(names[0], i)] for i in range(len(data[names[0]]))]
    reordered = set()  # (source, index) of records added out of order
    for t in names[1:]:
        new = data[t]
        gain = np.array(
            [
                [
                    -threshold if (v := group_sim(g, t, j)) is None else v - threshold
                    for j in range(len(new))
                ]
                for g in groups
            ],
            dtype=float,
        ).reshape(len(groups), len(new))
        pairs = sorted((int(i), int(j)) for i, j in solve(gain))
        _check_pairs(pairs)
        for i, j in _crossing(pairs):
            reordered.add((t, j))
        for i, j in pairs:
            groups[i] = groups[i] + [(t, j)]
        groups = _insert_unmatched(groups, new, pairs, source=t)

    return SourceAlignment(
        groups=[
            _finish(g, data, pair_sim, reordered=reordered, window=window)
            for g in groups
        ],
        sources=names,
        order=order,
        threshold=threshold,
    )


def _check_pairs(pairs):
    rows = [i for i, _ in pairs]
    cols = [j for _, j in pairs]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        raise ValueError(
            "the order strategy returned a row or a column twice; it must return a "
            "one-to-one set of (group, record) pairs"
        )


def _finish(group, data, pair_sim, *, reordered, window):
    """Build the :class:`SourceGroup`: pairwise scores, mean score and margin."""
    members = dict(group)
    records = {s: data[s][i] for s, i in group}
    scores = [
        (s, t, v)
        for a, (s, i) in enumerate(group)
        for t, j in group[a + 1 :]
        if (v := pair_sim(s, i, t, j)) is not None
    ]
    score = _mean(v for _, _, v in scores) if len(group) > 1 else None
    margin = None
    if len(group) > 1:
        margins = []
        for s, i in group:
            rest = [(t, j) for t, j in group if t != s]
            own = _mean(pair_sim(s, i, t, j) for t, j in rest)
            if own is None:
                continue
            lo, hi = (
                (0, len(data[s]))
                if window is None
                else (max(0, i - window), min(len(data[s]), i + window + 1))
            )
            rivals = [
                _mean(pair_sim(s, k, t, j) for t, j in rest)
                for k in range(lo, hi)
                if k != i
            ]
            best_rival = max((r for r in rivals if r is not None), default=0.0)
            margins.append(own - best_rival)
        margin = min(margins) if margins else None
    return SourceGroup(
        members=members,
        records=records,
        scores=scores,
        score=score,
        margin=margin,
        reordered=any((s, i) in reordered for s, i in group),
    )
