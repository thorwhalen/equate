"""Fuse aligned records field by field, weighting sources by trust, and report conflicts.

The stage after :func:`equate.align_sources` (data fusion, or "truth discovery" when the
source weights are learned): for each item, each field's candidate values are grouped
into **votes** (values that agree), each vote weighs the summed ``trust`` of the sources
behind it, and the heaviest vote wins. The value kept is the one from the most trusted
source in the winning vote, and its source is recorded as the field's provenance.

A field is a **conflict** when a losing vote weighs at least ``conflict_ratio`` of the
winning one: trusted sources disagree (a field no source is trusted for never is).
Numbers agree only when equal by default: give timing fields a tolerance, e.g.
``agree={'start': lambda a, b: abs(a - b) < 0.1}``. A noisy, low-trust source contradicting a trusted
one is expected and not reported; two trusted sources disagreeing is exactly what a
person should check.

>>> from equate import align_sources
>>> sheet = [{'text': 'bonjour', 'fr': 'hello'}, {'text': 'merci', 'fr': 'thanks'}]
>>> ocr = [{'text': 'bonjour'}, {'text': 'mercl'}]
>>> audio = [{'text': 'bonjou', 'start': 0.0}, {'text': 'merci', 'start': 1.5}]
>>> al = align_sources({'sheet': sheet, 'ocr': ocr, 'audio': audio})
>>> fused = fuse(al, trust={'audio': {'text': 0.3}})
>>> [item.values for item in fused]  # doctest: +NORMALIZE_WHITESPACE
[{'text': 'bonjour', 'fr': 'hello', 'start': 0.0},
 {'text': 'merci', 'fr': 'thanks', 'start': 1.5}]
>>> [(c.field, c.votes[1].sources) for c in fused[1].conflicts]
[('text', ['ocr'])]
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from equate.compare.vectorize import record_fields, record_value

__all__ = [
    "fuse",
    "Fusion",
    "FusedItem",
    "FieldVote",
    "FieldConflict",
    "default_agree",
    "DFLT_CONFLICT_RATIO",
    "DFLT_TRUST",
]

#: a losing vote weighing at least this fraction of the winner makes the field a conflict
DFLT_CONFLICT_RATIO = 0.5
#: the trust of a source (or a source's field) that ``trust`` does not mention
DFLT_TRUST = 1.0


def default_agree(a, b):
    """Whether two field values agree: ``==``, strings compared case- and
    whitespace-insensitively.

    >>> default_agree(' Merci  beaucoup', 'merci beaucoup')
    True
    """
    if isinstance(a, str) and isinstance(b, str):
        return " ".join(a.split()).casefold() == " ".join(b.split()).casefold()
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True  # NaN agrees with NaN
    try:
        return bool(a == b)
    except Exception:  # e.g. numpy arrays: ambiguous truth value
        return a is b


@dataclass
class FieldVote:
    """A value some sources agree on, and the summed trust behind it."""

    value: Any
    sources: list
    weight: float


@dataclass
class FieldConflict:
    """A field whose trusted sources disagree. ``votes`` is heaviest first; the first is
    the value kept."""

    field: str
    votes: list

    @property
    def value(self):
        """The value kept."""
        return self.votes[0].value


@dataclass
class FusedItem:
    """One item, fused.

    - ``values``: ``{field: value}``;
    - ``provenance``: ``{field: source}``, the source each value was taken from;
    - ``conflicts``: the :class:`FieldConflict` fields;
    - ``votes``: ``{field: [FieldVote, ...]}`` for every field, heaviest first (what a
      trust estimator or a review screen reads);
    - ``members``: ``{source: index}`` of the records fused (from the alignment);
    - ``missing``: the sources that have no record for this item;
    - ``margin`` / ``reordered``: carried over from the alignment (low margin = unsure).
    """

    values: dict
    provenance: dict
    conflicts: list = field(default_factory=list)
    votes: dict = field(default_factory=dict)
    members: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    margin: Optional[float] = None
    reordered: bool = False

    @property
    def needs_review(self):
        """Whether the item has a conflict or is out of order (margins are the caller's
        to threshold: their scale depends on the comparators)."""
        return bool(self.conflicts) or self.reordered


@dataclass
class Fusion:
    """The result of :func:`fuse`: one :class:`FusedItem` per aligned item, in order."""

    items: list
    sources: list

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        return self.items[k]

    @property
    def conflicts(self):
        """``(item_index, FieldConflict)`` for every conflict, in item order."""
        return [(k, c) for k, item in enumerate(self.items) for c in item.conflicts]

    @property
    def unmatched(self):
        """The items only one source has (the others lack them), when there are several
        sources."""
        if len(self.sources) < 2:
            return []
        return [item for item in self.items if len(item.members) == 1]


def _trust_of(trust, source, field_name):
    t = trust.get(source, DFLT_TRUST)
    if isinstance(t, Mapping):
        return float(t.get(field_name, DFLT_TRUST))
    return float(t)


def _agree_of(agree, field_name):
    if agree is None:
        return default_agree
    if isinstance(agree, Mapping):
        return agree.get(field_name, default_agree)
    if callable(agree):
        return agree
    raise TypeError(
        f"agree must be None, a callable (a, b) -> bool or a mapping {{field: callable}}, "
        f"got {type(agree).__name__}"
    )


def _votes(candidates, agree):
    """Group ``[(source, value, weight)]`` into votes; each value joins the first vote
    whose first value it agrees with. Heaviest first; ties keep source order."""
    votes = []
    for source, value, weight in candidates:
        for vote in votes:
            if agree(vote["value"], value):
                vote["sources"].append(source)
                vote["weight"] += weight
                if weight > vote["best"]:  # keep the most trusted source's value
                    vote["value"], vote["best"] = value, weight
                break
        else:
            votes.append(
                {"value": value, "sources": [source], "weight": weight, "best": weight}
            )
    votes.sort(key=lambda v: -v["weight"])  # stable: ties keep source order
    return [FieldVote(v["value"], v["sources"], v["weight"]) for v in votes]


def _value_source(vote, candidates):
    """The source the kept value came from: the most trusted one in the vote."""
    weight = {s: w for s, _, w in candidates}
    return max(vote.sources, key=lambda s: weight[s])  # max keeps the first on ties


def fuse(
    groups,
    *,
    trust=None,
    agree=None,
    conflict_ratio=DFLT_CONFLICT_RATIO,
    fields=None,
):
    """Fuse each aligned item's records into one, field by field.

    ``groups`` is a :class:`~equate.sources.SourceAlignment` (from
    :func:`~equate.align_sources`) or any iterable of its
    :class:`~equate.sources.SourceGroup` items.

    - ``trust``: ``{source: weight}`` or ``{source: {field: weight}}``; anything unstated
      weighs :data:`DFLT_TRUST` (so the default is equal trust). A weight of ``0`` keeps a
      source's value only when no other source has the field.
    - ``agree``: when two values of a field agree: ``None`` (:func:`default_agree`), a
      predicate ``(a, b) -> bool``, or ``{field: predicate}``.
    - ``conflict_ratio``: a field is a conflict when a losing vote weighs at least this
      fraction of the winning one.
    - ``fields``: the fields to fuse (default: every field of every record, in the order
      first seen).

    Returns a :class:`Fusion`: iterate it for :class:`FusedItem` items; ``.conflicts`` and
    ``.unmatched`` list what a person should look at.
    """
    trust = trust or {}
    sources = list(getattr(groups, "sources", None) or [])  # a SourceAlignment's
    groups = list(groups)
    if not sources:
        for g in groups:
            sources.extend(s for s in g.members if s not in sources)
    items = []
    for g in groups:
        names = fields
        if names is None:
            names = []
            for record in g.records.values():
                names.extend(f for f in record_fields(record) if f not in names)
        values, provenance, conflicts, all_votes = {}, {}, [], {}
        for f in names:
            candidates = [
                (s, v, _trust_of(trust, s, f))
                for s, record in g.records.items()
                if (v := record_value(record, f)) is not None
            ]
            if not candidates:
                continue
            votes = _votes(candidates, _agree_of(agree, f))
            all_votes[f] = votes
            values[f] = votes[0].value
            provenance[f] = _value_source(votes[0], candidates)
            top = votes[0].weight
            if len(votes) > 1 and top > 0 and votes[1].weight >= conflict_ratio * top:
                conflicts.append(FieldConflict(f, votes))
        items.append(
            FusedItem(
                values=values,
                provenance=provenance,
                conflicts=conflicts,
                votes=all_votes,
                members=dict(g.members),
                missing=[s for s in sources if s not in g.members],
                margin=g.margin,
                reordered=g.reordered,
            )
        )
    return Fusion(items=items, sources=sources)
