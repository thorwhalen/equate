"""The service layer: equate's matching core as JSON-in/JSON-out operations.

This module is the **surface-free** half of the HTTP service (doc 10 §5, decision D10).
Every operation is a plain method taking JSON-able arguments and returning a dataclass of
JSON-able fields, so any surface can project it: :func:`mk_app` mounts it as an HTTP API
with ``qh`` (``pip install 'equate[service]'``), and the review UI in ``ui/`` consumes the
same payloads through Zod schemas generated from :func:`json_schemas`.

Two kinds of operation:

- **stateless** — :meth:`MatchService.match`: two collections in, scored pairs out (the
  batch core, cached by content);
- **session-scoped** — the interactive loop of doc 08. :meth:`MatchService.open_session`
  scores the two collections *once* and keeps a :class:`~equate.interactive.CandidateStore`
  of top-k candidates; :meth:`~MatchService.candidates` reads (or refreshes) one item's
  top-k; :meth:`~MatchService.edit` applies confirm / reject / reassign edits and returns
  the locally re-optimized assignment (one constrained LAP over the cached scores, never a
  re-score); :meth:`~MatchService.label` is the active-learning ``Oracle`` endpoint: it
  records a human/LLM/crowd label and returns the next most-informative query.

Every response carries an :class:`Accounting` of the time and item counts each tier
(``score``, ``match``, ``reoptimize``, ...) cost, so a caller can budget expensive
cascades (doc 06). Sessions live in any ``MutableMapping`` (``sessions=``), so a
persistent ``dol`` store replaces the in-memory dict without touching this code.

>>> svc = MatchService()
>>> result = svc.match(['apple pie', 'banana split'], ['banana splitt', 'aple pie'])
>>> [(p.left, p.right) for p in result.pairs]
[('apple pie', 'aple pie'), ('banana split', 'banana splitt')]
"""

import hashlib
import json
import math
import numbers
import time
import threading
import uuid
from copy import deepcopy
from collections import defaultdict
from functools import wraps
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np
from scipy.sparse import coo_matrix

from equate.base import Explanation, Partition, ScoreMatrix
from equate.compare import resolve_comparator
from equate.facade import score_matrix
from equate.featurize.records import record_text
from equate.interactive import CandidateStore, ConstraintSet, reoptimize
from equate.matching import harden, resolve_matcher, soft_match

__all__ = [
    "MatchService",
    "SessionNotFound",
    "Accounting",
    "TierCost",
    "Edit",
    "Label",
    "FieldDiff",
    "CandidateView",
    "PairView",
    "ReviewItem",
    "Query",
    "MatchResult",
    "SessionView",
    "EditResult",
    "LabelResult",
    "Session",
    "explain_pair",
    "record_text",
    "to_jsonable",
    "json_schemas",
    "mk_app",
    "serve",
]

DFLT_K = 5  # candidates retained per item (the CandidateStore top-k)
DFLT_FIELD_COMPARE = "ratio"  # per-field similarity shown in a field diff
DFLT_MATCH_CACHE_SIZE = 128  # stateless /match results kept (LRU)
DFLT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")  # vite dev

EditAction = Literal["confirm", "reject", "reassign", "clear"]


class SessionNotFound(KeyError):
    """No session with that id (an HTTP 404)."""


# --- Accounting: per-tier cost / latency (doc 10 §5, doc 06 cascades) ----------------


@dataclass
class TierCost:
    """What one tier (``score``, ``match``, ``reoptimize``, ...) has cost so far."""

    calls: int = 0
    seconds: float = 0.0
    items: int = 0


@dataclass
class Accounting:
    """Per-tier cost/latency ledger. ``tiers`` maps a tier name to its :class:`TierCost`.

    Plain data (it is part of a session, which must pickle into any store); an external
    budget or metrics sink attaches through the ``hook`` argument of :meth:`record`.
    """

    tiers: dict = field(default_factory=dict)

    def record(self, tier, seconds, items=0, *, hook=None):
        """Add one call of ``tier`` that took ``seconds`` over ``items`` items, and tell
        ``hook(tier, seconds, items)`` if one is given."""
        cost = self.tiers.setdefault(tier, TierCost())
        cost.calls += 1
        cost.seconds += seconds
        cost.items += items
        if hook is not None:
            hook(tier, seconds, items)

    @contextmanager
    def timed(self, tier, items=0, *, hook=None):
        """Context manager recording the wall-clock time of its body under ``tier``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(tier, time.perf_counter() - t0, items, hook=hook)

    def merge(self, other):
        """Add ``other``'s tiers into this ledger (a committed operation's costs)."""
        for tier, c in other.tiers.items():
            mine = self.tiers.setdefault(tier, TierCost())
            mine.calls += c.calls
            mine.seconds += c.seconds
            mine.items += c.items
        return self

    def snapshot(self):
        """A copy, so a response never aliases the session's live ledger."""
        return Accounting({k: TierCost(**asdict(v)) for k, v in self.tiers.items()})


# --- Boundary types (JSON-able dataclasses; the UI's Zod schemas mirror these) -------


@dataclass
class Edit:
    """One user edit of the assignment.

    - ``confirm`` — force ``item -> candidate`` (a confirmed match);
    - ``reject`` — forbid ``item -> candidate``;
    - ``reassign`` — confirm a *different* candidate for ``item`` (its previous forced
      match, and any other item forced onto ``candidate``, are released);
    - ``clear`` — drop any constraint on ``item -> candidate`` (undo).

    ``item`` indexes the left collection, ``candidate`` the right one; ``source`` says who
    made the edit (``'human'``, ``'llm'``, ...) — provenance for a later precedence policy.
    """

    action: EditAction
    item: int
    candidate: int
    source: str = "human"


@dataclass
class Label:
    """An oracle's verdict on one ``item -> candidate`` pair (active learning)."""

    item: int
    candidate: int
    is_match: bool
    source: str = "human"  # 'human' | 'llm' | 'crowd' | ...


@dataclass
class FieldDiff:
    """One field of a side-by-side comparison: both values and their similarity."""

    field: str
    left: Any
    right: Any
    similarity: Optional[float]  # None when a side lacks the field


@dataclass
class CandidateView:
    """A scored candidate for an item, with the right-hand object and an explanation."""

    candidate: int
    right: Any
    score: float
    explanation: Optional[Explanation] = None


@dataclass
class PairView:
    """A matched pair: indices, both objects, the score, and whether a user fixed it."""

    item: int
    candidate: int
    left: Any
    right: Any
    score: Optional[float]
    status: Literal["proposed", "confirmed"] = "proposed"


@dataclass
class ReviewItem:
    """One row of the review queue: an item, its current match and its alternatives.

    ``margin`` is the top-1/top-2 score gap (``None`` when there is no rival) — small
    means ambiguous, which is why the queue puts it first.
    """

    item: int
    left: Any
    assigned: Optional[int]
    status: Literal["proposed", "confirmed", "unmatched"]
    margin: Optional[float]
    candidates: list


@dataclass
class Query:
    """An active-learning query: the pair an oracle should label next."""

    item: int
    left: Any
    candidate: CandidateView
    margin: Optional[float]


@dataclass
class MatchResult:
    """Stateless match output: ``pairs`` (or ``groups`` for ``how='cluster'``)."""

    pairs: list
    groups: Optional[list] = None
    accounting: Accounting = field(default_factory=Accounting)
    cached: bool = False


@dataclass
class SessionView:
    """A session's state: its current assignment and the review queue to work through."""

    session_id: str
    n_left: int
    n_right: int
    sense: str
    k: int
    assignment: list
    review: list
    edits: list
    labels: list
    accounting: Accounting
    floor: Optional[float] = None


@dataclass
class EditResult:
    """The result of an edit: the re-optimized assignment and what it changed.

    ``changed`` lists the items whose match moved, so a UI re-renders only that block.
    """

    session_id: str
    assignment: list
    changed: list
    accounting: Accounting


@dataclass
class LabelResult:
    """The result of a label: what was recorded, the edit result, and the next query."""

    label: Label
    result: EditResult
    next: Optional[Query]


@dataclass
class Session:
    """Server-side session state. Not a boundary type: :class:`SessionView` is its view.

    Everything is plain data (no open handles), so a session pickles into any ``dol``
    store passed as ``MatchService(sessions=...)``.
    """

    session_id: str
    A: list
    B: list
    scores: ScoreMatrix
    store: CandidateStore
    k: int
    assignment: list  # [(i, j), ...]
    edits: list = field(default_factory=list)  # [Edit, ...] — the replayable log
    labels: list = field(default_factory=list)  # [Label, ...]
    accounting: Accounting = field(default_factory=Accounting)
    floor: Optional[float] = None  # the score floor applied at open (equate#21)


# --- Helpers -------------------------------------------------------------------------


_MISSING = (None, "")  # a field absent, null or empty is "missing", not "dissimilar"


def _as_scorable(items, featurize, compare):
    """What the scorer sees of ``items``.

    A callable comparator (a record-aware Python function) gets the raw items. Otherwise
    records (mappings) have no text of their own, so a named strategy gets their joined
    values; and on the default text route (no comparator, a text featurizer) scalars are
    turned into text too. Sequences, graphs and numbers for a named comparator (``'dtw'``,
    ``'exp_decay'``) pass through untouched.
    """
    if callable(compare):
        return items
    text_route = compare is None and featurize in _TEXT_FEATURIZERS
    return [
        record_text(x)
        if isinstance(x, Mapping) or (text_route and isinstance(x, (int, float)))
        else x
        for x in items
    ]


_TEXT_FEATURIZERS = (None, "tfidf", "sbert", "openai")


def _fields_of(obj):
    return dict(obj) if isinstance(obj, Mapping) else {"value": obj}


def explain_pair(left, right, score, *, field_compare=DFLT_FIELD_COMPARE):
    """A structured :class:`~equate.base.Explanation` of a pair: a per-field diff.

    Records (mappings) are compared field by field over the union of their keys; any other
    object is one field named ``value``. A side that is absent, ``None`` or ``''`` has no
    similarity (``None``): missing is not the same as different. ``details['fields']`` is a list of
    :class:`FieldDiff` dicts — the payload a UI renders as a side-by-side diff.

    >>> e = explain_pair({'name': 'Jon'}, {'name': 'Jon', 'city': 'Paris'}, 0.8)
    >>> [(f['field'], f['similarity']) for f in e.details['fields']]
    [('name', 1.0), ('city', None)]
    """
    compare = resolve_comparator(field_compare)
    lf, rf = _fields_of(left), _fields_of(right)
    diffs = []
    for name in list(lf) + [k for k in rf if k not in lf]:
        lv, rv = lf.get(name), rf.get(name)
        sim = None
        if lv not in _MISSING and rv not in _MISSING:
            sim = round(float(compare(str(lv), str(rv))), 6)
        diffs.append(asdict(FieldDiff(str(name), lv, rv, sim)))
    return Explanation(summary=f"score {score:.3f}", details={"fields": diffs})


def _finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def to_jsonable(obj):
    """Convert a boundary dataclass (or nested containers of them) to plain JSON data.

    Non-finite floats become ``None`` (JSON has no ``inf``/``nan``); tuples become lists;
    numpy scalars become Python numbers.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f: to_jsonable(getattr(obj, f)) for f in obj.__dataclass_fields__}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return _finite_or_none(obj)
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return to_jsonable(obj.item())
    return obj


def _content_key(payload):
    """A hash of a JSON-able payload, or ``None`` if it is not JSON-able.

    Key order is **kept** (no ``sort_keys``): a record's text joins its values in key order,
    so two records differing only in key order really do score differently.
    """
    try:
        blob = json.dumps(payload, default=None)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(blob.encode()).hexdigest()


class _LRU(OrderedDict):
    """A tiny bounded mapping (least-recently-used eviction) — the default match cache."""

    def __init__(self, maxsize=DFLT_MATCH_CACHE_SIZE):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.maxsize:
            self.popitem(last=False)


def _effective_constraints(edits):
    """Fold an edit log into the :class:`ConstraintSet` it currently means.

    The log is append-only (replayable); this resolves it: the latest decision about a
    pair wins, an item has at most one confirmed match, and a candidate is confirmed for at
    most one item (a later confirm/reassign releases the earlier holder).
    """
    forced = {}  # item -> candidate
    forbidden = set()
    for e in edits:
        pair = (e.item, e.candidate)
        if e.action in ("confirm", "reassign"):
            for i, j in list(forced.items()):
                if i == e.item or j == e.candidate:
                    del forced[i]
            forced[e.item] = e.candidate
            forbidden.discard(pair)
        elif e.action == "reject":
            if forced.get(e.item) == e.candidate:
                del forced[e.item]
            forbidden.add(pair)
        elif e.action == "clear":
            if forced.get(e.item) == e.candidate:
                del forced[e.item]
            forbidden.discard(pair)
        else:
            raise ValueError(f"unknown edit action {e.action!r}")
    cs = ConstraintSet()
    for i, j in forced.items():
        cs.confirm(i, j)
    for i, j in sorted(forbidden):
        cs.reject(i, j)
    return cs


def _without_rejected(sm, forbidden):
    """``sm`` with the ``forbidden`` cells removed — a rejected pair becomes a hole.

    A reject is "this is not a candidate", which is exactly what blocking means, so it
    reuses the blocked-matching semantics (D11): holes are worst-cased with a big-M and
    dropped, making the re-solve lexicographic — as many non-rejected pairs as possible,
    then the best score. An item whose every candidate was rejected is simply left
    unmatched, where forbidding with ``inf`` (core :func:`~equate.interactive.reoptimize`)
    would make a square problem infeasible.
    """
    if not forbidden:
        return sm
    mask = sm.candidate_mask()
    for i, j in forbidden:
        mask[i, j] = False
    return _restricted(sm, mask)


def _past_floor_removed(sm, floor):
    """``sm`` keeping only the cells that pass ``floor`` (at or above it for a
    ``'maximize'`` sense, at or below for ``'minimize'``; NaN never passes): the others
    become holes.

    The same mechanism as a reject, so a session never *forces* a pair past the floor:
    the lexicographic re-solve leaves an item unmatched instead (equate#21).
    """
    if floor is None:
        return sm
    mask = sm.candidate_mask()
    if sm.is_sparse:
        coo = sm.data.tocoo()
        rows, cols, values = coo.row, coo.col, coo.data
    else:
        rows, cols = np.nonzero(mask)
        values = sm.data[rows, cols]
    with np.errstate(invalid="ignore"):
        fails = ~(values >= floor if sm.sense == "maximize" else values <= floor)
    mask[rows[fails], cols[fails]] = False
    return _restricted(sm, mask)


def _restricted(sm, mask):
    """``sm`` as a sparse matrix storing exactly the cells where ``mask`` is True."""
    rows, cols = np.nonzero(mask)
    values = np.asarray(sm.data[rows, cols]).ravel()
    # coo keeps explicitly-stored zeros, so a real 0.0 score stays a candidate
    data = coo_matrix((values, (rows, cols)), shape=sm.shape)
    return ScoreMatrix(
        data, sense=sm.sense, row_labels=sm.row_labels, col_labels=sm.col_labels
    )


def _resolve_edits(sm, edits):
    """The optimal 1:1 assignment of ``sm`` under an edit log (confirms forced, rejects
    removed). Raises ``ValueError`` if no assignment honours the confirms."""
    cs = _effective_constraints(edits)
    forced_only = ConstraintSet(forced=list(cs.forced))
    return reoptimize(_without_rejected(sm, cs.forbidden), forced_only, sense=sm.sense)


def _as_edit(e):
    if isinstance(e, Edit):
        return e
    if not isinstance(e, Mapping):
        raise ValueError(f"an edit is a mapping with action/item/candidate, got {e!r}")
    try:
        return Edit(**e)
    except TypeError as exc:
        raise ValueError(f"invalid edit {e!r}: {exc}") from None


# --- The service ---------------------------------------------------------------------


class MatchService:
    """equate's matching core as stateless + session-scoped JSON-able operations.

    Every seam is a keyword argument with a working default:

    - ``sessions`` — where sessions live (any ``MutableMapping``; default an in-memory
      dict; pass a pickle-valued ``dol`` store to persist them);
    - ``cache`` — where stateless match results are cached (default a bounded LRU; pass
      ``None`` to disable);
    - ``mk_id`` — session-id factory (default ``uuid4().hex``);
    - ``field_compare`` — the per-field comparator used in explanations;
    - ``on_cost`` — ``(tier, seconds, items)`` hook called on every accounted tier.
    """

    def __init__(
        self,
        *,
        sessions: Optional[MutableMapping] = None,
        cache: Optional[MutableMapping] = "default",
        mk_id: Callable[[], str] = lambda: uuid.uuid4().hex,
        field_compare=DFLT_FIELD_COMPARE,
        on_cost: Optional[Callable] = None,
    ):
        self.sessions = {} if sessions is None else sessions
        self.cache = _LRU() if cache == "default" else cache
        self.mk_id = mk_id
        # in-process concurrency: one lock per session (edits read-modify-write the edit
        # log), one for the cache. A multi-process store needs its own versioning.
        self._session_locks = defaultdict(threading.Lock)
        self._cache_lock = threading.Lock()
        self.field_compare = field_compare
        self.on_cost = on_cost

    # -- scoring ------------------------------------------------------------------

    def _score(self, A, B, acc, *, featurize, compare, block, sense):
        A_, B_ = (
            _as_scorable(A, featurize, compare),
            _as_scorable(B, featurize, compare),
        )
        with self._timed(acc, "score", len(A) * len(B)):
            sm = score_matrix(
                A_, B_, featurize=featurize, compare=compare, block=block, sense=sense
            )
        return ScoreMatrix(sm.data, sense=sm.sense, row_labels=A, col_labels=B)

    def _timed(self, acc, tier, items=0):
        return acc.timed(tier, items, hook=self.on_cost)

    # -- stateless ----------------------------------------------------------------

    def match(
        self,
        A: list,
        B: list,
        *,
        featurize: Optional[str] = None,
        compare: Optional[str] = None,
        block: Optional[str] = None,
        how: str = "assign",
        threshold: Optional[float] = None,
        sense: Optional[str] = None,
    ) -> MatchResult:
        """Match two collections (stateless). Mirrors :func:`equate.match`.

        ``how`` is ``'assign'`` (optimal 1:1), ``'greedy'``, ``'stable'``, ``'soft'`` or
        ``'cluster'`` (then ``groups`` lists the entity clusters over the pooled ``A + B``
        indices, ``B`` offset by ``len(A)``). Pairs below ``threshold`` (above it, for a
        ``'minimize'`` sense) are dropped. Results are cached by content.
        """
        A, B = list(A), list(B)
        config = dict(featurize=featurize, compare=compare, block=block, sense=sense)
        key = None
        if self.cache is not None:
            key = _content_key([A, B, config, how, threshold])
            with self._cache_lock:
                hit = self.cache.get(key) if key is not None else None
            if hit is not None:
                # a deep copy: a caller mutating its result must never corrupt the cache
                return MatchResult(**{**asdict_shallow(deepcopy(hit)), "cached": True})

        acc = Accounting()
        sm = self._score(A, B, acc, **config)
        with self._timed(acc, "match", len(A) + len(B)):
            if how == "cluster":
                thr = 0.5 if threshold is None else threshold
                from equate.cluster import resolve_clusterer

                n_a = len(A)
                edges = [(i, n_a + j, v) for i, j, v in sm.stored_entries()]
                partition: Partition = resolve_clusterer("connected_components")(
                    edges, n_a + len(B), threshold=thr, sense=sm.sense
                )
                groups = [sorted(g) for g in partition.groups().values()]
                result = MatchResult(pairs=[], groups=groups)
            else:
                if how == "soft":
                    pairs = sm.drop_holes(harden(soft_match(sm)))
                else:
                    pairs = list(resolve_matcher(how)(sm))
                views = [self._pair_view(sm, i, j) for i, j in sorted(pairs)]
                if threshold is not None:
                    views = [v for v in views if _passes(v.score, threshold, sm.sense)]
                result = MatchResult(pairs=views)
        result.accounting = acc.snapshot()
        if key is not None:
            with self._cache_lock:
                self.cache[key] = deepcopy(result)
        return result

    # -- sessions -----------------------------------------------------------------

    def open_session(
        self,
        A: list,
        B: list,
        *,
        k: int = DFLT_K,
        featurize: Optional[str] = None,
        compare: Optional[str] = None,
        block: Optional[str] = None,
        sense: Optional[str] = None,
        floor: Optional[float] = None,
    ) -> SessionView:
        """Score ``A`` against ``B`` once and open an interactive review session.

        The session keeps each item's top-``k`` candidates and the optimal 1:1
        assignment; every later edit re-solves from these cached scores.

        ``floor`` is the lowest acceptable score (the highest acceptable distance for a
        ``'minimize'`` sense). Pairs past it are not candidates at all: never proposed,
        never offered for review, and a reject never pushes an item onto one — the item is
        left unmatched. Like blocking (D11), the assignment is then *lexicographic*: as
        many acceptable pairs as possible, then the best total score — so a floor can
        change the opening assignment, not just trim it. That is unlike
        :meth:`match`'s ``threshold``, which solves first and then drops weak pairs.
        """
        k = _positive_int(k, "k")
        floor = _finite_or_error(floor, "floor")
        A, B = list(A), list(B)
        acc = Accounting()
        sm = self._score(
            A, B, acc, featurize=featurize, compare=compare, block=block, sense=sense
        )
        sm = _past_floor_removed(sm, floor)
        with self._timed(acc, "candidates", len(A)):
            store = CandidateStore.from_scores(sm.data, k=k, sense=sm.sense)
        with self._timed(acc, "reoptimize", len(A) + len(B)):
            assignment = reoptimize(sm, sense=sm.sense) if A and B else []
        session = Session(
            session_id=self.mk_id(),
            A=A,
            B=B,
            scores=sm,
            store=store,
            k=k,
            assignment=assignment,
            accounting=acc,
            floor=floor,
        )
        self._save(session)
        return self.view(session.session_id)

    def view(self, session_id: str, *, n: Optional[int] = None) -> SessionView:
        """The current state of a session (``n`` caps the review queue length)."""
        s = self._load(session_id)
        return SessionView(
            session_id=s.session_id,
            n_left=len(s.A),
            n_right=len(s.B),
            sense=s.scores.sense,
            k=s.k,
            assignment=self._assignment_views(s),
            review=self.review(session_id, n=n),
            edits=list(s.edits),
            labels=list(s.labels),
            accounting=s.accounting.snapshot(),
            floor=s.floor,
        )

    def review(
        self, session_id: str, *, n: Optional[int] = None, include_confirmed=True
    ) -> list:
        """The review queue: items ordered most-uncertain first (smallest margin).

        Confirmed items come last (a human already decided them); ``include_confirmed=
        False`` drops them. Each :class:`ReviewItem` carries its top-k candidates with
        field-diff explanations — everything a review row needs.
        """
        s = self._load(session_id)
        forced = dict(_effective_constraints(s.edits).forced)
        assigned = dict(s.assignment)

        def order(i):
            m = _margin(s, i)
            return (i in forced, m if math.isfinite(m) else math.inf, i)

        items = sorted(range(len(s.A)), key=order)
        if not include_confirmed:
            items = [i for i in items if i not in forced]
        if n is not None:
            items = items[:n]
        out = []
        for i in items:
            status = (
                "confirmed"
                if i in forced
                else ("proposed" if i in assigned else "unmatched")
            )
            out.append(
                ReviewItem(
                    item=i,
                    left=s.A[i],
                    assigned=assigned.get(i),
                    status=status,
                    margin=_finite_or_none(_margin(s, i)),
                    candidates=self._candidate_views(s, i),
                )
            )
        return out

    def candidates(
        self, session_id: str, item: int, *, k: Optional[int] = None
    ) -> list:
        """Item ``item``'s top-k candidates (best first), with explanations.

        With ``k`` (larger or smaller than the session's), the item's top-k is
        **refreshed** from the cached scores and stored — no re-scoring.
        """
        if k is None:
            s = self._load(session_id)
            return self._candidate_views(s, self._check_index(item, len(s.A), "item"))
        k = _positive_int(k, "k")
        with self._session_locks[session_id]:
            s = self._load(session_id)
            item = self._check_index(item, len(s.A), "item")
            with self._timed(s.accounting, "candidates", 1):
                row = CandidateStore.from_scores(
                    s.scores.data[item : item + 1], k=k, sense=s.scores.sense
                )
                s.store.candidates[item] = row.candidates[0]
            self._save(s)
            return self._candidate_views(s, item)

    def edit(self, session_id: str, edits: list) -> EditResult:
        """Apply ``edits`` (:class:`Edit` or dicts) and re-optimize the assignment.

        Atomic: if an edit is invalid or the edits leave no feasible assignment, a
        ``ValueError`` is raised and the session is unchanged. One constrained LAP over the
        cached scores — an edit is a cheap re-solve, never a re-score. Concurrent edits of
        one session (in this process) are serialized, so none is lost.
        """
        with self._session_locks[session_id]:
            return self._edit(session_id, edits)

    def _edit(self, session_id, edits, *, label=None):
        s = self._load(session_id)
        new = []
        for e in map(_as_edit, edits):
            item = self._check_index(e.item, len(s.A), "item")
            candidate = self._check_index(e.candidate, len(s.B), "candidate")
            new.append(Edit(e.action, item, candidate, str(e.source)))
        log = s.edits + new
        acc = Accounting()
        with self._timed(acc, "reoptimize", len(s.A) + len(s.B)):
            assignment = _resolve_edits(s.scores, log)
        before = dict(s.assignment)
        after = dict(assignment)
        changed = sorted(
            i for i in set(before) | set(after) if before.get(i) != after.get(i)
        )
        s.edits, s.assignment = log, assignment
        s.accounting.merge(acc)
        if label is not None:
            s.labels.append(
                label
            )  # in the same save as its edit: no half-applied label
        self._save(s)
        return EditResult(
            session_id=s.session_id,
            assignment=self._assignment_views(s),
            changed=changed,
            accounting=s.accounting.snapshot(),
        )

    def label(
        self,
        session_id: str,
        item: int,
        candidate: int,
        is_match: bool,
        *,
        source: str = "human",
    ) -> LabelResult:
        """Record an oracle label and apply it (a match confirms, a non-match rejects).

        Returns the edit result and the next active-learning :class:`Query` — the most
        uncertain item no label has decided yet (``None`` when none is left).
        """
        if not isinstance(is_match, bool):
            raise ValueError(f"is_match must be a boolean, got {is_match!r}")
        with self._session_locks[session_id]:
            s = self._load(session_id)
            item = self._check_index(item, len(s.A), "item")
            candidate = self._check_index(candidate, len(s.B), "candidate")
            lab = Label(item, candidate, is_match, str(source))
            action = "confirm" if is_match else "reject"
            result = self._edit(
                session_id, [Edit(action, item, candidate, lab.source)], label=lab
            )
            return LabelResult(lab, result, next=self.next_query(session_id))

    def next_query(self, session_id: str) -> Optional[Query]:
        """The next pair to label: uncertainty sampling over the not-yet-decided items.

        An item is decided once it is confirmed. The pair offered is the item's current
        match when it has one, else its best candidate that is neither rejected for it nor
        confirmed for another item — so labelling never steals a confirmed match back and
        forth. An item with no such candidate left is skipped.
        """
        s = self._load(session_id)
        cs = _effective_constraints(s.edits)
        forbidden = set(cs.forbidden)
        taken = {j for _, j in cs.forced}
        for row in self.review(session_id, include_confirmed=False):
            open_ = [
                c
                for c in row.candidates
                if (row.item, c.candidate) not in forbidden and c.candidate not in taken
            ]
            open_.sort(
                key=lambda c: c.candidate != row.assigned
            )  # stable: current first
            if open_:
                return Query(row.item, row.left, open_[0], row.margin)
        return None

    def close_session(self, session_id: str) -> bool:
        """Forget a session. Returns whether it existed."""
        return self.sessions.pop(session_id, None) is not None

    # -- internals ----------------------------------------------------------------

    def _load(self, session_id) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise SessionNotFound(f"no session {session_id!r}") from None

    def _save(self, session):
        # re-assign so a write-through store (dol) persists the mutated session
        self.sessions[session.session_id] = session

    @staticmethod
    def _check_index(x, n, what):
        if isinstance(x, bool) or not isinstance(x, int):
            try:
                x = int(x)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{what} must be an integer index, got {x!r}"
                ) from None
        if not 0 <= x < n:
            raise ValueError(f"{what} index {x} is out of range [0, {n})")
        return x

    def _pair_view(self, sm, i, j, status="proposed"):
        i, j = int(i), int(j)
        return PairView(
            item=i,
            candidate=j,
            left=sm.row_labels[i],
            right=sm.col_labels[j],
            # a user-forced pair on a non-candidate (blocked-out / rejected) cell has no
            # score: reading the raw cell would report a hole as a real 0
            score=_finite_or_none(sm.score_at(i, j)) if _is_stored(sm, i, j) else None,
            status=status,
        )

    def _assignment_views(self, s):
        forced = set(_effective_constraints(s.edits).forced)
        return [
            self._pair_view(
                s.scores, i, j, "confirmed" if (i, j) in forced else "proposed"
            )
            for i, j in sorted(s.assignment)
        ]

    def _candidate_views(self, s, item):
        views = []
        for j, score in s.store.top(item, n=len(s.store.candidates.get(item, []))):
            views.append(
                CandidateView(
                    candidate=j,
                    right=s.B[j],
                    score=_finite_or_none(score),
                    explanation=explain_pair(
                        s.A[item], s.B[j], score, field_compare=self.field_compare
                    ),
                )
            )
        return views


def _margin(s, i):
    """Item ``i``'s confidence margin: top-1 vs top-2 score gap, where a session floor
    counts as a virtual runner-up — an item left with one candidate just past the floor
    is a borderline call, not a certain one (it would otherwise read as ``inf``)."""
    m = s.store.margin(i)
    top = s.store.top(i)
    if s.floor is not None and top and not math.isnan(top[0][1]):
        m = min(m, abs(top[0][1] - s.floor))
    return m


def _is_stored(sm, i, j):
    """Whether ``(i, j)`` is a candidate cell of ``sm`` (every cell of a dense matrix)."""
    if not sm.is_sparse:
        return True
    lo, hi = sm.data.indptr[i], sm.data.indptr[i + 1]
    return j in sm.data.indices[lo:hi]


def _finite_or_error(x, what):
    if x is None:
        return None
    try:
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise TypeError
        value = float(x)  # a numpy scalar is a Real; a huge int overflows here
    except (TypeError, OverflowError):
        raise ValueError(f"{what} must be a finite number, got {x!r:.80}") from None
    if not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {x!r:.80}")
    return value


def _positive_int(x, what):
    if isinstance(x, bool) or not isinstance(x, int) or x < 1:
        raise ValueError(f"{what} must be a positive integer, got {x!r}")
    return x


def asdict_shallow(obj):
    """A dataclass's fields as a dict, without deep-copying (unlike ``asdict``)."""
    return {f: getattr(obj, f) for f in obj.__dataclass_fields__}


def _passes(score, threshold, sense):
    if score is None:
        return False
    return score >= threshold if sense == "maximize" else score <= threshold


# --- JSON schemas (the SSOT the UI's Zod schemas are checked against) ----------------

#: the boundary types a surface exchanges, by name
BOUNDARY_TYPES = {
    t.__name__: t
    for t in (
        Edit,
        Label,
        FieldDiff,
        Explanation,
        CandidateView,
        PairView,
        ReviewItem,
        Query,
        TierCost,
        Accounting,
        MatchResult,
        SessionView,
        EditResult,
        LabelResult,
    )
}


def json_schemas():
    """JSON Schemas of the boundary types, keyed by type name.

    Generated from the dataclasses with ``pydantic`` (part of ``equate[service]``). The UI
    keeps a snapshot of the *field names* (``ui/src/schema/fields.json``) and a test on
    each side fails when the Python types and the Zod schemas drift apart.
    """
    from equate._dependencies import require

    pydantic = require("pydantic", extra="service", purpose="JSON schema generation")
    return {
        name: pydantic.TypeAdapter(t).json_schema()
        for name, t in BOUNDARY_TYPES.items()
    }


def boundary_fields():
    """``{type name: [field names]}`` for the boundary types — the cross-language
    contract the UI's Zod schemas are tested against (no pydantic needed)."""
    return {name: list(t.__dataclass_fields__) for name, t in BOUNDARY_TYPES.items()}


# --- The HTTP surface (qh) -----------------------------------------------------------


def mk_app(service: Optional[MatchService] = None, *, cors_origins=DFLT_CORS_ORIGINS):
    """Mount a :class:`MatchService` as an HTTP API (FastAPI, built with ``qh``).

    Routes (JSON in, JSON out)::

        POST   /match                                       stateless match
        POST   /sessions                                    open a review session
        GET    /sessions/{session_id}                       session state + review queue
        DELETE /sessions/{session_id}                       close it
        GET    /sessions/{session_id}/candidates/{item}     an item's top-k
        POST   /sessions/{session_id}/candidates/{item}     refresh it with {"k": ...}
        POST   /sessions/{session_id}/edits                 {"edits": [...]} -> re-optimized
        POST   /sessions/{session_id}/label                 oracle label -> next query
        GET    /sessions/{session_id}/next                  the next query to label

    An unknown session is a 404; an invalid or infeasible edit a 422. ``cors_origins``
    lets the review UI's dev server (vite, port 5173) call it; pass ``()`` to disable.
    Requires ``pip install 'equate[service]'``.
    """
    from equate._dependencies import require

    qh = require("qh", extra="service", purpose="the HTTP service")
    HTTPException = require("fastapi", extra="service").HTTPException
    svc = service or MatchService()
    J = to_jsonable

    def http_errors(adapter):
        # qh turns any exception into a 500 but passes HTTPException through, so map
        # the service's error contract here: unknown session -> 404, bad input -> 422
        @wraps(adapter)
        def wrapped(*args, **kwargs):
            try:
                return adapter(*args, **kwargs)
            except SessionNotFound as e:
                raise HTTPException(404, detail=str(e.args[0]))
            except (KeyError, ValueError, TypeError) as e:
                # an unknown strategy name (registry KeyError), a bad value or shape
                raise HTTPException(422, detail=str(e.args[0] if e.args else e))

        return wrapped

    # thin adapters: HTTP-shaped signatures (path params arrive as strings) -> the service
    @http_errors
    def match(
        A: list,
        B: list,
        featurize: Optional[str] = None,
        compare: Optional[str] = None,
        block: Optional[str] = None,
        how: str = "assign",
        threshold: Optional[float] = None,
        sense: Optional[str] = None,
    ):
        return J(
            svc.match(
                A,
                B,
                featurize=featurize,
                compare=compare,
                block=block,
                how=how,
                threshold=threshold,
                sense=sense,
            )
        )

    @http_errors
    def open_session(
        A: list,
        B: list,
        k: int = DFLT_K,
        featurize: Optional[str] = None,
        compare: Optional[str] = None,
        block: Optional[str] = None,
        sense: Optional[str] = None,
        floor: Optional[float] = None,
    ):
        return J(
            svc.open_session(
                A,
                B,
                k=k,
                featurize=featurize,
                compare=compare,
                block=block,
                sense=sense,
                floor=floor,
            )
        )

    @http_errors
    def get_session(session_id: str):
        return J(svc.view(session_id))

    @http_errors
    def close_session(session_id: str):
        return {"closed": svc.close_session(session_id)}

    @http_errors
    def get_candidates(session_id: str, item: str):
        return J(svc.candidates(session_id, _int(item, "item")))

    @http_errors
    def refresh_candidates(session_id: str, item: str, k: int = DFLT_K):
        return J(svc.candidates(session_id, _int(item, "item"), k=k))

    @http_errors
    def post_edits(session_id: str, edits: list):
        return J(svc.edit(session_id, edits))

    @http_errors
    def post_label(
        session_id: str,
        item: int,
        candidate: int,
        is_match: bool,
        source: str = "human",
    ):
        return J(svc.label(session_id, item, candidate, is_match, source=source))

    @http_errors
    def next_query(session_id: str):
        return J(svc.next_query(session_id))

    R = qh.RouteConfig
    s = "/sessions/{session_id}"
    app = qh.mk_app(
        {
            match: R(path="/match", methods=["POST"]),
            open_session: R(path="/sessions", methods=["POST"]),
            get_session: R(path=s, methods=["GET"]),
            close_session: R(path=s, methods=["DELETE"]),
            get_candidates: R(path=s + "/candidates/{item}", methods=["GET"]),
            refresh_candidates: R(path=s + "/candidates/{item}", methods=["POST"]),
            post_edits: R(path=s + "/edits", methods=["POST"]),
            post_label: R(path=s + "/label", methods=["POST"]),
            next_query: R(path=s + "/next", methods=["GET"]),
        }
    )

    if cors_origins:
        cors = require("fastapi.middleware.cors", extra="service")
        app.add_middleware(
            cors.CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.state.match_service = svc
    return app


def _int(x, what):
    try:
        return int(x)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be an integer index, got {x!r}") from None


def serve(*, host="127.0.0.1", port=8000, service: Optional[MatchService] = None):
    """Run the HTTP service with uvicorn (``python -m equate.service serve``)."""
    from equate._dependencies import require

    uvicorn = require("uvicorn", extra="service", purpose="serving the HTTP API")
    uvicorn.run(mk_app(service), host=host, port=port)


if __name__ == "__main__":
    import sys

    cmd, *rest = sys.argv[1:] or ["serve"]
    if cmd == "serve":
        kw = dict(a.split("=", 1) for a in rest)  # host=0.0.0.0 port=8001
        serve(host=kw.get("host", "127.0.0.1"), port=int(kw.get("port", 8000)))
    elif cmd == "fields":
        print(json.dumps(boundary_fields(), indent=2))
    elif cmd == "schema":
        print(json.dumps(json_schemas(), indent=2))
    else:
        raise SystemExit(f"usage: python -m equate.service [serve|fields|schema]")
