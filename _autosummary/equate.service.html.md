# equate.service

The service layer: equate’s matching core as JSON-in/JSON-out operations.

This module is the **surface-free** half of the HTTP service (doc 10 §5, decision D10).
Every operation is a plain method taking JSON-able arguments and returning a dataclass of
JSON-able fields, so any surface can project it: [`mk_app()`](#equate.service.mk_app) mounts it as an HTTP API
with `qh` (`pip install 'equate[service]'`), and the review UI in `ui/` consumes the
same payloads through Zod schemas generated from [`json_schemas()`](#equate.service.json_schemas).

Two kinds of operation:

- **stateless** — [`MatchService.match()`](#equate.service.MatchService.match): two collections in, scored pairs out (the
  batch core, cached by content);
- **session-scoped** — the interactive loop of doc 08. [`MatchService.open_session()`](#equate.service.MatchService.open_session)
  scores the two collections *once* and keeps a [`CandidateStore`](equate.interactive.html.md#equate.interactive.CandidateStore)
  of top-k candidates; [`candidates()`](#equate.service.MatchService.candidates) reads (or refreshes) one item’s
  top-k; [`edit()`](#equate.service.MatchService.edit) applies confirm / reject / reassign edits and returns
  the locally re-optimized assignment (one constrained LAP over the cached scores, never a
  re-score); [`label()`](#equate.service.MatchService.label) is the active-learning `Oracle` endpoint: it
  records a human/LLM/crowd label and returns the next most-informative query.

Every response carries an [`Accounting`](#equate.service.Accounting) of the time and item counts each tier
(`score`, `match`, `reoptimize`, …) cost, so a caller can budget expensive
cascades (doc 06). Sessions live in any `MutableMapping` (`sessions=`), so a
persistent `dol` store replaces the in-memory dict without touching this code.

```pycon
>>> svc = MatchService()
>>> result = svc.match(['apple pie', 'banana split'], ['banana splitt', 'aple pie'])
>>> [(p.left, p.right) for p in result.pairs]
[('apple pie', 'aple pie'), ('banana split', 'banana splitt')]
```

### Functions

| [`explain_pair`](#equate.service.explain_pair)(left, right, score, \*[, ...])   | A structured [`Explanation`](equate.base.html.md#equate.base.Explanation) of a pair: a per-field diff.   |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [`record_text`](#equate.service.record_text)(obj)                              | Text of `obj`: a mapping's non-empty values joined by spaces, else `str(obj)`.                                                   |
| [`to_jsonable`](#equate.service.to_jsonable)(obj)                              | Convert a boundary dataclass (or nested containers of them) to plain JSON data.                                                  |
| [`json_schemas`](#equate.service.json_schemas)()                                | JSON Schemas of the boundary types, keyed by type name.                                                                          |
| [`mk_app`](#equate.service.mk_app)([service, cors_origins])               | Mount a [`MatchService`](#equate.service.MatchService) as an HTTP API (FastAPI, built with `qh`).                 |
| [`serve`](#equate.service.serve)(\*[, host, port, service])              | Run the HTTP service with uvicorn (`python -m equate.service serve`).                                                            |

### Classes

| [`MatchService`](#equate.service.MatchService)(\*[, sessions, cache, mk_id, ...])   | equate's matching core as stateless + session-scoped JSON-able operations.      |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`Accounting`](#equate.service.Accounting)([tiers])                               | Per-tier cost/latency ledger.                                                   |
| [`TierCost`](#equate.service.TierCost)([calls, seconds, items])                 | What one tier (`score`, `match`, `reoptimize`, ...) has cost so far.            |
| [`Edit`](#equate.service.Edit)(action, item, candidate[, source])           | One user edit of the assignment.                                                |
| [`Label`](#equate.service.Label)(item, candidate, is_match[, source])        | An oracle's verdict on one `item -> candidate` pair (active learning).          |
| [`FieldDiff`](#equate.service.FieldDiff)(field, left, right, similarity)         | One field of a side-by-side comparison: both values and their similarity.       |
| [`CandidateView`](#equate.service.CandidateView)(candidate, right, score[, ...])     | A scored candidate for an item, with the right-hand object and an explanation.  |
| [`PairView`](#equate.service.PairView)(item, candidate, left, right, score)     | A matched pair: indices, both objects, the score, and whether a user fixed it.  |
| [`ReviewItem`](#equate.service.ReviewItem)(item, left, assigned, status, ...)     | One row of the review queue: an item, its current match and its alternatives.   |
| [`Query`](#equate.service.Query)(item, left, candidate, margin)              | An active-learning query: the pair an oracle should label next.                 |
| [`MatchResult`](#equate.service.MatchResult)(pairs[, groups, accounting, cached])  | Stateless match output: `pairs` (or `groups` for `how='cluster'`).              |
| [`SessionView`](#equate.service.SessionView)(session_id, n_left, n_right, ...)     | A session's state: its current assignment and the review queue to work through. |
| [`EditResult`](#equate.service.EditResult)(session_id, assignment, changed, ...)  | The result of an edit: the re-optimized assignment and what it changed.         |
| [`LabelResult`](#equate.service.LabelResult)(label, result, next)                  | The result of a label: what was recorded, the edit result, and the next query.  |
| [`Session`](#equate.service.Session)(session_id, A, B, scores, store, k, ...)  | Server-side session state.                                                      |

### Exceptions

| [`SessionNotFound`](#equate.service.SessionNotFound)   | No session with that id (an HTTP 404).   |
|--------------------------------------------------------------------|------------------------------------------|

### *class* equate.service.Accounting(tiers=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per-tier cost/latency ledger. `tiers` maps a tier name to its [`TierCost`](#equate.service.TierCost).

Plain data (it is part of a session, which must pickle into any store); an external
budget or metrics sink attaches through the `hook` argument of [`record()`](#equate.service.Accounting.record).

#### merge(other)

Add `other`’s tiers into this ledger (a committed operation’s costs).

#### record(tier, seconds, items=0, , hook=None)

Add one call of `tier` that took `seconds` over `items` items, and tell
`hook(tier, seconds, items)` if one is given.

#### snapshot()

A copy, so a response never aliases the session’s live ledger.

#### timed(tier, items=0, , hook=None)

Context manager recording the wall-clock time of its body under `tier`.

### *class* equate.service.CandidateView(candidate, right, score, explanation=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A scored candidate for an item, with the right-hand object and an explanation.

### *class* equate.service.Edit(action, item, candidate, source='human')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One user edit of the assignment.

- `confirm` — force `item -> candidate` (a confirmed match);
- `reject` — forbid `item -> candidate`;
- `reassign` — confirm a *different* candidate for `item` (its previous forced
  match, and any other item forced onto `candidate`, are released);
- `clear` — drop any constraint on `item -> candidate` (undo).

`item` indexes the left collection, `candidate` the right one; `source` says who
made the edit (`'human'`, `'llm'`, …) — provenance for a later precedence policy.

### *class* equate.service.EditResult(session_id, assignment, changed, accounting)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of an edit: the re-optimized assignment and what it changed.

`changed` lists the items whose match moved, so a UI re-renders only that block.

### *class* equate.service.FieldDiff(field, left, right, similarity)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One field of a side-by-side comparison: both values and their similarity.

### *class* equate.service.Label(item, candidate, is_match, source='human')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An oracle’s verdict on one `item -> candidate` pair (active learning).

### *class* equate.service.LabelResult(label, result, next)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of a label: what was recorded, the edit result, and the next query.

### *class* equate.service.MatchResult(pairs, groups=None, accounting=Accounting(), cached=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Stateless match output: `pairs` (or `groups` for `how='cluster'`).

### *class* equate.service.MatchService(\*, sessions=None, cache='default', mk_id=<function MatchService.<lambda>>, field_compare='ratio', on_cost=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

equate’s matching core as stateless + session-scoped JSON-able operations.

Every seam is a keyword argument with a working default:

- `sessions` — where sessions live (any `MutableMapping`; default an in-memory
  dict; pass a pickle-valued `dol` store to persist them);
- `cache` — where stateless match results are cached (default a bounded LRU; pass
  `None` to disable);
- `mk_id` — session-id factory (default `uuid4().hex`);
- `field_compare` — the per-field comparator used in explanations;
- `on_cost` — `(tier, seconds, items)` hook called on every accounted tier.

#### candidates(session_id, item, , k=None)

Item `item`’s top-k candidates (best first), with explanations.

With `k` (larger or smaller than the session’s), the item’s top-k is
**refreshed** from the cached scores and stored — no re-scoring.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

#### close_session(session_id)

Forget a session. Returns whether it existed.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### edit(session_id, edits)

Apply `edits` ([`Edit`](#equate.service.Edit) or dicts) and re-optimize the assignment.

Atomic: if an edit is invalid or the edits leave no feasible assignment, a
`ValueError` is raised and the session is unchanged. One constrained LAP over the
cached scores — an edit is a cheap re-solve, never a re-score. Concurrent edits of
one session (in this process) are serialized, so none is lost.

* **Return type:**
  [`EditResult`](#equate.service.EditResult)

#### label(session_id, item, candidate, is_match, , source='human')

Record an oracle label and apply it (a match confirms, a non-match rejects).

Returns the edit result and the next active-learning [`Query`](#equate.service.Query) — the most
uncertain item no label has decided yet (`None` when none is left).

* **Return type:**
  [`LabelResult`](#equate.service.LabelResult)

#### match(A, B, , featurize=None, compare=None, block=None, how='assign', threshold=None, sense=None)

Match two collections (stateless). Mirrors `equate.match()`.

`how` is `'assign'` (optimal 1:1), `'greedy'`, `'stable'`, `'soft'` or
`'cluster'` (then `groups` lists the entity clusters over the pooled `A + B`
indices, `B` offset by `len(A)`). Pairs below `threshold` (above it, for a
`'minimize'` sense) are dropped. Results are cached by content.

* **Return type:**
  [`MatchResult`](#equate.service.MatchResult)

#### next_query(session_id)

The next pair to label: uncertainty sampling over the not-yet-decided items.

An item is decided once it is confirmed. The pair offered is the item’s current
match when it has one, else its best candidate that is neither rejected for it nor
confirmed for another item — so labelling never steals a confirmed match back and
forth. An item with no such candidate left is skipped.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Query`](#equate.service.Query)]

#### open_session(A, B, , k=5, featurize=None, compare=None, block=None, sense=None, floor=None)

Score `A` against `B` once and open an interactive review session.

The session keeps each item’s top-`k` candidates and the optimal 1:1
assignment; every later edit re-solves from these cached scores.

`floor` is the lowest acceptable score (the highest acceptable distance for a
`'minimize'` sense). Pairs past it are not candidates at all: never proposed,
never offered for review, and a reject never pushes an item onto one — the item is
left unmatched. Like blocking (D11), the assignment is then *lexicographic*: as
many acceptable pairs as possible, then the best total score — so a floor can
change the opening assignment, not just trim it. That is unlike
[`match()`](#equate.service.MatchService.match)’s `threshold`, which solves first and then drops weak pairs.

* **Return type:**
  [`SessionView`](#equate.service.SessionView)

#### review(session_id, , n=None, include_confirmed=True)

The review queue: items ordered most-uncertain first (smallest margin).

Confirmed items come last (a human already decided them); `include_confirmed=
False` drops them. Each [`ReviewItem`](#equate.service.ReviewItem) carries its top-k candidates with
field-diff explanations — everything a review row needs.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

#### view(session_id, , n=None)

The current state of a session (`n` caps the review queue length).

* **Return type:**
  [`SessionView`](#equate.service.SessionView)

### *class* equate.service.PairView(item, candidate, left, right, score, status='proposed')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A matched pair: indices, both objects, the score, and whether a user fixed it.

### *class* equate.service.Query(item, left, candidate, margin)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An active-learning query: the pair an oracle should label next.

### *class* equate.service.ReviewItem(item, left, assigned, status, margin, candidates)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One row of the review queue: an item, its current match and its alternatives.

`margin` is the top-1/top-2 score gap (`None` when there is no rival) — small
means ambiguous, which is why the queue puts it first.

### *class* equate.service.Session(session_id, A, B, scores, store, k, assignment, edits=list(), labels=list(), accounting=Accounting(), floor=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Server-side session state. Not a boundary type: [`SessionView`](#equate.service.SessionView) is its view.

Everything is plain data (no open handles), so a session pickles into any `dol`
store passed as `MatchService(sessions=...)`.

### *exception* equate.service.SessionNotFound

Bases: [`KeyError`](https://docs.python.org/3/builtins/exceptions.html#KeyError)

No session with that id (an HTTP 404).

### *class* equate.service.SessionView(session_id, n_left, n_right, sense, k, assignment, review, edits, labels, accounting, floor=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A session’s state: its current assignment and the review queue to work through.

### *class* equate.service.TierCost(calls=0, seconds=0.0, items=0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What one tier (`score`, `match`, `reoptimize`, …) has cost so far.

### equate.service.explain_pair(left, right, score, , field_compare='ratio')

A structured [`Explanation`](equate.base.html.md#equate.base.Explanation) of a pair: a per-field diff.

Records (mappings) are compared field by field over the union of their keys; any other
object is one field named `value`. A side that is absent, `None` or `''` has no
similarity (`None`): missing is not the same as different. `details['fields']` is a list of
[`FieldDiff`](#equate.service.FieldDiff) dicts — the payload a UI renders as a side-by-side diff.

```pycon
>>> e = explain_pair({'name': 'Jon'}, {'name': 'Jon', 'city': 'Paris'}, 0.8)
>>> [(f['field'], f['similarity']) for f in e.details['fields']]
[('name', 1.0), ('city', None)]
```

### equate.service.json_schemas()

JSON Schemas of the boundary types, keyed by type name.

Generated from the dataclasses with `pydantic` (part of `equate[service]`). The UI
keeps a snapshot of the *field names* (`ui/src/schema/fields.json`) and a test on
each side fails when the Python types and the Zod schemas drift apart.

### equate.service.mk_app(service=None, , cors_origins=('http://localhost:5173', 'http://127.0.0.1:5173'))

Mount a [`MatchService`](#equate.service.MatchService) as an HTTP API (FastAPI, built with `qh`).

Routes (JSON in, JSON out):

```default
POST   /match                                       stateless match
POST   /sessions                                    open a review session
GET    /sessions/{session_id}                       session state + review queue
DELETE /sessions/{session_id}                       close it
GET    /sessions/{session_id}/candidates/{item}     an item's top-k
POST   /sessions/{session_id}/candidates/{item}     refresh it with {"k": ...}
POST   /sessions/{session_id}/edits                 {"edits": [...]} -> re-optimized
POST   /sessions/{session_id}/label                 oracle label -> next query
GET    /sessions/{session_id}/next                  the next query to label
```

An unknown session is a 404; an invalid or infeasible edit a 422. `cors_origins`
lets the review UI’s dev server (vite, port 5173) call it; pass `()` to disable.
Requires `pip install 'equate[service]'`.

### equate.service.record_text(obj)

Text of `obj`: a mapping’s non-empty values joined by spaces, else `str(obj)`.

```pycon
>>> record_text({'name': 'Jon Smith', 'city': 'Paris', 'zip': None})
'Jon Smith Paris'
>>> record_text(42)
'42'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### equate.service.serve(, host='127.0.0.1', port=8000, service=None)

Run the HTTP service with uvicorn (`python -m equate.service serve`).

### equate.service.to_jsonable(obj)

Convert a boundary dataclass (or nested containers of them) to plain JSON data.

Non-finite floats become `None` (JSON has no `inf`/`nan`); tuples become lists;
numpy scalars become Python numbers.
