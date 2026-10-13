# equate.interactive.candidate_store

The CandidateStore: per left-item top-k candidates with scores — the SSOT for review
and interactive re-optimization.

Retaining the top-k candidates (not just the single best) is what lets a user edit a match
and have the system re-solve from the remembered alternatives (doc 08-interactive). It is a
plain dataclass so it serializes (e.g. persist with `dol.cache_this` to resume a session).

### Classes

| [`CandidateStore`](#equate.interactive.candidate_store.CandidateStore)(candidates[, sense, ...])   | Per left index `i`, its top-k `(right_index, score)` candidates, best-first.   |
|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|

### *class* equate.interactive.candidate_store.CandidateStore(candidates, sense='maximize', row_labels=None, col_labels=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Per left index `i`, its top-k `(right_index, score)` candidates, best-first.

Iterating yields the top-1 `(i, j)` pairs, so `dict(store)` gives the greedy 1:n
match. `row_labels` / `col_labels` name the two collections.

#### *classmethod* from_scores(scores, , k=5, sense='maximize', row_labels=None, col_labels=None)

Build a store of each row’s top-`k` candidates from a (dense or sparse) score
matrix.

A **sparse** matrix considers only its stored (candidate) cells — which is exactly
the blocking semantics (an absent cell is not a candidate), so for a blocked matrix
this intentionally differs from the dense all-cells behavior regardless of `sense`.

#### margin(i)

Confidence gap between item `i`’s top-1 and top-2 candidate scores.

A *small* margin means the top choice is uncertain (two close candidates) — the
signal for review triage and active-learning query selection. An item with 0 or 1
candidates is unambiguous (no rival to be confused with), so its margin is `+inf`
— it sorts LAST in a most-uncertain-first queue, never spuriously to the front.

#### top(i, n=1)

The top-`n` `(right_index, score)` candidates for left item `i`.
