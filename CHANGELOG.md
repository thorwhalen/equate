# Changelog

All notable changes to `equate` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions match the
releases CI publishes to PyPI.

## 0.0.14 — 2026-07-17

Hardening of **blocked (sparse) matching**: a blocker produces a sparse score
matrix whose absent cells are *non-candidates* (holes), not scores of `0`. This
release makes it structurally impossible for a matcher to treat a hole as a real
pair, and fixes a subtler defect where a hole could still be *preferred* over a
real one. See decision register **D11** (`docs/research/11-…`).

### Fixed

- **A blocked cell is never selected.** Every matcher now consumes the score
  matrix through sanctioned worst-casing views and drops any assignment that lands
  on a hole, so a structurally-absent cell can be neither preferred nor assigned —
  even when a full-cardinality solver (LAP / max-weight) is forced onto one. This
  closes a bug class where densifying a sparse matrix before cost conversion
  silently turned holes into real `0` scores.
- **The "optimal" matcher no longer loses to an available all-real matching.** The
  hole penalty was one unit worse than the worst real *cell*, but a matcher
  optimizes a *total*, so the solver would buy a hole to save more elsewhere and
  return a matching with fewer pairs and a worse score than one that was there for
  the taking (the default `how='assign'` matcher could be beaten by `greedy`). The
  penalty is now a big-M. This only affected **unbounded** score comparators
  (e.g. `dot`, which ships with `bounded=False`); `[0, 1]` comparators were safe.
- **`equate.util.maximal_matching` handles sparse input and orientation correctly.**
  It built its graph cell-by-cell (reading a hole as a real `0.0` edge) and unpacked
  networkx's *unordered* edges without orienting them (transposed pairs on
  asymmetric / rectangular input). It now builds edges from candidate cells only and
  orients each `(row, col)` pair.
- **The score matrix accepts every dense array-like and sparse format.** A list of
  rows, or a `coo` / `lil` / `dok` matrix, is normalized once at construction, so
  matchers no longer crash on inputs they used to accept.
- `equate.completion`'s matcher functions are now thin re-exports of the canonical
  `equate.util` ones, instead of drifting verbatim copies that still carried the
  transposed-pair and hole-as-`0` bugs.

### Changed

- **Blocked matching is now lexicographic:** it uses as many real candidate pairs as
  possible, then optimizes the score among those. An absent cell is not a bad option —
  it is *not an option*. Results can differ from earlier releases for blocked inputs
  with unbounded scores.
- **`resolve_matcher(spec)` returns a runner, not `spec` itself.** It previously passed
  a callable through unchanged (`resolve_matcher(f) is f`); it now returns a wrapper
  that enforces the hole-dropping invariant. The wrapper carries the matcher's
  `__name__` / `__doc__`, and the original is recoverable via `__wrapped__`.
- A user-supplied `how=<callable>` matcher receives the raw stored scores again (with
  only holes worst-cased). An interim build rescaled the whole matrix by `-max`, which
  silently changed results for matchers reading absolute score values; that is fixed.

## 0.0.13 and earlier

See the [commit history](https://github.com/thorwhalen/equate/commits/master).
