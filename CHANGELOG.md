# Changelog

All notable changes to `equate` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions match the
releases CI publishes to PyPI.

## Unreleased

**Surfaces: an HTTP service and a review UI** (#12; doc 10 §5, decision D10).

- `equate.service.MatchService` exposes the matching core as JSON-in/JSON-out operations. `match` is stateless and cached by content. A review session (`open_session`) scores the two collections once and keeps each item's top-k candidates. Then `candidates` reads or refreshes an item's top-k, `edit` applies `confirm` / `reject` / `reassign` / `clear` edits and returns the re-optimized assignment plus the items it changed, and `label` records an oracle's verdict and returns the next most uncertain pair (`next_query`). Edits re-solve from the cached scores; they never re-score. A rejected pair becomes a non-candidate, so rejecting every candidate of an item leaves it unmatched rather than making the problem infeasible.
- Every response carries an `Accounting` of calls, seconds and items per tier (`score`, `match`, `candidates`, `reoptimize`); `MatchService(on_cost=...)` is the hook for an external budget. Candidates come with an `Explanation` whose `details['fields']` is a per-field diff (`explain_pair`). Records (dicts) work on every named featurizer and comparator through their joined values (`record_text`).
- Sessions live in any `MutableMapping` (`sessions=`); a session pickles, so a `dol` store persists it. Edits to one session are serialized (a per-session lock), and an edit or a label is atomic: a failed one changes nothing. Each `Edit` records its `source` (`'human'`, `'llm'`, ...).
- `equate.service.mk_app()` mounts the service as an HTTP API with qh. It is the new extra `equate[service]` (qh, uvicorn) and is imported only when used. Run it with `python -m equate.service serve`. A missing session is a 404, and bad input or an unknown strategy is a 422.
- `equate.score_matrix(A, B, ...)` is the scoring half of `match`, now public. `match` itself is unchanged.
- `ui/` is a review UI built with React, zodal and shadcn: a review queue ordered by uncertainty, side-by-side field diffs, optimistic confirm/reject/reassign, and a labelling widget. Its Zod schemas are checked against the Python dataclasses on both sides.

**Graph families** (part of #10; doc 12 §3-4, decision D8): matching for *graphs*, at both arities.

- `equate.align(G1, G2, *, how='isorank', node_affinity=None, seeds=None)` adds the second matcher arity, a **quadratic matcher**. It aligns the *nodes* of two graphs and returns a `Matching` whose `.labeled_pairs()` gives `(G1 node, G2 node)` and whose `.plan` holds the node-pair scores. `'isorank'` is IsoRank in-house: numpy, an optional non-negative prior, and seeds that are kept and propagate. `'rrwm'` is pygmtools' reweighted random walk on the full quadratic affinity (new extra `equate[graph-matching]`; it uses the numpy backend and leaves the global backend alone). Both extract the final 1:1 correspondence with a **flat** matcher (`extract='optimal'` by default, any `resolve_matcher` spec). They live in a new `quadratic_matchers` registry, and the protocol is `equate.base.QuadraticMatcher`.
- `featurize='wl'` is the Weisfeiler-Lehman subtree feature map (`WLFeaturizer`, core). It is a graph kernel used as a featurizer, so `match(graphs_a, graphs_b, featurize='wl')` takes the ordinary indexable cosine route. `featurize='grakel'` exposes any GraKeL kernel as an empirical kernel map (new extra `equate[graph-kernels]`).
- Graphs can be networkx graphs (duck-typed; networkx is not required), `{node: neighbors}` mappings, or adjacency matrices (`equate._graph.as_graph`). Node labels come from `label=`, either an attribute name or a callable.
- Not included: a PLANETALIGN backend (it is not on PyPI), FINAL/REGAL/CONE-Align, and Gromov-Wasserstein in `soft_match`. These are tracked as a follow-up.

**Sequence-alignment comparators** (`equate.compare.sequence`, part of #10; doc 12, decision D8). Matching one level down: each one aligns the *elements* of two sequences, then scores the alignment. They are `direct(h)` comparators, and the collection-level matchers are unchanged.

- `dtw`: exact DTW inside a Sakoe-Chiba band by default (`window=0.1` of the longer series, at least one cell; an int is an inclusive half-width in cells, so `0` means no warping; `None` means unconstrained). It is not FastDTW. Steps can be scalars or vectors, and `local_cost=` accepts any objects. `subsequence_dtw` finds the best sub-span (directional). `soft_dtw` is Cuturi & Blondel's soft-minimum; it is not a distance.
- `needleman_wunsch` (global) and `smith_waterman` (local) align strings or token lists, with `match`/`mismatch`/`gap` scores or a `substitution(x, y)` callable.
- Each comparator returns a float. Its `*_alignment` twin (none for `soft_dtw`, which averages over all paths) (also `dtw.align` and so on) returns the new `Alignment(score, path, sense)` dataclass, which carries the element-level path, with `None` marking a gap.
- The core is pure numpy. `backend='dtaidistance'`/`'tslearn'` (new extra `equate[timeseries]`) and `backend='biopython'` (new extra `equate[bio]`) are drop-in engines, imported only on use.
- **Behaviour change:** `match(..., sense=None)` is the new default and reads the sense from the comparator's declared polarity. Every *distance* comparator passed as `compare=` is now minimized: `'dtw'`, `'levenshtein_distance'`, `'angular'`, `'haversine'`, and a `functools.partial` of any of them. Before, `match(A, B, compare='levenshtein_distance')` silently maximized the distance. The default vector route (no `compare=`) and similarity comparators are unaffected, and an explicit `sense=` still wins. `dedupe` uses the same `polarity_sense` helper.
- A comparator configured through the registry (`resolve_comparator('dtw', window=None)`) keeps its `.meta`.

**`equate.evaluate`**, an evaluation and benchmark harness (#11). It is dependency-light: numpy and scipy, with scikit-learn optional via `equate[eval]`.

- `Scores` is a frozen dataclass: precision, recall, F1 derived from them, and support. Every metric returns it, so F1 never travels alone.
- Pairwise: `pairwise_prf`. Cluster: `bcubed` (the default), `ceaf` (entity or mention; the alignment goes through the `matchers` registry), and `cluster_metrics` (adds ARI, V-measure, homogeneity and completeness, from sklearn when installed and numpy otherwise; the two agree on every non-empty input, and an empty one is vacuously 1.0).
- Granularity is explicit: `pairs_to_labels` (union-find closure), `labels_to_pairs`, and `cross_pairs_to_labels` for two tables.
- Threshold-free and ranking: `pr_auc` (tie-aware average precision), `ranking_metrics` (recall@k, MRR, MAP), and `select_threshold`, kept separate from scoring so it is chosen on validation data.
- Leakage: `entity_split` splits by entity, so test entities are unseen. `assert_no_leakage` is the post-hoc guard.
- Benchmarks: a `benchmarks` registry with a generated `toy_names`, plus `benchmark_matcher`. It scores the test part only, pairwise, and skips with a warning any seed whose test part has no gold pair. It fits a trainable matcher on train and validation only (after asserting no leakage), and reports a wall-clock cost column, the test base rate and the difficulty, with mean ± std over seeds. Loaders for Abt-Buy, Amazon-Google, DBLP-ACM and WDC Products are deferred to #15.
- `blocking_metrics` is re-exported from `equate.block`, the single implementation.

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
