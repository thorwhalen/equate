---
name: equate-dev-add-strategy
description: How to add a new strategy to equate — a featurizer, comparator, blocker, matcher, or clusterer — the equate way. Covers picking the stage/subpackage, satisfying the callable protocol + declaring strategy metadata, registering it in the name-keyed registry with a LAZY factory, wiring an optional heavy dependency as a pip extra with a check_requirements-style error (never a raw ImportError), and adding deterministic tests that skip when the optional dep is absent. Triggers when adding, registering, or wrapping a matching strategy or a third-party library behind an equate extra.
metadata:
  audience: developers
  project: equate
---

# Adding a strategy to equate

Adding a featurizer / comparator / blocker / matcher / clusterer is the most
common recurring task in equate. Every strategy follows the **same shape** so the
core stays open-closed and the default install stays light. Read
[[equate-dev-architecture]] first for the stage contracts.

## The invariants (every strategy obeys these)

1. **A strategy is a plain callable** satisfying its stage protocol (see
   [[equate-dev-architecture]]). A bare lambda must be a legal strategy — no
   subclassing required.
2. **It is reachable two ways:** by a **string name** via the stage registry, or by
   **passing the callable directly** to the facade. Both must work.
3. **Heavy dependencies are optional and lazy.** The import happens *inside* the
   factory/first-use, guarded by `check_requirements`; a missing dep raises an
   actionable error naming the extra + install command, never a bare `ImportError`.
   The default install must not gain a new hard dependency.
4. **It declares metadata** so the framework can pick a *legal* index/matcher: a
   featurizer declares output kind / dim / normalized / compatible-metric; a
   comparator declares polarity / bounded / is_metric / is_symmetric; a matcher
   declares the objective and required score `sense`.
5. **Scores respect the sense contract** — similarity is higher-is-better; matchers
   route through the shared `to_cost(scores, *, sense)` SSOT, never a bespoke
   conversion.
6. **A matcher consumes a `ScoreMatrix` and densifies *only* via its worst-casing views**
   — `dense_cost()` / `dense_similarity()` / `candidate_mask()` / `stored_entries()`; mark
   it `@scorematrix_matcher`. **NEVER `.toarray()` the raw scores and then call `to_cost`**
   — that drops the blocked matrix's hole-worst-casing (holes fill with `0` and out-rank
   real negatives / look cheapest; decision register **D11**). Drop any assignment landing
   on a `~candidate_mask()` cell. The registry-wide conformance sweep in
   `tests/test_matching.py` will fail your matcher if it densifies-then-`to_cost`.

## Recipe

### 1. Pick the stage and file
Map the strategy to a stage and drop it in that subpackage (see the module sketch
in [[equate-dev-architecture]]): `featurize/`, `compare/`, `block/`, `match/`, or
`cluster/`. New third-party wrap → its own module (e.g. `compare/string.py` hosts
`rapidfuzz`/`jellyfish` wrappers).

### 2. Write the callable + metadata
Satisfy the stage protocol exactly. Attach metadata (a small dataclass or
registry-entry fields). Keep pure-Python fallbacks dependency-free (e.g. `difflib`
lives in core; `rapidfuzz` is the optional fast path).

### 3. Register it with a LAZY factory
Register `name -> factory` in the stage registry. The factory returns the callable
and is only *called* on use, so the heavy import is deferred:

```python
# compare/__init__.py  (illustrative — match the real registry once it exists)
@register_comparator("jaro_winkler", polarity="similarity", bounded=True,
                     is_metric=False, is_symmetric=True)
def _jaro_winkler_factory():
    rapidfuzz = require("rapidfuzz", extra="fuzzy")   # check_requirements wrapper
    from rapidfuzz.distance import JaroWinkler
    return lambda a, b: JaroWinkler.normalized_similarity(a, b)
```

### 4. Wire the optional dependency (if any)
- Add the extra to `pyproject.toml` under `[project.optional-dependencies]`
  (e.g. `fuzzy = ["rapidfuzz"]`, `ann = ["hnswlib"]`, `ot = ["POT"]`).
- Import **only inside** the factory, via the `check_requirements`/`require(...)`
  helper that raises:
  `"<name> needs the '<extra>' extra — install: pip install 'equate[<extra>]'"`.
- **Never** add it to core `dependencies`. **Never** import a copyleft/heavyweight
  engine (Zingg AGPL, Spark) as a Python import — those go out-of-process behind a
  driver interface.

### 5. Add tests
- A **deterministic** test of the pure-Python path where possible (no network, no
  model download).
- For optional-dep strategies, guard with `pytest.importorskip("<lib>")` so CI
  without the extra skips rather than errors.
- If it's a matcher, assert the `sense` contract (a maximize-sense similarity yields
  the high-score assignment) and that it goes through `to_cost`.
- Keep tests under `tests/`; CI collects `tests/` (see `pyproject.toml`
  `[tool.pytest.ini_options]`).

### 6. Update the map only if the category is new
If you added a *new kind* of strategy or a new stage, update
[[equate-dev-architecture]] and `.claude/CLAUDE.md`. A new instance of an existing
stage needs no map change — that's the point of the registry (placement test:
*would deleting this change agent behavior?* new stage = yes; new instance = no).

## Anti-patterns (rejected in review)

- Adding `rapidfuzz`/`sentence-transformers`/`faiss`/`POT` to core `dependencies`.
- A raw `import faiss` at module top (breaks `import equate` when the extra is absent).
- A matcher that converts similarity→cost itself instead of `to_cost(..., sense=)`.
- **A matcher that `.toarray()`s a sparse score matrix before `to_cost`** — silently drops
  the blocked-cell worst-casing (D11). Use the `ScoreMatrix` worst-casing views instead.
- Returning printed strings instead of a structured `Candidate`/`Matching`/`Explanation`.
- Assuming a comparator is symmetric/metric when it isn't (silently corrupts blocking
  and assignment).
- A strategy that only works when passed by callable but isn't registered by name
  (or vice-versa).
