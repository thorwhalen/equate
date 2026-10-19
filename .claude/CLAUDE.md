# equate — Agent Instructions (map)

This file is the **index**, not the content store. Behavioral rules live here or in
the dev skills; deep content lives in `docs/research/` and is referenced from them.
*Placement test: if deleting a sentence wouldn't change agent behavior, it belongs
in a file, not here.*

## What this is

`equate` is a Python framework for **matching**: given collections of objects with
no reliable shared identifier, decide which objects *correspond* and emit that
correspondence as a set of (optionally scored) tuples. It is **not equality** —
`==` is the degenerate case (identity featurizer + boolean compare + `groupby`).

The whole design is one idea: matching factors into three composable,
independently-swappable stages — **featurize → compare → match/resolve** —
and equate owns the **orchestration** while **wrapping mature libraries** at each
seam. Reimplement almost nothing computational; own the seams, contracts, defaults.

> Guiding principle: `match(A, B)` Just Works with zero config; every stage is
> replaceable with one keyword arg; **no heavy library is ever a hard dependency.**

## Architecture — shipped

The injectable-stage architecture of
[`docs/research/10-design-implications-for-equate.md`](../docs/research/10-design-implications-for-equate.md)
**is the code** (the redesign epic shipped it, #2–#13): stage registries
(`featurize/`, `compare/`, `block/`, `matching/`, `cluster/`), the `match` / `dedupe` /
`resolve` / `align` / `score_matrix` facades, `interactive/`, `evaluate/`, `service.py`,
and `completion.py` (tabular facades over the same seams). The module map is in the
**equate-dev-architecture** skill. The pre-redesign names (`util.py`: `similarity_matrix`,
`match_keys_to_values`, `match_greedily`, the `*_matching` functions; `completion`'s
legacy helpers) are **kept working** — external PyPI users exist (see Release below);
never remove one without a deprecation cycle.

**Before touching the core, read the `equate-dev-architecture` skill.** It holds the
stage protocols and the resolved contracts (score `sense` SSOT, sparse `ScoreMatrix`,
directional/non-metric comparators, structured dataclass returns, transitivity-is-a-choice).
One load-bearing invariant it now enforces (register **D11**): a matcher consumes a
`ScoreMatrix` and densifies **only** via its worst-casing views — **never `.toarray()` a
sparse score matrix before `to_cost`** (drops the blocked-cell worst-casing; a registry
conformance test in `tests/test_matching.py` guards it). The hole fill is a **big-M**, so
blocked matching is **lexicographic** (as many real pairs as possible, then best score);
a per-cell fill lets the solver *buy* a hole. This is invisible to `[0,1]` scores — **test
blocking with unbounded ones.**

**"Matcher" spans structured objects at two arities (D8, shipped in #10).** A *flat*
matcher picks pairs from a `ScoreMatrix`; a *quadratic* matcher (`equate.align`,
`matching/quadratic.py`) aligns the nodes of two graphs and produces scores and
correspondence jointly. It never re-implements extraction: it hands its node-pair scores to
a flat matcher (`extract=`). Sequence/graph *comparators* (DTW, NW/SW) are ordinary
`direct` comparators that declare their polarity; graph *kernels* are featurizers. Before
adding a structured strategy, decide which of these three seams it belongs to.

## Surfaces (#12): the service and the review UI

- **`equate/service.py` is the surface-free core** — `MatchService` methods take JSON-able
  args and return JSON-able dataclasses (`PairView`, `ReviewItem`, `SessionView`,
  `EditResult`, ...). **No surface library is imported at module level**: `mk_app()` pulls
  `qh` lazily (`equate[service]`), so `import equate.service` stays numpy/scipy-only.
- **The JSON-dataclass boundary rule (D10):** a new field on a boundary type is a
  cross-language change. Regenerate `ui/src/schema/fields.json`
  (`python -m equate.service fields > ui/src/schema/fields.json`) and update the Zod schema
  in `ui/src/schema/equate.ts`; `tests/test_service.py` and `ui/src/schema/equate.test.ts`
  each fail until both sides agree. Non-finite floats go out as `null` (`to_jsonable`).
- **Sessions re-solve from cached scores; they never re-score.** A reject is a *hole* (the
  blocked-matching semantics, D11), not an `inf` — so rejecting every candidate leaves an
  item unmatched instead of making the LAP infeasible. Edits fold latest-wins
  (`_effective_constraints`).
- **Concurrency:** edits of one session are serialized by a per-session lock in-process;
  across processes, `Session.version` is checked on save (`SessionConflict` → 409). A
  session is two store keys: the mutable state (`sid`, saved on every edit) and the
  immutable `SessionData` (`sid.data`: A, B, scores — written once). HTTP handlers run in
  a worker thread (async endpoints), never on the event loop. An edit or a label is
  atomic — a failed one leaves the session (log, labels, accounting) untouched.
- **HTTP errors are mapped in `mk_app`**, because qh turns any exception into a 500:
  `SessionNotFound` → 404, other `KeyError`/`ValueError`/`TypeError` → 422.
- **`ui/`** is a Vite + React + zodal app with its own `npm test` / `npm run build`
  (`npm run ci-local` = both). Hosted CI runs them on any `ui/**` change
  (`.github/workflows/ui-ci.yml` → the wads npm workflow, config under `wads` in
  `ui/package.json`). `mk_app(ui_dir=…)` serves a built bundle at `/` with the API under
  `/api` — the same path the vite dev proxy uses, so the client's `/api` base is right in
  both.

## Benchmarks (#15)

- **Never commit benchmark data.** Loaders download into a store (`equate.evaluate.store`,
  default `~/.local/share/equate/benchmarks`); tests use tiny fixtures written in each
  publisher's format (`tests/test_suites.py`). The real-file check is opt-in:
  `EQUATE_NETWORK_TESTS=1 python -m pytest tests/test_suites.py -k real`.
- **A new loader pins the SHA-256 of every file it fetches** and states `license` honestly
  — "not stated" is an answer; a guessed license is not.
- Gold must be **complete** for whole-table scoring; a source whose gold is a labelled
  subset (DeepMatcher) says so in `meta['gold']`.

## The research corpus (`docs/research/`)

Start at [`00-taxonomy-and-terminology.md`](../docs/research/00-taxonomy-and-terminology.md)
(canonical map + cross-community glossary) and
[`10-design-implications-for-equate.md`](../docs/research/10-design-implications-for-equate.md)
(the architecture). [`11-design-decisions-and-open-questions.md`](../docs/research/11-design-decisions-and-open-questions.md)
is the decision register the roadmap draws from. Facet docs `01`–`09` and `12`–`18`
drill into each stage/concern; `docs/research/README.md` indexes everything. **These
are grounded, cited references — consult them instead of re-deriving the field.**

## Dev skills (in `skills/`, symlinked into `.claude/skills/`)

| Skill | Use when |
|---|---|
| **equate-dev-architecture** | Designing/reviewing/modifying the matching core: stage model, module layout, resolved contracts, wrap-vs-reimplement. Read before touching the public API or a pipeline stage. |
| **equate-dev-add-strategy** | Adding a featurizer / comparator / blocker / matcher / clusterer via the registry + lazy optional-dep pattern + tests. |

Dev skills are *living artifacts* (per `~/.claude/skills/dev-skills-workflow`):
revise them as the design changes; add one when a recurring dev task emerges. Real
files live in `skills/<name>/`; each is symlinked (relatively, per-skill) into
`.claude/skills/`. A newly-created `.claude/skills/` dir needs a **session restart**
to be watched before `/equate-dev-*` invocation works.

**Roadmap discipline:** for each roadmap sub-issue, first ask *"should a skill or
this CLAUDE.md be updated before the code?"* — update the map, then build.

## Conventions

- Functional-first; `collections.abc` interfaces over bespoke ABCs; `dataclasses`
  for configs/results; keyword-only args past the 2nd/3rd position.
- Every public boundary returns structured dataclasses (never printouts) so the same
  core dispatches to CLI, HTTP (`qh`), and a declarative UI (`zodal` + shadcn).
- Heavy deps (embeddings, ANN, OT, LLMs, fast string libs) are **optional extras**
  with lazy imports + `check_requirements`-style errors. Core stays `numpy`/`scipy`-light.
- Every module needs a top-level docstring (auto-extracted for docs; `ruff` D100 is on).

## Testing

```bash
python -m pytest tests/ -v      # CI collects tests/ (see pyproject [tool.pytest.ini_options])
ruff check equate
```
Prefer deterministic, dependency-light tests; guard optional-dep tests with
`pytest.importorskip`. CI is the wads uv stub → `i2mint/wads/.github/workflows/uv-ci.yml`.

## Git & release

- Packaging is modern (`pyproject.toml`, hatchling, uv CI stub). No `setup.cfg`.
- **No local ecosystem package depends on equate** (verified via `priv` dep_graph,
  direct+transitive = []) → the redesign is free to break internal APIs.
- **But equate is live on PyPI (~27.8k downloads all-time)** → ship breaking changes
  as a deliberate version bump with changelog notes; add cheap deprecation shims /
  back-compat re-exports where practical.
- CI auto-bumps the version and publishes on pushes to the default branch — do not
  push redesign work to `master` until it's meant to release.

## Roadmap

The redesign is tracked as a GitHub **epic issue** with linked sub-issues on
`thorwhalen/equate`. The epic is the SSOT for sequencing; each sub-issue names the
research doc(s) and decision(s) it implements.
