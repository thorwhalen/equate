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

## Architecture — current vs target

- **Current** = `equate/util.py` (`similarity_matrix` featurize+compare → a family
  of `matcher(similarity_matrix) → (i,j)` funcs; default `hungarian_matching`) +
  `match_greedily` (stdlib direct path) + `completion.py` (WIP tabular join-key).
- **Target** = the injectable-stage architecture in
  [`docs/research/10-design-implications-for-equate.md`](../docs/research/10-design-implications-for-equate.md)
  and the module sketch in the **equate-dev-architecture** skill. The redesign
  builds it incrementally via the roadmap epic, keeping current public names working
  (back-compat re-exports) — external PyPI users exist (see Release below).

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
