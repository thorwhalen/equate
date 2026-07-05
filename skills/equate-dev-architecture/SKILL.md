---
name: equate-dev-architecture
description: Read before designing, reviewing, or modifying equate's core matching architecture. Covers the featurize→compare→match/resolve pipeline, the module layout, the stage protocols, the resolved score-sense / ScoreMatrix / structured-return contracts, and the "own the orchestration, wrap the libraries" rule. Triggers when working on equate's public API, adding or adjusting a pipeline stage, choosing where a concern lives, or making any design decision about the matching core. This is the north-star for the redesign — the target architecture, not (yet) the current code.
metadata:
  audience: developers
  project: equate
---

# equate architecture — the north-star

`equate` is a framework for **matching**: given collections of objects with no
reliable shared identifier, decide which objects *correspond*, and emit that
correspondence as a set of (optionally scored) tuples. It is **not equality** —
`==` is the degenerate case (identity featurizer + boolean compare + `groupby`).

**One thesis governs everything:** matching factors into three composable,
independently-swappable stages — **featurize → compare → match/resolve** — and
`equate` owns the *orchestration* of those seams while **wrapping mature libraries
at each one**. Reimplement almost nothing computational; own the seams, the
contracts, and the defaults.

> **Guiding principle (progressive disclosure):** `match(A, B)` must Just Work
> with zero config; every stage must be replaceable with one keyword argument;
> **no heavy library is ever a hard dependency.**

## Current state vs target (do not confuse them)

- **Current code** = `equate/util.py` (a `similarity_matrix` featurize+compare
  feeding a family of `matcher(similarity_matrix) → (i,j)` functions: greedy,
  hungarian/JV default, max-weight, stable-marriage, kuhn-munkres) + `match_greedily`
  (direct stdlib path) + `completion.py` (WIP tabular join-key duplicate). No
  registries, no blocking seam, no interactive layer yet.
- **Target** = the module sketch below. The redesign (see the roadmap epic) builds
  it incrementally. When you touch the core, move *toward* the target; keep the
  current public names working (back-compat re-exports) so external PyPI users and
  existing doctests don't break silently.

## The three stages (and the Python idiom each generalizes)

| Stage | Contract (structural — a plain lambda satisfies it) | Generalizes |
|---|---|---|
| **① Featurize** `φ` | `Callable[[Iterable[Object]], Sequence[Repr]]` — batched; **`id` / any `key=` fn is valid** (`featurize=str.lower`) | `key=` in `sorted`/`groupby` |
| **② Compare** `g` | `Callable[[Repr, Repr], float]` (or batched `g(X,Y)->matrix`); two constructors — `featurized(φ, g=cosine)` (indexable) vs `direct(h)` (opaque, re-rank only) | `==`, `<` widened to a graded score |
| **② Block** | `Callable[[Seq[Repr], Seq[Repr]], Iterable[(i,j)]]` — **lazy candidate pairs, never a materialized n×m matrix** | "don't compute most of the similarity matrix" |
| **③ Match** `Σ` | `Callable[..., Iterable[(i,j)]]` — `(scores, *, sense=...) -> pairs`; a *family* (greedy, LAP, stable, soft/OT, clustering), not one algorithm | `groupby`/join widened to optimized correspondence |

Deep dives live in the research corpus — read them, don't re-derive:
`docs/research/00-taxonomy-and-terminology.md` (the canonical map + glossary),
`docs/research/10-design-implications-for-equate.md` (the full architecture),
`docs/research/11-design-decisions-and-open-questions.md` (the decision register —
the contracts below are resolved there). Facet docs `01`–`09`, `12`–`18` drill
into each stage/concern; `docs/research/README.md` is the index.

## Resolved core contracts (honor these everywhere)

1. **Score sense is explicit and centralized.** Scores default to **similarity,
   higher = better**. Every matcher shares one knob `sense: Literal['maximize','minimize'] = 'maximize'`
   and routes similarity→cost through the single `to_cost(scores, *, sense)` SSOT in
   `equate/base.py` (shipped in #2 — it replaced three inconsistent per-matcher
   conversions `S.max()-S`/`-S`/`1-S`, and also uncovered/fixed a broken hand-rolled
   Gale-Shapley). *Never* hand-roll a per-matcher conversion.
2. **One sparse `ScoreMatrix` SSOT** flows compare → match, carrying `sense` + row/col
   labels. Blocking = leaving cells uncomputed (sparse), not a separate data model.
3. **Comparators are not assumed symmetric or metric.** A comparator declares
   polarity (similarity vs distance), bounded?, is_metric, is_symmetric. cosine is a
   *similarity*, not a metric (use angular distance when the triangle inequality is
   needed); Monge-Elkan / SequenceMatcher / containment are directional.
4. **Strategies are name-keyed registries with LAZY factories** (open-closed):
   users pass a string (`match='hungarian'`) *or* a callable, and register their own
   without subclassing. The heavy import fires only on use.
5. **Heavy deps are optional extras behind `check_requirements`-style errors** —
   never a raw `ImportError`. Default install stays `numpy`/`scipy`-light. See
   [[equate-dev-add-strategy]].
6. **Every public boundary returns structured dataclasses, not strings/printouts**
   (`Candidate`, `Matching`, `Explanation`). This is what makes the same core
   dispatchable to a CLI, an HTTP service (`qh`), and a declarative UI (`zodal`+shadcn).
7. **Transitivity is a choice, never hard-wired.** Grading the comparison breaks
   transitivity (`a≈b`, `b≈c` ⇏ `a≈c`). Recovering groups (connected-components /
   correlation clustering) is a separate, swappable resolve stage with a documented
   entity-collapse failure mode — the matcher must not silently take transitive closure.

## The facade & return types (decision register D4/D8/D9/D10)

- **One facade, `how=` selects the correspondence algebra** — all are strategies over
  the same candidate+score structure:
  `match(A, B, *, how='assign'|'pairs'|'soft'|'cluster', k=…, threshold=…)`, default
  `how='assign'` (bipartite 1:1, scipy-JV — the optimized canonical path).
  `dedupe(A)` and `resolve(*collections)` are thin facades (`how='cluster'`).
- **Tiered return types where richer *iterates as* simpler** (so `dict(match(...))`
  always works): `Matching` (default hard; pairs+scores, optional `.plan` for soft,
  optional `.probability` when a calibrator/Fellegi-Sunter is in the pipeline) →
  `Partition` (records→entity ids; the canonical *clustering* output, iterates as its
  point-estimate pairs) → `PartitionPosterior` (`equate[bayes]`; iterates as its point
  estimate). All JSON-serializable dataclasses.
- **Two Matcher arities coexist:** flat `Matcher(scores, *, sense) → pairs` (unchanged)
  and `QuadraticMatcher.align(G1, G2, *, node_affinity, seeds) → correspondence` for
  network alignment (it produces scores *and* correspondence jointly; reuses flat LAP
  as the inner extraction step). Sequence/graph *comparators* (DTW, NW/SW, GED) are
  `direct(h)` scorers emitting `Alignment(score, path)`, routed `sense='minimize'`;
  graph *kernels* are Featurizers (graph→vector, reclaiming the indexable path).
- **Default text featurizer = char-n-gram TF-IDF (core, zero heavy deps);** a modern
  MIT/Apache embedder (bge-m3 / multilingual-e5-large; nomic small) is the default
  *only when `equate[embeddings]` is installed*. Featurizer metadata carries
  license/dim/prefix/truncatable/normalize/output_kinds so equate auto-applies E5
  prefixes, refuses non-commercial models in commercial contexts, and picks a legal index.

## Target module sketch (responsibilities are the point, names are suggestions)

```
equate/
├── __init__.py     # facade: match(), dedupe(), link(), resolve()
├── base.py         # stage protocols; to_cost SSOT; ScoreMatrix; Candidate/Matching/Explanation/Partition  ← #2
├── _dependencies.py # require()/have() + MissingDependencyError (lazy optional deps)  ← #3
├── registry.py     # generic name→lazy-factory Registry (open-closed dispatch)  ← #4
├── _vector.py      # numpy cosine + L2-normalize (drops the sklearn dep)  ← #4
├── featurize/      # featurizers registry + resolve_featurizer; identity/key-fn;  ← #4
│                   #   tfidf.py = pure numpy/scipy char-n-gram TF-IDF (core default);
│                   #   text.py = [sbert|openai|…] dense embedders behind extras (lazy)
├── compare/        # registry; direct() vs featurized(); string(difflib | rapidfuzz|jellyfish [extra]);
│                   #   numeric_geo; vector(cosine); vectorize(comparison-vector combiners); calibrate
├── block/          # Blocker protocol; all_pairs default; keyed blocking; index build/query;
│                   #   ann/lsh [extra]; metablock; PC/RR/PQ metrics
├── match/          # registry; assign(greedy, scipy LAP default, sparse routing);
│                   #   graph[extra]; stable; soft[POT extra]; kbest(Murty)
├── cluster/        # connected-components default; correlation; canonicalize (golden record)
├── interactive/    # CandidateStore (top-k+scores, persisted); constraints (local re-solve);
│                   #   active (QueryStrategy/Oracle); review (ReviewQueue/RiskModel/Explanation)
├── evaluate.py     # pairwise P/R/F1, B-Cubed; blocking PC/RR/PQ
├── service.py      # qh-friendly JSON-in/JSON-out facade
└── util.py         # existing helpers + back-compat re-exports
```

## The rules that keep the core clean

- **Own the orchestration; wrap single-purpose libraries** (rapidfuzz, hnswlib,
  datasketch, POT, sentence-transformers) behind optional extras. **Borrow designs**
  from end-to-end ER frameworks (dedupe, Splink, recordlinkage, Magellan — all share
  a Fellegi-Sunter core) but do **not** import copyleft/heavyweight engines (Zingg
  AGPL, Spark) — those stay out-of-process behind a driver interface.
- **Optimize the canonical case first:** two collections → (optionally scored) pairs,
  1:1, batch, exact. Everything else (1:n top-k, n:m, soft, clustering, incremental,
  cross-modal) is reachable by turning a parameter, not rewriting code.
- **LLM matchers are a cascade re-rank tier** (`equate[llm]`), never a default —
  TF-IDF + edit-distance is genuinely competitive on clean structured data.
- Favor functions over classes; `collections.abc` interfaces over bespoke ABCs;
  `dataclasses` for configs/results; keyword-only args past the 2nd/3rd position.

When a design question isn't answered here, it's in the decision register
(`docs/research/11-design-decisions-and-open-questions.md`). When adding a concrete
strategy, follow [[equate-dev-add-strategy]].
