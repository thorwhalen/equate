# Design Implications for `equate`: From Research to Architecture

*Capstone synthesis for the `equate` redesign. Reads the corpus in
`docs/research/` (`00`–`09`) and turns it into concrete, buildable architecture:
the core case to optimize first, the injectable stage seams, the scalability and
interactivity paths, service/UI attachment points, a module sketch, and a
wrap-vs-reimplement decision per layer. Start from the taxonomy in
[`00-taxonomy-and-terminology.md`](00-taxonomy-and-terminology.md).*

## Abstract

The research corpus converges on a single architectural thesis: **matching is not
one algorithm but a pipeline of three composable, independently-swappable stages —
featurize → compare → match — and `equate` should own the *orchestration* of those
seams while wrapping mature libraries at each one.** `equate` already has the right
skeleton: `similarity_matrix` (featurize + compare) feeding a family of matchers
that share a `matcher(similarity_matrix) → (i,j)` signature. The redesign
formalizes that skeleton into injectable strategy protocols, fixes a latent
score/cost-conversion inconsistency, adds a **blocking/candidate-generation** seam
so the O(n²) similarity matrix can go sparse, retains **top-k with scores** to
enable interactive re-optimization, and draws every heavy dependency (embeddings,
ANN, optimal transport, LLMs) behind **optional extras with lazy imports** so the
default install stays `numpy`/`scipy`-light. This document specifies each of those
moves, names the modules, and says — per layer — what to wrap versus reimplement.

Guiding principle (progressive disclosure): **`match(A, B)` must Just Work with
zero config; every stage must be replaceable with one keyword argument; no heavy
library is ever a hard dependency.**

---

## 1. The core canonical case to optimize first

**Optimize this one path before anything else:**

> Given two collections `A` and `B`, produce the (optionally scored) set of matched
> pairs, efficiently, with sensible defaults and no required configuration.

```python
match(A, B)                       # → iterator of (a, b) or (a, b, score)
match(A, B, k=3)                  # → top-3 candidates per a, with scores
match(A, B, threshold=0.8)        # → boolean cut; unmatched a's allowed
match(A, B, featurize=str.lower)  # → swap stage ① with a plain key function
```

This is a direct generalization of what `match_keys_to_values` already does. The
core case is **1:1, scored, batch, exact (all-pairs)** — the point in the
[dimension space](00-taxonomy-and-terminology.md#3-the-dimensions-along-which-matching-problems-vary)
that covers the majority of real requests (fuzzy-join two lists, reconcile two
catalogs, align two schemas). Defaults:

| Stage | Default (zero-dep, batteries-included) | Why |
|---|---|---|
| ① Featurize | TF-IDF via `grub` (current) for text; **identity** for pre-vectorized input | already shipped; competitive on clean structured data [6] |
| ② Compare | `cosine_similarity` (featurized path) / `difflib.ratio` (direct path) | already shipped; no heavy deps |
| ② Block | **none** (dense all-pairs) below a size threshold; warn + suggest a blocker above it | correctness baseline; blocking is opt-in |
| ③ Match | `scipy.optimize.linear_sum_assignment` (Jonker-Volgenant) | exact 1:1, rectangular, `maximize=True`, BSD, already a dep [3] |

**Immediate fix (SSOT correctness):** today each matcher converts
similarity→cost differently — `hungarian_matching` uses `S.max() − S`,
`kuhn_munkres_matching` uses `−S`, `stable_marriage_matching` uses `1 − S`. These
are *not* equivalent and are a latent bug. Centralize into one sense-aware
`_to_cost(scores, *, sense)` helper (per [`03`](03-assignment-and-graph-matching.md)),
and make `sense: Literal['maximize','minimize'] = 'maximize'` the single explicit
knob every matcher shares.

**Make `scipy` the default matcher, not `networkx`.** `maximal_matching` and
`kuhn_munkres_matching` build an O(n·m) `networkx` graph over *all* pairs — quadratic
and slow. `linear_sum_assignment` is exact, rectangular, faster, and already
imported. Keep the `networkx` matchers as opt-in strategies for their distinct
*semantics* (max-cardinality, blossom), not as defaults.

---

## 2. The featurize → compare → match separation as injectable stages

The whole design is the **strategy pattern + dependency injection** applied to the
three-stage triad from [`00`](00-taxonomy-and-terminology.md#1-the-canonical-decomposition).
Each stage is a plain **callable** with a documented contract, defaulted, and
overridable by keyword — favouring functions over classes, `collections.abc`
interfaces over bespoke ABCs, and `dataclasses` for the few stateful configs.

```python
# Stage protocols (structural; a plain lambda satisfies them)
Featurizer  = Callable[[Iterable[Object]], Sequence[Repr]]     # ① batched; id() is valid
Comparator  = Callable[[Repr, Repr], float]                    # ② or a batched g(X, Y) -> matrix
Blocker     = Callable[[Sequence[Repr], Sequence[Repr]], Iterable[tuple[int, int]]]  # ② candidate pairs
Matcher     = Callable[..., Iterable[tuple[int, int]]]         # ③ (scores, *, sense=...) -> pairs
```

Design rules distilled from the corpus:

1. **A plain key function must be a valid featurizer** ([`04`](04-featurization-and-representation.md)).
   `featurize=str.lower` or `featurize=operator.attrgetter('name')` must work with
   no ceremony — this is the bridge from Python's `key=` idiom to full matching.
2. **Two comparator constructors** ([`05`](05-comparison-and-similarity-functions.md)):
   `featurized(φ, g=cosine)` (indexable metric over representations) and
   `direct(h)` (opaque pairwise scorer, e.g. edit distance, cross-encoder). Only
   the former can drive an ANN/LSH index; the latter can only re-rank candidates.
3. **Attach metadata to strategies** ([`04`](04-featurization-and-representation.md),
   [`05`](05-comparison-and-similarity-functions.md)): a featurizer declares its
   output *kind* (vector/set/bitstring/scalar/structured), dimensionality, whether
   it L2-normalizes, and its compatible metric; a comparator declares
   polarity (similarity vs distance), bounded?, is_metric, is_symmetric. This lets
   the framework auto-select a *legal* index and matcher instead of hard-coding
   assumptions (e.g. cosine is a similarity, not a metric — use angular distance
   when the triangle inequality is needed).
4. **Name-keyed registries (SSOT, open-closed)** ([`09`](09-python-ecosystem-landscape.md)):
   users select strategies by string (`featurize='sbert'`, `compare='jaro_winkler'`,
   `match='hungarian'`, `block='minhash_lsh'`) *or* pass a callable, and register
   their own without subclassing. Registry factories are **lazy** — the heavy import
   fires only on use.
5. **Multi-field / structured records** ([`04`](04-featurization-and-representation.md),
   [`05`](05-comparison-and-similarity-functions.md)): a `{field: Comparator}` mapping
   produces a per-pair **comparison vector**, reduced by a pluggable combiner
   (`weighted_sum`, `mean`, `max`, `fellegi_sunter`, or a fitted sklearn estimator).
   This upgrades `equate` from string-pair to record matching without disturbing the
   string path.

### 2.1 Facades for text/image/audio featurization (heavy deps OPTIONAL)

Progressive disclosure at the featurizer level: string names resolve to lazy
factories; the heavy library is imported only when the name is used, and its
absence raises a `check_requirements`-style error naming the extra and the install
command — never a raw `ImportError` ([`04`](04-featurization-and-representation.md),
[`09`](09-python-ecosystem-landscape.md)).

| Modality | Featurizer names | Optional extra → wrapped library |
|---|---|---|
| Text (sparse) | `tfidf` (default) | core (`grub`/`sklearn`) |
| Text (dense) | `sbert`, `openai`, `cohere` | `equate[text-embeddings]` → sentence-transformers; `equate[api]` → openai/cohere |
| Image | `clip`, `phash`, `dhash` | `equate[image]` → open_clip / imagehash |
| Audio | `clap`, `chromaprint` | `equate[audio]` → CLAP / pyacoustid (needs `fpcalc` system binary) |
| String direct | `levenshtein`, `jaro_winkler`, `ratio` | `equate[fuzzy]` → rapidfuzz; `equate[phonetic]` → jellyfish |

The default install stays GPU-free and small (`numpy`, `scipy`, `sklearn`, `grub`,
stdlib `difflib`). Follow the existing pattern in `util.py` of importing
`networkx` *inside* the function that needs it.

---

## 3. Scalability path: avoid all-pairs before you optimize it

The single biggest lever is **blocking / candidate generation**: matching is
inherently O(n·m), so ~1M records ⇒ ~10¹² comparisons; blocking exists to make that
sub-quadratic ([`01`](01-entity-resolution-record-linkage.md),
[`02`](02-blocking-and-scalable-candidate-generation.md)). The key reframing:

> **Blocking is exactly "do not compute most of the similarity matrix."** The
> blocker decides *which cells* of `S` are populated; `equate`'s existing sparse
> similarity matrix is the right unifying data structure.

Design:

1. **One `Blocker` protocol**: `candidate_pairs(A, B) -> Iterable[(i, j)]` (plus a
   self-join form for dedup), returning a **lazy generator of pairs, never a
   materialized n×m matrix**. Ship a trivial `all_pairs` blocker as the correctness
   baseline (today's dense path becomes `block='all_pairs'`).
2. **Separate keying from grouping** ([`02`](02-blocking-and-scalable-candidate-generation.md)):
   standard/token/phonetic/q-gram blocking are *one* algorithm parameterized by an
   injected `key_fn: Object -> Iterable[key]`. Users get a different blocker by
   swapping a function.
3. **One retrieval/index protocol** — `build(vectors) → query(vec, k) → ids` — that
   MinHash-LSH, HNSW, IVF-PQ, and ScaNN all satisfy, making the ANN backend a
   configuration choice ([`02`](02-blocking-and-scalable-candidate-generation.md),
   [`04`](04-featurization-and-representation.md)). Match the index to the
   representation *kind*: ANN (HNSW/FAISS) for dense vectors, MinHash-LSH for token
   sets, multi-index Hamming for perceptual-hash bit-strings, inverted index /
   LSH-Ensemble for sparse TF-IDF and join-key containment ([`07`](07-schema-and-ontology-matching.md)).
4. **Meta-blocking as an optional `refine(candidates) -> candidates`** post-processor
   (block filtering, comparison propagation, graph edge-pruning) so recall-first
   blocking stays cheap and precision recovery is modular ([`02`](02-blocking-and-scalable-candidate-generation.md)).
5. **Blocking metrics are first-class outputs**: pair completeness (PC = recall,
   upper-bounds system recall), reduction ratio (RR), pairs quality (PQ = precision),
   via an evaluation utility taking ground-truth pairs — turning blocker/threshold
   selection into an empirical recall-vs-efficiency decision
   ([`02`](02-blocking-and-scalable-candidate-generation.md)).

Optional extras: `equate[lsh]` → datasketch; `equate[ann]` → hnswlib (zero-friction
default) / faiss / scann; `equate[dense]` → an encoder.

### 3.1 When the assignment-optimization layer is worth it

The global optimizer ([`03`](03-assignment-and-graph-matching.md)) earns its O(n³)
cost only under specific conditions:

- **Use the LAP optimizer** when the output must be a **globally coherent 1:1
  assignment** (each object matched at most once, minimizing total cost). Greedy is
  order-dependent and sub-optimal; the Hungarian/JV optimum can differ materially.
- **Skip it** for 1:n link-to-KB or top-k retrieval — there is no global
  constraint, so **top-k per row** (argmax / ANN query) is both correct and cheap.
- **Route sparse/large inputs to sparse solvers**
  (`scipy.sparse.csgraph.min_weight_full_bipartite_matching` or `lapmod`) rather than
  densifying — critical once blocking has made `S` sparse.
- **Expose the *objective*, not the algorithm** as the primary choice: `'optimal'`
  (LAP), `'stable'` (Gale-Shapley — document that it optimizes stability, not total
  score, and is proposer-optimal/receiver-pessimal), `'greedy'`,
  `'max_cardinality'`, `'soft'` (optimal transport).
- **Soft matching gets its own parallel interface** (`soft_match(...) -> plan`, a
  fractional transport-plan matrix, plus a `harden(plan)` helper) — the home for
  graded, partial, unequal-mass, and cross-modal (Gromov-Wasserstein) matching
  ([`03`](03-assignment-and-graph-matching.md)). Extra: `equate[ot]` → POT.

---

## 4. The interactive verification story

Human-in-the-loop is the payoff of retaining **top-k candidates with (calibrated)
scores**. Split human involvement into **three independently-optional loops** behind
small protocols — do not conflate them ([`08`](08-interactive-active-learning-and-hitl.md)):

| Loop | Protocol | Default | Extra |
|---|---|---|---|
| **Train** (active learning) | `QueryStrategy.rank(pool, state) -> ordered pairs` + `Oracle.label(pairs) -> labels` | margin uncertainty | `equate[active]` → modAL/scikit-activeml |
| **Review** (verification) | `ReviewQueue` ranked by calibrated confidence or a `RiskModel` | confidence triage | — |
| **Re-optimize** (interactive edits) | `ConstraintSet` (append-only force/forbid/must-link/cannot-link) → constrained re-solve | Murty k-best reuse | `equate[kbest]` → Murty |

The mechanics the corpus prescribes:

1. **`CandidateStore` as SSOT** — per item, its top-k candidates with scores +
   provenance/explanation. Backs both review triage and k-best reuse; persist it
   (`dol.cache_this`) so a session resumes.
2. **Calibrated scores are a prerequisite** ([`08`](08-interactive-active-learning-and-hitl.md),
   [`05`](05-comparison-and-similarity-functions.md)): a swappable Platt/isotonic
   step, because both review triage and re-optimization treat the score as a match
   probability.
3. **Every edit is a constraint, not a recompute**: a *confirm* forces an edge, a
   *reject* forbids one (cost = +∞); many-to-many edits become must-link/cannot-link
   constraints consumed by a constrained/correlation clusterer.
4. **Re-optimization must be LOCAL**: an edit re-solves only the affected
   block/connected component (warm-started), never a global recompute by default.
   This locality is the performance contract that makes editing feel instant.
5. **k-best via Murty** wrapping the inner LAP solver, with force/forbid cell
   constraints as the primitive for user edits ([`03`](03-assignment-and-graph-matching.md),
   [`08`](08-interactive-active-learning-and-hitl.md)). No well-maintained pure-Python
   Murty exists on PyPI — this is likely in-house code in `equate[kbest]`.
6. **Explanations as structured payloads** (rules / feature attributions / side-by-side
   field diffs), *not* printouts, so any UI can render them declaratively.

---

## 5. Where an HTTP service and a UI attach

The strategy-seam architecture is exactly what makes `equate` dispatchable to a
service and a UI without rework (aligns with the user's `python-dispatching` skill
and declarative/schema-based UI preference).

**HTTP service (via `qh`).** Expose the core facade and the interactive loop as
endpoints; `qh` turns typed Python functions into an HTTP API, so keeping the public
surface as **plain keyword-argument functions returning JSON-serializable
dataclasses** is the enabling constraint:
- `POST /match` — `{A, B, featurize?, compare?, match?, k?, threshold?}` → matched
  tuples with scores. Stateless; the batch core.
- `GET/POST /candidates/{item}` — read/refresh the `CandidateStore` top-k (§4.1).
- `POST /edits` — append a `ConstraintSet` event (confirm/reject/must-link) and
  return the **locally re-optimized** result (§4.4). This is the stateful,
  session-scoped endpoint; back it with a persisted `CandidateStore`
  (`dol`/`cache_this`) keyed by session id.
- `POST /label` — the active-learning `Oracle` endpoint (a human/LLM/crowd labeller
  behind one interface).

Because scoring can be expensive (embeddings, LLMs), add a **cost/latency accounting
hook** and aggressive caching/batching on `encode`/`score_pairs` so the service can
budget cascades ([`06`](06-deep-learning-and-llm-entity-matching.md)).

**UI (via `zodal` + shadcn).** The review/edit loop is a **declarative,
schema-driven** UI — precisely `zodal`'s sweet spot. The structured `Explanation`
and `Candidate` payloads become Zod schemas; the UI renders:
- a **review queue** (ranked candidate list with scores + explanation), rendering
  side-by-side field diffs from the structured payload;
- **confirm / reject / reassign** controls that POST `ConstraintSet` edits and
  optimistically re-render the locally re-optimized block;
- an **active-learning labelling widget** (the `Oracle` UI) presenting the
  most-uncertain pair.

The contract that makes both attachments clean: **return structured dataclasses, not
strings or printouts**, at every public boundary.

---

## 6. Recommended module / abstraction sketch

Functional-first, `collections.abc` interfaces, `dataclasses` for configs/results,
keyword-only args past the 2nd/3rd position, lazy heavy imports. Names are
suggestions; responsibilities are the point.

```
equate/
├── __init__.py          # facade: match(), dedupe(), link(), resolve() + re-exports
├── base.py              # Featurizer/Comparator/Blocker/Matcher protocols;
│                        #   ScoreMatrix (sparse SSOT: data + sense + row/col labels);
│                        #   Candidate / Matching / Explanation dataclasses
├── featurize/
│   ├── __init__.py      # registry (name -> lazy factory); identity; metadata contract
│   ├── text.py          # tfidf (default, grub); [sbert|openai|cohere] behind extras
│   ├── image.py         # [clip|phash|dhash]  (extra)
│   ├── audio.py         # [clap|chromaprint]  (extra)
│   └── structured.py    # per-field featurizer products; nested records
├── compare/
│   ├── __init__.py      # registry; direct() vs featurized() constructors; metadata
│   ├── string.py        # difflib (core); [rapidfuzz|jellyfish|py_stringmatching] extras
│   ├── numeric_geo.py   # decay funcs (step/linear/exp/gauss), haversine
│   ├── vector.py        # cosine (default), dot, angular-distance
│   ├── vectorize.py     # comparison-vector composition + combiners (weighted_sum,
│   │                    #   mean, max, fellegi_sunter, fitted estimator)
│   └── calibrate.py     # threshold / Platt / isotonic  → [0,1] probability
├── block/
│   ├── __init__.py      # Blocker protocol; all_pairs (default); block metrics PC/RR/PQ
│   ├── key_blocking.py   # keyed blocking parameterized by key_fn (standard/token/phonetic/qgram/SNM)
│   ├── index.py         # build/query retrieval protocol; brute-force default
│   ├── ann.py           # [hnswlib|faiss|scann]  (extra)
│   ├── lsh.py           # [datasketch] MinHash/SimHash/LSH-Ensemble  (extra)
│   └── metablock.py     # refine(candidates): block-filtering, comparison-propagation, graph pruning
├── match/
│   ├── __init__.py      # Matcher registry; _to_cost(scores, *, sense)  ← SSOT fix
│   ├── assign.py        # greedy, hungarian/JV (scipy default), sparse LAP routing
│   ├── graph.py         # [networkx] max-cardinality (Hopcroft-Karp), blossom  (extra)
│   ├── stable.py        # Gale-Shapley (objective = stability, documented)
│   ├── soft.py          # [POT] Sinkhorn/EMD/unbalanced/partial + harden(plan)  (extra)
│   └── kbest.py         # [kbest] Murty enumeration; force/forbid primitives
├── cluster/
│   ├── __init__.py      # Clusterer protocol; connected-components (union-find) default
│   ├── correlation.py   # correlation / constrained clustering (must/cannot-link)
│   └── canonicalize.py  # golden-record merge policies (majority/most-complete/source-trust)
├── interactive/
│   ├── candidate_store.py  # top-k + scores + provenance, persisted (dol.cache_this)
│   ├── constraints.py      # append-only ConstraintSet events; local re-solve
│   ├── active.py           # QueryStrategy + Oracle protocols  ([active] extras)
│   └── review.py           # ReviewQueue, RiskModel, structured Explanation payloads
├── evaluate.py          # pairwise P/R/F1, B-Cubed, blocking PC/RR/PQ; ground-truth adapter
├── service.py           # qh-friendly facade functions (JSON-in/JSON-out dataclasses)
├── registry.py          # capability detection: pick fastest installed backend; install hints
└── util.py              # existing helpers (keep back-compat re-exports)
```

Backbone decisions ([`09`](09-python-ecosystem-landscape.md)):
- **One sparse `ScoreMatrix` SSOT** flowing from compare → match, carrying an
  explicit `sense` flag and row/col labels (what `util.ensure_sparse` +
  `linear_sum_assignment` already gesture at).
- **Capability detection + strategy registry**: a bare install works everywhere;
  `equate` auto-selects the fastest *installed* backend (rapidfuzz > difflib,
  hnswlib/faiss > brute force, lapx > scipy on large sparse) and emits actionable
  install hints for the rest.
- **Adapters so no library type leaks**: normalize pure-function libs, builder/index
  libs (faiss/hnswlib/datasketch via build→add→query), and stateful pipelines
  (dedupe/Splink via a coarse `fit`/`predict` facade) behind the same protocols.

---

## 7. Wrap vs reimplement, per layer

The corpus is emphatic ([`09`](09-python-ecosystem-landscape.md)): **reimplement
almost nothing computational; own only the orchestration.** Keep only `numpy`/`scipy`
(and current `grub`/`sklearn`) as hard deps; everything else is a pip extra behind a
lazy import.

| Layer | Reimplement (own it) | Wrap (optional extra) | Do NOT depend on |
|---|---|---|---|
| ① Featurize (text) | identity, key-fn adapter, TF-IDF via `grub` | `equate[text-embeddings]`→sentence-transformers; `equate[api]`→openai/cohere; `equate[image]`→open_clip/imagehash; `equate[audio]`→CLAP/pyacoustid | — |
| ② Compare (string) | `difflib` fallback, comparison-vector composition, combiners, calibration | `equate[fuzzy]`→rapidfuzz; `equate[phonetic]`→jellyfish; `equate[stringmatching]`→py_stringmatching | GPL fuzzywuzzy |
| ② Block / index | `all_pairs`, keyed blocking, block metrics, meta-blocking glue | `equate[lsh]`→datasketch; `equate[ann]`→hnswlib/faiss/scann; `equate[schema]`→Valentine (join-key discovery, [`07`](07-schema-and-ontology-matching.md)) | — |
| ③ Match (hard) | greedy, `_to_cost` sense SSOT, Murty k-best (`equate[kbest]`) | `scipy` LAP (default, already dep); `equate[fast-lap]`→lapx; `equate[graph]`→networkx | — |
| ③ Match (soft) | `harden(plan)` helper | `equate[ot]`→POT | — |
| ③ Cluster | connected-components (union-find), canonicalize merge policies | correlation clustering (own or wrap) | — |
| Classify (probabilistic) | Fellegi-Sunter combiner, EM (small) | — | full ER frameworks as *imports* |
| Interactive | CandidateStore, ConstraintSet, local re-solve, ReviewQueue, RiskModel | `equate[active]`→modAL/scikit-activeml; `equate[llm]`→openai/transformers | — |
| Orchestration | facade, ScoreMatrix SSOT, registries, capability detection, blocking-key DSL | — | — |

Hard rules from [`09`](09-python-ecosystem-landscape.md):
- **Never import copyleft/heavyweight engines.** Zingg (AGPL-3.0, Spark) and
  Splink-on-Spark stay strictly out-of-process behind a driver interface, never a
  Python import, so `equate` stays permissively licensed and light.
- **Borrow designs from end-to-end ER frameworks** (dedupe, Splink, recordlinkage,
  Magellan — all share a Fellegi-Sunter core) but reserve *wrapping* for
  single-purpose libraries.
- **LLM matchers** ([`06`](06-deep-learning-and-llm-entity-matching.md)) are a
  `PairScorer` (or set-level `select`) strategy behind `equate[llm]`, added as a
  **cascade re-rank tier** (cheap blocker → cheap scorer → LLM on top-k), never a
  default — the evidence shows TF-IDF + edit-distance is genuinely competitive on
  clean structured data.

---

## 8. Migration from the current code (concrete first steps)

1. **Extract `_to_cost(scores, *, sense)` SSOT** and route all matchers through it;
   fix the three inconsistent conversions. *(Correctness; no API change.)*
2. **Make `linear_sum_assignment` the default matcher**; demote the two `networkx`
   all-pairs matchers to opt-in strategies with lazy imports. *(Performance.)*
3. **Introduce the `Featurizer`/`Comparator`/`Matcher` registries** and reframe
   `similarity_matrix`'s `obj_to_vect`/`similarity_func` as the first two seams;
   keep current defaults so existing doctests pass. *(No behaviour change.)*
4. **Add the `Blocker` seam** with `all_pairs` default and a `ScoreMatrix` that can be
   sparse; wire the sparse path already present in `ensure_sparse`.
5. **Add `k=` (top-k retention) and `threshold=`** to the facade; introduce
   `CandidateStore`. *(Enables §4.)*
6. **Layer the optional extras** (`fuzzy`, `text-embeddings`, `ann`, `lsh`, `ot`,
   `kbest`, `active`, `llm`) with `check_requirements`-style errors.
7. **`completion.py`** (tabular join-key discovery) becomes a *facade over* the same
   seams with a containment comparator and top-k selector, per
   [`07`](07-schema-and-ontology-matching.md) — not a parallel implementation.

---

## 9. Open design questions carried forward

Distilled from the facet docs' open questions; these need a decision during
implementation:

- **Primary cardinality target** — is `equate`'s spine strict 1:1 (LAP/Murty),
  1:n retrieval, or clustering? This drives whether k-best or constrained clustering
  is the core re-optimization solver ([`03`](03-assignment-and-graph-matching.md),
  [`08`](08-interactive-active-learning-and-hitl.md)).
- **One normalized `recall_target` dial** auto-mapped to each backend's native params
  (efSearch, nprobe, LSH bands/rows, SNM window) vs exposing native params — progressive
  disclosure vs calibration cost ([`02`](02-blocking-and-scalable-candidate-generation.md)).
- **Default clustering** — connected-components (simple, but entity-collapse-prone) vs
  correlation clustering (robust, costlier) ([`01`](01-entity-resolution-record-linkage.md)).
- **Unifying return type** for hard (pair lists) and soft (plan matrices) matchers — a
  `Matching` object exposing both `.pairs` and `.plan`? ([`03`](03-assignment-and-graph-matching.md)).
- **Where Fellegi-Sunter lives** — is it an L3 scorer, an L4 matcher, or a distinct
  `ClassifierMatcher` semantics that fuses scoring and decision? ([`09`](09-python-ecosystem-landscape.md)).
- **Ship or wrap an embedding model** for dense blocking, or only define the encoder
  interface and leave the model to the user? ([`02`](02-blocking-and-scalable-candidate-generation.md),
  [`04`](04-featurization-and-representation.md)).

---

## References

1. Christophides V, Efthymiou V, Palpanas T, Papadakis G, Stefanidis K. An Overview of End-to-End Entity Resolution for Big Data. *ACM Computing Surveys* 53(6), Art. 127, 2020. [https://dl.acm.org/doi/10.1145/3418896](https://dl.acm.org/doi/10.1145/3418896)
2. Papadakis G, Skoutas D, Thanos E, Palpanas T. Blocking and Filtering Techniques for Entity Resolution: A Survey. *ACM Computing Surveys* 53(2), Art. 31, 2020. [https://arxiv.org/abs/1905.06167](https://arxiv.org/abs/1905.06167)
3. Crouse DF. On Implementing 2D Rectangular Assignment Algorithms. *IEEE Trans. Aerospace and Electronic Systems* 52(4):1679-1696, 2016. [https://doi.org/10.1109/TAES.2016.140952](https://doi.org/10.1109/TAES.2016.140952)
4. Jonker R, Volgenant A. A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems. *Computing* 38(4):325-340, 1987. [https://link.springer.com/article/10.1007/BF02278710](https://link.springer.com/article/10.1007/BF02278710)
5. Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP* 2019. [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
6. Papadakis G, et al. A Critical Re-evaluation of Benchmark Datasets for (Deep) Learning-Based Matching Algorithms. *ICDE* 2024 / arXiv:2307.01231, 2023. [https://arxiv.org/abs/2307.01231](https://arxiv.org/abs/2307.01231)
7. Thirumuruganathan S, et al. Deep Learning for Blocking in Entity Matching: A Design Space Exploration (DeepBlocker). *PVLDB* 14(11), 2021. [https://dl.acm.org/doi/10.14778/3476249.3476294](https://dl.acm.org/doi/10.14778/3476249.3476294)
8. Malkov YuA, Yashunin DA. Efficient and Robust ANN Search Using Hierarchical Navigable Small World Graphs (HNSW). *IEEE TPAMI* 42(4), 2020. [https://arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
9. Flamary R, Courty N, et al. POT: Python Optimal Transport. *JMLR* 22(78):1-8, 2021. [https://jmlr.org/papers/v22/20-451.html](https://jmlr.org/papers/v22/20-451.html)
10. Murty KG. An Algorithm for Ranking All the Assignments in Order of Increasing Cost. *Operations Research* 16(3):682-687, 1968. [https://pubsonline.informs.org/doi/abs/10.1287/opre.16.3.682](https://pubsonline.informs.org/doi/abs/10.1287/opre.16.3.682)
11. Meduri V, Popa L, Sen P, Sarwat M. A Comprehensive Benchmark Framework for Active Learning Methods in Entity Matching. *SIGMOD* 2020. [https://arxiv.org/abs/2003.13114](https://arxiv.org/abs/2003.13114)
12. Konda P, Das S, Doan A, et al. Magellan: Toward Building Entity Matching Management Systems. *PVLDB* 9(12):1197-1208, 2016. [https://dl.acm.org/doi/10.14778/2994509.2994535](https://dl.acm.org/doi/10.14778/2994509.2994535)
13. Zhu E, Nargesian F, Pu KQ, Miller RJ. LSH Ensemble: Internet-Scale Domain Search. *PVLDB* 9(12), 2016. [https://arxiv.org/abs/1603.07410](https://arxiv.org/abs/1603.07410)
14. Koutras C, et al. Valentine: Evaluating Matching Techniques for Dataset Discovery. *IEEE ICDE* 2021. [https://ieeexplore.ieee.org/document/9458921](https://ieeexplore.ieee.org/document/9458921)
15. Peeters R, Steiner A, Bizer C. Entity Matching using Large Language Models. *EDBT* 2025 / arXiv:2310.11244. [https://arxiv.org/abs/2310.11244](https://arxiv.org/abs/2310.11244)
