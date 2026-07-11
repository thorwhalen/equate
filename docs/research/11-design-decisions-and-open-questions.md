# Design Decisions & Open Questions: the `equate` Decision Register

*Round-2 capstone for the `equate` redesign. Where
[`00-taxonomy-and-terminology.md`](00-taxonomy-and-terminology.md) draws the map and
[`10-design-implications-for-equate.md`](10-design-implications-for-equate.md)
sketches the architecture, this document **decides**. It converts the tensions and
contradictions surfaced across the corpus (`00`–`18`) into a numbered register of
resolved design decisions, each with options, a recommended default, and the
concrete public-API consequence. It is written to be **sub-issue-able**: the
roadmap epic should be able to lift each `Dn` into a tracked issue verbatim.*

## Abstract

The research corpus is broad and, in places, self-contradictory — deliberately so,
because the contradictions are the real decisions the redesign must make. Blocking
is framed both as "leave most cells of the similarity matrix uncomputed" and as a
distinct upstream pipeline object; scores live in four incompatible spaces
(similarity, cost, log-odds, probability); comparators are assumed symmetric by the
assignment layer yet several key ones are directional; the core data model is
argued as strict 1:1 assignment, as many-to-many/typed correspondence, and as
collective clustering over an entity graph — all at once. This document is the
**decision register** that resolves those tensions into one coherent design, under
a single governing philosophy: *progressive disclosure* — the canonical
`match(A, B)` case is a one-line pure function with zero heavy dependencies, and
every stage, score space, cardinality, and backend is reachable by turning one
keyword or injecting one callable. It ranges over ten core decisions (`D1`–`D10`),
proposes a **resolved score & data-model contract** (a higher-is-better similarity
convention with an explicit sense-aware adapter policy, a directional-by-default /
symmetrize-at-the-boundary rule, and a layered model in which pair-scoring,
assignment, and clustering are all *strategies over one candidate+score
structure*), and closes with a prioritized list of deeper **open research
questions** that are tracked but not blocking. The roadmap epic draws its sub-issues
from this register.

The governing rule, restated so every decision can be checked against it:

> **`match(A, B)` must Just Work — 1:1, scored, batch, exact, zero config, no heavy
> deps. Every axis of variation is a keyword or an injected strategy. No heavy
> library is ever a hard dependency. Return structured data, never printouts.**

---

## How to read this register

Each decision has four parts:

- **(a) Decision** — the question, stated as a fork.
- **(b) Options & tradeoffs** — grounded in the corpus, citing the facet doc(s) that
  argue each side.
- **(c) Recommended default** — the call, with rationale tied to the project
  philosophy (functional-first; strategy pattern + dependency injection; a lean
  `numpy`/`scipy` core; `collections.abc` + `dataclasses`; keyword-only args past the
  2nd/3rd position; optional heavy deps behind lazy-imported extras).
- **(d) API / abstraction implication** — what it fixes in the public surface.

Decisions `D1`–`D7` correspond to the round-1 completeness critic's contradictions;
`D8`–`D10` are additional core decisions the corpus forces (structured-object
matchers, the default featurizer, and the unifying return type). The current public
surface being redesigned is `equate/__init__.py`'s
`{match_keys_to_values, match_greedily, similarity_matrix, hungarian_matching}` over
`equate/util.py`.

---

## D1. Sparse similarity matrix *vs* a separate candidate-generation stage

**(a) Decision.** Is blocking modeled as "the sparse similarity matrix" (a matcher
that never populates most cells), or as a distinct upstream pipeline object that
emits candidate pairs *before* any matrix exists? The two framings imply different
core APIs: a matrix-centric core vs a stream-of-pairs core.

**(b) Options & tradeoffs.**

- **Sparse-matrix-as-unifying-structure.**
  [`00`](00-taxonomy-and-terminology.md#12-the-expanded-field-standard-pipeline) and
  [`10`](10-design-implications-for-equate.md#3-scalability-path-avoid-all-pairs-before-you-optimize-it)
  frame blocking as *"precisely the decision to leave most cells of that matrix
  uncomputed."* This is conceptually elegant — one `ScoreMatrix` SSOT flows compare
  → match, `ensure_sparse` already gestures at it, and sparse LAP solvers
  (`min_weight_full_bipartite_matching`, `lapmod`) consume it directly
  ([`03`](03-assignment-and-graph-matching.md#35-sparse-and-rectangular-lap)).
- **Candidate generation as a distinct object.**
  [`02`](02-blocking-and-scalable-candidate-generation.md#10-design-implications-for-equate)
  is emphatic that the generator must return *"a **lazy iterator/generator** of
  pairs, never a materialized n×m matrix — this is the whole point"*, aligning with
  the iterables discipline. Materializing even a sparse `n×m` matrix to represent
  "which pairs to score" is backwards: at 10⁶×10⁶ you cannot hold the index of
  candidate cells, let alone the cells. Vector-DB backends
  ([`17`](17-vector-databases-and-scale-out.md)) return *ids*, not matrices.

The tension is only apparent: they describe **two ends of the same edge**. The
blocker's *output* is a stream of candidate pairs; the scorer's *output over that
stream* is a sparse matrix. The mistake is making either one the *only* core type.

**(c) Recommended default.** **Both, layered, with the pair-stream as the load-bearing
primitive and the sparse matrix as its scored materialization.** A `Blocker` is a
distinct stage returning `Iterable[tuple[int, int]]` (lazy). The default blocker is
`all_pairs` — the correctness baseline that makes the dense path a *degenerate
blocker*, not a separate code path. Scoring a candidate stream produces a
`ScoreMatrix` that is dense when the blocker is `all_pairs` and sparse otherwise.
This honors [`02`](02-blocking-and-scalable-candidate-generation.md)'s streaming
requirement *and* [`10`](10-design-implications-for-equate.md)'s one-SSOT-matrix goal
without contradiction: the matrix is a *view over scored candidates*, not the thing
the blocker builds.

**(d) API / abstraction implication.**

```python
Blocker = Callable[[Sequence[Repr], Sequence[Repr]], Iterable[tuple[int, int]]]
# default: all_pairs; swap by block='minhash_lsh' | 'ann' | a callable
# self-join form block(A) for dedup
```

The core pipeline is `featurize → block → compare(over candidates) → match`. Below a
size threshold `match()` uses `all_pairs` silently; above it, it warns and names the
extra (`equate[ann]`, `equate[lsh]`). The `ScoreMatrix` dataclass carries `data`
(dense ndarray or `scipy.sparse`), `sense`, and row/col labels — the compare→match
contract — but is *produced by* scoring the candidate stream, never authored by the
blocker.

---

## D2. Score-space inconsistency (similarity vs cost vs log-odds vs probability)

**(a) Decision.** There is no single score contract in the corpus: similarity in
`[0,1]`, sign-flipped distance as assignment cost, Fellegi-Sunter log-odds match
weights in `(-∞, ∞)`, LLM probabilities, and raw unbounded distances (edit counts,
km, DTW cost) all appear. Today each matcher converts differently
(`hungarian_matching` does `S.max() - S`, `kuhn_munkres_matching` does `-S`,
`stable_marriage_matching` does `1 - S`) — *"not equivalent and a latent bug"*
([`10`](10-design-implications-for-equate.md#1-the-core-canonical-case-to-optimize-first)).
What is the canonical score, and how do the other spaces adapt into it without
corrupting non-metric comparators?

**(b) Options & tradeoffs.**

- **Assume similarity ∈ [0,1] everywhere.** Simplest, but *wrong*:
  [`05`](05-comparison-and-similarity-functions.md#12-similarity-vs-distance-bounded-vs-raw-graded-vs-boolean)
  warns Monge-Elkan, Affine-Gap, Needleman-Wunsch, Smith-Waterman and Soft-TF-IDF
  *"don't normalize cleanly"* (py_stringmatching deliberately omits their
  `get_sim_score`), and [`12`](12-sequence-and-graph-structure-matching.md) notes
  DTW/soft-DTW are *dissimilarities* and soft-DTW *can be negative*. A `1 - s`
  adapter applied blindly to an unbounded distance or a log-odds weight is nonsense.
- **Make cost the canonical space** (assignment-native). Natural for LAP
  ([`03`](03-assignment-and-graph-matching.md)) but alien to users, who think in
  "how alike," and to the retrieval/embedding tiers
  ([`13`](13-llm-and-modern-embedding-matching.md), cosine similarity, higher =
  closer) and to Fellegi-Sunter (positive weight = evidence *for*).
- **One canonical similarity + explicit, sense-aware adapters.**
  [`03`](03-assignment-and-graph-matching.md#11-design-implications-for-equate) and
  [`05`](05-comparison-and-similarity-functions.md#7-design-implications-for-equate)
  both converge here: a single `sense` parameter and one `_to_cost(scores, *, sense)`
  SSOT, plus a comparator that *declares its own polarity/bounds* as metadata so the
  framework adapts instead of the caller hard-coding `1 - s`.

**(c) Recommended default.** **Canonical public score is a similarity, higher = more
alike, retained as a float; the *matcher* — not the comparator, not the user —
converts to cost through one SSOT `_to_cost(scores, *, sense='maximize')`.** Crucially,
`_to_cost` must be **polarity- and bounds-aware, not a fixed algebraic flip**:

- A comparator declares metadata (`polarity: similarity|distance`, `bounded: bool`,
  `is_metric`, `is_symmetric`) per
  [`05`](05-comparison-and-similarity-functions.md#7-design-implications-for-equate).
- `sense='maximize'` on a bounded similarity → negate (`-S`) or complement, exactly
  one canonical choice, applied identically by every matcher.
- **Distances stay distances**: an alignment/edit/DTW dissimilarity is routed with
  `sense='minimize'` straight to the cost side — *never* forced through `1 - S`
  ([`12`](12-sequence-and-graph-structure-matching.md#7-design-implications-for-equate)
  §2).
- **Log-odds and probabilities are separate, explicit adapters**, not silent
  coercions: a Fellegi-Sunter combiner emits a match weight; `calibrate` (Platt /
  isotonic) maps *raw scores of one comparator* to `P(match)`; neither is applied
  automatically across comparator boundaries
  ([`05`](05-comparison-and-similarity-functions.md#51-thresholding-and-calibration)).

The load-bearing warning, promoted to a documented contract:

> **Never min-max, `1-s`, or cross-normalize scores from *different* comparators, and
> never treat a non-metric similarity (Jaro-Winkler, cosine, Monge-Elkan,
> `SequenceMatcher.ratio`, DTW) as a metric.** Normalization is *per comparator*
> (a decay function, a calibrator), attached to that comparator; the matcher only
> ever applies the one `sense` flip.

**(d) API / abstraction implication.** `match/__init__.py` owns
`_to_cost(scores, *, sense)`; every matcher takes `sense: Literal['maximize',
'minimize'] = 'maximize'` and calls it. Comparators are constructed by `direct(h)` or
`featurized(φ, g)` and carry declared metadata; `calibrate` and numeric/geo `decay`
adapters live in `compare/calibrate.py` and are opt-in, never auto-inserted. The
`ScoreMatrix` carries its `sense` so a downstream matcher can never guess wrong.

---

## D3. Symmetric-matrix assumption *vs* asymmetric (directional) comparators

**(a) Decision.** Bipartite assignment treats `S` as a rectangular affinity where
`S[i,j]` is *the* score for the pair — direction-free. But several first-class
comparators are directional: Monge-Elkan (`ME(a,b) ≠ ME(b,a)`), Tversky's `α,β`
asymmetry, `SequenceMatcher.ratio` order-sensitivity
(`ratio('tide','diet')=0.25` vs `0.5`), and set **containment** `|Q∩X|/|Q|` for
size-skewed columns ([`05`](05-comparison-and-similarity-functions.md),
[`07`](07-schema-and-ontology-matching.md) join-key discovery). Do we symmetrize
(min/max/mean) or preserve direction?

**(b) Options & tradeoffs.**

- **Symmetrize eagerly at the comparator.** Keeps the matrix square-symmetric and
  every matcher simple, but *destroys real signal*: containment's whole point in
  schema matching ([`07`](07-schema-and-ontology-matching.md)) is that a small column
  can be contained in a large one; averaging that away defeats join-key discovery.
- **Preserve direction everywhere.** Faithful, but most assignment/stable/OT solvers
  ([`03`](03-assignment-and-graph-matching.md)) *require* a single scalar per pair;
  Gale-Shapley already derives two preference lists, so it can consume both
  directions, but LAP cannot.
- **Directional at the comparator, symmetrized at the matcher boundary by an explicit
  policy.** The comparator declares `is_symmetric`; when it is false, `equate`
  retains both `S` and `Sᵀ` (or computes both directions) and applies a *named,
  visible* reconciliation only where the chosen matcher demands a scalar.

**(c) Recommended default.** **Preserve direction at the comparator; symmetrize only
at the matcher boundary, via an explicit policy defaulting to `mean`, and only when
the matcher requires it.** Rationale: this keeps the *modelling* honest (containment
and Monge-Elkan stay directional for retrieval/schema use), while making the
*loss of information* a deliberate, logged step rather than a silent default — exactly
the "numerical honesty in the API" stance of
[`03`](03-assignment-and-graph-matching.md#11-design-implications-for-equate) and
[`05`](05-comparison-and-similarity-functions.md). Matchers that can *use* both
directions (Gale-Shapley's two preference lists) do; matchers that cannot (LAP) get
the symmetrized scalar with `symmetrize='mean'|'min'|'max'` surfaced as a keyword.
For 1:n retrieval and schema/join-key discovery
([`07`](07-schema-and-ontology-matching.md)) the *directional* score is the output —
no symmetrization at all.

**(d) API / abstraction implication.** `Comparator.is_symmetric` metadata; the
`ScoreMatrix` may hold a direction flag or a paired transpose. Matchers requiring a
scalar accept `symmetrize: Literal['mean','min','max'] = 'mean'`; the default facade
warns once when it symmetrizes a declared-asymmetric comparator. Containment and
Tversky ship in `compare/` as explicitly directional, documented as retrieval/schema
comparators, not LAP inputs.

---

## D4. The contested core data model (1:1 assignment vs n:m/typed vs clustering graph)

**(a) Decision.** The single most consequential fork: *what is a `Matcher`,
fundamentally?* Three answers coexist in the corpus — a **pair-scorer** producing
independent scored links; a **bipartite 1:1 assignment** (LAP/Murty over a cost
matrix); and a **collective/clustering** resolver propagating decisions through an
entity graph into a partition. This choice drives whether the re-optimization core is
Murty/LAP or constrained clustering, and whether the output is edges, an assignment,
or a partition.

**(b) Options & tradeoffs.**

- **Strict 1:1 assignment as the spine.** The current code
  (`hungarian_matching`, `match_greedily` remove-matched-value) and
  [`03`](03-assignment-and-graph-matching.md) center here; it is the exactly-solvable,
  globally-coherent core (JV in `scipy`), and Sadinle's Bayesian bipartite prior
  ([`15`](15-collective-incremental-and-bayesian-er.md#33-bipartite--partition-priors-sadinle))
  is its statistical twin. But it is *not an equivalence relation*
  ([`00`](00-taxonomy-and-terminology.md#4-the-equivalence-relation-framing-why-equate-is-not-equality))
  and is wrong for dedup and link-to-KB.
- **Many-to-many / subsumption / typed correspondences.** Schema matching returns
  n:m correspondence *sets* with typed relations (equivalence, subsumption)
  ([`07`](07-schema-and-ontology-matching.md)); capacitated/GAP matching
  ([`03`](03-assignment-and-graph-matching.md#7-many-to-many-and-generalized-assignment))
  is NP-hard and needs flow/ILP. Powerful, but a poor *default* — most requests are
  1:1 or top-k.
- **Collective / clustering / partition as SSOT.**
  [`15`](15-collective-incremental-and-bayesian-er.md#45-adopt-the-partition-as-the-ssot-output-generalise-to-n-way)
  argues the **partition of a pooled record set** (records → latent entities) is the
  natural single-source-of-truth: it subsumes pairwise links (edges within a
  cluster), 1:1 assignment (bipartite, one record per file per cluster), and
  within/across-file dedup uniformly, and generalizes `match(A, B)` to
  `resolve(*collections)`. But transitive closure is fragile (one spurious edge
  collapses two entities — [`00`](00-taxonomy-and-terminology.md#42-transitive-closure-and-clustering)),
  and clustering is the wrong algebra for genuine 1:n/n:m tasks.

**(c) Recommended default.** **The `Matcher` is fundamentally a *strategy that turns a
candidate+score structure into a correspondence* — and the *algebra of that
correspondence is itself a strategy*. Default to bipartite 1:1 (scipy-JV) as the
optimized canonical path; expose 1:n top-k, soft plans, and clustering/partition as
sibling strategies over the same score structure; make `Partition` the canonical
*clustering* output (not the canonical *matching* output).** This is a layered
model, not a single winner:

- **Layer 0 — scoring** produces the candidate+score structure (D1/D2).
- **Layer 1 — matching strategy** is one of: `pair` (threshold/top-k, no global
  constraint — the 1:n and link-to-KB case, cheap argmax/ANN per row); `assign`
  (bipartite 1:1 LAP — the default); `soft` (OT transport plan); `cluster` (partition
  via connected-components or correlation clustering, for dedup/collective).
- The **spine is 1:1** because it is the common request, exactly solvable, and
  already shipped — but it is *one strategy*, and the framework never hard-wires it
  (nor hard-wires transitive closure —
  [`00`](00-taxonomy-and-terminology.md#43-when-to-enforce-transitivity--and-when-not-to),
  [`15`](15-collective-incremental-and-bayesian-er.md#74-global-consistency--transitivity-stay-in-the-matchcluster-stage)).

Collective ER is a *match-stage strategy that additionally consumes a relational
graph* and stays behind an optional extra
([`15`](15-collective-incremental-and-bayesian-er.md#73-collective-matching-is-a-match-stage-strategy-that-consumes-a-graph));
it is never the default. Statefulness (streaming `Resolver`) is an *opt-in facade
over the same stages*, never a tax on the batch core
([`15`](15-collective-incremental-and-bayesian-er.md#71-keep-the-core-a-pure-batch-function-add-a-resolver-facade-for-state)).

**(d) API / abstraction implication.** The surface facade is
`match(A, B, *, how='assign'|'pairs'|'soft'|'cluster', k=…, threshold=…)` with
`how='assign'` default; `dedupe(A)` and `resolve(*collections)` are thin facades
selecting `how='cluster'`. Output types: `assign` → `Matching` (pairs + scores);
`pairs` → top-k `Candidate` lists; `soft` → transport `plan`; `cluster` →
`Partition` (records → entity ids). A `CollectiveMatcher(scores, graph) → Partition`
protocol exists for the relational strategy but defaults to ignoring the graph. See
D10 for how these return types unify.

---

## D5. Placement of the classify / decide stage (Fellegi-Sunter's home)

**(a) Decision.** The corpus gives Fellegi-Sunter classification three inconsistent
homes: a distinct optional pipeline stage
([`00`](00-taxonomy-and-terminology.md#12-the-expanded-field-standard-pipeline)
"Classify / decide"), folded into comparison as a *combiner* over the comparison
vector ([`05`](05-comparison-and-similarity-functions.md#52-the-comparison-vector-and-multi-field-composition)),
and a distinct `ClassifierMatcher` semantics that fuses scoring and decision
([`09`](09-python-ecosystem-landscape.md#6-design-implications-for-equate) L4;
[`10`](10-design-implications-for-equate.md#9-open-design-questions-carried-forward)).
Where does it live?

**(b) Options & tradeoffs.**

- **As a comparison *combiner*.** Fellegi-Sunter *is* a principled way to reduce a
  per-field comparison vector to a scalar match weight
  ([`05`](05-comparison-and-similarity-functions.md#4-learned--parameterized-comparators)),
  EM-estimable unsupervised. Placing it in `compare/` keeps "score" and "decide"
  cleanly separable and lets the *same* combiner feed a threshold, a rule, or a
  learned model.
- **As a distinct classify stage.** Conceptually
  ([`00`](00-taxonomy-and-terminology.md)) it maps a comparison vector →
  match/non-match/*possible-match* (the abstain/clerical-review band), which is a
  *decision*, not a score — and the possible-match band is the hook for
  human-in-the-loop ([`08`](08-interactive-active-learning-and-hitl.md)).
- **As a `ClassifierMatcher`.** dedupe/Splink/recordlinkage all fuse "score the
  vector" and "decide match" in one trained object
  ([`09`](09-python-ecosystem-landscape.md#41-end-to-end-er-frameworks-l1l5)); a
  supervised classifier over the comparison vector *is* both at once.

**(c) Recommended default.** **Split the concern along the score→normalize→decide seam
([`05`](05-comparison-and-similarity-functions.md#7-design-implications-for-equate)
§3): the Fellegi-Sunter *log-odds combination* is a comparison-vector combiner in
`compare/`; the *thresholded three-way decision* (match / possible-match / non-match)
is a thin, optional `classify`/`decide` step in the match layer; a fully supervised
`ClassifierMatcher` is one registered matcher strategy that happens to fuse both.**
The value is that the *same* field comparators feed (i) a raw matrix, (ii) a
Fellegi-Sunter weight, or (iii) a fitted sklearn estimator interchangeably — the
architectural move
[`05`](05-comparison-and-similarity-functions.md#52-the-comparison-vector-and-multi-field-composition)
calls "the key." Keep the default path score-only (no classification) so simple
things stay simple; classification is opt-in and returns a *typed decision with the
possible-match band preserved* for review.

**(d) API / abstraction implication.**
`compare/vectorize.py` holds `FieldComparators` → comparison vector and combiners
(`weighted_sum`, `mean`, `max`, `fellegi_sunter`, fitted estimator). A `classify`
step (in `match/` or a small `decide.py`) maps a scalar/weight to
`{match, possible_match, non_match}` with two thresholds `Tμ > Tλ`, defaulting to
off. `ClassifierMatcher` is a registry entry, not a privileged class. Fellegi-Sunter
EM is owned in-house (small); heavy end-to-end frameworks are wrapped behind
`equate[er]`, never imported by default
([`09`](09-python-ecosystem-landscape.md#6-design-implications-for-equate)).

---

## D6. One embedding space for both blocking and matching *vs* mandatory cross-encoder rerank

**(a) Decision.** Can a single contrastive bi-encoder embedding serve *both* blocking
(ANN candidate generation) and final matching (pairwise score), or must blocking
never reuse the matcher's similarity — with a mandatory cross-encoder / LLM re-rank
on top (coupling risk vs accuracy)?

**(b) Options & tradeoffs.**

- **One embedding, both stages.** [`13`](13-llm-and-modern-embedding-matching.md)
  shows modern embedders (E5, BGE-M3, GTE) give LLM-quality retrieval from one
  forward pass; BGE-M3 emits dense + learned-sparse + multi-vector in *one* model,
  mapping *"unusually cleanly onto a matcher's blocking + scoring stages."* Reusing
  the embedder as both blocker and scorer is cheap and simple.
- **Never reuse; always cross-encode the survivors.**
  [`06`](06-deep-learning-and-llm-entity-matching.md) /
  [`13`](13-llm-and-modern-embedding-matching.md#2-the-two-tiers-restated-for-2025)
  establish the bi-encoder / cross-encoder split: *"cheap independently-embeddable
  scorers (bi-encoders) drive blocking; expensive joint scorers (LLMs/cross-encoders)
  drive final decisions."* A bi-encoder that both blocks *and* decides can share its
  blind spots on both stages (the same jargon/part-number failure mode —
  [`13`](13-llm-and-modern-embedding-matching.md#35-the-benchmark-to-point-users-at-mteb--mmteb)),
  and blocking recall *caps* end-to-end recall regardless of a strong re-ranker
  ([`02`](02-blocking-and-scalable-candidate-generation.md#2-evaluating-candidate-generation-the-recallefficiency-trade-off)).

**(c) Recommended default.** **Allow one embedding to serve both stages (the cheap,
correct default), but make the *cascade* — block with a bi-encoder, re-rank the top-k
survivors with an optional stronger scorer — the idiomatic composition, and never
make the cross-encoder mandatory.** Rationale: TF-IDF + edit distance is *genuinely
competitive on clean structured data*
([`10`](10-design-implications-for-equate.md#7-wrap-vs-reimplement-per-layer),
[`13`](13-llm-and-modern-embedding-matching.md#48-cost--accuracy--latency-and-when-llms-beat-fine-tuned-plms)),
so a mandatory LLM re-rank is wrong for the common case. The coupling risk is real
but is a *tuning* concern, not an architectural one: `equate` exposes both, lets the
same `Embedder` play both roles (one embedder, two roles —
[`13`](13-llm-and-modern-embedding-matching.md#6-make-prompting-strategy-a-swappable-object-on-the-llm-scorer)),
and — critically — ships PC/RR/PQ blocking metrics
([`02`](02-blocking-and-scalable-candidate-generation.md#7-make-pc--rr--pq-first-class-measurable-outputs))
so a user can *measure* whether reusing the embedding for blocking is capping recall
on *their* data. The re-rank tier is a `cascade([...])` combinator with per-tier cost
accounting ([`13`](13-llm-and-modern-embedding-matching.md#9-bake-the-cascade-in-with-per-tier-cost-accounting)),
opt-in.

**(d) API / abstraction implication.** An `Embedder` can be injected once and
referenced by both `block=` and `compare=`. `cascade([block_embed, cheap_scorer,
llm_rerank])` is a first-class combinator; the LLM/cross-encoder scorer is a
`direct(h)` comparator behind `equate[llm]` that *cannot* drive an index (declared
non-indexable metadata, D2). Multi-functional embedders (BGE-M3) expose
`output_kinds={dense, sparse, multi_vector}`; the blocker consumes dense+sparse, the
optional re-ranker consumes multi-vector — one featurizer, several primitives
([`13`](13-llm-and-modern-embedding-matching.md#3-model-multi-functional-embedders-bge-m3-as-a-featurizer-that-emits-several-comparison-primitives)).

---

## D7. Ship-vs-wrap stance on the heavy ML footprint

**(a) Decision.** How heavy is `equate` out of the box? Define only the encoder/index
interfaces and leave all models/backends to the user; ship DeepBlocker/UniBlocker-style
models and a default embedder in-framework; or bundle a default ANN backend so "it
just scales" on install? The corpus states the boundary inconsistently across
[`02`](02-blocking-and-scalable-candidate-generation.md),
[`04`](04-featurization-and-representation.md),
[`06`](06-deep-learning-and-llm-entity-matching.md),
[`13`](13-llm-and-modern-embedding-matching.md), and
[`17`](17-vector-databases-and-scale-out.md).

**(b) Options & tradeoffs.**

- **Interface-only (thinnest).** Define `Embedder`, `VectorIndex`, `Blocker`,
  `Matcher` protocols and ship *nothing* heavy. Maximally light and license-clean
  ([`09`](09-python-ecosystem-landscape.md#5-cross-cutting-observations)), but a bare
  install can't do dense blocking or embedding at all — the user must assemble every
  piece.
- **Ship models in-framework.** Bundle a DeepBlocker/UniBlocker encoder and a default
  sentence-transformer so dense matching works on `pip install equate`. Best OOB
  experience, but drags in `torch` (heavyweight), GPU concerns, model-license review
  (Jina v3 is CC-BY-NC —
  [`13`](13-llm-and-modern-embedding-matching.md#32-open-weight-families)), and breaks
  the "no heavy library is ever a hard dependency" rule.
- **Lean core + optional extras + capability detection.**
  [`09`](09-python-ecosystem-landscape.md#6-design-implications-for-equate),
  [`10`](10-design-implications-for-equate.md#7-wrap-vs-reimplement-per-layer),
  [`13`](13-llm-and-modern-embedding-matching.md#10-optional-dependency-map-extend-doc-04-93--doc-06-74),
  and [`17`](17-vector-databases-and-scale-out.md#8-wrap-vs-embed--the-verdict-for-equates-blocking-layer)
  all converge: core is `numpy`/`scipy`/`sklearn`/`grub`/stdlib; every heavy backend
  is a lazy-imported extra; a capability detector auto-selects the fastest *installed*
  backend and emits an actionable install hint otherwise. *"Reimplement almost
  nothing computational; own the orchestration."*

**(c) Recommended default.** **Lean core, wrap-not-ship, capability-detected. Define
the interfaces and ship a zero-dependency default at every seam; ship *no* model
weights and *no* heavy backend as a hard dep; wrap best-in-class libraries behind
`pip` extras with lazy imports and `check_requirements`-style errors.** This is the
strongest cross-corpus consensus and the direct expression of progressive disclosure.
Concretely: the default text featurizer is char-n-gram TF-IDF (D9); the default
matcher is scipy-JV; the default blocker is `all_pairs`; DeepBlocker/UniBlocker,
sentence-transformers (BGE-M3/E5/Nomic), FAISS/hnswlib, POT, LLM clients, and every
vector-DB are optional extras. Copyleft/heavy engines (Zingg AGPL, Spark) stay
strictly **out-of-process behind a driver interface, never imported**
([`09`](09-python-ecosystem-landscape.md#5-cross-cutting-observations),
[`17`](17-vector-databases-and-scale-out.md#8-wrap-vs-embed--the-verdict-for-equates-blocking-layer)).
The one thing `equate` *ships and owns*: the orchestration, the protocols, the
`ScoreMatrix`/`CandidateStore` SSOTs, the registries, and capability detection.

**(d) API / abstraction implication.** A `registry.py` capability detector picks
`rapidfuzz > difflib`, `hnswlib/faiss > brute force`, `lapx > scipy` on large sparse,
etc., and prints install hints. Extras map (consolidated from
[`09`](09-python-ecosystem-landscape.md), [`13`](13-llm-and-modern-embedding-matching.md),
[`17`](17-vector-databases-and-scale-out.md)): `equate[fuzzy]`, `equate[phonetic]`,
`equate[embeddings]`, `equate[api]`, `equate[ann]`, `equate[lsh]`, `equate[ot]`,
`equate[kbest]`, `equate[graph]`, `equate[llm]`/`[llm-local]`, `equate[lancedb]`/
`[pgvector]`/`[qdrant]`/…, `equate[duckdb]`/`[dask]`/`[spark]`, `equate[gpu]`,
`equate[collective]`, `equate[bayes]`. No extra is imported until used.

---

## D8. How structured (sequence / graph) matcher families fit the `Matcher` abstraction

**(a) Decision.** [`12`](12-sequence-and-graph-structure-matching.md) adds matcher
families the round-1 flat pipeline did not anticipate: sequence alignment (DTW,
soft-DTW, Needleman-Wunsch, Smith-Waterman) and graph/network alignment (GED, VF2,
WL kernels, IsoRank/FINAL/REGAL/CONE-Align). Do these bolt onto the existing seams,
or do they force a new `Matcher` arity?

**(b) Options & tradeoffs.** [`12`](12-sequence-and-graph-structure-matching.md#5-cross-cutting-map-to-the-equate-abstractions)
resolves this cleanly, and the resolution is worth adopting verbatim because it keeps
the flat core untouched:

- **Alignment comparators** (DTW/soft-DTW/NW/SW, GED, MCS, VF2) are **`direct(h)`
  comparators that also emit an internal `Alignment(score, path)`** — opaque, *not
  ANN-indexable*, they can only re-rank a candidate set; the collection-level matcher
  is unchanged (build `S[i,j]`, hand to flat LAP/greedy/soft, route as
  `sense='minimize'` because alignment scores are dissimilarities).
- **Graph kernels** (Weisfeiler-Lehman, graphlet) are **`Featurizer`s** (graph →
  vector), restoring the indexable featurize-then-compare path — reuse cosine + ANN/LSH
  blocking for graphs with no match-layer special-casing.
- **Network aligners** (IsoRank, FINAL, REGAL, CONE-Align, KG entity alignment)
  *cannot* be expressed as `matcher(scores)` because they **produce the scores and
  the correspondence jointly** from two structured objects — they need a distinct
  arity: `QuadraticMatcher.align(G1, G2, *, node_affinity=None, seeds=None) →
  correspondence`, with the flat LAP as the reusable *inner* extraction step.

**(c) Recommended default.** **Adopt [`12`](12-sequence-and-graph-structure-matching.md)'s
two-new-families layering: alignment comparators (feeding the *unchanged* flat
assignment layer), graph kernels reclaimed as featurizers, and a distinct
`QuadraticMatcher` protocol for network alignment — every heavy engine behind a lazy
extra, reimplementing almost nothing.** The default sequence dissimilarity is a
**Sakoe-Chiba-banded exact DTW** (via `dtaidistance`), *not* FastDTW, per Wu & Keogh
([`12`](12-sequence-and-graph-structure-matching.md#22-dynamic-time-warping-dtw)).
This is the strongest expression of `equate`'s "equality relaxed, *recursively*"
thesis: matching a collection of structured objects = compare each pair by matching
their sub-parts → assign at the collection level — the same triad one level down
([`12`](12-sequence-and-graph-structure-matching.md#7-design-implications-for-equate)
§7).

**(d) API / abstraction implication.** Two `Matcher` arities coexist:
`Matcher(scores, *, sense) → pairs` (flat, unchanged) and
`QuadraticMatcher(G1, G2, *, node_affinity, seeds) → correspondence` (structured).
A `Comparator` may optionally return `Alignment(score, path)`; alignment scores
declare `polarity='distance'`. Extras: `equate[timeseries]` (dtaidistance/tslearn),
`equate[bio]` (Biopython/parasail), `equate[graph]` (networkx VF2/GED),
`equate[graph-kernels]` (GraKeL), `equate[graph-matching]` (pygmtools),
`equate[network-align]` (PLANETALIGN). Soft-DTW and Gromov-Wasserstein fold into the
one `soft_match(...) → plan` + `harden(plan)` seam alongside vector OT (D10).

---

## D9. The canonical default text featurizer (TF-IDF core vs shipped modern embedder)

**(a) Decision.** What is the *default* featurizer on a bare install: keep char-n-gram
TF-IDF (current `grub`-backed path), or default to a modern contrastive embedder?
[`13`](13-llm-and-modern-embedding-matching.md) shows SBERT-era defaults are
superseded, but a strong embedder means `torch`.

**(b) Options & tradeoffs.**

- **TF-IDF core default.** Zero heavy deps, genuinely competitive on clean/structured
  name/SKU matching
  ([`13`](13-llm-and-modern-embedding-matching.md#37-recommended-equate-text-featurizer-defaults),
  [`10`](10-design-implications-for-equate.md#1-the-core-canonical-case-to-optimize-first)),
  already shipped. Misses semantic/paraphrase/cross-lingual similarity.
- **Modern embedder default.** Much stronger retrieval, but pulls `torch`, needs model
  download, raises license questions (do *not* hard-default to Jina v3, CC-BY-NC —
  [`13`](13-llm-and-modern-embedding-matching.md#37-recommended-equate-text-featurizer-defaults)),
  violates the lean-core rule (D7).

**(c) Recommended default.** **Keep char-n-gram TF-IDF as the *core zero-dependency*
default; make a MIT-licensed, long-context, multilingual embedder the default *when
`equate[embeddings]` is installed* — `bge-m3` or `multilingual-e5-large`, with
`nomic-embed-text-v1.5` (Apache-2.0, reproducible, MRL→64) as the small option.**
This is the honest "simple things simple" baseline plus a strong opt-in, and it
directly follows [`13`](13-llm-and-modern-embedding-matching.md#5-design-implications-for-equate)
§1. Crucially, license/dimension/context/prefix/truncation are **declared metadata on
the `Featurizer`** so `equate` can (a) refuse a non-commercial model in a commercial
context, (b) auto-apply the E5-style `query:`/`passage:` prefix (the silent-prefix
footgun — [`13`](13-llm-and-modern-embedding-matching.md#31-the-recipe-that-unified-them)),
(c) offer `dimensions=k` Matryoshka truncation, and (d) auto-pick a legal metric/index.

**(d) API / abstraction implication.** `featurize/text.py` registers `tfidf` (default,
core) and `[sbert|bge-m3|e5|nomic|openai|cohere|voyage]` behind
`equate[embeddings]`/`equate[api]`. `Featurizer` metadata: `license`, `dim`,
`max_seq_len`, `query_prefix`/`passage_prefix`, `truncatable_to`, `normalize`,
`output_kinds`. Int8/binary quantization is a supported vector-path option bridging to
the bit-string/Hamming machinery
([`13`](13-llm-and-modern-embedding-matching.md#4-first-class-quantization-on-the-vector-path)).

---

## D10. The unifying return type across hard / soft / partition / uncertainty tiers

**(a) Decision.** The strategies chosen in D4/D8 return structurally different things:
hard matchers → pair lists; soft/OT → a fractional plan matrix
([`03`](03-assignment-and-graph-matching.md#6-soft--graded-matching-via-optimal-transport-sinkhorn));
clustering/collective → a partition
([`15`](15-collective-incremental-and-bayesian-er.md)); Bayesian ER → a *posterior
over partitions*. [`10`](10-design-implications-for-equate.md#9-open-design-questions-carried-forward)
flags "a unifying return type" as an open question. What is the output contract?

**(b) Options & tradeoffs.**

- **One rigid type.** Forcing everything into `Iterable[(i,j)]` loses the plan, the
  partition, and all uncertainty — and forcing everything into `Partition` taxes the
  95% "give me scored pairs" case
  ([`15`](15-collective-incremental-and-bayesian-er.md#78-what-not-to-do)).
- **Unrelated types per strategy.** Honest but unusable — every caller branches on
  the strategy.
- **A small tiered hierarchy where richer types *iterate as* the simpler one.**
  [`15`](15-collective-incremental-and-bayesian-er.md#72-make-the-output-contract-uncertainty-capable-but-not-uncertainty-requiring)
  proposes exactly this: Tier 0 `(i,j[,score])`; Tier 1 attaches a marginal link
  probability where available (cheap, no heavy deps); Tier 2 a full
  `PartitionPosterior` behind `equate[bayes]`. And
  [`03`](03-assignment-and-graph-matching.md#8-give-soft-matching-its-own-parallel-interface)
  gives soft matching a *parallel* `soft_match(...) → plan` + `harden(plan)` seam that
  interconverts with hard matching (argmax / LAP on `−plan`).

**(c) Recommended default.** **A small dataclass hierarchy with the rule that every
richer return *iterates as* the simpler one, so uncertainty and structure are
*expressible but never required*.** Concretely:

- **`Matching`** (default, hard): pairs + retained scores; iterating yields `(i, j)`.
  This is Tier 0/1 — it may carry a per-pair calibrated `probability` when a
  calibrator or Fellegi-Sunter combiner is in the pipeline (D2/D5), a pure
  down-shadow of the Bayesian contract with no heavy deps.
- **`Matching.plan`** (soft): the OT/Sinkhorn/GW/soft-DTW transport plan lives on the
  *same* object via the parallel `soft_match` seam; `harden(plan)` recovers pairs.
- **`Partition`** (clustering/collective/n-way): records → entity ids; iterating
  yields the MAP/point-estimate pairs so a `Partition` is drop-in where pairs are
  expected ([`15`](15-collective-incremental-and-bayesian-er.md#75-adopt-the-partition-as-the-ssot-output-generalise-to-n-way)).
- **`PartitionPosterior`** (Tier 2, `equate[bayes]`): MCMC samples + `link_probability`,
  `n_entities_posterior`, loss-aware `point_estimate`; iterating yields the point
  estimate.

All are JSON-serializable dataclasses so the HTTP service (via `qh`) and the
declarative UI (via `zodal`) attach with no rework
([`10`](10-design-implications-for-equate.md#5-where-an-http-service-and-a-ui-attach)).
This makes the D4 layering *usable*: a caller can ignore the tier it doesn't need.

**(d) API / abstraction implication.** `base.py` owns `Candidate`, `Matching`
(with optional `.plan` and `.probability`), `Partition`, `Explanation`, and
(behind the extra) `PartitionPosterior`. The iteration contract — *richer iterates as
simpler* — is the single rule that lets `match(A, B)` always return "something you can
`dict(...)`" regardless of `how=`.

---

## D11. Making the sparse hole-worst-casing structural (matchers consume `ScoreMatrix`)

**(a) Decision.** D2 resolved that a blocked (sparse) score matrix routes through the
`to_cost` SSOT, which worst-cases the structurally-absent (non-candidate) cells so the
optimizer never prefers an unscored hole over a real pair. But `to_cost` dispatches on
`issparse(scores)` and the worst-casing lives *only* in the sparse branch. Because
`scipy.sparse` uses `0` as its fill value, **any caller that `.toarray()`s the raw scores
before `to_cost` silently takes the dense branch and loses the worst-casing** — holes
become real `0`s that can out-rank negative similarities or look cheapest for a minimize
matcher. This one bug class recurred across stages (HIGHs in #7's sparse-LAP and #9's
reoptimize; and it was found *shipped* in the `max_weight`/`kuhn_munkres` matchers, which
densified-then-`to_cost`). Documentation alone does not fix an API shaped like a trap.

**(b) Options.** (i) *Document the gotcha* — relies on every author remembering; keeps
failing. (ii) *Fix each site + a conformance test* — catches present + future matchers but
leaves the `.toarray()` temptation in place. (iii) *Make the sanctioned densify the only
path* — matchers consume the `ScoreMatrix` and densify **only** through its worst-casing
views, so the wrong array is never obtainable inside a matcher.

**(c) Resolution — (iii), with (ii) as the backstop.** The `Matcher` input is the
`ScoreMatrix` SSOT (already the compare→match carrier, D1/D2). It owns the fill semantics
via `dense_cost()` (worst-cased cost, min-solvers), `dense_similarity()` (worst-cased
similarity, argmax/max-weight), `candidate_mask()`, `stored_entries()` (iterate real
candidates without densifying), and `drop_holes(pairs, *, keep=())` — the **one** place the
"never select a hole" filter lives. A matcher never calls `.toarray()`.

**Two distinct guarantees are needed — worst-casing alone is not enough.** Worst-casing
stops a hole being *preferred*; it does not stop a hole being *assigned* when a
full-cardinality solver is forced onto one (a row with no candidate; rows sharing their only
candidate column). So every path also **drops** hole assignments, and that drop is enforced
at the `resolve_matcher` **boundary** — not per-matcher — so the invariant never depends on
an individual (or third-party) matcher's discipline. Per-matcher drops remain as idempotent
defense in depth. A **registry-wide conformance test** sweeps every registered matcher over
asymmetric, rectangular, and *forced-partial* blocked fixtures, asserting the result is
hole-free **and injective** — so the bug class is un-shippable for future matchers (e.g.
#10's soft-DTW / OT / GW families).

> Both halves were learned in adversarial review: the first cut worst-cased correctly but
> forgot to *drop* on the soft/OT, `reoptimize`, and legacy-matcher paths, and its fixture
> (a transpose-symmetric 2×2 anti-diagonal admitting a complete matching) could catch
> neither that nor a transposed-pair orientation bug. **A fixture that never forces a hole
> does not test hole-dropping** — verify a conformance guard by mutating the code it guards.

**(d) The worst-case fill must be a big-M, because a matcher optimizes a *total*.** The
original fill — "one unit worse than the worst real *cell*" (`max_real_cost + 1`) — is not
strong enough, and this is the subtlest half of D11. A LAP/max-weight solver compares whole
*assignments*, so a hole priced just above the worst cell is a bargain the solver will
happily buy: it takes one hole to save more than `1` elsewhere, `drop_holes` then deletes
that pair, and the returned matching is **strictly dominated** by an all-real matching that
was there for the taking — *fewer pairs and a worse score*. Reproduced on the default
`how='assign'` path: the "optimal" matcher lost to the `greedy` heuristic (2 pairs / 23.0 vs
3 pairs / 32.0) on ~3% of random blocked matrices. It bites exactly when the smallest stored
similarity exceeds `1.0` — i.e. any **unbounded** comparator (`dot`, BM25, counts,
Fellegi-Sunter log-odds; equate *ships* `dot` with `bounded=False`) — and never for a
`[0,1]` comparator, which is why every existing `[0,1]` fixture missed it, and why the
randomized conformance sweep now generates unbounded scores.

The fill is therefore `_hole_fill()`: strictly beyond the largest *swing* any all-real
assignment can have (`k·(|max| + |min|) + 1` over `k = min(n, m)` assigned cells). This
makes the optimum **lexicographic**, which is the semantics blocking always meant:
**use as many real candidate pairs as possible, then optimize the score among those.** An
absent cell is not a bad option — it is *not an option*.

**(e) API / back-compat implication.** The native contract is `Matcher(ScoreMatrix) →
pairs` (marked with `@scorematrix_matcher`; it reads `sense` off the matrix). The legacy
`(scores, *, sense) → pairs` raw-array contract is still accepted — `resolve_matcher` hands
a legacy callable a hole-worst-cased dense array (`ScoreMatrix.legacy_view()`), so even a
third-party matcher that does its own `to_cost`/argmax stays correct. Shipped matcher
*functions* still accept a raw array + `sense=` (via `ScoreMatrix.coerce`), so external
PyPI call sites are unchanged.

> **`legacy_view()` must preserve the stored values.** The first cut returned
> `dense_similarity()` (`S - S.max()`), which for a *hole-free* dense matrix is a pure
> gratuitous rescale to all-nonpositive — silently breaking every `how=<callable>` matcher
> that reads *absolute* scores (a threshold, a sign test, a `[0,1]` assumption), with no
> error and a green test suite. Worst-casing only ever required the **holes** to be worse
> than the real cells; it never required touching the real cells. The view now hands back
> raw stored values and rewrites *only* the holes. Generalized lesson: **a fix that
> normalizes data on a back-compat path is a breaking change wearing a correctness costume.**

---

## Resolved score & data-model contract

This section consolidates D2/D3/D4/D10 into the two contracts every module must honor.

### The canonical score contract

1. **Public scores are similarities, higher = more alike, retained as floats.** The
   user-facing mental model is "how alike," matching the `key=`/`==` idiom
   ([`00`](00-taxonomy-and-terminology.md#41-from-equality-to-graded-correspondence),
   [`05`](05-comparison-and-similarity-functions.md#11-the-central-duality-featurize-then-compare-vs-direct-pairwise-compare)).
2. **Do not assume `[0,1]`.** A comparator declares `polarity`, `bounded`, `is_metric`,
   `is_symmetric` as metadata; the framework adapts. Bounded similarities may be
   compared directly; unbounded distances (edit counts, km, DTW/GED cost) stay
   distances and are routed with `sense='minimize'`.
3. **One SSOT adapter: `_to_cost(scores, *, sense)`.** Every matcher converts through
   it — never a per-matcher `1-S` / `-S` / `max-S`. `sense='maximize'` applies exactly
   one canonical similarity→cost flip; `sense='minimize'` passes a distance straight
   through. This kills the current three-way inconsistency
   ([`10`](10-design-implications-for-equate.md#1-the-core-canonical-case-to-optimize-first),
   [`03`](03-assignment-and-graph-matching.md#11-design-implications-for-equate)).
4. **Cost, log-odds, and probability are *explicit adapters, per comparator*, never
   silent cross-normalizations.** Fellegi-Sunter log-odds is a comparison-vector
   combiner (D5); `calibrate` (Platt/isotonic) maps one comparator's raw scores to
   `P(match)` (D2, D10-Tier1); numeric/geo `decay` maps a domain distance to a `[0,1]`
   similarity ([`05`](05-comparison-and-similarity-functions.md#3-non-string-comparators)).
5. **Non-metric warning is a documented contract.** Jaro-Winkler, cosine, Monge-Elkan,
   `SequenceMatcher.ratio`, DTW, soft-DTW are *not* metrics (no triangle inequality;
   soft-DTW can be negative). `equate` must never (i) min-max or `1-s` across
   *different* comparators, (ii) feed a raw distance/log-odds into a `[0,1]`-assuming
   path, or (iii) use a triangle-inequality index (VP/ball tree) on a non-metric.
   Normalization is *per comparator*, attached to it.

### The symmetric-vs-directional policy

- **Directional at the comparator; symmetrized only at the matcher boundary, only when
  required, via an explicit named policy defaulting to `mean`** (D3). Containment,
  Tversky, and Monge-Elkan stay directional for retrieval and schema/join-key
  discovery ([`07`](07-schema-and-ontology-matching.md)); LAP-family matchers receive a
  symmetrized scalar with `symmetrize=` surfaced and a one-time warning when a
  declared-asymmetric comparator is collapsed.

### What the `Matcher` abstraction fundamentally is

**A `Matcher` is a *strategy that turns a common candidate+score structure into a
correspondence*, and the *algebra of the correspondence is itself a chosen strategy*.**
There is no single "true" data model — pair-scoring, 1:1 assignment, soft transport,
and clustering/partition are **sibling strategies over the same `ScoreMatrix` /
`CandidateStore`** (D4):

| Strategy `how=` | Correspondence algebra | Core solver | Default? |
|---|---|---|---|
| `pairs` | independent scored links / top-k (1:n) | argmax / ANN per row | no |
| `assign` | bipartite partial matching (1:1) | scipy-JV LAP | **yes** |
| `soft` | fractional transport plan | Sinkhorn/EMD (POT) | no |
| `cluster` | equivalence partition (records→entities) | connected-components / correlation clustering | no |

The **spine is `assign`** — exactly solvable, globally coherent, already shipped — but
it is *one strategy*, never hard-wired, and transitive closure is never baked into
scoring. The **canonical *clustering* output is `Partition`** (subsuming links, 1:1,
and dedup, and generalizing `match(A, B)` to `resolve(*collections)` —
[`15`](15-collective-incremental-and-bayesian-er.md#75-adopt-the-partition-as-the-ssot-output-generalise-to-n-way)),
while the **canonical *matching* output is `Matching`** (D10). Structured objects add a
second *arity* (`QuadraticMatcher`, D8) but not a second *philosophy*: the inner flat
matcher is reused as the extraction step. Statefulness (streaming `Resolver`) and
uncertainty (`PartitionPosterior`) are *facades and tiers over the same stages*, never
the default ([`15`](15-collective-incremental-and-bayesian-er.md#7-design-implications-for-equate)).

---

## Open research questions (tracked, not blocking the core)

These are the deeper unknowns distilled from the facet docs' own open-questions
sections. None blocks the canonical `match(A, B)` path; each should be a tracked issue
so the design stays honest about what it defers.

**Prioritized (higher = decide sooner because it shapes public types):**

1. **Incremental / stateful `Resolver` semantics.** When does the batch core graduate
   to a live `Resolver` with `add`/`update`/`remove`, and is *order-independence*
   (Swoosh **ICAR** — Idempotence/Commutativity/Associativity/Representativity) the
   documented correctness precondition for affected-region re-resolution? Union-find
   *with split* is the enabling structure
   ([`15`](15-collective-incremental-and-bayesian-er.md#5-incremental-streaming--temporal-er)).
   Shapes whether the public surface is only a function or also an object.
2. **Uncertainty as a first-class, cheap output.** Should even non-Bayesian matchers
   always emit a *marginal link probability* (Tier 1, D10) via a mandatory calibration
   step, or keep raw scores by default? Calibration is a prerequisite for principled
   review triage and re-optimization
   ([`08`](08-interactive-active-learning-and-hitl.md),
   [`05`](05-comparison-and-similarity-functions.md#51-thresholding-and-calibration)).
3. **One normalized `recall_target` dial vs native backend params.** Auto-map a single
   recall knob to each blocker's native params (efSearch, nprobe, LSH bands/rows, SNM
   window, DiskANN) for progressive disclosure, vs exposing native params for control —
   the calibration-cost tradeoff
   ([`02`](02-blocking-and-scalable-candidate-generation.md#8-expose-the-recallefficiency-knob-uniformly),
   [`17`](17-vector-databases-and-scale-out.md#96-capability-detection--measured-selection)).
4. **Default clustering algebra.** Connected-components (cheap, but entity-collapse-prone
   on one spurious edge) vs correlation clustering (robust, costlier) as the default
   `cluster` strategy ([`00`](00-taxonomy-and-terminology.md#43-when-to-enforce-transitivity--and-when-not-to),
   [`01`](01-entity-resolution-record-linkage.md)). Interacts with D4/D10.
5. **k-best re-optimization core: Murty vs constrained clustering.** For interactive
   force/forbid edits, is the re-solve primitive Murty over LAP (bipartite world) or
   constrained/correlation clustering (partition world)? No maintained pure-Python
   Murty exists on PyPI — likely in-house `equate[kbest]`
   ([`03`](03-assignment-and-graph-matching.md#5-k-best-assignments-murty-and-interactive-re-optimization),
   [`08`](08-interactive-active-learning-and-hitl.md)).

**Deeper / longer-horizon (track, revisit post-v1):**

6. **Collective / relational joint inference.** MLN/PSL/relational-clustering as an
   opt-in `CollectiveMatcher` consuming an entity graph — heavy, domain-specific, and
   requires *relational* featurization (neighbourhood overlap), so clearly
   optional-dependency territory
   ([`15`](15-collective-incremental-and-bayesian-er.md#2-collective--relational-entity-resolution)).
7. **Full Bayesian ER backend.** A `PartitionPosterior` (blink/d-blink-class) behind
   `equate[bayes]` — posterior over partitions, microclustering priors, folding
   blocking into the generative model. Research-grade; the *interface* should exist so
   a backend is drop-in, the *default* should not require it
   ([`15`](15-collective-incremental-and-bayesian-er.md#3-bayesian-entity-resolution-a-posterior-over-partitions)).
8. **Cross-modal / cross-space alignment.** Gromov-Wasserstein for matching collections
   with no shared feature space (graphs, or text↔image), unified under the `soft_match`
   seam ([`03`](03-assignment-and-graph-matching.md#6-soft--graded-matching-via-optimal-transport-sinkhorn),
   [`12`](12-sequence-and-graph-structure-matching.md#45-the-optimal-transport--gromov-wasserstein-connection)).
9. **Multi-source (n-way, k>2) linkage as the general case.** Generalize
   `match(A, B)` → `resolve(*collections)` with a structured partition prior over a
   pooled record set, making multi-file *not* a bolt-on
   ([`15`](15-collective-incremental-and-bayesian-er.md#4-multi-source-n-way-k--2-collections-linkage)).
10. **Temporal ER / side-channel comparator context.** Entities whose attributes drift
    over time need time-aware comparison — an argument for keeping the comparator
    signature extensible to a side-channel (timestamps), not just two representations
    ([`15`](15-collective-incremental-and-bayesian-er.md#54-temporal-er-entities-whose-attributes-drift)).
11. **LLM set-level assignment consistency.** Letting the assigner consume *set-level*
    LLM judgements (ComEM-style select-from-candidates, BoostER uncertainty-targeted
    verification) with at-most-one-match consistency — the assigner accepts either a
    score matrix *or* a callable `(key, candidates) → chosen`
    ([`13`](13-llm-and-modern-embedding-matching.md#8-let-the-assigner-consume-set-level-llm-judgements)).
12. **Evaluation harness as a shipped default.** Ship PC/RR/PQ (blocking), pairwise
    P/R/F1, B³ (clustering) with *unseen-entity splits* and a *cost column* so users
    measure `bge-m3 + LLM-rerank` vs `TF-IDF + AnyMatch` on their own data/budget
    before paying ([`14`](14-evaluation-benchmarks-and-methodology.md),
    [`13`](13-llm-and-modern-embedding-matching.md#11-evaluate-before-paying)).

---

## References

Drawn from the corpus (`00`–`18`); Vancouver style.

1. Christophides V, Efthymiou V, Palpanas T, Papadakis G, Stefanidis K. An Overview of End-to-End Entity Resolution for Big Data. *ACM Computing Surveys* 53(6), Art. 127, 2020. [https://dl.acm.org/doi/10.1145/3418896](https://dl.acm.org/doi/10.1145/3418896)
2. Papadakis G, Skoutas D, Thanos E, Palpanas T. Blocking and Filtering Techniques for Entity Resolution: A Survey. *ACM Computing Surveys* 53(2), Art. 31, 2020. [https://arxiv.org/abs/1905.06167](https://arxiv.org/abs/1905.06167)
3. Burkard R, Dell'Amico M, Martello S. *Assignment Problems*, Revised Reprint. SIAM, 2012. [https://epubs.siam.org/doi/book/10.1137/1.9781611972238](https://epubs.siam.org/doi/book/10.1137/1.9781611972238)
4. Fellegi IP, Sunter AB. A Theory for Record Linkage. *Journal of the American Statistical Association* 64(328):1183-1210, 1969. [https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501049](https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501049)
5. Crouse DF. On Implementing 2D Rectangular Assignment Algorithms. *IEEE Trans. Aerospace and Electronic Systems* 52(4):1679-1696, 2016. [https://doi.org/10.1109/TAES.2016.140952](https://doi.org/10.1109/TAES.2016.140952)
6. Jonker R, Volgenant A. A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems. *Computing* 38(4):325-340, 1987. [https://link.springer.com/article/10.1007/BF02278710](https://link.springer.com/article/10.1007/BF02278710)
7. Gale D, Shapley LS. College Admissions and the Stability of Marriage. *American Mathematical Monthly* 69(1):9-15, 1962. [https://doi.org/10.1080/00029890.1962.11989827](https://doi.org/10.1080/00029890.1962.11989827)
8. Murty KG. An Algorithm for Ranking All the Assignments in Order of Increasing Cost. *Operations Research* 16(3):682-687, 1968. [https://pubsonline.informs.org/doi/abs/10.1287/opre.16.3.682](https://pubsonline.informs.org/doi/abs/10.1287/opre.16.3.682)
9. Cuturi M. Sinkhorn Distances: Lightspeed Computation of Optimal Transport. *NeurIPS* 26, 2013. [https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)
10. Flamary R, Courty N, et al. POT: Python Optimal Transport. *JMLR* 22(78):1-8, 2021. [https://jmlr.org/papers/v22/20-451.html](https://jmlr.org/papers/v22/20-451.html)
11. Elmagarmid AK, Ipeirotis PG, Verykios VS. Duplicate Record Detection: A Survey. *IEEE TKDE* 19(1):1-16, 2007. [https://ieeexplore.ieee.org/document/4016511](https://ieeexplore.ieee.org/document/4016511)
12. Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP* 2019. [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
13. Wang L, Yang N, Huang X, Jiao B, Yang L, Jiang D, Majumder R, Wei F. Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5). arXiv:2212.03533, 2022. [https://arxiv.org/abs/2212.03533](https://arxiv.org/abs/2212.03533)
14. Chen J, Xiao S, Zhang P, Luo K, Lian D, Liu Z. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings. arXiv:2402.03216, 2024. [https://arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216)
15. Peeters R, Steiner A, Bizer C. Entity Matching using Large Language Models. *EDBT* 2025; arXiv:2310.11244. [https://arxiv.org/abs/2310.11244](https://arxiv.org/abs/2310.11244)
16. Li Y, Li J, Suhara Y, Doan A, Tan W-C. Deep Entity Matching with Pre-Trained Language Models (Ditto). *PVLDB* 14(1):50-60, 2020. [https://arxiv.org/abs/2004.00584](https://arxiv.org/abs/2004.00584)
17. Thirumuruganathan S, et al. Deep Learning for Blocking in Entity Matching: A Design Space Exploration (DeepBlocker). *PVLDB* 14(11), 2021. [https://dl.acm.org/doi/10.14778/3476249.3476294](https://dl.acm.org/doi/10.14778/3476249.3476294)
18. Malkov YuA, Yashunin DA. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (HNSW). *IEEE TPAMI* 42(4), 2020. [https://arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
19. Cuturi M, Blondel M. Soft-DTW: a Differentiable Loss Function for Time-Series. *ICML* 2017 (PMLR 70:894-903). [https://arxiv.org/abs/1703.01541](https://arxiv.org/abs/1703.01541)
20. Wu R, Keogh EJ. FastDTW is Approximate and Generally Slower than the Algorithm it Approximates. *IEEE Trans. Knowledge and Data Engineering* 34(8), 2022. [https://arxiv.org/abs/2003.11246](https://arxiv.org/abs/2003.11246)
21. Riesen K, Bunke H. Approximate Graph Edit Distance Computation by Means of Bipartite Graph Matching. *Image and Vision Computing* 27(7):950-959, 2009. [https://doi.org/10.1016/j.imavis.2008.04.004](https://doi.org/10.1016/j.imavis.2008.04.004)
22. Wang R, Guo Z, Pan W, et al. Pygmtools: A Python Graph Matching Toolkit. *JMLR* 25(33):1-7, 2024. [https://jmlr.org/papers/v25/23-0572.html](https://jmlr.org/papers/v25/23-0572.html)
23. Singh R, Xu J, Berger B. Global Alignment of Multiple Protein Interaction Networks (IsoRank). *PNAS* 105(35):12763-12768, 2008. [https://www.pnas.org/doi/10.1073/pnas.0806627105](https://www.pnas.org/doi/10.1073/pnas.0806627105)
24. Bhattacharya I, Getoor L. Collective Entity Resolution in Relational Data. *ACM TKDD* 1(1), Art. 5, 2007. [https://dl.acm.org/doi/10.1145/1217299.1217304](https://dl.acm.org/doi/10.1145/1217299.1217304)
25. Benjelloun O, Garcia-Molina H, Menestrina D, Su Q, Whang SE, Widom J. Swoosh: a generic approach to entity resolution (ICAR). *The VLDB Journal* 18(1):255-276, 2009. [https://link.springer.com/article/10.1007/s00778-008-0098-x](https://link.springer.com/article/10.1007/s00778-008-0098-x)
26. Steorts RC, Hall R, Fienberg SE. A Bayesian Approach to Graphical Record Linkage and Deduplication. *JASA* 111(516):1660-1672, 2016. [https://arxiv.org/abs/1312.4645](https://arxiv.org/abs/1312.4645)
27. Sadinle M. Bayesian Estimation of Bipartite Matchings for Record Linkage. *JASA* 112(518):600-612, 2017. [https://www.tandfonline.com/doi/abs/10.1080/01621459.2016.1148612](https://www.tandfonline.com/doi/abs/10.1080/01621459.2016.1148612)
28. Gruenheid A, Dong XL, Srivastava D. Incremental Record Linkage. *PVLDB* 7(9):697-708, 2014. [http://www.vldb.org/pvldb/vol7/p697-gruenheid.pdf](http://www.vldb.org/pvldb/vol7/p697-gruenheid.pdf)
29. Koutras C, et al. Valentine: Evaluating Matching Techniques for Dataset Discovery. *IEEE ICDE* 2021. [https://ieeexplore.ieee.org/document/9458921](https://ieeexplore.ieee.org/document/9458921)
30. Wang T, et al. Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching (ComEM). *COLING* 2025; arXiv:2405.16884. [https://arxiv.org/abs/2405.16884](https://arxiv.org/abs/2405.16884)

---

*Cross-links:* [`00-taxonomy-and-terminology.md`](00-taxonomy-and-terminology.md) ·
[`02-blocking-and-scalable-candidate-generation.md`](02-blocking-and-scalable-candidate-generation.md) ·
[`03-assignment-and-graph-matching.md`](03-assignment-and-graph-matching.md) ·
[`05-comparison-and-similarity-functions.md`](05-comparison-and-similarity-functions.md) ·
[`07-schema-and-ontology-matching.md`](07-schema-and-ontology-matching.md) ·
[`08-interactive-active-learning-and-hitl.md`](08-interactive-active-learning-and-hitl.md) ·
[`09-python-ecosystem-landscape.md`](09-python-ecosystem-landscape.md) ·
[`10-design-implications-for-equate.md`](10-design-implications-for-equate.md) ·
[`12-sequence-and-graph-structure-matching.md`](12-sequence-and-graph-structure-matching.md) ·
[`13-llm-and-modern-embedding-matching.md`](13-llm-and-modern-embedding-matching.md) ·
[`14-evaluation-benchmarks-and-methodology.md`](14-evaluation-benchmarks-and-methodology.md) ·
[`15-collective-incremental-and-bayesian-er.md`](15-collective-incremental-and-bayesian-er.md) ·
[`17-vector-databases-and-scale-out.md`](17-vector-databases-and-scale-out.md)
