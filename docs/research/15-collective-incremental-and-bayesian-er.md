# Collective, Incremental & Bayesian Entity Resolution: Does the Matcher Have State?

*Research note for the `equate` redesign — a round-2 gap-filling doc in
`docs/research/`. Builds on the round-1 corpus (`00`–`10`); read
[`00-taxonomy-and-terminology.md`](00-taxonomy-and-terminology.md) §3.6 (pairwise
vs collective), §3.8 (batch vs incremental) and §4.2 (transitive closure) first.*

## Abstract

Round-1 treated matching as a *stateless batch function* over two collections:
`match(A, B) → pairs`. Three mature research threads challenge that framing and
force an architectural decision. **Collective / relational ER** makes match
decisions *interdependent* — resolving one pair changes the evidence for its
graph neighbours, so the matcher reasons over a whole entity graph at once
(Markov logic, probabilistic soft logic, Swoosh's merge-domination). **Bayesian
ER** replaces a single answer with a *posterior distribution over partitions*,
turning the output contract from "the matching" into "a sample of matchings with
calibrated uncertainty" (blink, d-blink, bipartite/partition priors).
**Incremental / streaming / temporal ER** keeps a *live* resolution that updates
as records arrive, drift, or are corrected — an *updatable object*, not a pure
function. This note surveys the algorithms, situates them, and argues that
`equate` should keep the stateless batch core as the default while defining two
clean extension seams: an optional **probabilistic/collective output contract**
and an optional **stateful `Resolver` facade** — both reusing the same
featurize → compare → match stages, both behind optional dependencies.

---

## 1. The question this doc exists to answer

The round-1 architecture ([`10-design-implications-for-equate.md`](10-design-implications-for-equate.md))
settled on a pure function whose signature is essentially
`match(A, B, *, featurize, compare, matcher) → Iterable[pair]`. That is correct
for the dominant case. But three literatures each answer "no, not always" to one
of three latent assumptions baked into that signature:

| Baked-in assumption | Challenged by | What breaks it |
|---|---|---|
| Pairs are scored **independently** | **Collective / relational ER** | resolving `a≈b` is evidence that `a`'s co-authors ≈ `b`'s co-authors; scores are coupled through a graph [1,2,3] |
| The output is **one** matching (or one partition) | **Bayesian ER** | the honest answer is a *posterior over partitions* — many plausible matchings with probabilities [7,8,9] |
| The inputs are **two fixed collections resolved once** | **Incremental / streaming / multi-source ER** | records stream in, attributes drift, and there may be `k > 2` sources to co-resolve [13,14,15] |

Each challenge maps to a concrete API question for `equate`:

1. **Is scoring local or global?** → does the *match* stage get to see a graph /
   the other decisions, or only a similarity matrix?
2. **Is the output a point estimate or a distribution?** → does the return type
   carry uncertainty (marginal link probabilities, posterior samples)?
3. **Is the matcher a function or an object?** → is there a stateful `Resolver`
   with `add` / `update` / `remove`, or only a batch call?

The rest of this note answers each from the literature, then §7 converts the
answers into extension points. The thesis, stated up front: **keep the stateless
batch function as the zero-config default; add statefulness and uncertainty as
*opt-in* facades over the same three stages, never as a tax on the common case.**

---

## 2. Collective & relational entity resolution

### 2.1 The core idea: decisions propagate through a graph

**Pairwise ER** scores each candidate pair in isolation and thresholds it.
**Collective ER** (a.k.a. *relational* or *joint* ER) recognises that records live
in a graph of relationships — papers cite papers, authors co-author, addresses
share households — and that a match decision on one pair is *evidence* for
matches on related pairs [1]. Resolving "*J. Smith*" and "*John Smith*" to one
author raises the probability that their respective co-authors are also the same
people, which in turn feeds back. The decisions are **mutually reinforcing**, so
they must be inferred *together* rather than one at a time.

This is the concrete instantiation of the taxonomy's "global/collective" axis
([`00`](00-taxonomy-and-terminology.md) §3.6). It is strictly more powerful than
pairwise ER on relational data — and strictly more expensive, because you can no
longer decompose the problem into independent pair decisions.

### 2.2 Bhattacharya & Getoor: relational clustering (RC-ER)

The canonical, algorithm-first treatment is Bhattacharya & Getoor, *"Collective
Entity Resolution in Relational Data"* (ACM TKDD, 2007) [1]. Their **relational
clustering (RC-ER)** algorithm is a **greedy agglomerative clustering** driven by
a combined similarity:

```
sim(c_i, c_j) = (1 − α)·sim_attr(c_i, c_j)  +  α·sim_rel(c_i, c_j)
```

- `sim_attr` — ordinary attribute similarity (string/token, per
  [`05`](05-comparison-and-similarity-functions.md)).
- `sim_rel` — **neighbourhood similarity**: how much the *cluster neighbourhoods*
  of `c_i` and `c_j` overlap once you account for already-merged clusters. They
  study several set-overlap measures over hyper-edges — common neighbours,
  **Jaccard**, and the **Adamic/Adar** coefficient (down-weighting
  high-degree "hub" neighbours).

The algorithm keeps a **priority queue** of candidate cluster-pair merges. When it
merges the top pair, only the similarities of pairs *sharing a neighbourhood* with
the merged cluster need re-computation — so relational evidence propagates locally
and incrementally rather than triggering a global recompute. Blocking bootstraps
the initial candidate pairs, keeping the whole procedure sub-quadratic in
practice. The key architectural lesson: **collective ER is agglomerative
clustering whose edge weights are recomputed as clusters form** — the matcher is
*stateful within a single resolution run*.

Related lineage: **query-time / on-demand** collective ER resolves only the
neighbourhood relevant to a query rather than the whole graph (Bhattacharya &
Getoor, KDD 2006) — the collective analogue of lazy evaluation.

### 2.3 Markov Logic Networks (Singla & Domingos)

Singla & Domingos, *"Entity Resolution with Markov Logic"* (ICDM 2006) [2], cast
collective ER declaratively. A **Markov Logic Network (MLN)** attaches a real
**weight** to each first-order-logic formula; the formulas are *templates* that,
grounded over the data, define a Markov random field whose nodes are the truth
values of atoms like `SameEntity(x, y)`. Rules such as

```
SameEntity(a1, a2) ∧ SameEntity(v1, v2)  ⇒  SameEntity(p1, p2)     (weighted)
```

encode transitivity and relational propagation ("same authors + same venue ⇒
likely same paper"). MAP inference over the ground network yields a globally
consistent set of match decisions; the framework subsumes earlier collective
approaches (McCallum & Wellner's CRF coreference). MLNs give **expressiveness and
principled joint inference** at the cost of grounding blow-up and inference that
is generally intractable without approximation. Open-source home: the *Alchemy*
system; modern Python-adjacent options exist but are heavyweight.

### 2.4 Probabilistic Soft Logic & HL-MRFs (Bach, Getoor et al.)

**Probabilistic Soft Logic (PSL)** [3,5] is the scalable, continuous-valued
cousin of MLNs and is *explicitly motivated by ER*. Its logical rules compile to a
**Hinge-Loss Markov Random Field (HL-MRF)**, in which every atom takes a **soft
truth value in [0,1]** and the rules become **hinge-loss potentials**. The
decisive property: **MAP inference is a *convex* optimization** (solved by a
consensus ADMM message-passing algorithm), so inference scales to millions of
ground atoms — orders of magnitude beyond discrete MLN inference [3]. Because ER
similarities are naturally continuous (`sim ∈ [0,1]`), PSL models them directly
rather than thresholding early. Pujara, Miao, Getoor & Cohen's **Knowledge Graph
Identification** [4] uses PSL to jointly perform entity resolution, link
prediction, and ontology-constraint enforcement over noisy extractions at scale —
a flagship demonstration that collective ER, collective classification, and link
prediction are *one joint inference problem* when expressed in PSL. PSL ships as an
Apache-licensed Java/Groovy framework [5].

**Takeaway for a matcher framework:** collective ER wants (a) *relational
features* (edges between records, not just attribute vectors), and (b) a *joint
inference* stage that reads the whole candidate graph. Both are heavyweight and
domain-specific — clearly an **optional strategy behind an optional dependency**,
not part of the default path.

### 2.5 Swoosh & generic ER: the algebra of match + merge (ICAR)

Benjelloun, Garcia-Molina, Menestrina, Su, Whang & Widom, *"Swoosh: a generic
approach to entity resolution"* (VLDB Journal 18(1), 2009) [6], take a different,
*merge-centric* view that is deeply relevant to `equate`'s API design. They treat
**`match(r1, r2) → bool`** and **`merge(r1, r2) → r`** as opaque **black-box
functions** and ask: *what algebraic properties must they satisfy for ER to be
efficient and order-independent?* The answer is four properties, the **ICAR**
family:

| Property | Meaning (on the black-box match/merge) |
|---|---|
| **I — Idempotence** | `r` matches `r`, and `merge(r, r) = r`. |
| **C — Commutativity** | `r1` matches `r2` ⇔ `r2` matches `r1`; and `merge(r1, r2) = merge(r2, r1)`. |
| **A — Associativity** | `merge(r1, merge(r2, r3)) = merge(merge(r1, r2), r3)`. |
| **R — Representativity** | if `r3 = merge(r1, r2)` then anything that matched `r1` also matches `r3` (the merged record *represents* its parts). |

When ICAR holds, ER results are **independent of the order** records are processed
and the resolution is finite; this licenses the efficient **R-Swoosh** and
**F-Swoosh** algorithms (F-Swoosh additionally caches per-feature comparisons to
avoid redundant `match` calls). When ICAR fails, you must fall back to the
brute-force **G-Swoosh**, which considers all merge orders — dramatically more
expensive. The lesson generalises far beyond Swoosh: **order-independence is a
property of your match/merge functions, and it is exactly the property that makes
*incremental* and *distributed* resolution correct.** If `equate` ever exposes a
merge-based dedup mode, ICAR is the contract to document as the "fast-path"
guarantee (and D-Swoosh extends the family to the distributed setting).

---

## 3. Bayesian entity resolution: a posterior over partitions

### 3.1 Why Bayesian — the output contract changes

Everything above returns *a* matching. Bayesian ER argues that with noisy,
ambiguous data the honest deliverable is a **probability distribution over the
space of partitions** (clusterings) of the records — because many partitions are
plausible and downstream analyses should propagate that uncertainty rather than
condition on one guessed linkage [7,8]. Concretely, a Bayesian resolver returns
**posterior MCMC samples of the linkage structure**, from which you can read:

- **marginal match probabilities** per pair (`P(a ≈ b | data)`),
- the **posterior distribution of the number of entities**,
- **credible sets** of whole partitions,
- and, crucially, **error propagation** into any downstream regression /
  capture-recapture estimate (linkage uncertainty becomes part of the model).

This is the strongest possible statement of the taxonomy's "scored vs boolean"
axis: not a score, but a *calibrated posterior*.

### 3.2 The generative "linkage-structure" model (blink, graphical linkage)

Steorts, Hall & Fienberg, *"A Bayesian Approach to Graphical Record Linkage and
Deduplication"* (JASA 2016) [8], introduce the representation the whole modern
Bayesian-ER line rests on: **records link to *latent true entities*, not directly
to each other.** The pattern of links is a **bipartite graph** between observed
records and a latent population of entities; each entity has true field values,
and each record is a *distorted copy* of its entity's values. This latent-entity
formulation:

- handles **arbitrarily many files at once** and **duplicates within files**
  uniformly (multi-source is free — see §4),
- makes the *partition* (which records share an entity) the object of inference,
- guarantees **transitivity by construction** (records grouped via a shared
  latent entity), avoiding the fragile transitive-closure step
  ([`00`](00-taxonomy-and-terminology.md) §4.2).

Steorts, *"Entity Resolution with Empirically Motivated Priors"* (Bayesian
Analysis 2015) [7], packages this as **blink**: an **empirical-Bayes** prior takes
the empirical distribution of the observed data as the prior for latent entity
values, sidestepping hand-tuned priors, and a **string pseudo-likelihood** models
distortion of string fields. blink ships as an R package [7]. Later work replaces
the uniform partition prior with **exchangeable random partition priors** (e.g.
Ewens–Pitman / microclustering families) so that cluster sizes stay small as the
population grows — the *microclustering* property that ordinary clustering priors
violate [19].

### 3.3 Bipartite / partition priors (Sadinle)

Sadinle, *"Bayesian Estimation of Bipartite Matchings for Record Linkage"* (JASA
2017) [10], gives the **two-file, duplicate-free** special case a clean Bayesian
treatment: a **prior on the bipartite matching** enforces the one-to-one linkage
constraint (each record in file A matches ≤ 1 record in file B), and a chosen
**loss function** turns the posterior into a point estimate — a *Bayes estimate of
the linkage* rather than an arbitrary threshold. This is the Bayesian counterpart
of the assignment problem in [`03`](03-assignment-and-graph-matching.md): the
one-to-one constraint lives in the *prior*, and the reject option (no match) is
first-class. Shipped as the **BRL** ("Beta Record Linkage") R package [10].

### 3.4 d-blink: making Bayesian ER scale

The historic objection to Bayesian ER is that MCMC over partitions is slow and
quadratic. Marchant, Kaplan, Elazar, Rubinstein & Steorts, *"d-blink: Distributed
End-to-End Bayesian Entity Resolution"* (JCGS 2021) [9], attack this **without
compromising posterior correctness** via four ideas:

1. an **auxiliary-variable representation** that *induces* a partition of entities
   and records into **blocks** — so blocking becomes *part of the generative
   model* rather than a lossy pre-filter (jointly performing blocking and ER);
2. **well-balanced blocks via k-d trees** for even distributed load;
3. a **distributed partially-collapsed Gibbs sampler** with improved mixing; and
4. **fast Gibbs-update algorithms** per block.

The payoff: a reported **~200× speed-up**, scaling to **> 1 million records**,
validated on six datasets including a 2010 U.S. Decennial Census case study [9].
The implementation, **dblink**, is an **Apache Spark / Scala** package that emits
posterior clustering samples as Parquet [9]. The architectural point for `equate`:
**blocking and matching need not be separate stages** — d-blink folds them into
one model — but that is a research-grade move; the default should keep them as the
independent seams of round-1.

### 3.5 The Bayesian output contract, distilled

A Bayesian resolver's return type is **not** `Iterable[(i, j)]`. It is closer to:

```python
@dataclass
class PartitionPosterior:
    samples: Sequence[Partition]          # MCMC draws over clusterings
    def link_probability(self, i, j) -> float: ...   # marginal P(i ≈ j)
    def n_entities_posterior(self) -> Distribution: ...
    def point_estimate(self, *, loss='B-cubed') -> Partition: ...
```

`equate` cannot make this the *default* return (it needs `pymc`/`numpyro`-class
machinery and minutes-to-hours of compute), but it **can** define the *interface*
so a Bayesian backend is a drop-in strategy, and so even cheap matchers can
optionally emit **marginal link probabilities** as a lightweight down-shadow of
the same contract.

---

## 4. Multi-source (n-way, k > 2 collections) linkage

Round-1's `match(A, B)` is two-collection. Real linkage is often **k-file**: link
a survey, an administrative register, and a health record — with **duplicates
within each** — simultaneously. Two families handle `k > 2`:

- **Generalized Fellegi–Sunter.** Sadinle & Fienberg, *"A Generalized
  Fellegi–Sunter Framework for Multiple Record Linkage"* (JASA 2013) [11], extend
  the classical two-file comparison-vector model to **k files**, resolving
  matches across all files jointly (motivated by de-duplicating homicide record
  systems). Frequentist Fellegi–Sunter extends to `k > 2` but its complexity
  grows combinatorially and it degrades for moderately large `k`.
- **Partition / latent-entity models.** The graphical Bayesian model of §3.2 is
  *natively* multi-file: it links every record to a latent entity regardless of
  source, so "3 files" vs "2 files" is not a special case at all [8].
  Aleshin-Guendel & Sadinle, *"Multifile Partitioning for Record Linkage and
  Duplicate Detection"* (JASA 2023) [12], give a **structured partition prior**
  over *all* records from *all* files at once, encoding prior knowledge about the
  data-collection processes (e.g. within-file dedup rates).

**Design consequence:** the **partition of a pooled record set** — records →
latent entities — is the natural **single-source-of-truth representation** for
n-way matching, subsuming pairwise links, 1:1 assignment, and within/across-file
dedup. `equate`'s clustering-output type should be *this partition*, and the API
should generalise cleanly from `match(A, B)` to `resolve(*collections)`.

---

## 5. Incremental, streaming & temporal ER

### 5.1 Incremental record linkage (the affected-region principle)

When data updates arrive at high velocity, re-running batch ER from scratch is
wasteful and quickly stale. Gruenheid, Dong & Srivastava, *"Incremental Record
Linkage"* (PVLDB 2014) [14], give the canonical end-to-end framework: on an
update (insert / delete / change), **re-resolve only the affected clusters** and
their neighbourhood, not the whole dataset. Critically, they let new evidence
**retroactively fix previous linkage errors** (a merge can later be split, a split
re-merged) rather than treating past decisions as frozen. This is the operational
version of the taxonomy's batch-vs-incremental axis
([`00`](00-taxonomy-and-terminology.md) §3.8): the unit of work is the **connected
component touched by the update**. Union-find with support for *splits* (not just
merges) is the enabling data structure.

### 5.2 Progressive / pay-as-you-go ER (best partial answer under a budget)

A related "time" axis is **progressive** or **pay-as-you-go** ER: emit the *best
possible partial resolution within a compute/latency budget*, maximising matches
found early (pioneered by Whang et al.; surveyed in [15]). Recent systems make it
online: Simonini et al., *"Entity Resolution On-Demand"* (PVLDB 2022) [16], and
Gazzarri & Herschel, *"Progressive Entity Resolution over Incremental Data"* (EDBT
2023) [17], schedule comparisons so that early output has high recall. This
matters for `equate`'s **interactive** story
([`08`](08-interactive-active-learning-and-hitl.md)): a `match` that can *yield*
its best pairs first and refine on demand is more useful in a UI than one that
blocks until globally optimal.

### 5.3 Streaming *Bayesian* linkage

The two threads meet in Taylor, Kaplan & Betancourt, *"Fast Bayesian Record
Linkage for Streaming Data Contexts"* (JCGS 2024) [13]: files arrive
**sequentially**, and rather than refit the full joint posterior on every arrival,
they **update** link estimates incrementally, achieving *near-equivalent posterior
inference at a fraction of the compute* of the batch Gibbs sampler. This is the
Bayesian analogue of §5.1 — the *posterior* is the state that gets updated — and
it is exactly why a stateful `Resolver` object (not a pure function) is the right
abstraction for the streaming case.

### 5.4 Temporal ER: entities whose attributes drift

A distinct wrinkle: entities **change over time** — people move, companies
rename, products revise. **Temporal ER** models attribute *evolution* so that two
records that look dissimilar *now* can still be the same entity observed at
different times (and, conversely, so that a shared value across a long gap is
weaker evidence). This needs **time-aware comparison** (decay/transition models on
attributes) and is surveyed within the big-data ER overview [15]. For `equate` it
reinforces that the **comparator may need side-channel context** (timestamps),
not just the two representations — an argument for keeping the comparator
signature extensible.

---

## 6. Synthesis: three axes, one architectural fork

The three literatures collapse into three orthogonal *optional* capabilities, each
of which can be layered onto the round-1 core without disturbing the default:

| Capability | Changes the… | Cost / dependency | Default? |
|---|---|---|---|
| **Collective / relational** | *match* stage (reads a graph, joint inference) | MLN/PSL/relational-clustering; heavy | opt-in strategy |
| **Bayesian / probabilistic** | *output contract* (posterior over partitions) | MCMC (`numpyro`/Spark); minutes–hours | opt-in backend + interface |
| **Incremental / streaming** | *lifecycle* (function → stateful object) | modest; union-find with split | opt-in facade |

Crucially these are **composable and independent**: you can have collective-but-
batch (RC-ER), Bayesian-but-batch (blink), incremental-but-pairwise (Gruenheid),
or all three (streaming d-blink-style). So `equate` should **not** pick one
"advanced mode"; it should expose *three separate seams* and let them combine.

---

## 7. Design implications for `equate`

Extends [`10-design-implications-for-equate.md`](10-design-implications-for-equate.md).
Guiding principle unchanged: **`match(A, B)` Just Works, stateless, zero-config;
statefulness and uncertainty are opt-in and never taxed onto the common case.**

### 7.1 Keep the core a pure batch function; add a `Resolver` *facade* for state

Do **not** make the core matcher stateful. Instead, define a thin **stateful
facade** that composes the *same* featurize → compare → match/cluster stages:

```python
# Stateless default (round-1) — unchanged, the 95% case
match(A, B) -> Iterable[pair]

# Opt-in stateful facade for incremental / streaming
class Resolver:                       # wraps the same 3 stages + a live partition
    def add(self, records) -> None: ...        # insert; re-resolve affected component only
    def update(self, id, record) -> None: ...  # attribute drift / correction
    def remove(self, id) -> None: ...
    def resolution(self) -> Partition: ...      # current best clustering
```

`Resolver` is where union-find-with-split, affected-region re-resolution [14], and
streaming posterior updates [13] live. It **reuses** the stage protocols — it is an
orchestration wrapper, not a parallel implementation (SSOT). Batch `match` can be
implemented as `Resolver().add(A); .add(B); .resolution()` internally, or kept
separate for speed; either way the *stages* are shared.

### 7.2 Make the output contract *uncertainty-capable* but not uncertainty-*requiring*

Define a small hierarchy of return types so uncertainty is *expressible* at every
tier without being mandatory:

- **Tier 0 (default):** `Iterable[(i, j)]` or `(i, j, score)` — today's contract.
- **Tier 1 (cheap uncertainty):** attach a **marginal link probability** per pair
  where the matcher can produce one (calibrated Fellegi–Sunter posterior, logistic
  score → probability). A pure down-shadow of the Bayesian contract; no heavy deps.
- **Tier 2 (full Bayesian):** a `PartitionPosterior` (§3.5) with MCMC samples,
  `link_probability`, `n_entities_posterior`, and a loss-aware `point_estimate`.
  Behind an optional `[bayes]` extra (`numpyro`/`pymc`, or a Spark/`dblink`
  bridge).

The extension point is: **the match stage may return richer objects that are still
consumable as plain pairs** (iterate → yields `(i, j)` for the MAP/point estimate).
This lets a UI ask "how sure are you?" without forcing every caller to care.

### 7.3 Collective matching is a *match-stage strategy* that consumes a graph

Round-1's `Matcher` sees only a similarity matrix. Add an **optional relational
seam**: a collective matcher additionally receives **inter-record edges**
(relationships) and may perform **joint inference**:

```python
CollectiveMatcher = Callable[[SimilarityMatrix, RelationalGraph], Partition]
```

- Default matchers ignore the graph (pass `None`) — no change.
- A relational-clustering strategy (RC-ER-style [1]) recomputes edge weights as
  clusters form.
- MLN/PSL backends [2,3,4] are **optional-dependency strategies** (`[collective]`
  extra: `pslpython`/`pgmpy`), never imported by default.

Keep the *featurizer* extensible to emit **relational features** (neighbourhood
overlap, shared-edge sets) — same protocol, richer representation.

### 7.4 Global consistency & transitivity stay in the match/cluster stage

Reinforces [`00`](00-taxonomy-and-terminology.md) §4.2: never hard-wire transitive
closure into scoring. The latent-entity/partition model [8] gives transitivity *by
construction*; correlation clustering gives it *robustly*; connected-components
gives it *cheaply-but-fragile*. All three are **swappable cluster strategies** with
the same output type — a **`Partition`** (records → entity ids).

### 7.5 Adopt the *partition* as the SSOT output; generalise to n-way

Make `Partition` (a labelling of a pooled record set into latent entities) the
**canonical clustering output** (§4). It subsumes:
- pairwise links (edges within a cluster),
- 1:1 assignment (bipartite, one record per file per cluster — Sadinle's
  constraint [10]),
- within- and across-file dedup uniformly [8,12].

Then generalise the surface API from `match(A, B)` to a variadic
`resolve(*collections)` (or `match(A, B, *more)`), with the two-collection
bipartite case as the optimised special path. Multi-source is then *not* a bolt-on.

### 7.6 If you offer merge-based dedup, expose match+merge and document ICAR

For a canonicalisation / golden-record mode
([`01`](01-entity-resolution-record-linkage.md)), expose **`match`** and
**`merge`** as injectable black boxes (Swoosh style [6]). Document the **ICAR**
properties (§2.5) as the contract that unlocks **order-independent, incremental,
distributable** resolution — and warn that a non-ICAR merge forces the expensive
all-orders path. ICAR is also the *correctness precondition* for §7.1's incremental
`Resolver`: order-independence is exactly what makes affected-region re-resolution
sound.

### 7.7 Optional-dependency boundaries (the strategy/extra map)

| Extra | Pulls in | Enables |
|---|---|---|
| *(none, default)* | `numpy`, `scipy`, `scikit-learn` | batch pairwise + assignment + connected-components |
| `[collective]` | `pslpython` / `pgmpy` / graph libs | MLN/PSL/relational-clustering joint inference |
| `[bayes]` | `numpyro` / `pymc` (or Spark `dblink` bridge) | posterior over partitions, marginal probabilities |
| `[incremental]` | *(lightweight)* union-find-with-split | `Resolver` streaming/incremental facade |

Every heavy capability is **lazy-imported behind an extra**; the default install
stays `numpy`/`scipy`-light, per round-1.

### 7.8 What *not* to do

- Don't make the default matcher stateful or Bayesian — the common case is batch,
  pairwise, point-estimate, and must stay a one-line pure function.
- Don't fuse blocking into matching by default (d-blink does, but that's a modelling
  commitment); keep them independent seams and let a research backend fuse them.
- Don't force a `Partition` return where a caller wants raw scored pairs — offer
  both; make the richer type *iterate as* the simpler one.

---

## 8. Glossary

| Term | Definition |
|---|---|
| **Collective / relational / joint ER** | Resolving co-occurring references *together*, propagating match evidence through a graph of relationships, rather than scoring pairs independently [1]. |
| **Relational clustering (RC-ER)** | Greedy agglomerative clustering whose edge weight blends attribute similarity with *neighbourhood* similarity, recomputed as clusters merge [1]. |
| **Markov Logic Network (MLN)** | Weighted first-order formulas that template a Markov random field; supports joint MAP inference over match atoms; expressive but inference-hard [2]. |
| **Probabilistic Soft Logic (PSL)** | Logic rules compiled to a Hinge-Loss MRF with soft [0,1] truth values; MAP inference is *convex* and scales to millions of atoms; motivated by ER [3,5]. |
| **HL-MRF** | Hinge-Loss Markov Random Field — the continuous graphical model underlying PSL [3]. |
| **Swoosh** | Generic ER framework treating `match`/`merge` as black boxes; R-/F-/G-Swoosh algorithms [6]. |
| **ICAR** | Idempotence, Commutativity, Associativity, Representativity — properties of match/merge that make ER order-independent and efficient [6]. |
| **Bayesian ER** | Inferring a *posterior distribution over partitions* of records into entities, yielding calibrated uncertainty rather than a single matching [7,8]. |
| **Linkage structure / graphical record linkage** | Generative model linking each record to a *latent true entity*; the partition is the inference target; transitivity is automatic [8]. |
| **blink** | Empirically-motivated-prior Bayesian ER model + R package (Steorts 2015) [7]. |
| **d-blink** | Distributed, end-to-end Bayesian ER that folds blocking into the generative model; ~200× speed-up, > 1M records; Spark/Scala [9]. |
| **Bipartite / partition prior** | A prior encoding the one-to-one (Sadinle [10]) or general k-file (Aleshin-Guendel & Sadinle [12]) matching constraint. |
| **Posterior partition uncertainty** | Marginal link probabilities, entity-count distribution, and credible partitions read off MCMC samples [7,8,9]. |
| **Multi-source / n-way linkage** | Co-resolving `k > 2` files (with within-file duplicates) jointly [8,11,12]. |
| **Incremental ER** | Updating an existing resolution on inserts/updates/deletes by re-resolving only affected clusters, allowing past errors to be fixed [14]. |
| **Streaming ER** | Incremental ER over an unbounded, high-velocity stream; a *stateful* resolver [13,15]. |
| **Progressive / pay-as-you-go ER** | Emitting the best partial resolution within a compute/latency budget, matches-early [15,16,17]. |
| **Temporal ER** | ER accounting for entity attributes that *drift* over time via time-aware comparison [15]. |
| **Partition** | A labelling of a pooled record set into latent-entity clusters; the SSOT output type subsuming links, 1:1 assignment, and dedup. |

---

## References

1. Bhattacharya I, Getoor L. Collective Entity Resolution in Relational Data. *ACM Transactions on Knowledge Discovery from Data (TKDD)* 1(1), Art. 5, 2007. [https://dl.acm.org/doi/10.1145/1217299.1217304](https://dl.acm.org/doi/10.1145/1217299.1217304) · PDF: [https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf)
2. Singla P, Domingos P. Entity Resolution with Markov Logic. *IEEE ICDM* 2006:572-582. [https://alchemy.cs.washington.edu/papers/singla06b/singla06b.pdf](https://alchemy.cs.washington.edu/papers/singla06b/singla06b.pdf)
3. Bach SH, Broecheler M, Huang B, Getoor L. Hinge-Loss Markov Random Fields and Probabilistic Soft Logic. *Journal of Machine Learning Research* 18(109):1-67, 2017. [https://jmlr.org/papers/v18/15-631.html](https://jmlr.org/papers/v18/15-631.html) · arXiv: [https://arxiv.org/abs/1505.04406](https://arxiv.org/abs/1505.04406)
4. Pujara J, Miao H, Getoor L, Cohen W. Knowledge Graph Identification. *International Semantic Web Conference (ISWC)* 2013:542-557. [https://link.springer.com/chapter/10.1007/978-3-642-41335-3_34](https://link.springer.com/chapter/10.1007/978-3-642-41335-3_34) · PDF: [https://linqs.org/assets/resources/pujara-slg13.pdf](https://linqs.org/assets/resources/pujara-slg13.pdf)
5. PSL — Probabilistic Soft Logic (LINQS). Project homepage & Apache-licensed Java implementation. [https://psl.linqs.org/](https://psl.linqs.org/) · Source: [https://github.com/linqs/psl](https://github.com/linqs/psl)
6. Benjelloun O, Garcia-Molina H, Menestrina D, Su Q, Whang SE, Widom J. Swoosh: a generic approach to entity resolution. *The VLDB Journal* 18(1):255-276, 2009. [https://link.springer.com/article/10.1007/s00778-008-0098-x](https://link.springer.com/article/10.1007/s00778-008-0098-x) · Keynote overview (PDF): [https://www.cl.cam.ac.uk/teaching/0910/ConcDistS/MW09-HectorKeynote.pdf](https://www.cl.cam.ac.uk/teaching/0910/ConcDistS/MW09-HectorKeynote.pdf)
7. Steorts RC. Entity Resolution with Empirically Motivated Priors. *Bayesian Analysis* 10(4):849-875, 2015. [https://projecteuclid.org/journals/bayesian-analysis/volume-10/issue-4/Entity-Resolution-with-Empirically-Motivated-Priors/10.1214/15-BA965SI.full](https://projecteuclid.org/journals/bayesian-analysis/volume-10/issue-4/Entity-Resolution-with-Empirically-Motivated-Priors/10.1214/15-BA965SI.full) · arXiv: [https://arxiv.org/abs/1409.0643](https://arxiv.org/abs/1409.0643) · `blink` R package: [https://github.com/cleanzr/blink](https://github.com/cleanzr/blink)
8. Steorts RC, Hall R, Fienberg SE. A Bayesian Approach to Graphical Record Linkage and Deduplication. *Journal of the American Statistical Association* 111(516):1660-1672, 2016. [https://www.tandfonline.com/doi/abs/10.1080/01621459.2015.1105807](https://www.tandfonline.com/doi/abs/10.1080/01621459.2015.1105807) · arXiv: [https://arxiv.org/abs/1312.4645](https://arxiv.org/abs/1312.4645)
9. Marchant NG, Kaplan A, Elazar DN, Rubinstein BIP, Steorts RC. d-blink: Distributed End-to-End Bayesian Entity Resolution. *Journal of Computational and Graphical Statistics* 30(2):406-421, 2021. [https://www.tandfonline.com/doi/abs/10.1080/10618600.2020.1825451](https://www.tandfonline.com/doi/abs/10.1080/10618600.2020.1825451) · arXiv: [https://arxiv.org/abs/1909.06039](https://arxiv.org/abs/1909.06039) · `dblink` Spark package: [https://github.com/cleanzr/dblink](https://github.com/cleanzr/dblink)
10. Sadinle M. Bayesian Estimation of Bipartite Matchings for Record Linkage. *Journal of the American Statistical Association* 112(518):600-612, 2017. [https://www.tandfonline.com/doi/abs/10.1080/01621459.2016.1148612](https://www.tandfonline.com/doi/abs/10.1080/01621459.2016.1148612) · `BRL` R package: [https://github.com/msadinle/BRL](https://github.com/msadinle/BRL)
11. Sadinle M, Fienberg SE. A Generalized Fellegi–Sunter Framework for Multiple Record Linkage with Application to Homicide Record Systems. *Journal of the American Statistical Association* 108(502):385-397, 2013. [https://arxiv.org/abs/1205.3217](https://arxiv.org/abs/1205.3217)
12. Aleshin-Guendel S, Sadinle M. Multifile Partitioning for Record Linkage and Duplicate Detection. *Journal of the American Statistical Association* 118(543):1786-1795, 2023. [https://arxiv.org/abs/2110.03839](https://arxiv.org/abs/2110.03839)
13. Taylor I, Kaplan A, Betancourt B. Fast Bayesian Record Linkage for Streaming Data Contexts. *Journal of Computational and Graphical Statistics* 33(3):833-844, 2024. [https://www.tandfonline.com/doi/abs/10.1080/10618600.2023.2283571](https://www.tandfonline.com/doi/abs/10.1080/10618600.2023.2283571) · arXiv: [https://arxiv.org/abs/2307.07005](https://arxiv.org/abs/2307.07005)
14. Gruenheid A, Dong XL, Srivastava D. Incremental Record Linkage. *Proceedings of the VLDB Endowment* 7(9):697-708, 2014. [https://dl.acm.org/doi/10.14778/2732939.2732943](https://dl.acm.org/doi/10.14778/2732939.2732943) · PDF: [http://www.vldb.org/pvldb/vol7/p697-gruenheid.pdf](http://www.vldb.org/pvldb/vol7/p697-gruenheid.pdf)
15. Christophides V, Efthymiou V, Palpanas T, Papadakis G, Stefanidis K. An Overview of End-to-End Entity Resolution for Big Data. *ACM Computing Surveys* 53(6), Art. 127, 2020 (incremental/streaming, temporal, progressive ER). [https://dl.acm.org/doi/10.1145/3418896](https://dl.acm.org/doi/10.1145/3418896) · arXiv: [https://arxiv.org/abs/1905.06397](https://arxiv.org/abs/1905.06397)
16. Simonini G, Papadakis G, Palpanas T, Bergamaschi S. Entity Resolution On-Demand. *Proceedings of the VLDB Endowment* 15(7):1506-1518, 2022. [https://www.vldb.org/pvldb/vol15/p1506-simonini.pdf](https://www.vldb.org/pvldb/vol15/p1506-simonini.pdf)
17. Gazzarri L, Herschel M. Progressive Entity Resolution over Incremental Data. *EDBT* 2023. [https://openproceedings.org/2023/conf/edbt/paper-180.pdf](https://openproceedings.org/2023/conf/edbt/paper-180.pdf)
18. Getoor L, Machanavajjhala A. Entity Resolution: Theory, Practice & Open Challenges (tutorial). *Proceedings of the VLDB Endowment* 5(12):2018-2019, 2012. [http://vldb.org/pvldb/vol5/p2018_lisegetoor_vldb2012.pdf](http://vldb.org/pvldb/vol5/p2018_lisegetoor_vldb2012.pdf)
19. Marchant NG, Rubinstein BIP, Steorts RC. Bayesian Graphical Entity Resolution Using Exchangeable Random Partition Priors. *Journal of Survey Statistics and Methodology* 11(3):569-596, 2023. [https://academic.oup.com/jssam/article/11/3/569/6969687](https://academic.oup.com/jssam/article/11/3/569/6969687) · arXiv: [https://arxiv.org/abs/2301.02962](https://arxiv.org/abs/2301.02962)

---

*Cross-links:* [`00-taxonomy-and-terminology.md`](00-taxonomy-and-terminology.md)
(collective vs pairwise §3.6, batch vs incremental §3.8, transitive closure §4.2) ·
[`01-entity-resolution-record-linkage.md`](01-entity-resolution-record-linkage.md)
(pipeline, clustering/canonicalization) ·
[`03-assignment-and-graph-matching.md`](03-assignment-and-graph-matching.md)
(bipartite 1:1 as the frequentist twin of Sadinle's partition prior) ·
[`08-interactive-active-learning-and-hitl.md`](08-interactive-active-learning-and-hitl.md)
(progressive output, human-in-the-loop) ·
[`09-python-ecosystem-landscape.md`](09-python-ecosystem-landscape.md)
(what to wrap vs reimplement; optional extras) ·
[`10-design-implications-for-equate.md`](10-design-implications-for-equate.md)
(the base architecture this note extends).
