# Data Fusion, Truth Discovery & Canonicalization: The Golden-Record Step

*Round-2 gap-filling facet for the `equate` redesign. Where the round-1 corpus
(`00`–`10`) covers how objects get **grouped** — blocking, comparison, assignment,
clustering, transitive closure — this doc covers what happens **after** a group
exists: collapsing the members of a resolved cluster into **one clean,
representative record**. It is the hand-off point where entity resolution
([`01`](01-entity-resolution-record-linkage.md)) meets Master Data Management, and
where three research traditions — database **data fusion**, data-mining **truth
discovery**, and industry **canonicalization / survivorship** — solve the same
problem under different names.*

## Abstract

Once matching has decided *which records co-refer*, a second, largely independent
problem remains: the co-referent records **disagree** on attribute values (three
sources give three birthdates), and a downstream consumer needs a single answer.
This document surveys the three literatures that address it — **data fusion**
(conflict-resolution strategies over an integrated relation), **truth discovery**
(jointly estimating source reliability and true values), and
**canonicalization / survivorship** (golden-record construction in MDM) — with
concrete algorithms (TruthFinder, Dawid–Skene, LTM, Accu, CRH, Knowledge-Based
Trust), their assumptions, complexity, and empirical track record. It then frames
this whole stage for `equate` as an **optional, pluggable post-resolution step**:
a `Canonicalizer` that consumes the clusters produced by
[`01`](01-entity-resolution-record-linkage.md)/[`03`](03-assignment-and-graph-matching.md)
and emits fused records, with fusion **policy** as an injectable strategy and the
heavier source-quality estimators behind an optional extra.

The one-line framing: **matching answers *"are these the same?"*; fusion answers
*"then what is the truth about the thing they describe?"* — and `equate` should
own the seam between them without forcing every user across it.**

---

## 1. Where this stage sits in the pipeline

The canonical ER pipeline ([`00`](00-taxonomy-and-terminology.md#2-the-canonical-pipeline),
[`01`](01-entity-resolution-record-linkage.md)) runs
`preprocess → block → compare → classify → cluster → **canonicalize**`. The first
five stages are the subject of round-1; this doc is the sixth. The reframing that
makes it a *distinct* stage and not an afterthought:

> Matching produces a **partition** of records into co-reference groups. Fusion
> produces, per group, a **single tuple of attribute values**. These are different
> problems with different failure modes: matching errs by splitting/merging
> entities; fusion errs by choosing a wrong value for a correctly-formed entity.

Three communities converge here, each with its own vocabulary and its own default
assumption about *what makes a value right* ([`00`](00-taxonomy-and-terminology.md#4-a-rosetta-stone-of-vocabularies)):

| Community | Name for the step | "Right value" is decided by | Key sources |
|---|---|---|---|
| **Databases / data integration** | **Data fusion** | a declared *resolution function* over the present values (vote, max, most-recent, preferred source) | Bleiholder & Naumann [1]; Dong et al. [2,3] |
| **Data mining / Web** | **Truth discovery** | *inferred source reliability* — trust the value the trustworthy sources assert | Li et al. survey [4]; TruthFinder [5]; LTM [7]; Accu [8]; CRH [9] |
| **Industry / MDM** | **Canonicalization / survivorship / golden record** | hand-authored *survivorship rules* at attribute granularity + governance | Loshin [14]; Binette & Steorts [15] |

The distinction from the *matching* clustering step is important and is stressed by
Binette & Steorts, who argue canonicalization is a **first-class step beyond
matching**, with its own tension between locally-good pairwise decisions and a
globally-coherent merged record [15]. This doc treats it as the natural extension
of the clustering/transitive-closure discussion in
[`01`](01-entity-resolution-record-linkage.md#24-clustering--canonicalization--from-pairs-to-entities)
and [`00`](00-taxonomy-and-terminology.md#5-the-intransitivity-problem).

### 1.1 Fusion vs. featurization/comparison — a different axis

Do not confuse **value fusion** (this doc) with the **combiner** that reduces a
per-field *comparison vector* into one match score
([`05`](05-comparison-and-similarity-functions.md)). The combiner fuses
*similarities* to make a *match decision*; the canonicalizer fuses *values* after
the match decision is already made. They share the "weighted combination of
per-field signals" shape but run at different pipeline positions with different
outputs (a score vs. a record).

---

## 2. Data Fusion: conflict-resolution strategies

The database view (Bleiholder & Naumann's ACM Computing Surveys survey [1], and
Dong, Berti-Équille & Srivastava's survey chapter [2]) treats fusion as the last
operator of data integration, after **schema mapping** (align the columns —
[`07`](07-schema-and-ontology-matching.md)) and **duplicate detection** (align the
rows — [`01`](01-entity-resolution-record-linkage.md)). What is left is a set of
rows known to describe the same object but disagreeing on values.

### 2.1 The three goals and the conflict taxonomy

Dong & Naumann frame data fusion's objective as producing data that is **complete**
(don't lose values), **concise** (one row per entity, one value per attribute), and
**consistent** (no contradictions) [3]. Achieving all three simultaneously is
impossible in general, so a fusion system picks trade-offs.

Conflicts are classified [1,2] into:

- **Uncertainty** — a value vs. a *null* (one source knows the birthdate, another
  is silent). Usually resolvable by "take the non-null value."
- **Contradiction** — two *non-null, different* values for the same attribute of
  the same entity (1961 vs. 1962). The hard case; requires a decision.

A second axis is **schema-level** vs. **instance-level** conflicts; schema-level
conflicts are the province of schema matching
([`07`](07-schema-and-ontology-matching.md)), so fusion proper is about
instance-level contradictions.

### 2.2 The three strategy classes

Bleiholder & Naumann's taxonomy of **conflict-handling strategies** [1] is the
backbone abstraction — and maps cleanly onto pluggable policies:

1. **Conflict-ignoring** — make no decision. *Pass It On* returns all values
   (e.g. as a set/list); *Consider All Possibilities* enumerates cross-products.
   Cheap, defers the problem to the consumer, sacrifices conciseness.
2. **Conflict-avoiding** — apply a uniform rule that sidesteps inspecting the
   values. *Take The Information* prefers non-null over null (handles uncertainty);
   *Trust Your Friends* prefers values from a designated authoritative source.
   Instance-independent — no per-value reasoning.
3. **Conflict-resolving** — actually look at the conflicting values and/or their
   metadata and decide. Split into:
   - **deciding** strategies, which return **one of the present values** — *Cry With
     The Wolves* (take the value the most sources agree on → **majority vote**),
     *Keep Up To Date* (most-recent by timestamp), *Roll The Dice* (random);
   - **mediating** strategies, which return a **new value not present in any
     source** — *Meet In The Middle* (average/median of numerics), or a computed
     summary.

These named strategies are the vocabulary of the SQL **`FUSE BY`** operator and the
Fusionplex/HumMer line of systems [1,2]; they are exactly the "fusion policy"
extension points `equate` should expose (§6).

### 2.3 Concrete resolution functions (the pluggable primitives)

| Resolution function | Class | Applies to | Needs |
|---|---|---|---|
| `first` / `any` | ignoring/avoiding | any | nothing |
| `coalesce` (first non-null) | avoiding | any | null-awareness |
| `most_frequent` (vote) | deciding | categorical | value counts |
| `longest` / `most_complete` | deciding | strings, records | length/coverage |
| `most_recent` | deciding | any | a timestamp/version column |
| `preferred_source` | avoiding | any | a source-priority order |
| `max` / `min` | deciding | ordinal/numeric | ordering |
| `average` / `median` | mediating | numeric | aggregability |
| `group` (return all) | ignoring | any | a set/list container |
| `highest_quality` | avoiding/deciding | any | per-value metadata (accuracy, freshness) |

Two properties determine what a function needs from the pipeline: whether it is
**metadata-based** (needs source priority, timestamps, or quality scores carried
alongside the value) and whether it is **value-based** (needs only the multiset of
values). This distinction dictates what provenance `equate` must *carry through*
the pipeline to make a policy expressible (§6).

### 2.4 Attribute independence — the default and its cost

Classic data fusion decides **each attribute independently** ("column-at-a-time"),
which is simple and parallel but can synthesize an **inconsistent golden record**:
taking `most_recent` phone from source A and `most_recent` address from source B can
produce a person who never existed in any single source. Consistency-aware fusion
(e.g. "keep values that co-occurred in some real record", Bleiholder &
Naumann's *No Gossiping*/consistency constraints [1]) trades conciseness for
coherence. This is the fusion analogue of the local-vs-global tension in matching
([`03`](03-assignment-and-graph-matching.md)).

---

## 3. Truth Discovery: source-quality-aware fusion

Data fusion's resolution functions assume you already know which source to trust (or
that majority rules). **Truth discovery** removes that assumption: it **jointly and
iteratively estimates source reliability and true values from the data itself**, on
the mutual-reinforcement intuition [4]:

> A **source** is trustworthy if it asserts many true values; a **value** is likely
> true if trustworthy sources assert it. Start uniform, iterate to a fixed point.

This breaks the tie that naive voting cannot: three unreliable sources copying one
error should not outvote one reliable source. The Li et al. **SIGKDD Explorations
survey (2016)** [4] is the standard map of the field and organizes methods by (a)
the source-reliability model, (b) how claims and sources are jointly inferred, and
(c) which problem dimensions they handle.

### 3.1 Problem dimensions (what a truth-discovery method must commit to)

- **Single-truth vs. multi-truth.** Does an object have exactly one true value
  (a person's birthdate) or possibly several (a book's authors)? Multi-truth
  changes the model from "pick one" to per-value true/false [4,7].
- **Data type.** Categorical (equality/vote), continuous (a wrong value can be
  *close*, so distance matters), or heterogeneous mixes [9].
- **Source-quality granularity.** A single scalar trust per source, vs.
  **two-sided** quality (false-positive vs. false-negative rates), vs. per-object or
  per-attribute quality [4,7].
- **Source dependence / copying.** Are sources independent, or do some **copy**
  others (so agreement is not independent evidence)? [8].
- **Priors & difficulty.** Long-tail sources with few claims; per-object difficulty;
  streaming/dynamic data [4].

### 3.2 TruthFinder — the canonical iterative method

Yin, Han & Yu's **TruthFinder** (KDD 2007; IEEE TKDE 2008) [5] is the archetype.
It iterates two updates to convergence:

- **source trustworthiness** `t(s)` = average **confidence** of the facts `s`
  asserts;
- **fact confidence** `c(f)` = combination of the trustworthiness of the sources
  asserting `f`.

Its refinements are what made it work and are worth reusing conceptually:

- combine via **log-space** sums (treat `1 − t(s)` as an error probability and sum
  `−ln(1 − t(s))`), so many mediocre sources cannot trivially dominate one good one;
- **implication between facts** — similar values reinforce each other (if "J. Smith"
  and "John Smith" are near-duplicates, evidence for one supports the other), so a
  similarity function feeds the confidence update. This couples truth discovery back
  to the comparison layer ([`05`](05-comparison-and-similarity-functions.md));
- a **dampening factor** for correlated sources and to prevent overconfidence.

Complexity is `O(iterations × |claims|)` — linear per iteration in the number of
source-value assertions; convergence is typically fast (a handful of iterations).
It is single-truth, categorical, and assumes source independence.

### 3.3 Dawid–Skene — the latent-truth / confusion-matrix model

Dawid & Skene's 1979 EM method [6] predates the "truth discovery" name but is its
statistical foundation and the workhorse of **crowdsourced labeling**. Model:

- each object has a **latent true label**;
- each source/annotator has a **confusion matrix** — class-conditional error rates
  `Pr(observed = j | truth = k)`;
- **EM** alternates estimating latent labels (E-step) and confusion matrices
  (M-step) to a local maximum-likelihood fixed point.

It gives per-source, per-class error rates rather than a single trust scalar — much
richer than TruthFinder — at the cost of more parameters and a categorical-label
assumption. It is the direct ancestor of the probabilistic models below and is the
right default when **each source labels the same object schema repeatedly** (the
crowdsourcing / multiple-annotator setting), which is structurally identical to the
**human-verification oracle-fusion** problem in
[`08`](08-interactive-active-learning-and-hitl.md) (combine several labellers'
judgments about the same candidate pair).

### 3.4 Probabilistic / Bayesian latent-truth models: LTM and beyond

Zhao, Rubinstein, Gemmell & Han's **Latent Truth Model (LTM)** (VLDB 2012) [7]
generalizes Dawid–Skene into a full Bayesian graphical model for **multi-truth**
data integration:

- the truth of **each claimed value** is a latent Bernoulli variable (so an object
  can have several true values);
- each source has **two-sided quality** — separate **sensitivity** (recall: how
  often it asserts a true value) and **specificity** (1 − false-positive rate: how
  often it avoids asserting false values), because a source can be reliable at
  confirming truth but noisy at avoiding falsehoods;
- inference via **collapsed Gibbs sampling**, converging quickly in roughly
  **linear time** in the data size, with **batch and online/streaming** modes and a
  slot for **prior domain knowledge** of source quality.

LTM and its relatives (LCA — Latent Credibility Analysis; Zhao & Han's other
generative models; multi-truth successors [4]) are the "principled" wing:
interpretable source-quality parameters, natural uncertainty estimates, easy
incorporation of priors — traded against heavier inference than the iterative
methods.

### 3.5 Source dependence and copy detection — the Accu family

Dong, Berti-Équille & Srivastava's **"Integrating Conflicting Data: The Role of
Source Dependence"** (VLDB 2009) [8] adds the insight that most other methods
ignore: **sources are not independent — some copy others.** If ten websites all
copy one wrong value, naive voting (and even TruthFinder) counts ten votes for a
falsehood. Their models:

- **Accu** — Bayesian accuracy estimation using per-source **accuracy** and the
  number of competing false values;
- **AccuCopy / AccuSim** — additionally detect **copying dependence** between
  sources (Bayesian analysis of shared errors: two sources sharing *the same wrong
  value* is strong evidence of copying) and **discount copied votes**, optionally
  folding value similarity (AccuSim) into the analysis.

This dependence-aware line is what scales truth discovery to the open Web, where
copying is rampant, and is a distinguishing design axis in the survey [4].

### 3.6 CRH — an optimization framework for heterogeneous data

Li, Li, Gao, Zhao, Fan & Han's **CRH** ("Conflicts to Harmony", SIGMOD 2014) [9]
recasts truth discovery as a single **optimization problem**: minimize the total
**source-weighted deviation** between the inferred truth and each source's claims,

```
minimize  Σ_sources  w(source) · Σ_objects  d( truth(object), claim(source, object) )
subject to a constraint on the weights (e.g. Σ exp(−w) = 1)
```

solved by **block coordinate descent** alternating (i) fix weights → update truths,
(ii) fix truths → update weights. Its key contribution is **pluggable loss
functions `d`** per attribute type — 0/1 loss for categorical, normalized squared
loss for continuous — so a **single framework fuses heterogeneous records** (a row
with both a categorical `city` and a numeric `price`). This "one objective, swappable
per-type loss" structure is directly analogous to `equate`'s per-field
comparison-vector design ([`05`](05-comparison-and-similarity-functions.md)) and is
the most `equate`-shaped of the truth-discovery formulations.

### 3.7 Knowledge-Based Trust — web-scale, extraction-aware

Dong et al.'s **Knowledge-Based Trust (KBT)** (VLDB 2015) [10] estimates a web
source's trustworthiness from the **correctness of the facts it states** (endogenous
signal) rather than hyperlinks (exogenous, PageRank-style). Its notable move for a
real pipeline: a **multi-layer probabilistic model that separates *extraction*
errors from *source* errors** — a fact may be wrong because the source is wrong, or
because the extractor misread a correct source — jointly inferring both. Applied to
2.8B facts across 119M webpages. The lesson for `equate`: **the observation you fuse
is itself noisy (parsing/extraction/normalization can introduce the conflict)**, and
a serious fusion layer should not blame the source for a pipeline artifact.

### 3.8 The empirical reality check

Waguih & Berti-Équille's experimental evaluation of **12 truth-discovery algorithms**
[11] is the sobering benchmark: **there is no universal winner.** Performance depends
heavily on the **source-quality distribution** (how many good vs. bad sources, how
correlated their errors are), and — crucially — **simple majority voting is a
surprisingly strong baseline** that sophisticated methods only beat in specific
regimes (few reliable sources, heavy copying, skewed coverage). This mirrors the
round-1 finding that TF-IDF + edit distance is competitive with deep matchers on
clean data ([`06`](06-deep-learning-and-llm-entity-matching.md),
[`10`](10-design-implications-for-equate.md)). **Design consequence for `equate`:
ship voting/most-recent as the zero-config default; make source-reliability truth
discovery an *opt-in* that users reach for when they can articulate why voting is
failing.**

### 3.9 Python tooling (what exists to wrap)

Truth discovery has **thin, research-grade** library support, unlike the mature ER
ecosystem ([`09`](09-python-ecosystem-landscape.md)):

- **`truthdiscovery`** (joesingo) — a Python 3 library with several algorithms
  (Sums, Average·Log, Investment, PooledInvestment, TruthFinder), a graph
  source→claim→object model, plus CLI and web interfaces.
- **`spectrum`** (totucuong) — a data-fusion library covering discrete and
  continuous values.
- Reference implementations from the benchmark papers (Waguih & Berti-Équille's Java
  suite [11]) and per-paper code for LTM/CRH.

None is a maintained, broadly-adopted standard. This argues for `equate` to **own a
small, dependency-light truth-discovery core** (voting, weighted voting, a
TruthFinder-style iteration, a Dawid–Skene EM) and treat heavier/probabilistic
methods as optional wraps — mirroring the "own the orchestration, wrap the heavy
compute" rule from [`09`](09-python-ecosystem-landscape.md) and
[`10`](10-design-implications-for-equate.md).

---

## 4. Canonicalization & Survivorship (the MDM framing)

Industry Master Data Management (MDM) reaches the same destination via
**deterministic, governed rules** rather than statistical inference. Loshin's
*Master Data Management* [14] and vendor practice frame the output as the **golden
record** (a.k.a. *single version of truth*, *single customer view*, *master
record*) and the rules that build it as **survivorship** (a.k.a. *consolidation* or
*merge* rules).

### 4.1 Golden record and survivorship

The **golden record** is the single, most-complete, most-trusted representation of
an entity, assembled from all its matched source records. **Survivorship** is the
process of deciding, **per attribute**, which source value "survives" into the
golden record. The typical MDM lifecycle is `ingest → match/link → survivorship →
validate → steward/maintain`, with survivorship the fusion step.

### 4.2 Attribute-level survivorship rules

Survivorship in MDM is almost always **attribute-level** (not record-level): each
column can survive from a *different* source. Common rule primitives — which are the
same resolution functions as §2.3, wearing an enterprise hat:

- **Source-of-record / trusted-source priority** — a ranked source list per
  attribute (CRM wins for `email`, ERP wins for `billing_address`). The MDM analogue
  of `preferred_source` / *Trust Your Friends*.
- **Most-recent** — newest `last_updated` wins (`most_recent` / *Keep Up To Date*).
- **Most-complete / longest** — prefer the non-null, richest value.
- **Most-frequent** — majority across sources.
- **Rule-based / conditional** — domain logic (e.g. "a validated address beats an
  unvalidated one regardless of recency").
- **Aggregate** — max/min/sum for numerics.

Two structural facts distinguish the MDM view: rules are **governed and auditable**
(a data steward can override, and every survived value must trace to its source —
**lineage/provenance**), and there is an explicit **merge/unmerge** requirement
(golden records can be *split* again when a bad match is discovered — the fusion
analogue of the "every edit is a constraint" principle in
[`08`](08-interactive-active-learning-and-hitl.md)).

### 4.3 Implementation styles (architecture, not algorithm)

MDM practice distinguishes *where the golden record lives*, which affects how fusion
couples to the rest of a system:

- **Registry** — sources stay authoritative; the hub stores only match links and
  computes the golden record **virtually/on-read** (fusion is a query-time view).
- **Consolidation** — golden records are **materialized** in a hub for
  analytics/reporting, read-mostly.
- **Coexistence / centralized (transaction)** — the hub is authoritative and writes
  flow back to sources.

For a *library* like `equate` the relevant takeaway is the **virtual-vs-materialized**
choice: fusion can be a **lazy, on-demand reduction over a cluster** (compute the
golden record when asked) or a **materialized artifact** — and the API should not
force one (§6).

### 4.4 Canonicalization in knowledge bases

The knowledge-base / NLP community faces canonicalization for **open information
extraction**, where the "records" are `(subject, relation, object)` triples with
**uncanonical surface strings** ("Barack Obama" vs. "Obama"; "was born in" vs.
"place of birth"). Galárraga, Heitz, Murphy & Suchanek's **Canonicalizing Open
Knowledge Bases** (CIKM 2014) [13] does this by **clustering synonymous noun phrases**
(hierarchical agglomerative clustering over similarity features — token overlap,
IDF-weighted words, shared types, co-occurring relations) for the *subjects*, and a
separate clustering for the *relations*. This is *matching feeding canonicalization*
in miniature — the featurize→compare→cluster→canonicalize pipeline
([`00`](00-taxonomy-and-terminology.md#1-the-canonical-decomposition)) applied to
strings, and a useful validation that `equate`'s stage decomposition generalizes
beyond tabular records.

### 4.5 From data fusion to *knowledge* fusion

Dong et al.'s **"From Data Fusion to Knowledge Fusion"** (VLDB 2014) [12] marks the
generalization from fusing *values of a known schema* to fusing *extracted triples*
where both the extraction and the source can err (the Knowledge Vault setting). It
formalizes why fusion at Web/KG scale must model the **extractor** as a source in
its own right — reinforcing the §3.7 point that the noisy observation, not just the
source, must be part of the model. For `equate` this is mostly context: it shows the
outer envelope of the design space (fusion over uncertain, machine-extracted facts)
and confirms the same source-reliability machinery carries over.

---

## 5. Glossary

- **Data fusion** — merging multiple records that represent the same real-world
  object into a single, complete, concise, consistent representation; the database
  community's name for the golden-record step [1,3].
- **Truth discovery** — jointly inferring **source reliability** and **true values**
  from conflicting multi-source data, via iterative or probabilistic
  mutual-reinforcement between source trust and value belief [4].
- **Source reliability / trustworthiness** — a per-source quality estimate (scalar
  accuracy, or two-sided sensitivity/specificity, or a full confusion matrix) used
  to weight that source's assertions [4,6,7].
- **Conflict resolution** — choosing among (or computing from) contradicting values;
  classified as conflict-*ignoring*, conflict-*avoiding*, or conflict-*resolving*
  (deciding vs. mediating) [1].
- **Resolution function** — the concrete operator applied to a set of conflicting
  values (vote, max, most-recent, average, preferred-source, coalesce) [1,2].
- **Canonicalization** — producing the single representative form for a resolved
  cluster (of records, or of KB surface strings) [13,15].
- **Golden record** — the single, most-trusted, most-complete representation of an
  entity in MDM; a.k.a. single version of truth / single customer view [14].
- **Survivorship** — the MDM process deciding, per attribute, which source value
  "survives" into the golden record [14].
- **Master Data Management (MDM)** — the governance + technology discipline that
  creates and maintains golden records for core enterprise entities [14].
- **TruthFinder** — the archetypal iterative truth-discovery algorithm (source trust
  ↔ fact confidence, log-space combination, value-implication) [5].
- **Dawid–Skene** — the 1979 EM model estimating per-annotator confusion matrices
  over a latent true label; foundation of crowdsourced-label aggregation [6].
- **LTM (Latent Truth Model)** — a Bayesian multi-truth model with two-sided source
  quality, inferred by collapsed Gibbs sampling [7].
- **Accu / AccuCopy** — accuracy-based truth discovery that also detects and
  discounts **copying** between sources [8].
- **CRH** — an optimization-framework truth-discovery method with pluggable
  per-type loss functions for heterogeneous data [9].
- **Provenance / lineage** — the record of which source (and version/time) each
  fused value came from; prerequisite for metadata-based resolution and auditability
  [14].

---

## 6. Design implications for `equate`

Fusion/canonicalization is the natural **L5+** extension of the clustering back-end
already anticipated in [`01`](01-entity-resolution-record-linkage.md#3-design-implications)
and [`10`](10-design-implications-for-equate.md) (the `cluster/canonicalize.py`
module sketch). The round-1 corpus already reserved a home for it —
[`10` §6](10-design-implications-for-equate.md) lists
`canonicalize.py  # golden-record merge policies (majority/most-complete/source-trust)`.
This section specifies that home.

### 6.1 A `Canonicalizer` is an optional post-resolution stage

Keep matching and fusion **decoupled**. The default `match(A, B)` returns pairs and
must not force a fusion step. Fusion attaches only when the user asks for **one
record per entity**:

```python
resolve(records, *, block=..., compare=..., cluster=..., canonicalize=None)
# canonicalize=None  → return clusters (groups of co-referent records)   [today's behaviour]
# canonicalize=policy → return one fused golden record per cluster        [opt-in]
```

The stage consumes the output of clustering/transitive-closure
([`01`](01-entity-resolution-record-linkage.md), [`00` §5](00-taxonomy-and-terminology.md#5-the-intransitivity-problem))
— a `Cluster` = an iterable of source records — and emits a `GoldenRecord`. It never
runs on 1:1 `match(A, B)` output unless the user explicitly clusters first.

### 6.2 The core abstraction: attribute-level resolution policy

Following the survey convergence (§2.3, §4.2) that fusion is **per-attribute**, make
the primary extension point a mapping from field to a **resolution function**, with a
sensible default:

```python
Resolution = Callable[[Sequence[Value], ResolutionContext], Value]
#   values from the cluster's records for ONE attribute
#   context carries per-value provenance: source id, timestamp, quality, is_null

Canonicalizer = Callable[[Cluster], GoldenRecord]

def survivorship(
    rules: Mapping[str, Resolution] | Resolution = most_frequent,  # per-field or one default
    *,
    default: Resolution = coalesce,        # for fields not named in `rules`
    consistency: Literal['per_field', 'row_coherent'] = 'per_field',
) -> Canonicalizer: ...
```

Ship the §2.3 primitives as named, registry-selectable strategies (SSOT +
open-closed, per [`10` §2](10-design-implications-for-equate.md)):
`most_frequent` (vote), `most_recent`, `most_complete`/`longest`, `coalesce`,
`preferred_source(order)`, `max`/`min`, `average`/`median`, `group` (return all —
the conflict-*ignoring* escape hatch). `canonicalize='majority'` or a callable, same
dual string-or-callable pattern as every other `equate` seam.

### 6.3 Provenance is the load-bearing prerequisite

Metadata-based resolution (`most_recent`, `preferred_source`, `highest_quality`)
**cannot be expressed unless the pipeline carries provenance** alongside each value
(§2.3, §4.2). This is a concrete, early requirement: the `ScoreMatrix`/record
representation flowing through `equate` must be able to tag each attribute value with
`(source_id, timestamp?, quality?)`. Design the `Cluster`/`GoldenRecord` dataclasses
to carry a **per-value provenance envelope** from the start, even if the default
policy ignores it — retrofitting provenance later is expensive.

### 6.4 Truth discovery as an optional, source-aware policy tier

Layer the sophistication as the §3.8 evidence dictates — **cheap default, opt-in
inference**:

| Tier | Policy | What it needs | Boundary |
|---|---|---|---|
| 0 | `most_frequent` / `coalesce` / `most_recent` | values (+ maybe timestamps) | **core** (stdlib only) |
| 1 | `weighted_vote(source_weights)` | user-supplied source weights | **core** |
| 2 | `truthfinder()` / Dawid–Skene EM | many objects × sources; iterate/EM | **core, numpy** — small, own it |
| 3 | LTM / CRH / Accu (copy-aware, Bayesian, heterogeneous) | Gibbs/optimization; more deps | `equate[truth]` — optional extra, wrap `truthdiscovery`/`spectrum` or port |

Rationale straight from the benchmarks [11]: voting is a strong baseline, so tiers
0–1 are the default; the estimator tiers earn their complexity only under skewed
source quality or copying. Expose the **objective/assumption**, not the algorithm
name, as the primary knob (`sources_independent: bool`, `multi_truth: bool`,
`value_type: 'categorical'|'continuous'`) and let it select a legal method —
mirroring the "expose the objective, not the algorithm" rule for matchers in
[`10` §3.1](10-design-implications-for-equate.md).

### 6.5 Coupling to clustering and to human verification

- **To clustering / transitive closure.** Fusion inherits clustering's errors: an
  over-merged cluster fuses records that are not the same entity, silently corrupting
  the golden record. Expose a **cluster-quality signal** (size, internal cohesion,
  presence of hard conflicts) so a policy can *abstain* — return the group unfused,
  or flag it for review — rather than fabricate a value. This is the fusion side of
  the split/merge tension Binette & Steorts flag [15].
- **To human verification** ([`08`](08-interactive-active-learning-and-hitl.md)).
  Fusion is a natural **review trigger**: surface golden records whose attributes had
  **high-conflict, low-agreement** resolutions (many distinct values, no majority, or
  a mediating strategy invented a value) into the same `ReviewQueue`. A steward's
  override becomes an **append-only constraint** ("for entity E, `email` survives
  from source S"), re-solved locally — identical machinery to the confirm/reject
  constraints in [`08`](08-interactive-active-learning-and-hitl.md). And the
  **Dawid–Skene / oracle-fusion** model (§3.3) is the *same* estimator that combines
  **multiple human labellers'** verdicts on one candidate pair — so the truth-discovery
  core does double duty as the label-aggregation core for active learning.

### 6.6 Lazy vs. materialized (virtual golden record)

Per §4.3, do not bake in materialization. Make the canonicalizer a **pure reduction
over a cluster** so it can be called **lazily on-read** (registry style — compute a
golden record when queried) *or* materialized and cached
(`dol.cache_this`, per the persistence pattern in
[`08`](08-interactive-active-learning-and-hitl.md) and
[`10`](10-design-implications-for-equate.md)). Keeping it a pure function of
`(cluster, policy)` also makes it trivially unit-testable and re-runnable when a
cluster changes.

### 6.7 What to build vs. wrap (per [`09`](09-python-ecosystem-landscape.md)/[`10`](10-design-implications-for-equate.md))

| Concern | Build (own it) | Wrap (optional extra) | Avoid |
|---|---|---|---|
| Resolution functions (§2.3) | all of them — trivial, stdlib/`numpy` | — | over-abstracting |
| Provenance envelope + policy dispatch | yes — it's orchestration | — | — |
| Voting / weighted voting / TruthFinder-style / Dawid–Skene EM | yes — small, dependency-light core | — | — |
| Bayesian/optimization TD (LTM, CRH, Accu, copy detection) | interface only | `equate[truth]` → `truthdiscovery`/`spectrum` or a port | hard dep |
| MDM governance (stewardship UI, unmerge history, audit) | expose the hooks (provenance, override-as-constraint) | leave the app/UI to `zodal` ([`10` §5](10-design-implications-for-equate.md)) | building an MDM product |

Bottom line: **own the fusion *orchestration* and the light default policies; keep
source-reliability inference and MDM governance as optional/out-of-scope**, exactly
the boundary the round-1 corpus drew for every other heavy capability.

---

## 7. Open questions carried forward

- **Default policy.** Is the zero-config canonicalizer `most_frequent` (robust,
  needs ≥3 sources to break ties) or `coalesce`/`most_recent` (works with 2 sources,
  needs a timestamp)? Depends on `equate`'s primary use case
  ([`10` §9](10-design-implications-for-equate.md)).
- **Row-coherence.** Does `equate` guarantee the golden record is a *real* row from
  some source (conservative, consistent) or allow best-of-breed per-attribute
  synthesis (complete but possibly inconsistent, §2.4)? A `consistency=` flag, and
  what its default is.
- **Where does value-fusion's source model live** relative to the match-scoring
  combiner ([`05`](05-comparison-and-similarity-functions.md))? They share weighting
  machinery but run at different stages — unify or keep separate?
- **Cluster-error propagation.** What cluster-quality signal is cheap to compute and
  actionable enough for a policy to abstain on a bad merge (§6.5)?
- **Scope boundary with `completion.py`.** The existing missing-data-completion
  work (join-based value filling) overlaps fusion (both pick a value from multiple
  candidate sources); should completion be re-expressed as a degenerate canonicalizer
  over a 1-record-per-source "cluster"?

---

## References

1. Bleiholder J, Naumann F. Data Fusion. *ACM Computing Surveys* 41(1), Art. 1, 2008. [https://dl.acm.org/doi/10.1145/1456650.1456651](https://dl.acm.org/doi/10.1145/1456650.1456651)
2. Dong XL, Berti-Équille L, Srivastava D. Data Fusion: Resolving Conflicts from Multiple Sources. In: *Handbook of Data Quality — Research and Practice*, Springer, 2013 / arXiv:1503.00310. [https://arxiv.org/abs/1503.00310](https://arxiv.org/abs/1503.00310)
3. Dong XL, Naumann F. Data Fusion — Resolving Data Conflicts for Integration (tutorial). *PVLDB* 2(2):1654-1655, 2009. [http://www.vldb.org/pvldb/vol2/vldb09-tutorial1.pdf](http://www.vldb.org/pvldb/vol2/vldb09-tutorial1.pdf)
4. Li Y, Gao J, Meng C, Li Q, Su L, Zhao B, Fan W, Han J. A Survey on Truth Discovery. *ACM SIGKDD Explorations Newsletter* 17(2):1-16, 2016 / arXiv:1505.02463. [https://arxiv.org/abs/1505.02463](https://arxiv.org/abs/1505.02463) · DOI: [https://dl.acm.org/doi/10.1145/2897350.2897352](https://dl.acm.org/doi/10.1145/2897350.2897352)
5. Yin X, Han J, Yu PS. Truth Discovery with Multiple Conflicting Information Providers on the Web (TruthFinder). *KDD* 2007 / *IEEE TKDE* 20(6):796-808, 2008. [http://hanj.cs.illinois.edu/pdf/kdd07_xyin.pdf](http://hanj.cs.illinois.edu/pdf/kdd07_xyin.pdf)
6. Dawid AP, Skene AM. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. *Journal of the Royal Statistical Society: Series C (Applied Statistics)* 28(1):20-28, 1979. [https://rss.onlinelibrary.wiley.com/doi/10.2307/2346806](https://rss.onlinelibrary.wiley.com/doi/10.2307/2346806)
7. Zhao B, Rubinstein BIP, Gemmell J, Han J. A Bayesian Approach to Discovering Truth from Conflicting Sources for Data Integration (Latent Truth Model). *PVLDB* 5(6):550-561, 2012 / arXiv:1203.0058. [http://vldb.org/pvldb/vol5/p550_bozhao_vldb2012.pdf](http://vldb.org/pvldb/vol5/p550_bozhao_vldb2012.pdf)
8. Dong XL, Berti-Équille L, Srivastava D. Integrating Conflicting Data: The Role of Source Dependence (Accu / AccuCopy). *PVLDB* 2(1):550-561, 2009. [http://www.vldb.org/pvldb/vol2/vldb09-pvldb47.pdf](http://www.vldb.org/pvldb/vol2/vldb09-pvldb47.pdf)
9. Li Q, Li Y, Gao J, Zhao B, Fan W, Han J. Resolving Conflicts in Heterogeneous Data by Truth Discovery and Source Reliability Estimation (CRH). *SIGMOD* 2014:1187-1198. [https://dl.acm.org/doi/10.1145/2588555.2610509](https://dl.acm.org/doi/10.1145/2588555.2610509)
10. Dong XL, Gabrilovich E, Murphy K, Dang V, Horn W, Lugaresi C, Sun S, Zhang W. Knowledge-Based Trust: Estimating the Trustworthiness of Web Sources. *PVLDB* 8(9):938-949, 2015 / arXiv:1502.03519. [https://www.vldb.org/pvldb/vol8/p938-dong.pdf](https://www.vldb.org/pvldb/vol8/p938-dong.pdf)
11. Waguih DA, Berti-Équille L. Truth Discovery Algorithms: An Experimental Evaluation. arXiv:1409.6428, 2014. [https://arxiv.org/abs/1409.6428](https://arxiv.org/abs/1409.6428)
12. Dong XL, Gabrilovich E, Heitz G, Horn W, Murphy K, Sun S, Zhang W. From Data Fusion to Knowledge Fusion. *PVLDB* 7(10):881-892, 2014 / arXiv:1503.00302. [https://arxiv.org/abs/1503.00302](https://arxiv.org/abs/1503.00302)
13. Galárraga L, Heitz G, Murphy K, Suchanek FM. Canonicalizing Open Knowledge Bases. *CIKM* 2014. [https://suchanek.name/work/publications/cikm2014.pdf](https://suchanek.name/work/publications/cikm2014.pdf)
14. Loshin D. *Master Data Management.* Morgan Kaufmann (The MK/OMG Press), 2008. [https://www.sciencedirect.com/book/monograph/9780123742254/master-data-management](https://www.sciencedirect.com/book/monograph/9780123742254/master-data-management)
15. Binette O, Steorts RC. (Almost) All of Entity Resolution. *Science Advances* 8(12):eabi8021, 2022. [https://www.science.org/doi/10.1126/sciadv.abi8021](https://www.science.org/doi/10.1126/sciadv.abi8021)
16. Canalle GK, Salgado AC, Lóscio BF. A Survey on Data Fusion: What For? In What Form? What Is Next? *Journal of Intelligent Information Systems* 57(1):25-50, 2021. [https://link.springer.com/article/10.1007/s10844-020-00627-4](https://link.springer.com/article/10.1007/s10844-020-00627-4)
17. Dong XL, Srivastava D. *Big Data Integration.* Synthesis Lectures on Data Management, Morgan & Claypool, 2015. [https://link.springer.com/book/10.1007/978-3-031-01853-4](https://link.springer.com/book/10.1007/978-3-031-01853-4)
18. Singo J. `truthdiscovery` — Python 3 truth-discovery library. [https://github.com/joesingo/truthdiscovery](https://github.com/joesingo/truthdiscovery) · To T-C. `spectrum` — a Python data-fusion library. [https://github.com/totucuong/spectrum](https://github.com/totucuong/spectrum)
