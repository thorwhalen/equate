# Evaluation, Benchmarks & Methodology for Matching: How to Measure a Matcher Honestly

*Research note for the `equate` redesign — one of a corpus in `docs/research/`.
Round-2 gap-filler. Depends on the metric primitives sketched in
[`01-entity-resolution-record-linkage.md`](01-entity-resolution-record-linkage.md)
(§4, pairwise vs. cluster) and the blocking metrics in
[`02-blocking-and-scalable-candidate-generation.md`](02-blocking-and-scalable-candidate-generation.md).
Situate everything in the vocabulary of
[`00-taxonomy-and-terminology.md`](00-taxonomy-and-terminology.md); the concrete
API recommendations extend
[`10-design-implications-for-equate.md`](10-design-implications-for-equate.md).*

## Abstract

Matching is one of the rare problems where the *evaluation* is harder and more
error-prone than the *method*: a matcher that reports 95% F1 on a public
benchmark is routinely a matcher that will disappoint in production, because the
benchmark was too easy, the labels leaked across the train/test split, or the
wrong granularity of metric was reported. This note is a practitioner's manual
for honest measurement. It defines the metric families — **pairwise**
precision/recall/F1, **cluster-level** metrics (B³, CEAF, Generalized Merge
Distance, Adjusted Rand Index, homogeneity/completeness/V-measure),
**threshold-free** curves (PR-AUC, ROC-AUC) and **ranking** metrics (recall@k,
MAP, MRR), and **blocking** metrics (pair completeness, reduction ratio, pairs
quality) — says *which one lies to you when*, and catalogs the standard
benchmarks (Magellan/DeepMatcher, WDC Products, Machamp, Alaska, Valentine,
OpenEA). It then covers the methodology that determines whether any of those
numbers mean anything: ground-truth construction, leakage-safe splitting, class
imbalance, quality estimation without full labels, and transfer/generalization.
It closes with a concrete `equate.evaluate` design — the metrics, the
benchmark registry, the leakage-aware splitter, and the self-benchmark harness
`equate` should ship, all behind optional-dependency boundaries.

---

## 1. Why evaluation is the hard part of matching

Three structural facts make matching evaluation uniquely treacherous, and every
recommendation below follows from one of them.

**(a) Extreme class imbalance.** Matching is a needle-in-a-haystack problem. For
two sources of *n* and *m* records there are *n·m* candidate pairs but only
O(max(n, m)) true matches, so the positive class is often 10⁻³–10⁻⁶ of the
space. A "matcher" that predicts *non-match* for everything scores ~99.99%
**accuracy**. Accuracy is therefore useless; every honest metric focuses on the
positive (match) class or on the induced partition [1,2]. This single fact kills
accuracy and ROC-AUC as headline numbers (§4) and forces care about how the
negative class is sampled (§7.3).

**(b) The unit of truth is ambiguous.** Is the ground truth a set of *pairs*, or
a *partition* of records into entities? A prediction can be perfect at the pair
level and wrong at the entity level, and vice versa (§2.2). Reporting the wrong
granularity is the most common way published numbers mislead [3].

**(c) Ground truth is expensive, incomplete, and itself uncertain.** Labeling all
*n·m* pairs is quadratic and infeasible; real "gold" sets are sampled, often only
over the candidate pairs a *particular* blocker produced — which biases every
downstream number and, if reused as a test set, leaks (§7.1–7.2). Human labels
also disagree; inter-annotator agreement on hard corner-cases is far below 100%.

> **Consequence for `equate`.** Evaluation cannot be an afterthought bolted onto
> a demo. It must be a first-class, reusable subsystem (`equate.evaluate`) with
> the same care given to the matchers themselves — because the *only* way a user
> learns their matcher is bad is by measuring it correctly.

---

## 2. The two granularities: pairwise vs. cluster-level metrics

Fix notation. Let the gold standard be a set of true match pairs **G** (or,
equivalently, a partition of records into gold entities). Let the system output
be predicted pairs **P** (or a predicted partition). "Matching" outputs pairs;
"resolution/clustering" outputs a partition, typically the transitive closure of
the pairs. Both granularities matter and they disagree [3].

### 2.1 Pairwise precision / recall / F1

Treat matching as binary classification over the *set of pairs*.

- **Precision** = |P ∩ G| / |P| — of the pairs we called matches, how many were
  real. Falls when we over-merge (false positives).
- **Recall** = |P ∩ G| / |G| — of the real matches, how many we found. Falls when
  we miss pairs (false negatives).
- **F1** = 2·P·R / (P + R), the harmonic mean — the single most-reported ER
  number [1,2]. Use **Fβ** (β>1 weights recall) when a domain penalizes misses
  more than false merges, or vice versa.

Pairwise metrics are cheap (set intersection), interpretable, and the right
choice when the deliverable genuinely *is* pairs — a fuzzy join, a 1:1
alignment, a link-discovery task. They are the natural fit for `equate`'s core
`match(A, B) → pairs` signature.

### 2.2 Why pairwise metrics mislead on clusters

When the deliverable is *entities* (groups), pairwise metrics distort in two
ways [3]:

1. **Quadratic cluster-size weighting.** A gold entity with *k* records
   contributes k(k−1)/2 pairs. A single 100-record cluster contributes ~4,950
   pairs — it dominates the metric, so getting a few huge clusters right can mask
   thousands of wrong small ones. Pairwise recall is biased toward large
   entities.
2. **Transitivity blind spots.** Suppose gold entity {A, B, C}. If the system
   predicts A–B and B–C but *not* A–C, and you evaluate the raw pairs, you are
   penalized for a missing A–C that the *transitive closure* would supply. If you
   evaluate the closure, a single spurious edge can chain-merge two large correct
   clusters into one catastrophic blob — pairwise precision collapses in a way
   that overstates the damage relative to how a human perceives the partition.

The canonical demonstration is Menestrina, Whang & Garcia-Molina [3]: pairwise
F1, cluster F1, and other measures **rank the same algorithms differently**, so
"which matcher is best" is metric-dependent. You must pick the granularity that
matches your deliverable and report it explicitly.

### 2.3 Cluster-level metrics

These judge the predicted *partition* against the gold partition. `equate`
should offer a representative set spanning the three metric families (set-based,
counting-pairs, information-theoretic) because they satisfy different formal
properties (§2.4).

**B³ (B-cubed)** [4,5]. Computed *per record*, then averaged, so it avoids the
quadratic large-cluster bias of raw pairwise. For record *r* with gold cluster
*L(r)* and predicted cluster *C(r)*:

- B³ precision(r) = |C(r) ∩ L(r)| / |C(r)|
- B³ recall(r)    = |C(r) ∩ L(r)| / |L(r)|

Average over all records; take the harmonic mean for B³ F1. B³ is the de facto
standard for entity-clustering and coreference because it satisfies all of
Amigó's formal constraints (§2.4) [4]. It is O(N) given a hash from record →
cluster on each side. **Recommended default cluster metric for `equate`.**

**CEAF (Constrained Entity-Aligned F-measure)** [5,6]. Finds the *optimal
one-to-one alignment* between predicted and gold clusters (a maximum-weight
bipartite matching — Hungarian algorithm, O(k³) in the number of clusters) and
scores similarity of aligned clusters. Two variants: **CEAFₘ** (mention-based,
entity size counts) and **CEAFₑ** (entity-based, each entity weighted equally).
CEAF corrects a known B³ quirk (B³ can reward putting every mention in its own
singleton). Note the cubic cost in cluster count and that it — like all
alignment metrics — reuses the same **assignment** machinery `equate` already
has for matching (see [`03-assignment-and-graph-matching.md`](03-assignment-and-graph-matching.md));
`linear_sum_assignment` can compute it.

**Generalized Merge Distance (GMD) / cluster-F** [3]. An edit-distance over
partitions: the minimum cost of `split` and `merge` operations to turn the
predicted partition into the gold one, with user-supplied per-operation cost
functions. With unit costs it reduces to familiar measures; Menestrina et al.
show pairwise-F and cluster-F are special cases, and give a linear-time
algorithm. Its value is *tunability*: a domain that fears over-merging can charge
merges more than splits.

**Adjusted Rand Index (ARI)** [7]. The Rand index counts pairs the two partitions
*agree* on (both-together or both-apart) over all pairs; ARI corrects it for
chance so random labelings score ≈0 and identical partitions score 1 (it can go
slightly negative for worse-than-random). ARI is symmetric, needs no
cluster-alignment, and is a good *single-number* summary — but being pair-based
it satisfies fewer of Amigó's constraints than B³ (§2.4) and inherits some
large-cluster sensitivity. Available directly as
`sklearn.metrics.adjusted_rand_score` [8].

**Homogeneity, Completeness, V-measure** [8]. Information-theoretic, from the
clustering literature (Rosenberg & Hirschberg 2007). **Homogeneity**: each
predicted cluster contains only members of a single gold entity (the "precision"
direction). **Completeness**: all members of a gold entity land in the same
predicted cluster (the "recall" direction). **V-measure** is their (weighted)
harmonic mean, exactly analogous to F1 but over conditional entropies.
Convenient because `sklearn.metrics.homogeneity_completeness_v_measure` returns
all three, and they map cleanly onto the precision/recall intuition users
already have.

### 2.4 Amigó's formal constraints — *which metric to trust*

Amigó, Gonzalo, Artiles & Verdejo [4] evaluate clustering metrics against four
**formal constraints** a good extrinsic metric should satisfy:

1. **Cluster homogeneity** — do not mix items of different gold entities.
2. **Cluster completeness** — keep items of one gold entity together.
3. **Rag bag** — a small error in a large heterogeneous cluster is less harmful
   than the same error in a clean cluster.
4. **Cluster-size vs. quantity** — many small errors should be penalized more
   than one large one.

Their headline result: **B³ is the only family that satisfies all four**;
counting-pairs measures (Rand / ARI) satisfy only the first two; purely
set-based measures (like MUC from coreference) miss others. This is the
principled reason to make **B³ the default cluster metric** and to offer ARI /
V-measure as complements rather than substitutes. Ship this table so users pick
deliberately.

| Metric | Family | Alignment needed | Chance-corrected | All 4 constraints | Cost |
|---|---|---|---|---|---|
| Pairwise P/R/F1 | counting pairs | no | no | no | O(\|P\|+\|G\|) |
| ARI | counting pairs | no | **yes** | no | O(N) via contingency |
| B³ P/R/F1 | set / per-record | no | no | **yes** | O(N) |
| CEAFₑ/ₘ | alignment | **yes** (Hungarian) | no | most | O(k³) in #clusters |
| GMD / cluster-F | edit distance | no | no | tunable | O(N) |
| Homogeneity/Completeness/V | information-theoretic | no | no (V not chance-corrected) | most | O(N) |

### 2.5 Choosing a metric — a decision rule

- **Deliverable is pairs / a 1:1 alignment** (fuzzy join, schema match, KG
  links) → **pairwise P/R/F1** (and ranking metrics, §4).
- **Deliverable is entities / clusters** (dedup, canonicalization) → **B³ F1** as
  the headline, plus **one chance-corrected number (ARI)** and **one tunable
  number (GMD)** to expose over/under-merge asymmetry.
- **Always report precision and recall separately, not just F1** — F1 hides the
  operating point, and downstream cost (a bad merge vs. a missed link) is rarely
  symmetric.

---

## 3. Threshold selection is part of the method, not a footnote

Almost every matcher emits a *score*; a threshold turns scores into decisions.
Two failure modes to guard against, both of which `equate`'s harness should make
hard to commit:

- **Tuning the threshold on the test set.** The threshold is a learned parameter.
  Select it on a validation split; report the frozen threshold's performance on
  test (§7.2).
- **Reporting only the best-threshold F1.** This is a *maximum over a random
  variable* and is optimistically biased. Report either a threshold-free summary
  (§4) *or* a validation-selected threshold — ideally both.

---

## 4. Threshold-free and ranking metrics

Threshold-free metrics summarize a scored matcher across *all* operating points,
decoupling "is the score any good?" from "is the cut in the right place?".

**Precision–Recall AUC (PR-AUC / Average Precision).** Sweep the threshold,
plot precision vs. recall, integrate. **This is the right threshold-free metric
for matching** because it ignores true negatives, which the base-rate problem
(§1a) makes overwhelming and uninformative. `sklearn.metrics.average_precision_score`
computes the sample-weighted step-function version (preferred over trapezoidal
interpolation, which is optimistic) [8].

**ROC-AUC.** Sweeps TPR vs. FPR. **Misleading under heavy imbalance**: FPR =
FP/(FP+TN) has a gigantic TN denominator, so even thousands of false positives
barely move it, and a useless matcher can post ROC-AUC ≈ 0.95. Report it only
alongside PR-AUC, never instead of it [1]. The gap between a high ROC-AUC and a
low PR-AUC is itself a useful diagnostic of imbalance severity.

**Ranking / retrieval metrics** matter when matching is framed as *retrieval*:
for each record on side A, rank candidates on side B (the natural framing for
1:n matching, entity linking, blocking, and `match(A, B, k=...)` top-k output).

- **Recall@k** — is the true match among the top-k retrieved. Directly measures a
  blocker or a top-k matcher; this is what `equate`'s `k=` API should be scored
  with.
- **MRR (Mean Reciprocal Rank)** — 1/rank of the first correct match, averaged.
  Rewards putting the right answer first (entity-linking staple).
- **MAP (Mean Average Precision)** and **nDCG** — for the 1:n case with multiple
  correct matches per query, and when graded relevance matters.

These connect evaluation to blocking (§5-cross-link) and to the top-k design
`equate` already retains (see [`10-design-implications-for-equate.md`](10-design-implications-for-equate.md)).

---

## 5. Blocking / candidate-generation metrics

Blocking is evaluated *separately* from matching because its job is different:
shrink the O(n·m) space while keeping true matches reachable. The three standard
measures (detailed in
[`02-blocking-and-scalable-candidate-generation.md`](02-blocking-and-scalable-candidate-generation.md))
[9,20], where **C** = candidate pairs the blocker emits and **D** = true
duplicates:

- **Pair Completeness (PC)** = |D ∩ C| / |D|. *This is recall of the blocker.* It
  **upper-bounds the whole system's recall** — no matcher can recover a pair the
  blocker discarded — so blocking is tuned to keep PC ≈ 1.0.
- **Reduction Ratio (RR)** = 1 − |C| / (|A|·|B|). How much the comparison space
  shrank vs. brute force; RR → 1 means near-total pruning.
- **Pairs Quality (PQ)** = |D ∩ C| / |C|. *This is precision of the candidate
  set* — the fraction of emitted pairs that are real matches.

The fundamental tension: **PC (recall) trades against RR and PQ (efficiency)**.
Because the matcher can only *lose* recall, the received wisdom is to prioritize
PC in blocking and let the matcher improve precision downstream [9]. Composite
summaries exist (the harmonic mean of PC and RR is sometimes called **F*** or
just reported as a pair), but PC and RR should be reported *separately* — a
single blended number hides the operating point exactly the way F1 does.

> **Methodological trap (leakage via the gold set).** If your gold pairs were
> only labeled *among the candidates a specific blocker produced*, then PC
> measured against that gold set is definitionally 1.0 and meaningless. Blocking
> recall must be estimated against an *independently* constructed sample of true
> matches (§7.1).

---

## 6. The benchmark catalog

The field has converged on a handful of public suites. `equate` should ship a
**loader/registry** for the most-used ones (§8) so users can baseline their
matcher in one line. Below, grouped by task, with what each is good and bad for.

### 6.1 Magellan / DeepMatcher ER suite — the de facto standard

The workhorse benchmarks come from two UW-Madison efforts: the **Magellan**
project (`py_entitymatching`, Konda et al. 2016 [10]) which curated the datasets,
and **DeepMatcher** (Mudgal et al., SIGMOD 2018 [11]) which fixed the canonical
train/validation/test splits (typically 3:1:1) and the "structured / textual /
dirty" taxonomy that everyone still cites. The core tables [12]:

| Dataset | Domain | Type | Notes |
|---|---|---|---|
| **DBLP–ACM** | bibliographic | structured | clean; nearly *solved* (F1 > 98%) |
| **DBLP–Scholar** | bibliographic | structured/dirty | noisier bibliographic |
| **Amazon–Google** | software products | structured | harder; F1 ~70s historically |
| **Abt–Buy** | products | **textual** | long text attrs; a genuine hard case |
| **Walmart–Amazon** | electronics | structured/dirty | attribute misalignment |
| **Fodors–Zagat, iTunes–Amazon, Beer, DBLP-*, Company** | restaurants/music/misc | various | small; several are trivially easy |

**Strengths:** ubiquitous, small, fixed splits → perfect for regression tests and
apples-to-apples comparison. **Weaknesses:** most are *too easy* and *saturated*
(§7), balanced by construction (unrealistic base rate), single-domain, and
English-only. Treat them as **sanity checks and CI fixtures, not proof of
generalization**.

### 6.2 WDC Products — the modern, difficulty-controlled benchmark

**WDC Products** (Peeters, Der & Bizer, EDBT 2024 [13]) is the current
best-practice EM benchmark, built from real e-commerce data with schema.org
annotations. Its contribution is to make difficulty a *controlled variable*
along three dimensions: (i) **amount of corner-cases** (hard positives/negatives),
(ii) **generalization to unseen entities** (test entities absent from training),
and (iii) **development-set size**. It ships 9 train × 9 val × 9 test → 27
variants, and — crucially — provides **both a pair-wise and a multi-class
formulation of the same task**, and **strictly separates offers across
train/val/test so an offer appears in exactly one split** (a built-in defense
against the leakage of §7.2). This is the benchmark to reach for when the
question is "does my matcher generalize?" rather than "does it fit?".

### 6.3 Machamp & Alaska — generalized / multi-task matching

- **Machamp** (Wang, Li & Hirota, CIKM 2021 [14]): **Generalized Entity
  Matching (GEM)** — matching across *structured, semi-structured, and
  unstructured* records, not just two clean relational tables. Seven tasks with
  fixed splits; the benchmark for testing whether a matcher handles schema
  heterogeneity rather than assuming aligned attributes.
- **Alaska** (Crescenzi et al., 2021 [15]): ~70k product specs from 71
  e-commerce sites with thousands of heterogeneous attributes; the first
  **real-world** benchmark supporting *multiple* integration tasks — **schema
  matching *and* entity resolution** — on the same corpus, enabling honest
  end-to-end pipeline evaluation instead of stage-in-isolation numbers.

### 6.4 Valentine — the schema-matching benchmark

**Valentine** (Koutras et al., ICDE 2021 [16]) is an extensible experiment suite
for **schema matching / dataset discovery** — matching *columns* across tables,
the problem covered in [`07-schema-and-ontology-matching.md`](07-schema-and-ontology-matching.md).
It bundles baseline matchers (COMA, Cupid, Similarity Flooding,
Distribution-based, Jaccard-Levenshtein) and, importantly, a **dataset
fabricator** that generates matching scenarios (unionable / joinable / view /
semantically-joinable pairs) with controlled noise, plus effectiveness metrics
tailored to schema matching (**Precision@ground-truth-size**,
**Recall@ground-truth-size**, and their F1). It is the reference point for the
schema-alignment stage of `equate`.

### 6.5 OpenEA — knowledge-graph entity alignment

**OpenEA** (Sun et al., PVLDB 2020 [17]) benchmarks **embedding-based entity
alignment** across knowledge graphs (align entities between two KGs given seed
alignments). It ships a KG-sampling algorithm producing datasets with controlled
heterogeneity/degree distributions (e.g., EN-FR, EN-DE, D-W, D-Y at 15K/100K
sizes) and an open library of 12+ methods. The standard metrics here are
**Hits@k** and **MRR** (§4) — this is the ranking-metric world. Relevant to
`equate` if it grows a graph/relational-matching mode
([`03-assignment-and-graph-matching.md`](03-assignment-and-graph-matching.md)).

### 6.6 Benchmark selection guide

| If your task is… | Use | Metrics |
|---|---|---|
| CI sanity / regression | DBLP-ACM, Fodors-Zagat (easy, fixed splits) | pairwise F1 |
| "Does it generalize?" | **WDC Products** (unseen-entity variant) | pairwise F1, PR-AUC |
| Schema heterogeneity | Machamp, Alaska | pairwise F1 |
| Schema/column matching | **Valentine** | P@GT, R@GT |
| KG entity alignment | OpenEA | Hits@k, MRR |
| Dedup/clustering quality | any + gold partition | **B³ F1**, ARI |

---

## 7. Methodology: whether the numbers mean anything

Metrics and benchmarks are necessary but not sufficient. The methodology below is
what separates a trustworthy F1 from a press-release F1. The field's own
reckoning with this — Primpeli & Bizer's *Profiling* [18], the *Critical
Re-evaluation* of Papadakis et al. [19], and *Bridging the Gap* [21] — is
essentially a catalog of these mistakes made at scale.

### 7.1 Ground-truth construction

- **Pairs vs. partition.** Decide up front. If labeling clusters, the gold is a
  partition; derive pair labels by transitive closure. If labeling pairs, beware
  that a hand-labeled pair set may be *transitively inconsistent* (A–B and B–C
  labeled match, A–C labeled non-match) — resolve before computing cluster
  metrics.
- **The completeness problem.** You cannot label all n·m pairs. The dangerous
  shortcut is to label only pairs *some blocker surfaced* and treat unlabeled
  pairs as non-matches. This **caps measured recall at the blocker's recall** and
  bakes the blocker's bias into "ground truth." Mitigations: label a **uniform
  random sample** of the full pair space to estimate the true positive rate, or
  pool candidates from *multiple diverse* blockers before labeling.
- **Label noise & agreement.** Report inter-annotator agreement on a subset;
  corner-cases (the pairs that actually differentiate matchers) are where humans
  disagree most, so a benchmark's ceiling is its label quality.

### 7.2 Train/dev/test splitting and **leakage**

Leakage — test information contaminating training — is the single most common way
matching results are inflated. Matching has leakage modes ordinary ML does not:

- **Entity/record leakage (the big one).** If the *same record* (or the same
  underlying entity) appears in both the training and test pairs, a model can
  memorize it. A random split *over pairs* almost guarantees this, because one
  popular entity generates many pairs that scatter across splits. **Split over
  entities/records, not over pairs** — assign each record to exactly one split and
  derive pairs. This is precisely why WDC Products enforces per-offer split
  separation [13], and why "generalization to unseen entities" is a *separate*,
  harder axis than the standard split.
- **Model selection on the test set.** Choosing hyperparameters, the decision
  threshold, or the best epoch by test performance leaks. Li et al. flagged
  exactly this in prior EM work — doing model selection on the test set makes
  results "slightly unfit for comparison" [11]. Freeze everything on validation;
  touch test **once**.
- **Blocking leakage.** If blocking is fit on the full dataset (including test)
  before splitting, block statistics leak. Fit blocking on train only, or block
  within each split.
- **Preprocessing/embedding leakage.** IDF weights, learned encoders, or
  normalization statistics fit on all data leak the test distribution. Fit on
  train, apply to test.

### 7.3 Class imbalance — train balancing vs. test realism

DeepMatcher-style benchmarks are often **balanced** (≈1:1 match:non-match) for
training convenience, but the real world is ~1:100 to 1:10⁶. Two rules:

1. **You may balance (or oversample) the *training* set** to help a classifier
   learn — this is legitimate.
2. **You must *not* balance the *test* set** if you want production-representative
   precision. Precision is exquisitely sensitive to the negative base rate: a
   matcher with 90% precision at 1:1 can drop below 10% precision at 1:1000
   because false positives scale with the (huge) negative pool. Report the test
   base rate alongside precision, or report PR-AUC which is base-rate-explicit
   [1]. A matcher tuned and reported only on balanced data is a matcher whose
   production precision is unknown.

### 7.4 Estimating quality *without* full labels

For large corpora, exhaustive gold is impossible. Principled estimators exist and
`equate` should expose them:

- **Stratified / importance sampling of pairs** to estimate precision and recall
  with confidence intervals rather than point values. Barnes et al. give
  **performance bounds for pairwise ER** under sampling [22]; **OASIS** (Marchant
  & Rubinstein, PVLDB 2017 [23]) does *adaptive* importance sampling to estimate F1
  with far fewer labels than uniform sampling. These turn "we can't measure recall" into
  "we can measure recall ± ε with N labels."
- **Report confidence intervals**, not bare point estimates, whenever the gold set
  is a sample. A 0.02 F1 gap on a 200-pair test set is noise.

### 7.5 Transfer, generalization, and cross-dataset evaluation

A matcher's headline in-distribution F1 says little about a *new* dataset. The
honest questions:

- **Unseen entities** — test entities absent from training (WDC's dimension (ii)
  [13]).
- **Cross-dataset / zero-shot transfer** — train on dataset X, test on dataset Y
  with no Y labels. A 2025 EDBT deep-dive [24] finds most EM methods are *not*
  designed for this and are usually evaluated on a few arbitrarily-chosen held-out
  sets; it also reports the now-common finding that **generative LLMs generalize
  better zero-shot than fine-tuned PLMs, and that fine-tuning *reduces*
  cross-dataset generalization** — a direct trade-off worth surfacing to users.
  (Cross-links [`06-deep-learning-and-llm-entity-matching.md`](06-deep-learning-and-llm-entity-matching.md).)
- **Ablations.** To claim a component helps, ablate it and re-measure on the same
  split. Report the delta, not just the final number.
- **Significance.** With multiple runs (random seeds, split resamples), report
  mean ± std and a paired test; a 0.5-point F1 win inside run-to-run variance is
  not a win.

### 7.6 The "benchmarks are too easy" critique — take it seriously

A coherent body of recent work argues the standard suites overstate progress:

- **Primpeli & Bizer, *Profiling Entity Matching Benchmark Tasks*** (CIKM 2020)
  [18]: define profiling dimensions (corner-case density, schema complexity,
  textuality, development-set size) and show the popular tasks cluster into a few
  easy profiles — a matcher can "win" without handling the hard cases.
- **Papadakis, Kirielle, Christen & Palpanas, *A Critical Re-evaluation of
  Benchmark Datasets*** (2023) [19]: introduce *linearity* and *complexity*
  measures and show that **most popular datasets pose easy, near-linearly-separable
  tasks** — a *linear* matcher nearly matches a deep one on them, so they cannot
  discriminate methods; they propose harder tasks.
- **Wang et al., *Bridging the Gap between Reality and Ideality of Entity
  Matching*** (IJCAI 2022) [21]: standard benchmarks assume **closed-world
  (restricted entities), balanced labels, single-modality**; relaxing these to
  open entities, imbalanced labels, and multi-modal records makes SOTA methods
  degrade sharply — benchmarks "conceal the main challenges and overestimate
  progress."

**Design takeaway:** never let a user conclude "my matcher is good" from
DBLP-ACM alone. `equate`'s harness should (a) *report task difficulty* (linearity
/ corner-case profile) alongside the score, and (b) default the "generalization"
question to a hard, unseen-entity variant when one is available.

---

## 8. Design implications for `equate`

The corpus thesis is that `equate` owns *orchestration of swappable stages*
(featurize → block → compare → match) and wraps mature libraries at each seam
([`10-design-implications-for-equate.md`](10-design-implications-for-equate.md)).
Evaluation is a **fourth first-class subsystem** with the same shape: a small,
dependency-light core with strategy seams and optional heavy backends.

### 8.1 An `equate.evaluate` module (dependency-light core)

Ship pure-`numpy`/stdlib implementations of the metrics that have no heavy deps,
so *every* install can self-evaluate:

```python
from equate.evaluate import pairwise_prf, bcubed, cluster_metrics, blocking_metrics, pr_auc, ranking_metrics

pairwise_prf(pred_pairs, true_pairs)            # → Scores(precision, recall, f1, support)
bcubed(pred_labels, true_labels)                # → Scores(precision, recall, f1)   B³, the default cluster metric
cluster_metrics(pred_labels, true_labels)       # → {b3_f1, ari, v_measure, homogeneity, completeness, gmd}
blocking_metrics(candidate_pairs, true_pairs, n_a, n_b)  # → {pair_completeness, reduction_ratio, pairs_quality}
pr_auc(scores, y_true)                           # → average precision (threshold-free, imbalance-safe)
ranking_metrics(ranked_candidates_per_query, gold)  # → {recall_at_k, mrr, map}
```

Design rules, each grounded above:

- **Return precision *and* recall, never F1 alone** (§2.5). A frozen
  `Scores` dataclass with `precision, recall, f1, support` fields — SSOT for how
  a score is represented, so every matcher and metric speaks the same type.
- **Two granularities are explicit, not implicit** (§2.2). `pairwise_prf` takes
  pair sets; `bcubed`/`cluster_metrics` take label arrays (record → cluster id).
  A helper `pairs_to_labels(pairs)` (transitive closure via union-find) and its
  inverse make the conversion one call and force the user to *choose* a
  granularity.
- **B³ is the default cluster metric**, with ARI + V-measure + GMD as
  complements (§2.4), because B³ alone satisfies all of Amigó's constraints.
- **Threshold handling is a method, not a magic number.** Provide
  `select_threshold(scores, y_true, *, optimize='f1', on='validation')` that is
  *separate* from scoring, so tuning-on-test is something the user has to do on
  purpose, not by accident (§3, §7.2).
- **Reuse existing machinery.** CEAF is a maximum-weight assignment → route it
  through the same `linear_sum_assignment`/matcher seam `equate` already owns
  ([`03`](03-assignment-and-graph-matching.md)); don't reimplement Hungarian.
  ARI/V-measure can delegate to `sklearn.metrics` **iff** sklearn is present, else
  fall back to the numpy implementation — a lazy-import optional boundary.

### 8.2 A leakage-aware splitter

Make the *right* split the *easy* split (progressive disclosure):

```python
from equate.evaluate import entity_split

train, val, test = entity_split(records, pairs, *, by='entity', ratios=(0.6, 0.2, 0.2), unseen_test=True, seed=0)
```

- Split **over records/entities, not pairs** by default (§7.2) — this is the SSOT
  fix for the most common leakage. A `by='pair'` mode exists but *warns* it is
  leak-prone.
- `unseen_test=True` guarantees test entities are absent from train (the WDC
  generalization axis [13]).
- Provide `assert_no_leakage(train, test)` as a cheap post-hoc guard usable in CI.

### 8.3 A benchmark registry (optional-dependency, lazy-fetch)

A `equate.benchmarks` registry mapping names → loaders that download-and-cache on
first use (behind a `[benchmarks]` extra, never a hard dep):

```python
from equate.benchmarks import load, list_benchmarks

list_benchmarks()                 # ['dblp_acm', 'abt_buy', 'walmart_amazon', 'wdc_products', ...]
ds = load('wdc_products', variant='unseen_80pair')   # → Benchmark(train, val, test, gold, meta)
```

- Loaders return a uniform `Benchmark` container (train/val/test frames + gold
  pairs/labels + metadata incl. **base rate and difficulty profile**), so the
  same evaluation code runs across all suites.
- Cover the standard suites (§6): Magellan/DeepMatcher tables, WDC Products,
  Machamp, Valentine (schema), OpenEA (KG). Each behind a strategy so adding a
  benchmark is registering a loader, not editing core.
- **Report difficulty** (linearity / corner-case profile per [18,19]) in
  `Benchmark.meta` so a user sees "this benchmark is easy" next to their 98% F1.

### 8.4 A self-benchmark harness

One entry point that ties it together and is the honest-evaluation "happy path":

```python
from equate.evaluate import benchmark_matcher

report = benchmark_matcher(
    my_matcher,
    on=['dblp_acm', 'wdc_products'],   # CI sanity + generalization
    metrics=['pairwise_f1', 'pr_auc', 'b3_f1'],
    threshold='select_on_val',          # never on test
    n_seeds=5,                           # mean ± std, significance
    estimate_ci=True,                    # sampling-based CIs for large corpora (§7.4)
)
report.to_markdown()                     # per-benchmark, per-metric, with base rate + difficulty
```

Behaviors that encode the methodology as defaults: selects threshold on
validation, refuses to balance the test set, reports the test base rate,
runs multiple seeds with mean ± std, and prints benchmark difficulty next to the
score. The harness is where "measure honestly" becomes the path of least
resistance.

### 8.5 Optional-dependency boundaries (summary)

| Capability | Default (no heavy dep) | Optional backend |
|---|---|---|
| Pairwise P/R/F1, B³, PC/RR/PQ, PR-AUC, ranking | pure numpy/stdlib | — |
| ARI, V-measure, homogeneity/completeness | numpy fallback | `sklearn.metrics` [8] |
| CEAF assignment | `scipy.linear_sum_assignment` (already a dep) | — |
| Benchmark datasets | not installed | `equate[benchmarks]` (lazy download+cache) |
| CI estimators (OASIS-style sampling) | uniform-sample estimator | `equate[eval-extras]` |

**Guiding principle (progressive disclosure):** `pairwise_prf(pred, gold)` and
`bcubed(pred, gold)` must Just Work in the base install; every richer capability
(chance-corrected metrics, benchmark downloads, sampled CIs) is one optional
extra away; and the *honest* protocol — entity-split, threshold-on-val,
report-with-base-rate — is what the high-level harness does by default so users
fall into the pit of success.

---

## 9. Glossary

- **Precision (pairwise)** — |P ∩ G| / |P|; fraction of predicted matches that are
  correct.
- **Recall (pairwise)** — |P ∩ G| / |G|; fraction of true matches recovered.
- **F1 / Fβ** — harmonic mean of precision and recall (β weights recall).
- **B³ (B-cubed)** — per-record precision/recall averaged over records; the
  default cluster metric; satisfies all of Amigó's formal constraints.
- **CEAF** — Constrained Entity-Aligned F-measure; scores an optimal one-to-one
  alignment between predicted and gold clusters (mention `CEAFₘ` / entity `CEAFₑ`).
- **Generalized Merge Distance (GMD) / cluster-F** — minimum tunable-cost
  split/merge edits between predicted and gold partitions.
- **Adjusted Rand Index (ARI)** — chance-corrected count of pair agreements
  between partitions; ~0 random, 1 identical.
- **Homogeneity / Completeness / V-measure** — information-theoretic
  precision-analog / recall-analog / their harmonic mean for clusterings.
- **Pairwise vs. cluster metrics** — over the set of predicted *pairs* vs. over
  the induced *partition* into entities; they can rank matchers differently.
- **PR-AUC / Average Precision** — area under precision–recall curve;
  threshold-free, imbalance-safe (ignores true negatives).
- **ROC-AUC** — area under TPR–FPR curve; misleading under class imbalance.
- **Recall@k / MRR / MAP / nDCG** — ranking metrics for top-k / 1:n / retrieval
  framings.
- **Pair Completeness (PC)** — blocker recall: fraction of true matches kept in
  the candidate set; upper-bounds system recall.
- **Reduction Ratio (RR)** — fraction of the O(n·m) space a blocker pruned.
- **Pairs Quality (PQ)** — blocker precision: fraction of candidate pairs that are
  true matches.
- **Data leakage** — test information contaminating training (shared entities
  across split, model selection on test, blocking/preprocessing fit on all data);
  inflates reported performance.
- **Class imbalance / base rate** — the tiny fraction of pairs that are matches;
  drives the choice of metric and the train-balance-but-not-test rule.
- **Magellan / DeepMatcher benchmarks** — UW-Madison curated ER tables (DBLP-ACM,
  Abt-Buy, Amazon-Google, Walmart-Amazon…) with fixed splits.
- **WDC Products** — difficulty-controlled EM benchmark varying corner-cases,
  unseen entities, and dev-set size; both pairwise and multi-class formulations.
- **Machamp / Alaska** — generalized (structured/semi/unstructured) and
  multi-task (schema matching + ER) real-world benchmarks.
- **Valentine** — schema-matching / dataset-discovery benchmark suite with a
  dataset fabricator; metrics are Precision@ground-truth and Recall@ground-truth.
- **OpenEA** — KG entity-alignment benchmark; metrics Hits@k and MRR.

---

## References

[1] Christophides V, Efthymiou V, Palpanas T, Papadakis G, Stefanidis K. *An
Overview of End-to-End Entity Resolution for Big Data.* ACM Computing Surveys
53(6):127, 2020.
[https://dl.acm.org/doi/10.1145/3418896](https://dl.acm.org/doi/10.1145/3418896)
(open access:
[hal.science/hal-02955445](https://hal.science/hal-02955445/document))

[2] Binette O, Steorts RC. *(Almost) All of Entity Resolution.* Science Advances
8(12):eabi8021, 2022 (arXiv:2008.04443).
[https://arxiv.org/abs/2008.04443](https://arxiv.org/abs/2008.04443)

[3] Menestrina D, Whang SE, Garcia-Molina H. *Evaluating Entity Resolution
Results.* Proceedings of the VLDB Endowment 3(1):208–219, 2010 (pairwise vs.
cluster F, Generalized Merge Distance).
[http://vldb.org/pvldb/vol3/R18.pdf](http://vldb.org/pvldb/vol3/R18.pdf)

[4] Amigó E, Gonzalo J, Artiles J, Verdejo F. *A Comparison of Extrinsic
Clustering Evaluation Metrics Based on Formal Constraints.* Information Retrieval
12:461–486, 2009 (B³ vs. Rand vs. others; the four constraints).
[https://dblp.org/rec/journals/ir/AmigoGAV09a.html](https://dblp.org/rec/journals/ir/AmigoGAV09a.html)

[5] Zheng Y, et al. *Machine Learning for Coreference Resolution: survey of
metrics* (MUC, B³ [Bagga & Baldwin 1998], CEAF [Luo 2005], BLANC).
[https://www.cs.cmu.edu/~yimengz/papers/Coreference_survey.pdf](https://www.cs.cmu.edu/~yimengz/papers/Coreference_survey.pdf)

[6] Recasens M, Hovy E. *BLANC: Implementing the Rand Index for Coreference
Evaluation* (Rand-index-style coreference metric; context for CEAF/B³).
[https://www.cs.cmu.edu/~hovy/papers/10BLANC-coref-metric.pdf](https://www.cs.cmu.edu/~hovy/papers/10BLANC-coref-metric.pdf)

[7] Hubert L, Arabie P. *Comparing Partitions.* Journal of Classification
2:193–218, 1985 (Adjusted Rand Index). DOI:
[https://doi.org/10.1007/BF01908075](https://doi.org/10.1007/BF01908075)

[8] scikit-learn. *Metrics and scoring: quantifying the quality of predictions*
(ARI, homogeneity/completeness/V-measure, average_precision_score, ROC-AUC).
[https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

[9] Papadakis G, Skoutas D, Thanos E, Palpanas T. *Blocking and Filtering
Techniques for Entity Resolution: A Survey.* ACM Computing Surveys 53(2):31,
2020 (PC/RR/PQ definitions).
[https://dl.acm.org/doi/abs/10.1145/3377455](https://dl.acm.org/doi/abs/10.1145/3377455)

[10] Konda P, Das S, Suganthan G C P, Doan A, et al. *Magellan: Toward Building
Entity Matching Management Systems.* Proceedings of the VLDB Endowment
9(12):1197–1208, 2016.
[http://www.vldb.org/pvldb/vol9/p1581-konda.pdf](http://www.vldb.org/pvldb/vol9/p1581-konda.pdf)
· repo:
[github.com/anhaidgroup/py_entitymatching](https://github.com/anhaidgroup/py_entitymatching)

[11] Mudgal S, Li H, Rekatsinas T, Doan A, Park Y, Krishnan G, Deep R, Arcaute E,
Raghavendra V. *Deep Learning for Entity Matching: A Design Space Exploration.*
SIGMOD 2018 (DeepMatcher; canonical splits & structured/textual/dirty taxonomy).
[https://dl.acm.org/doi/10.1145/3183713.3196926](https://dl.acm.org/doi/10.1145/3183713.3196926)

[12] anhaidgroup. *DeepMatcher benchmark datasets* (Abt-Buy, Amazon-Google,
DBLP-ACM, DBLP-Scholar, Walmart-Amazon, Fodors-Zagat, iTunes-Amazon, …).
[https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)

[13] Peeters R, Der R C, Bizer C. *WDC Products: A Multi-Dimensional Entity
Matching Benchmark.* EDBT 2024 (arXiv:2301.09521).
[https://arxiv.org/abs/2301.09521](https://arxiv.org/abs/2301.09521)
· data:
[webdatacommons.org/largescaleproductcorpus/wdc-products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/)

[14] Wang J, Li Y, Hirota W. *Machamp: A Generalized Entity Matching Benchmark.*
CIKM 2021 (arXiv:2106.08455).
[https://dl.acm.org/doi/10.1145/3459637.3482008](https://dl.acm.org/doi/10.1145/3459637.3482008)
· repo:
[github.com/megagonlabs/machamp](https://github.com/megagonlabs/machamp)

[15] Crescenzi V, De Angelis A, Firmani D, Mazzei M, Merialdo P, Piai F,
Srivastava D. *Alaska: A Flexible Benchmark for Data Integration Tasks* (schema
matching + entity resolution on one real-world corpus). arXiv:2101.11259, 2021.
[https://arxiv.org/abs/2101.11259](https://arxiv.org/abs/2101.11259)

[16] Koutras C, Siachamis G, Ionescu A, Psarakis K, Brons J, Fragkoulis M, Lofi
C, Bonifati A, Katsifodimos A. *Valentine: Evaluating Matching Techniques for
Dataset Discovery.* IEEE ICDE 2021:468–479.
[https://github.com/delftdata/valentine](https://github.com/delftdata/valentine)
· demo (PVLDB 14):
[http://vldb.org/pvldb/vol14/p2871-koutras.pdf](http://vldb.org/pvldb/vol14/p2871-koutras.pdf)

[17] Sun Z, Zhang Q, Hu W, Wang C, Chen M, Akrami F, Li C. *A Benchmarking Study
of Embedding-based Entity Alignment for Knowledge Graphs.* Proceedings of the
VLDB Endowment 13(11):2326–2340, 2020 (OpenEA; Hits@k, MRR).
[https://dblp.org/rec/journals/pvldb/SunZHWCAL20.html](https://dblp.org/rec/journals/pvldb/SunZHWCAL20.html)
· repo:
[github.com/nju-websoft/OpenEA](https://github.com/nju-websoft/OpenEA)

[18] Primpeli A, Bizer C. *Profiling Entity Matching Benchmark Tasks.* CIKM 2020
(profiling dimensions; benchmarks cluster into easy profiles).
[https://dl.acm.org/doi/10.1145/3340531.3412781](https://dl.acm.org/doi/10.1145/3340531.3412781)
· PDF:
[uni-mannheim.de/…/CIKM2020_Primpeli_Bizer.pdf](https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Research/Web-based_Systems/pub/CIKM2020_Primpeli_Bizer.pdf)

[19] Papadakis G, Kirielle N, Christen P, Palpanas T. *A Critical Re-evaluation of
Benchmark Datasets for (Deep) Learning-Based Matching Algorithms.* arXiv:2307.01231,
2023 (linearity/complexity; most datasets are too easy).
[https://arxiv.org/abs/2307.01231](https://arxiv.org/abs/2307.01231)

[20] Papadakis G, Ioannou E, Thanos E, Palpanas T. *The Four Generations of
Entity Resolution.* Synthesis Lectures on Data Management, Morgan & Claypool,
2021 (evaluation chapter; benchmark lineage).
[https://dblp.org/rec/series/synthesis/2021Papadakis.html](https://dblp.org/rec/series/synthesis/2021Papadakis.html)
· companion:
[entityresolution4g.com](https://entityresolution4g.com/)

[21] Wang T, Lin H, Fu C, Han X, Sun L, Xiong F, Chen H, Lu M, Zhu X. *Bridging
the Gap between Reality and Ideality of Entity Matching: A Revisiting and
Benchmark Re-Construction.* IJCAI 2022 (arXiv:2205.05889) (open entities,
imbalanced labels, multi-modality).
[https://arxiv.org/abs/2205.05889](https://arxiv.org/abs/2205.05889)

[22] Barnes M, Miller K, Dubrawski A. *Performance Bounds for Pairwise Entity
Resolution.* arXiv:1509.03302, 2015 (estimating pairwise P/R under sampling).
[https://arxiv.org/abs/1509.03302](https://arxiv.org/abs/1509.03302)

[23] Marchant NG, Rubinstein BIP. *In Search of an Entity Resolution OASIS:
Optimal Asymptotic Sequential Importance Sampling.* Proceedings of the VLDB
Endowment 10(11):1322–1333, 2017 (adaptive F-measure estimation with few labels).
[http://www.vldb.org/pvldb/vol10/p1322-rubinstein.pdf](http://www.vldb.org/pvldb/vol10/p1322-rubinstein.pdf)

[24] Zhang Z, Groth P, Calixto I, Schelter S. *A Deep Dive Into Cross-Dataset
Entity Matching with Large and Small Language Models.* EDBT 2025 (cross-dataset
generalization; fine-tuning reduces transfer).
[https://openproceedings.org/2025/conf/edbt/paper-224.pdf](https://openproceedings.org/2025/conf/edbt/paper-224.pdf)
