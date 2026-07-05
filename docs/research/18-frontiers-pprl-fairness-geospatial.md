# Responsible & Specialized Frontiers: PPRL, Fairness, Geospatial & Uncertainty

*Round-2 gap-filling facet for the `equate` redesign corpus in `docs/research/`.
Where the round-1 docs (`00`–`10`) build the core matching pipeline, this doc covers
four **specialized-but-load-bearing** frontiers that a *general* matching framework
must not ignore: matching **without seeing the data** (privacy-preserving record
linkage), matching **without discriminating** (fairness), matching **things that live
in space and move** (geospatial/trajectory), and matching that **honestly reports how
sure it is** (uncertainty quantification). Read
[`01-entity-resolution-record-linkage.md`](01-entity-resolution-record-linkage.md)
and [`02-blocking-and-scalable-candidate-generation.md`](02-blocking-and-scalable-candidate-generation.md)
first; this doc extends them.*

## Abstract

A framework that decides which objects *correspond* inherits four duties its
individual algorithms do not: to link records **without either party seeing the
other's raw identifiers** (PPRL), to **not encode disparate error rates** into
protected groups (fairness), to exploit the **metric structure of space and time**
when the objects are places or trajectories (geospatial matching), and to **quantify
and propagate** its own uncertainty rather than emitting hard pairs as if they were
facts (UQ). This document surveys the canonical methods in each frontier — Bloom-filter
(CLK) encoding and its cryptanalysis, secure multiparty computation and differential
privacy; per-group error-rate auditing and name-matching bias; grid/geohash/R-tree/S2/H3
spatial-join blocking, POI conflation, and Hidden-Markov map matching (Newson–Krumm);
and calibrated match probabilities with downstream linkage-error propagation — and states,
per frontier, what `equate` should support **natively** versus behind **optional-dependency
extension points**, closing with the ethical duty-of-care a matching library owes its users.

---

## 1. Orientation: four frontiers, one duty of care

The round-1 corpus optimizes the *mechanics* of matching. These four frontiers are
about the **consequences** of matching, and they share a structural property: each is a
*cross-cutting concern* that reshapes one of the three canonical stages
([`00-taxonomy-and-terminology.md`](00-taxonomy-and-terminology.md)) rather than adding
a new one.

| Frontier | Reshapes stage | Core question | `equate` posture |
|---|---|---|---|
| **PPRL** | Featurize (encode so raw data never leaves) | Can two parties link without seeing each other's PII? | Extension point (encoder + protocol); never a hard dep |
| **Fairness** | Match/decide + Evaluate | Are error rates equal across protected groups? | Native audit hooks; group-aware evaluation |
| **Geospatial** | Featurize + Block + Compare | Objects are places/paths with metric structure | Native decay comparators + blocker seam; heavy libs optional |
| **Uncertainty** | Compare/Classify + downstream | How sure is each match, and what does that do to my analysis? | Native calibration + typed probabilities; propagation as extension |

The unifying **duty of care**: a general matching library is *infrastructure*. Its
defaults become other people's production systems. Silent choices — an uncalibrated
score treated as a probability, a name comparator tuned only on Anglo names, a plaintext
join key logged to disk — become silent harms downstream. The design stance this doc
argues for is therefore not "implement everything" but **make the responsible choice the
easy one, and make the irresponsible one impossible-by-accident**: typed probabilities
that advertise whether they are calibrated, evaluation utilities that *require* a group
label to be *optionally* supplied, and encoders that carry their own privacy metadata.

---

## 2. Privacy-Preserving Record Linkage (PPRL)

**PPRL** is record linkage across databases held by *different parties* such that no
party learns anything about the other's records beyond the fact of a match (and often
not even that) — no plaintext quasi-identifiers (names, birth dates, addresses) are ever
exchanged [1,2]. It is mandatory in health, census, crime/fraud, and national-security
data integration, where the same person must be tracked across custodians who are legally
barred from sharing raw PII [2].

### 2.1 The parties and the threat model

The canonical setting is **two (or more) database owners** plus, usually, a **linkage
unit (LU)** — a third party that performs the comparison on *encoded* records and returns
only matched record-id pairs [1,2]. Vatsalan, Christen & Verykios's **taxonomy** [1]
organizes the field along ~15 dimensions grouped into privacy aspects, linkage
techniques, theoretical analysis, evaluation, and practical aspects; the follow-up
big-data survey [2] tracks the "generations" of methods from exact-hash → SMC →
perturbation → **Bloom-filter encoding**. The dominant threat model is the
**honest-but-curious** (semi-honest) adversary: parties follow the protocol but try to
infer PII from what they see. Malicious and collusion models exist but are costlier to
defend.

### 2.2 Bloom-filter encoding (Cryptographic Long-term Keys)

The workhorse of practical PPRL is **Bloom-filter (BF) encoding**, introduced by
**Schnell, Bachteler & Reiher (2009)** [3]. The construction:

1. Split each attribute value into a set of **q-grams** (e.g. bigrams of `"peter"` →
   `{_p, pe, et, te, er, r_}`).
2. Hash each q-gram with `k` independent keyed hash functions into an `l`-bit array,
   setting those bit positions to 1.
3. Compare two BFs with a **token-set similarity on bits** — the **Dice coefficient**
   `2·|A∧B| / (|A| + |B|)` (popcount of the AND over the popcounts) — which *approximates*
   the q-gram Jaccard/Dice similarity of the underlying strings, so **approximate
   (typo-tolerant) matching survives encoding**. This is the key property a plain
   cryptographic hash lacks.

A **Cryptographic Long-term Key (CLK)** packs *all* identifying attributes of a record
into a *single* BF (via record-level or field-weighted hashing), yielding one bitstring
per entity [3]. Comparison is O(`l`) popcount per pair (word-parallel, cache-friendly);
blocking still applies (§2.4, and the Hamming/LSH machinery of
[`02-blocking-and-scalable-candidate-generation.md`](02-blocking-and-scalable-candidate-generation.md)).
Practical `l` is 500–1000 bits, `k`≈15–30, `q`=2. This is the encoding shipped by the
Australian **anonlink/clkhash** stack (§2.6).

### 2.3 Attacks and hardening

BFs are **not** cryptographically secure; they leak frequency structure. **Cryptanalysis
attacks** re-identify encoded values by (a) matching the frequency of set bit-patterns to
the known frequency of common names/values, or (b) **pattern mining** co-occurring bit
positions that correspond to frequent q-grams [4]. Christen, Schnell, Vatsalan &
Ranbaduge's **"Efficient Cryptanalysis of Bloom Filters"** [4] demonstrated an attack
that is *independent of the encoding function and its parameters*, re-identifies values
from large real databases in **seconds-to-minutes**, and succeeds *even against several
hardening techniques*.

**Hardening techniques** (each trading similarity-preservation for privacy) include:
**salting** with record-specific keys (defeats global frequency but breaks cross-record
comparability unless the salt is a shared blocking key), **balancing** (concatenate the
negated filter so every BF has the same Hamming weight), **XOR-folding** (fold the array
to halve length and destroy positional structure), **random noise / bit-flipping**
(BLIP), **rule-90 / diffusion**, and **record-level BF** re-hashing. Ranbaduge &
Schnell's **"Securing Bloom Filters"** [5] surveys and evaluates these. The authoritative
book-length treatment is **Christen, Ranbaduge & Schnell, *Linking Sensitive Data*
(Springer 2020)** [6], whose later chapters are devoted entirely to Bloom-filter encoding,
comparison, and the attack/harden arms race. **Design consequence for `equate`:** BF
encoding is *usable* but *dangerous by default*; any BF encoder must ship with
attack-aware parameter guidance and hardening as first-class options, not an afterthought.

### 2.4 Secure multiparty computation & private set intersection

**Secure multiparty computation (SMC/MPC)** lets parties jointly compute a function of
their private inputs while learning *only the output* [2,7]. The linkage-relevant
primitive is **private set intersection (PSI)**: two parties compute the intersection of
their sets (e.g. exact-match join keys) without revealing non-matching elements [7].
PSI variants build on oblivious transfer, oblivious pseudo-random functions, or
**homomorphic encryption** (compute on ciphertext). SMC gives **provable** privacy
(unlike BFs) but historically at high computational/communication cost; modern OT-based
PSI is near-linear and practical for millions of records. A common hybrid is **LSH-blocking
+ PSI on candidate keys** to cut the quadratic comparison cost while keeping cryptographic
guarantees [7]. Tradeoff axis: **BF encoding = fast, approximate, attackable; SMC/PSI =
provable, slower, and classically limited to *exact* set membership** (fuzzy SMC matching
is an active research area).

### 2.5 Differential privacy in linkage

**Differential privacy (DP)** bounds how much any single record can change an algorithm's
output distribution (parameter `ε`), giving a formal, composable privacy guarantee.
Applied to PPRL, DP is used to perturb encodings or released similarity statistics
[2,8]. A crucial caveat, argued in *"The use of differential privacy for PPRL:
Protecting the bits but not the people"* [8]: naïvely DP-perturbing an *encoding* can
satisfy the DP definition **on the bits** while still allowing **re-identification of the
people**, because the linkage task's utility requires preserving exactly the similarity
signal DP is meant to blur. The honest position: **DP and record linkage are in tension** —
linkage needs the signal that DP destroys — so DP in PPRL is best applied to *downstream
released aggregates*, not as a drop-in encoder. This connects directly to §5's downstream
error propagation.

### 2.6 Python tooling

The reference open-source stack is from **CSIRO's Data61**:

| Tool | Role | Notes |
|---|---|---|
| **`clkhash`** | Client: PII → CLK Bloom filters | Schema-driven field hashing; the *encoder* |
| **`anonlink`** | Server: similarity + greedy solving over CLKs | popcount Dice + candidate selection |
| **`anonlink-client` / `anonlink-entity-service`** | End-to-end multi-party service | REST service, blocking support |

These implement exactly the §2.2 CLK pipeline. **For `equate`**, they are the natural
`equate[pprl]` extension: a `clkhash`-style featurizer producing bitstrings, comparable
by the *same* Hamming/Dice comparator and Hamming-LSH blocker `equate` already needs for
perceptual hashes ([`04-featurization-and-representation.md`](04-featurization-and-representation.md),
[`02-blocking-and-scalable-candidate-generation.md`](02-blocking-and-scalable-candidate-generation.md)).
PPRL is thus **mostly a featurize-stage plug-in over existing seams**, not a parallel
engine — with the important addition of privacy metadata and hardening.

---

## 3. Fairness & bias in matching

Matching decisions allocate real outcomes — which patients are counted, which claims are
paid, which people are flagged. When the *error rate* of matching differs systematically
across demographic groups, the system encodes **disparate impact** even if no protected
attribute is used as a feature.

### 3.1 How matching becomes unfair

The mechanism is subtle: matching error is driven by **string/feature statistics that
correlate with demographics**. Records from some groups match at **significantly lower or
higher rates** than others, with direct consequences (under/over-counting prevalence in a
population, systematically dropping people from a linked cohort) [9]. Sources of disparate
error include: **name diversity** (a comparator tuned on high-frequency Anglo names
mis-scores names from under-represented cultures), **data quality asymmetry** (some groups
have sparser or noisier records), **transliteration/romanization** variance, and **name
change** patterns correlated with gender (marriage) and migration. Because blocking
*silently drops* pairs it never generates, **unfair recall loss can happen in the blocker,
invisibly**, before any comparator runs — a first-class reason to make blocking auditable
([`02-blocking-and-scalable-candidate-generation.md`](02-blocking-and-scalable-candidate-generation.md)).

### 3.2 Name-matching bias (cultural & gender)

Name matching is where bias concentrates. **Name-ethnicity classifiers** show large
accuracy disparities: *"Equal accuracy for Andrew and Abubakar"* [10] demonstrates that
popular name-ethnicity tools (EthnicityEstimator, NamePrism, Ethnicolr) have accuracy
biases tied to **global naming conventions and the training-name distribution** — a
classifier balanced on ethnicity may be unbalanced on gender/age, and vice-versa.
Purpose-built fair name matchers now exist: **FairNM** [11] reduces bias via token-based
similarity, a Siamese-network short-name module, and name weighting, improving fairness
while keeping fuzzy-match accuracy. **Design consequence:** a general framework must not
hard-code one culture's name model as *the* default string comparator; name comparison is
a **swappable, auditable strategy** with documented bias characteristics, and phonetic
encoders (Soundex/NYSIIS/Double-Metaphone) carry known Anglocentric assumptions that must
be *stated*, not hidden.

### 3.3 Fairness metrics & auditing for entity resolution

The state-of-the-art audit framework is **"Through the Fairness Lens: Experimental
Analysis and Evaluation of Entity Matching"** (Shahbazi, Danevski, Nargesian, Asudeh &
Srivastava, PVLDB 2023) [12], with the companion tool **FairEM360** (PVLDB 2024) [13].
Their contribution: adapt classification-fairness measures to the *pairwise* ER setting
by conditioning standard metrics on the demographic group of the records in each pair.
Key measures:

- **Accuracy/error parity** — equal F1 (or match accuracy) across groups.
- **True-Positive-Rate parity (Equal Opportunity)** — among *true* matches, equal
  recall across groups (do we find real duplicates equally well for everyone?).
- **False-Positive / False-Negative rate parity** — equal wrongful-merge and wrongful-split
  rates.
- **Equalized odds** — TPR *and* FPR parity jointly.

FairEM360 [13] operationalizes this as a *suite*: it measures per-group ER performance,
flags disparities, and helps diagnose their source. A subtlety unique to ER: pairs span
*two* records, so "the group of a pair" needs a policy (both-same-group, either, or
cross-group pairs as their own stratum) — a design choice the audit tool must expose.

### 3.4 Mitigation

Mitigation strategies, in rough order of invasiveness: **(a) group-specific thresholds** —
tune the match cutoff per group so error rates equalize (post-processing; effective and
model-agnostic, studied in [12] and *"Fairness and Cost Constrained Privacy-Aware Record
Linkage"* [14]); **(b) group-aware training** — per-group models or reweighted training
data; **(c) fair representation / comparator design** — as in FairNM [11]; **(d) fair
blocking** — ensure recall parity in candidate generation, not just in the final decision.
Threshold tuning is the cheapest lever and composes naturally with a framework that keeps
**scores and thresholds explicit** rather than baking in a single global cutoff.

### 3.5 The fairness–privacy interaction

Fairness and privacy are **entangled**: PPRL encodings destroy exactly the attributes
needed to *audit* fairness, and fairness mitigations may need group labels that privacy
forbids collecting. *"Fairness and Cost Constrained Privacy-Aware Record Linkage"* [14]
studies this three-way tension explicitly. `equate`'s honest posture is to **surface the
tradeoff** (audit hooks that accept optional group labels *when the user is allowed to
supply them*) rather than pretend it away.

---

## 4. Geospatial & trajectory matching

When the objects are **places** (POIs) or **paths** (GPS traces), matching gains a
metric structure that both *helps* (space is a natural, cheap blocking key) and *demands*
specialized methods (a trajectory matches a road *network*, not another trajectory). This
is a specialization of the featurize/block/compare stages, not a new pipeline.

### 4.1 Spatial-join blocking

The spatial analogue of blocking ([`02-blocking-and-scalable-candidate-generation.md`](02-blocking-and-scalable-candidate-generation.md))
is the **spatial join**: pair only objects that are near in space. Two families:

**(a) Tree indexes.** The **R-tree** (Guttman, 1984) [15] groups nearby geometries into
hierarchically nested **minimum bounding rectangles**; a spatial join is a synchronized
traversal (or plane-sweep) that visits only overlapping MBRs, giving roughly **O(log n)**
point/range queries and near-linear joins in practice. **STR bulk-loading** and the
**STRtree** variant give fast static indexes. This is what **GeoPandas `sjoin`/`sjoin_nearest`**
use under the hood (via Shapely's `STRtree`/GEOS) with a `predicate` (`intersects`,
`within`, `dwithin`, …) [16]. Quadtrees are the space-partitioning cousin.

**(b) Discrete global grids (tessellation keys).** Impose a grid and use the **cell id as
a blocking key** — objects sharing (or adjacent to) a cell are candidates. This turns
spatial blocking into *exactly the standard-blocking machinery* `equate` already has, with
a spatial key function:

| System | Cell shape / curve | Hierarchy | Property | Watch-out |
|---|---|---|---|---|
| **Geohash** | Base-32 Z-order (Morton) on lat/lon | prefix = containment | 1-D sortable key, B-tree-friendly | **edge effect**: near points across a cell boundary get different prefixes → must probe 8 neighbors |
| **S2** (Google) | Cube faces + **Hilbert curve**, 30 levels | yes | strong locality (Hilbert), cell-covering of any region; level ~16 ≈ 1 km² | square cells → non-uniform neighbor distances |
| **H3** (Uber) | **Hexagons** (icosahedron; 12 pentagons), 16 resolutions | yes (approx.) | uniform neighbor distance, low directional bias | not perfectly hierarchical/containment (hex can't tile hex exactly) |

Geohash is simplest and universally supported; S2 excels at region-covering queries; H3's
hexagons give the most consistent adjacency for aggregation/mobility analysis. **For
`equate`, all three are just injectable `key_fn`s** feeding the existing keyed blocker —
the "edge effect" is handled by a key function that emits a cell *and its neighbors*
(redundancy-positive blocking, cf. [`02`](02-blocking-and-scalable-candidate-generation.md)).

**Compare stage for coordinates:** the natural comparator is a **distance-decay** function
over **great-circle (haversine)** distance — step / linear / exponential / Gaussian decay —
already anticipated as `numeric_geo.py` in
[`10-design-implications-for-equate.md`](10-design-implications-for-equate.md).

### 4.2 POI matching / conflation

**POI (point-of-interest) conflation** merges place records (name, address, category,
lat/lon, description) from multiple sources — the geospatial face of entity resolution
[17,18]. The distinguishing feature is **multi-modal comparison**: a match needs *both*
spatial proximity *and* textual/semantic agreement (a "Starbucks" 5 m from another
"Starbucks" is a match; two different shops in the same mall are not). The systematic
review by Low et al. [17] catalogs matching methods across name similarity, address
parsing, category ontologies, and spatial distance, combined by weighted rules or learned
classifiers. Modern end-to-end frameworks [18] add transformer-based text encoders over
name/address attributes fused with a spatial score. **This maps cleanly onto `equate`'s
multi-field comparison-vector model** ([`05-comparison-and-similarity-functions.md`](05-comparison-and-similarity-functions.md)):
a `{name: string_sim, addr: string_sim, geo: haversine_decay, category: set_sim}` field
map with a pluggable combiner — POI matching is a *configuration*, not a new engine.

### 4.3 Trajectory & map matching

**Map matching** snaps a noisy, time-stamped GPS trajectory onto the road *network* —
recovering the actual route driven. The canonical method is **Hidden-Markov map matching**
by **Newson & Krumm (ACM SIGSPATIAL GIS 2009)** [19]: model the road segments as **hidden
states** and GPS fixes as **noisy observations**, with

- **emission probability** ∝ a Gaussian over the great-circle distance from the GPS point
  to a candidate road segment (measurement noise), and
- **transition probability** ∝ an exponential over the difference between the great-circle
  distance and the on-road (routing) distance between consecutive candidates (penalizing
  physically implausible jumps),

then decode the most-likely segment sequence with the **Viterbi** algorithm —
**O(T·N²)** for T fixes and N candidate segments per fix. The HMM formulation elegantly
handles both **noise** and **sparseness** (low sampling rate), and is now the dominant
paradigm. Open-source implementations: **FMM (Fast Map Matching)** — Yang & Gidófalvi,
*IJGIS* 32(3):547-570, 2018 [20] — combines the HMM with **precomputation** of an
upper-bounded origin–destination routing table for large speedups, and ships **Python and
C++ APIs**; **Valhalla's Meili** and **GraphHopper**, **OSRM**, and **Barefoot** are
production HMM-based matchers. **For `equate`, map matching is out-of-core**: it is
sequence-to-network alignment with a routing engine and a road graph, not
collection-to-collection object matching. The right posture is an **extension point that
delegates to a real map-matcher** (e.g. `equate[geo]` → FMM), while `equate` natively owns
only the *generic* HMM/Viterbi *sequence-alignment* primitive if a shared abstraction
emerges — but does **not** ship a road-network engine.

---

## 5. Uncertainty quantification

A match score is not a probability, and a set of hard pairs is not the truth. A responsible
matching framework must (a) turn scores into **calibrated** probabilities and (b) let that
uncertainty **flow downstream** into whatever analysis consumes the linked data.

### 5.1 Calibrated match probabilities

A classifier or similarity is **calibrated** if, among pairs it scores near `p`, a fraction
≈ `p` are truly matches. Raw cosine similarities, edit ratios, and even ML classifier
outputs are usually **mis-calibrated**, so thresholding them, or feeding them to an
optimizer that treats them as probabilities, is unsound. Standard fixes (already flagged as
`compare/calibrate.py` in [`10`](10-design-implications-for-equate.md) and prerequisite to
interactive triage in [`08-interactive-active-learning-and-hitl.md`](08-interactive-active-learning-and-hitl.md)):

- **Platt scaling** (logistic fit on scores), **isotonic regression** (non-parametric,
  monotone), **beta calibration** (better for bounded [0,1] scores).
- The **Fellegi–Sunter** probabilistic model already yields a *principled* match weight
  from per-field **m-probabilities** (agreement given a true match) and **u-probabilities**
  (agreement given a non-match); its posterior is interpretable *if* m/u are well-estimated
  (EM or labeled data) [21]. **Splink** operationalizes exactly this with calibration
  diagnostics [21].
- **False-Match Rate (FMR)** and **False-Non-Match Rate (FNMR)** are the linkage-quality
  duals of precision/recall and the natural knobs for a **three-way decision** (match /
  non-match / **possible-match → clerical review**), the Fellegi–Sunter abstain region.

Calibration is a **native** `equate` responsibility because everything downstream — review
triage, thresholding, optimal assignment cost, uncertainty propagation — assumes the score
means what it says.

### 5.2 Propagating linkage error downstream

The deepest issue: when linked data feeds a **downstream analysis** (a regression, a
prevalence estimate, a training set), **linkage errors bias that analysis**, and ignoring
them understates uncertainty. Two literatures address it:

- **Correction-based** (frequentist): model the **linkage error rates** and correct the
  downstream estimator. **Kim & Chambers** [22] and the review by **Wang, Kim et al.** [23]
  give bias-corrected regression under (possibly correlated) linkage error, using estimated
  false-match/false-non-match rates as exchangeable error parameters.
- **Joint Bayesian** (propagation): model linkage *and* the downstream task jointly so
  linkage uncertainty flows into the posterior **exactly**. **Generalized Bayesian Record
  Linkage with Exact Error Propagation** [24] and the broad survey **"(Almost) All of Entity
  Resolution"** (Binette & Steorts) [25] cover Bayesian ER, its exact uncertainty
  propagation, and a feedback mechanism that de-biases the downstream estimate. A pragmatic
  middle path is **multiple imputation** over the linkage (sample several plausible linkages,
  analyze each, pool the variance).

**For `equate`:** the framework cannot own every downstream model, but it can make
propagation *possible* by (a) never discarding the probability, (b) emitting **top-k
candidates with calibrated scores** (the `CandidateStore` of
[`10`](10-design-implications-for-equate.md)) so downstream code can marginalize or
multiply-impute, and (c) exposing estimated **FMR/FNMR** from the evaluation utility. The
anti-pattern to prevent: a `match()` that returns hard pairs and throws the uncertainty
away.

---

## 6. Glossary

- **PPRL (privacy-preserving record linkage)** — linking records across parties without
  revealing raw quasi-identifiers; only matched id-pairs are learned [1,2].
- **Bloom-filter encoding / CLK** — q-grams of identifiers hashed into a bit array;
  Dice-coefficient of bit vectors approximates string similarity, enabling *fuzzy* matching
  on encoded data [3].
- **Cryptanalysis (of BFs)** — re-identifying encoded values via bit-frequency or pattern
  mining; BFs are not cryptographically secure [4].
- **Hardening** — salting/balancing/XOR-folding/noise added to a BF to resist cryptanalysis,
  trading similarity fidelity for privacy [5,6].
- **Secure multiparty computation (SMC/MPC)** — jointly compute a function of private inputs,
  revealing only the output [2,7].
- **Private set intersection (PSI)** — SMC primitive returning only the intersection of two
  sets [7].
- **Differential privacy (DP)** — `ε`-bounded guarantee that one record barely changes the
  output distribution; in tension with linkage utility [8].
- **Fairness in ER** — parity of matching error rates across protected groups
  (accuracy/TPR/FPR/equalized-odds parity) [12,13].
- **Name-matching bias** — systematic accuracy disparity of name comparators/classifiers
  across cultures, genders, ages [10,11].
- **Spatial join** — pair spatially-near objects using a spatial index (R-tree) or grid key
  [15,16].
- **Geohash / S2 / H3** — discrete global grid systems (Z-order / Hilbert-cube-squares /
  hexagons) usable as spatial blocking keys.
- **R-tree** — hierarchical minimum-bounding-rectangle index; O(log n) spatial queries [15].
- **POI conflation** — entity resolution over places, fusing spatial proximity with
  name/address/category similarity [17,18].
- **Map matching** — snapping a GPS trajectory to a road network; HMM+Viterbi is canonical
  [19,20].
- **Calibration** — property that predicted match probabilities match empirical match
  frequencies (Platt/isotonic/beta; Fellegi–Sunter m/u) [21].
- **FMR / FNMR** — false-match / false-non-match rate; linkage-quality duals of
  precision/recall.
- **Linkage-error propagation** — carrying (or correcting for) linkage uncertainty in the
  downstream analysis [22,23,24,25].

---

## 7. Design implications for `equate`

The consistent lesson across all four frontiers: **none needs a new engine; each needs a
seam plus honest metadata.** They land on the *existing* featurize → block → compare →
match/evaluate stages of [`10-design-implications-for-equate.md`](10-design-implications-for-equate.md).

### 7.1 Native vs. extension-point boundary

| Capability | Native (core, no heavy dep) | Extension point (optional extra) | Never (hard dep / owned) |
|---|---|---|---|
| **PPRL** | Bitstring featurizer *interface*; Dice/Hamming comparator; Hamming-LSH blocker (shared with perceptual hashes) | `equate[pprl]` → `clkhash`/`anonlink` CLK encoding + hardening; `equate[psi]` → an OT-PSI lib | Do not implement crypto; do not ship a hardening scheme as "secure" without citing its attack status |
| **Fairness** | Group-aware `evaluate()` (optional `group` labels → per-group P/R/F1, TPR/FPR parity); explicit scores+thresholds enabling per-group cutoffs | `equate[fair]` → FairEM360-style audit report; fair name comparator (FairNM) | Do not hard-code one culture's name/phonetic model as *the* default and hide it |
| **Geospatial** | Haversine + decay comparators; spatial **key_fn**s (geohash/S2/H3) feeding the keyed blocker; POI as a multi-field comparison-vector config | `equate[geo]` → `geopandas`/`shapely` STRtree spatial join, `h3`/`s2` cell keys; `equate[mapmatch]` → FMM/Valhalla | Do not ship a road-network routing engine |
| **Uncertainty** | Calibration transforms (Platt/isotonic/beta); Fellegi–Sunter m/u scorer; typed `Probability` carrying `is_calibrated`; top-k `CandidateStore`; FMR/FNMR in `evaluate()` | `equate[bayes]` → Bayesian ER / exact error propagation; multiple-imputation helper | Do not return hard pairs that silently discard the probability |

### 7.2 Concrete abstractions to add

1. **`PrivacyMeta` on encoders.** Any featurizer may carry a `privacy` descriptor
   (`plaintext` | `hashed` | `bloom_filter(hardening=…)`), so the framework can *warn* when
   a plaintext key is about to cross a party boundary or be persisted, and so a linkage-unit
   mode can *require* encoded input. Turns §2's duty-of-care into a type-level guardrail.
2. **Group-aware evaluation is the single highest-leverage fairness feature.** Extend the
   evaluation utility ([`10`](10-design-implications-for-equate.md), `evaluate.py`) to accept
   an **optional** `groups: Mapping[id, label]` and emit per-group metrics + parity gaps.
   Optional-by-design (privacy-respecting), auditable-when-available. Because blocking can
   drop recall unfairly (§3.1), the **blocking metrics (PC/RR/PQ)** must also be computable
   *per group*.
3. **Spatial keys are ordinary `key_fn`s.** No special path — a geohash/S2/H3 encoder is a
   `key_fn` returning `(cell, *neighbors)` for the keyed blocker; a haversine-decay
   comparator is an ordinary `Comparator`. POI matching = a documented field-map recipe.
   Everything heavy (`shapely`/`geopandas`/`h3`) stays behind `equate[geo]`.
4. **Map matching is a delegated extension, not a core matcher.** It is
   sequence-to-network, not collection-to-collection; expose it as `equate[mapmatch]`
   wrapping FMM/Valhalla, and only pull a shared `viterbi`/HMM primitive into core if a
   second sequence-alignment use-case justifies it.
5. **A typed, calibration-aware probability.** Make the score type advertise whether it is
   calibrated and under which method, so downstream consumers (assignment cost, review
   triage, propagation) can refuse to treat a raw similarity as `P(match)`. This is the
   uncertainty analogue of the `sense` flag the SSOT `ScoreMatrix` already carries
   ([`10`](10-design-implications-for-equate.md)).
6. **Keep uncertainty *out of the pipe but reachable*.** Default `match()` stays simple
   (hard pairs); `match(..., return_scores=True)` / the `CandidateStore` retain calibrated
   top-k so multiple-imputation or Bayesian propagation is *possible* without bloating the
   common case — progressive disclosure applied to UQ.

### 7.3 The duty-of-care summary (defaults that protect)

- **Privacy:** never log or persist plaintext join keys by default; make BF encoding
  available but *loudly* attack-aware; treat "secure" as a claim that must cite a threat
  model.
- **Fairness:** make per-group auditing a one-argument opt-in; never present a single
  culture's name/phonetic model as neutral; document each comparator's known bias surface.
- **Honesty:** never emit an uncalibrated score as a probability; never throw away
  uncertainty the user might need downstream; surface FMR/FNMR alongside precision/recall.

These are cheap to build *because* the round-1 architecture already factors matching into
swappable stages — responsibility is a matter of **which metadata the seams carry**, not a
separate subsystem.

---

## References

1. Vatsalan D, Christen P, Verykios VS. A taxonomy of privacy-preserving record linkage techniques. *Information Systems* 38(6):946-969, 2013. [https://www.sciencedirect.com/science/article/abs/pii/S0306437912001470](https://www.sciencedirect.com/science/article/abs/pii/S0306437912001470)
2. Vatsalan D, Sehili Z, Christen P, Rahm E. Privacy-Preserving Record Linkage for Big Data: Current Approaches and Research Challenges. In *Handbook of Big Data Technologies*, Springer, pp. 851-895, 2017. [https://link.springer.com/chapter/10.1007/978-3-319-49340-4_25](https://link.springer.com/chapter/10.1007/978-3-319-49340-4_25)
3. Schnell R, Bachteler T, Reiher J. Privacy-preserving record linkage using Bloom filters. *BMC Medical Informatics and Decision Making* 9:41, 2009. [https://link.springer.com/article/10.1186/1472-6947-9-41](https://link.springer.com/article/10.1186/1472-6947-9-41)
4. Christen P, Schnell R, Vatsalan D, Ranbaduge T. Efficient Cryptanalysis of Bloom Filters for Privacy-Preserving Record Linkage. *PAKDD* 2017. [https://link.springer.com/chapter/10.1007/978-3-319-57454-7_49](https://link.springer.com/chapter/10.1007/978-3-319-57454-7_49)
5. Ranbaduge T, Schnell R. Securing Bloom Filters for Privacy-preserving Record Linkage. *CIKM* 2020. [https://dl.acm.org/doi/10.1145/3340531.3412105](https://dl.acm.org/doi/10.1145/3340531.3412105)
6. Christen P, Ranbaduge T, Schnell R. *Linking Sensitive Data: Methods and Techniques for Practical Privacy-Preserving Information Sharing.* Springer, 2020. [https://link.springer.com/book/10.1007/978-3-030-59706-1](https://link.springer.com/book/10.1007/978-3-030-59706-1) (companion site: [https://dmm.anu.edu.au/lsdbook2020/](https://dmm.anu.edu.au/lsdbook2020/))
7. Karapiperis D, et al. Privacy-preserving record linkage using local sensitive hash and private set intersection. arXiv:2203.14284, 2022. [https://arxiv.org/abs/2203.14284](https://arxiv.org/abs/2203.14284)
8. Vatsalan D, Rahm E, et al. The use of differential privacy for privacy-preserving record linkage: Protecting the bits but not the people. Database Group Leipzig. [https://dbs.uni-leipzig.de/research/publications/the-use-of-differential-privacy-for-privacy-preserving-record-linkage](https://dbs.uni-leipzig.de/research/publications/the-use-of-differential-privacy-for-privacy-preserving-record-linkage)
9. FairEM360 / Through the Fairness Lens (see [12,13]) — disparate per-group matching error rates and their population-level consequences.
10. Lockhart JW, Sheller M, et al. Equal accuracy for Andrew and Abubakar — detecting and mitigating bias in name-ethnicity classification algorithms. *AI & Society*, 2023. [https://link.springer.com/article/10.1007/s00146-022-01619-4](https://link.springer.com/article/10.1007/s00146-022-01619-4)
11. FairNM: Fairness in Name Matching. Springer, 2025. [https://link.springer.com/chapter/10.1007/978-3-031-97144-0_4](https://link.springer.com/chapter/10.1007/978-3-031-97144-0_4)
12. Shahbazi N, Danevski N, Nargesian F, Asudeh A, Srivastava D. Through the Fairness Lens: Experimental Analysis and Evaluation of Entity Matching. *PVLDB* 16(11):3279-3292, 2023. [https://www.vldb.org/pvldb/vol16/p3279-shahbazi.pdf](https://www.vldb.org/pvldb/vol16/p3279-shahbazi.pdf) (arXiv: [https://arxiv.org/abs/2307.02726](https://arxiv.org/abs/2307.02726))
13. Moslemi S, Shahbazi N, Nargesian F, Asudeh A, et al. FairEM360: A Suite for Responsible Entity Matching. *PVLDB* 17, 2024. [https://dl.acm.org/doi/10.14778/3685800.3685889](https://dl.acm.org/doi/10.14778/3685800.3685889) (arXiv: [https://arxiv.org/abs/2404.07354](https://arxiv.org/abs/2404.07354))
14. Fairness and Cost Constrained Privacy-Aware Record Linkage. arXiv:2206.15089, 2022. [https://arxiv.org/abs/2206.15089](https://arxiv.org/abs/2206.15089)
15. Guttman A. R-trees: A Dynamic Index Structure for Spatial Searching. *ACM SIGMOD* 1984, pp. 47-57. [https://dl.acm.org/doi/10.1145/602259.602266](https://dl.acm.org/doi/10.1145/602259.602266)
16. GeoPandas: `geopandas.sjoin` (spatial join over Shapely `STRtree`/GEOS R-tree). [https://geopandas.org/en/stable/docs/reference/api/geopandas.sjoin.html](https://geopandas.org/en/stable/docs/reference/api/geopandas.sjoin.html)
17. Low R, Tekler ZD, Cheah L. Conflating point of interest (POI) data: A systematic review of matching methods. *Computers, Environment and Urban Systems*, 2023. [https://www.sciencedirect.com/science/article/abs/pii/S0198971523000406](https://www.sciencedirect.com/science/article/abs/pii/S0198971523000406)
18. Low R, Tekler ZD, Cheah L. An End-to-End Point of Interest (POI) Conflation Framework. *ISPRS Int. J. Geo-Information* 10(11):779, 2021. [https://www.mdpi.com/2220-9964/10/11/779](https://www.mdpi.com/2220-9964/10/11/779) (arXiv: [https://arxiv.org/abs/2109.06073](https://arxiv.org/abs/2109.06073))
19. Newson P, Krumm J. Hidden Markov Map Matching Through Noise and Sparseness. *Proc. 17th ACM SIGSPATIAL Int. Conf. on Advances in Geographic Information Systems (GIS '09)*, pp. 336-343, 2009. [https://dl.acm.org/doi/10.1145/1653771.1653818](https://dl.acm.org/doi/10.1145/1653771.1653818)
20. Yang C, Gidófalvi G. Fast map matching, an algorithm integrating hidden Markov model with precomputation. *International Journal of Geographical Information Science* 32(3):547-570, 2018. [https://doi.org/10.1080/13658816.2017.1400548](https://doi.org/10.1080/13658816.2017.1400548) (code: [https://github.com/cyang-kth/fmm](https://github.com/cyang-kth/fmm))
21. The Fellegi-Sunter Model (m/u probabilities, calibration diagnostics). Splink documentation. [https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html](https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html)
22. Kim G, Chambers R. Regression analysis under incomplete linkage. *Computational Statistics & Data Analysis* 56(9):2756-2770, 2012. [https://www.sciencedirect.com/science/article/abs/pii/S0167947312000771](https://www.sciencedirect.com/science/article/abs/pii/S0167947312000771)
23. Wang Z, Kim JK, et al. Regression with linked datasets subject to linkage error. *WIREs Computational Statistics* 14(4):e1570, 2022. [https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.1570](https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.1570)
24. Tancredi A, Steorts R, Liseo B. Generalized Bayesian Record Linkage and Regression with Exact Error Propagation. arXiv:1810.04808, 2018. [https://arxiv.org/abs/1810.04808](https://arxiv.org/abs/1810.04808)
25. Binette O, Steorts RC. (Almost) All of Entity Resolution. *Science Advances* 8(12), 2022. [https://arxiv.org/abs/2008.04443](https://arxiv.org/abs/2008.04443)
