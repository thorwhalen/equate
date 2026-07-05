# Entity Resolution & Record Linkage: The Canonical Matching Pipeline

*Research note for the `equate` redesign — one of a corpus in `docs/research/`.*

## Abstract

Entity Resolution (ER) — known under a dozen near-synonymous names across the
database, statistics, NLP, and semantic-web communities — is the mature,
canonical framing of *matching*: deciding which records refer to the same
real-world entity when no shared identifier exists. Over five decades the field
has converged on a remarkably stable **end-to-end pipeline** — schema alignment,
blocking/indexing, pairwise comparison, match classification, and
clustering/canonicalization — that decomposes an inherently *quadratic* problem
into scalable stages. This note surveys that pipeline, its foundational
probabilistic model (Fellegi–Sunter), the supervised/unsupervised/deep-learning
methods that fill each stage, the evaluation metrics (pairwise vs. cluster-level),
and the terminology that lets us translate between communities. It closes with
concrete abstractions and extension points these fifty years of practice suggest
for `equate` as a general matching *framework*.

---

## 1. What ER is, and why the name multiplies

Entity Resolution is the task of identifying, and linking or grouping, the
descriptions ("records", "profiles", "mentions", "references") that correspond to
the same real-world entity, in the *absence* of a reliable shared key [1]. The
core difficulty is that duplicate records do not share a common key and contain
errors — typos, missing values, formatting differences, abbreviations,
transpositions — that make exact equality useless [3]. ER is therefore
fundamentally about *fuzzy correspondence*, which is exactly the problem `equate`
targets.

The same problem was discovered independently by many communities, so it carries
many names. These are **(near-)synonyms**, and mapping them is a prerequisite for
a framework that means to unify the field [1,7]:

| Term | Community / origin | Nuance |
|---|---|---|
| **Record linkage** | Statistics, epidemiology, demography, official statistics (Newcombe 1959; Fellegi–Sunter 1969) | Classically the *linkage* setting: matching across two clean sources. Emphasis on probabilistic decision theory. |
| **Entity resolution** | Databases / data integration, ~2000s | The broad umbrella term; covers dedup, linkage, and canonicalization together. |
| **Data matching / fuzzy matching** | Data engineering, industry | Operational term for approximate joins. |
| **Deduplication (dedup) / duplicate detection** | Databases | The *single-source* setting: finding duplicates *within* one dataset. |
| **Merge/purge** | Databases (Hernández & Stolfo, 1995) | Early industrial framing; where the **Sorted Neighborhood Method** originated. |
| **Reference reconciliation** | Semantic web / knowledge bases | Resolving references within relational/RDF data with rich relationships. |
| **Coreference resolution** | NLP | Grouping textual *mentions* that refer to the same entity; the unstructured-text sibling of ER. |
| **Entity disambiguation / entity linking** | NLP / IR | Linking a mention to a canonical entry in a knowledge base (a *many-to-one* variant). |
| **Instance matching / link discovery** | Semantic web (LOD) | Matching RDF instances / minting `owl:sameAs` links. |
| **Identity resolution** | Industry (MDM, AdTech) | ER restricted to entities with a sense of identity (people, orgs). |
| **List washing, householding, golden-record / MDM** | Industry | Downstream/adjacent operations built on ER. |

**Design consequence:** these are *the same abstract operation under different
data shapes and output conventions*. A general framework should model the
operation once and treat "dedup within one set" vs. "link across two sets" vs.
"link mentions to a KB" as configurations, not separate products.

---

## 2. The canonical end-to-end pipeline

The recent flagship survey — Christophides, Efthymiou, Palpanas, Papadakis &
Stefanidis, *"An Overview of End-to-End Entity Resolution for Big Data"* (ACM
Computing Surveys, 2020) [1] — organizes the field around a workflow that is
essentially universal across surveys [1,3,5,7]. Binette & Steorts frame the same
pipeline statistically as four steps [7], and libraries (recordlinkage, dedupe,
Splink) implement it almost verbatim [9,10,11]. The stages:

```
   raw sources
       │
 (0) SCHEMA ALIGNMENT / preprocessing   → common attributes, normalization
       │
 (1) BLOCKING / INDEXING                 → candidate pairs  (beats O(n²))
       │
 (1b) BLOCK / COMPARISON CLEANING        → prune redundant/superfluous pairs (meta-blocking)
       │
 (2) COMPARISON                          → similarity/comparison vector per pair
       │
 (3) CLASSIFICATION / MATCH DECISION     → match / non-match / possible-match
       │
 (4) CLUSTERING / CANONICALIZATION       → entity clusters + merged golden records
       │
   resolved entities
```

### 2.0 Schema alignment (attribute / schema matching)

Before records can be compared they must be brought into a common representation:
deciding which columns correspond (name↔full_name, dob↔birth_date), normalizing
formats, tokenizing, standardizing. In the semantic-web / dirty-data setting this
is non-trivial and is itself a matching problem ("schema matching"). The Big-Data
survey distinguishes **schema-aware** methods (exploit known attribute
correspondences and semantics) from **schema-agnostic** methods (treat every
record as a bag of tokens regardless of attribute, trading precision for
robustness to heterogeneity) [1]. This axis is one of the survey's primary
taxonomic dimensions.

### 2.1 Blocking / indexing — taming quadratic cost

Naïvely, resolving *n* records requires comparing every pair: **O(n²)**
comparisons (≈ n·m for two sources of size n, m). At 1M records that is ~10¹²
comparisons — infeasible [1,3]. **Blocking** (a.k.a. indexing) restores
scalability by only comparing records likely to match: records are assigned to
**blocks** via a **blocking key** (e.g. "first 3 letters of surname +
postcode"), and comparisons happen only *within* blocks. The dedicated survey is
Papadakis, Skoutas, Thanos & Palpanas, *"Blocking and Filtering Techniques for
Entity Resolution: A Survey"* (ACM Computing Surveys, 2020) [2], which reviews the
space under two frameworks: **Blocking** (redundancy-positive, one key can place a
record in several blocks) and **Filtering** (similarity-join style, e.g. prefix/
length filtering for a threshold).

Canonical blocking / indexing methods and their tradeoffs:

| Method | Idea | Cost / property |
|---|---|---|
| **Standard blocking** | One block per blocking-key value; compare all pairs sharing a value | Simple; block-size skew causes huge blocks. Recall depends entirely on key quality. |
| **Sorted Neighborhood Method (SNM)** | Sort by key, slide a window of size *w*, compare only co-windowed records (Hernández & Stolfo, merge/purge) [3] | Sorting O(n log n) + windowing O(w·n). Robust to a single key error only via multi-pass; window size trades recall vs. cost. |
| **Q-gram / suffix-array indexing** | Blocking keys from character q-grams or suffixes → tolerant to typos | Higher recall on dirty keys; larger candidate sets. |
| **Canopy clustering** (with cheap metric + thresholds T_in, T_ex) | Cheap similarity forms overlapping "canopies"; expensive compare only within | Overlapping canopies raise recall; threshold tuning sensitive [1]. |
| **LSH / MinHash / SimHash** | Locality-sensitive hashing: similar records collide with high probability | Sub-quadratic, probabilistic recall guarantees; natural for high-dim/text/embedding blocking. |
| **Token blocking (schema-agnostic)** | Block on every token, no schema needed | Maximizes recall on heterogeneous/dirty data; produces massive redundancy → needs block processing [1,2]. |
| **(Deep) embedding / dense blocking (nearest-neighbor)** | Embed records as vectors, ANN search (e.g. HNSW, FAISS) for neighbors | State of the art recall on semantic matches; needs a vectorizer + ANN index. |

**Block processing / meta-blocking (stage 1b).** Redundancy-positive blocking
(especially token blocking) generates *redundant* comparisons (same pair in many
blocks) and *superfluous* ones (non-matches). **Block cleaning** purges oversized
blocks; **comparison cleaning / meta-blocking** builds a *blocking graph* whose
nodes are records and whose edge weights count co-occurrences, then prunes
low-weight edges (e.g. **Weighted Edge Pruning** removes edges below the average
weight) [1]. Meta-blocking is a signature contribution of the Papadakis/JedAI line
of work [2,6].

### 2.2 Comparison — the comparison (feature) vector

For each candidate pair, a set of **comparison functions** produces a
**comparison vector** (a.k.a. feature vector): one similarity/agreement value per
compared attribute. Standard string similarities [3]:

- **Edit-based:** Levenshtein / Damerau-Levenshtein (edit distance), Jaro,
  **Jaro–Winkler** (favors common prefixes; strong on names).
- **Token/set-based:** Jaccard, Dice, overlap, cosine over TF-IDF or q-gram
  token vectors (robust to word order and insertions).
- **Phonetic:** Soundex, NYSIIS, Metaphone/Double-Metaphone (match by sound).
- **Numeric/date/geo:** absolute/relative difference, day gaps, Haversine
  distance.
- **Embedding-based:** cosine over learned/pretrained vector embeddings
  (fastText, SBERT) — captures semantic rather than surface similarity.

The Python Record Linkage Toolkit exposes exactly this menu (`jaro`,
`jarowinkler`, `levenshtein`, `damerau_levenshtein`, `qgram`, `cosine`) via a
`Compare` object that emits a per-pair feature vector [12].

### 2.3 Classification — match / non-match / possible-match

Given comparison vectors, decide the class of each pair. The three-way decision —
**match**, **non-match**, and **possible-match** (send to a human) — comes
straight from Fellegi–Sunter (§3) [4]. Approaches:

- **Deterministic / rule-based:** boolean rules over comparisons ("exact SSN OR
  (Jaro-Winkler(name) > 0.9 AND same dob)"). Transparent, brittle, hard to tune.
- **Probabilistic (Fellegi–Sunter):** likelihood-ratio scoring with m/u
  probabilities; threshold into the three regions (§3). Can be **unsupervised**
  (parameters via EM) — this is why it dominates official statistics and powers
  Splink [4,10].
- **Supervised ML:** train a classifier on labeled pairs — logistic regression
  (dedupe's default is L2-regularized logistic regression [11]; the recordlinkage
  toolkit ships `LogisticRegressionClassifier`, SVM, naive Bayes [12]), random
  forests, gradient boosting. Needs labeled training pairs, which are scarce and
  class-imbalanced.
- **Active learning:** iteratively ask a human to label the *most informative*
  uncertain pairs, minimizing labeling effort — the design of **dedupe**,
  **DedupliPy**, and **Zingg** [11,13,14].
- **Unsupervised clustering of pairs:** e.g. K-means / EM over comparison vectors
  (recordlinkage) to separate match vs. non-match clouds without labels [12].
- **Deep learning:** **DeepMatcher** (Mudgal et al., SIGMOD 2018) framed EM as a
  design-space of neural architectures over attribute embeddings [15]; **Ditto**
  (Li et al., PVLDB 2020) casts EM as sequence-pair classification fine-tuning a
  pretrained Transformer (BERT/RoBERTa/DistilBERT), reporting up to ~29% F1 gain
  over prior SOTA and matching earlier accuracy with ≤½ the labels [16]. Cost:
  GPUs, training data, latency. Newer work uses LLMs zero/few-shot.

### 2.4 Clustering / canonicalization — from pairs to entities

Pairwise decisions are *locally* inconsistent: a classifier may say A≈B and B≈C
but A≉C. To produce actual entities we must resolve pairs into a **partition** of
records into entity clusters [1,7]. Methods:

- **Transitive closure / connected components:** treat "match" as an edge, take
  connected components. Simplest and most common; but a single wrong edge can
  *collapse* two distinct entities ("black hole" clusters). Union-Find, near-linear.
- **Correlation clustering:** optimize agreement with ± edge signals; NP-hard,
  solved with approximations; supports incremental variants [1].
- **Markov Clustering (MCL):** simulate flow to strengthen intra-cluster and weaken
  inter-cluster edges [1].
- **Hierarchical agglomerative (single/complete/average linkage):** merge by
  similarity; linkage choice trades chaining vs. tightness [1,7].
- **Collective / relational ER:** matching decisions influence each other through
  the entity graph — resolving papers helps resolve their authors and vice-versa.
  Getoor & Machanavajjhala's tutorial frames this as *relational* and *collective*
  ER, the third major paradigm beyond attribute-based and pairwise [5].

**Canonicalization (a.k.a. merging / fusion / golden-record).** Once a cluster is
formed, produce a single representative record: pick or synthesize the most
reliable value per attribute (majority vote, most-recent, most-complete, source
trust) [7]. Binette & Steorts stress canonicalization as a first-class step beyond
matching, and note the tension between pairwise decisions and globally coherent
merges [7]. This is where ER hands off to Master Data Management.

---

## 3. The Fellegi–Sunter probabilistic model (foundational)

Fellegi & Sunter, *"A Theory for Record Linkage"* (JASA, 1969) [4], formalized
Newcombe's 1959 intuition into a decision-theoretic framework that remains the
most widely used record-linkage method. It is worth stating precisely because it
is the reference model any framework should be able to express.

For a candidate pair, a **comparison vector** γ records agreement/disagreement
per field. Two conditional distributions govern each comparison pattern:

- **m-probability:** `m_i = P(agreement on field i | the pair is a true Match)`.
  High m = the field agrees when records truly match (reflects data quality/reliability).
- **u-probability:** `u_i = P(agreement on field i | the pair is a Non-match)`.
  Small u = agreement is unlikely by chance (reflects the field's cardinality;
  agreeing on a rare surname is stronger evidence than agreeing on sex).

Assuming conditional independence across fields, the **match weight** is the
log-likelihood ratio, and (in Splink's Bayes-factor formulation [10]) the total
score adds a prior term:

```
match_weight = log2( λ/(1−λ) )  +  Σ_i log2( m_i / u_i )
Pr(Match | γ) = 2^match_weight / (1 + 2^match_weight)
```

where **λ** is the prior probability that a random pair matches. Each field
contributes an additive weight (positive when it agrees, negative when it
disagrees); the Bayes factor `K = m/u` per field quantifies its evidential
strength.

**The fundamental theorem / optimal decision rule.** Fellegi & Sunter proved that
ranking pairs by the likelihood ratio and applying two thresholds is *optimal*:
it minimizes the region of "possible matches" (records sent to clerical review)
subject to fixed bounds on the two error rates — μ (false-link rate: linking
true non-matches) and λ (false-non-link rate: missing true matches) [4]. This
yields the canonical **three regions**:

```
   score ≥ T_upper   → MATCH
   T_lower < score < T_upper → POSSIBLE MATCH (clerical review / human)
   score ≤ T_lower   → NON-MATCH
```

**Parameter estimation.** m and u can be set from training data, from field
frequencies, or — crucially — learned **unsupervised** via
**Expectation–Maximization (EM)**, treating match/non-match as a latent variable.
This is what makes probabilistic linkage work *without labels*, and is the engine
of Splink [10] and the recordlinkage toolkit's ECM classifier [12].
**Frequency-based** refinements make u term-specific (rare values weigh more).

---

## 4. Evaluation: pairwise vs. cluster-level

ER is a highly **imbalanced** problem (matches are a tiny fraction of all pairs),
so accuracy is meaningless; precision/recall/F1 dominate. There are two distinct
granularities, and conflating them is a classic error [1,7,17].

**Pairwise metrics.** Treat the set of predicted matching *pairs* vs. the true
matching pairs:

- **Pairwise precision** = correct predicted pairs / all predicted pairs.
- **Pairwise recall (a.k.a. pair completeness, PC)** = correct predicted pairs /
  all true pairs.
- **Pairwise F1** = harmonic mean; the single most-reported ER metric [17].
- Blocking is scored separately by **Pair Completeness** (recall of the candidate
  set) vs. **Reduction Ratio (RR)** / **Pairs Quality (PQ)** (how much of the
  quadratic space was pruned) — blocking must maximize recall while shrinking the
  candidate set [1,2].

*Pairwise's blind spot:* it scores links in isolation. If A↔B and B↔C are found
but A↔C is missed, pairwise metrics still reward the (structurally broken) result
[17].

**Cluster-level metrics** judge the final *partition* into entities:

- **B³ (B-Cubed) precision/recall/F1** (Bagga & Baldwin): per *record*, precision
  = fraction of its predicted cluster that is truly co-referent; recall =
  fraction of its true co-referents captured; average over records [17]. Robust,
  interpretable, widely used in coreference and ER.
- **Cluster/MUC-style, CEAF, closest-cluster/variation-of-information**, and
  **purity / inverse purity** measures of cluster homogeneity [1,7].
- Recent work (Binette et al., *"How to Evaluate Entity Resolution Systems"*,
  2024) expresses many of these as **over-clustering (OCE)** and
  **under-clustering (UCE)** error and warns about estimating metrics from
  *sampled* ground truth [17].

**Design consequence:** a framework must report *both* — a match-decision quality
(pairwise) and an entity-formation quality (cluster) — and must score the
*blocking* stage on its own (recall/reduction) since a matcher can never recover
pairs blocking discarded.

---

## 5. Reference tools (concrete capabilities & tradeoffs)

| Tool | Language / scale | Classifier / model | Blocking | Notes |
|---|---|---|---|---|
| **Splink** [10] | Python over SQL backends: **DuckDB, Spark, AWS Athena, Postgres** | Fellegi–Sunter + **unsupervised EM**; no labels needed | Blocking rules (SQL predicates) | DuckDB ~1M records in <2 min on a laptop; Spark/Athena for 100M+. MoJ (UK gov). Term-frequency adjustments. |
| **dedupe** [11] | Python | **L2-regularized logistic regression** | **Learned predicate blocking** | **Active learning** UI to label pairs; does dedup *and* linkage. |
| **DedupliPy** [11] | Python | logistic regression | string-similarity blocking | Active learning (modAL/sklearn); convergence notifications. |
| **Python Record Linkage Toolkit** (recordlinkage) [12] | Python / pandas | LogisticRegression, SVM, NaiveBayes (supervised); **K-means, ECM (EM)** unsupervised | `Index` (block, sortedneighbourhood, full, random) | Clean pedagogical mapping of the 5 stages: clean→index→compare→classify→evaluate. |
| **Zingg** [13] | Scala/Spark, Python API | ML + **active learning** | smart blocking | Native to Databricks/Snowflake/Glue; MDM/golden-record; millions of records. |
| **JedAI / pyJedAI** [6,14] | Java / Python | full meta-blocking + matching + clustering | token blocking + **meta-blocking** | End-to-end, schema-agnostic; strong on dirty/heterogeneous & RDF data; INFORMS J. Computing 2024. |
| **Magellan** [18] | Python (data-science stack) | design-space of learners | rule + ML blocking | "EM management system": treats ER as guided *workflow* development, not one algorithm; pandas-native. |
| **DeepMatcher / Ditto** [15,16] | Python (PyTorch) | neural / pretrained Transformer | (external) | Highest accuracy on hard/textual EM; needs GPUs + training data. |

Papadakis et al.'s *"Four Generations of Entity Resolution"* (2021) [6] organizes
this evolution as four generations responding to Big Data's "Vs": (1) schema-aware
clean-clean linkage; (2) schema-agnostic blocking + meta-blocking for Volume/
Variety; (3) budget-aware/progressive (pay-as-you-go) for Velocity; (4) deep
learning for Variety/Veracity.

---

## 6. Design implications for `equate`

ER's fifty years of practice map cleanly onto abstractions a general matching
*framework* should expose. `equate` already has the right instinct — a
`similarity_matrix` + a pluggable "matcher" over it, with `match_greedily` as the
simple default (see README). The literature sharpens this into a **staged
pipeline of swappable strategies**, each an independent extension point:

1. **Model the pipeline as explicit, composable stages, not one function.** The
   universal `preprocess → block → compare → classify → cluster → canonicalize`
   decomposition [1,7] should be `equate`'s backbone. Each stage is a strategy
   object/callable with a stable interface; users assemble a pipeline and swap any
   stage. `match_greedily` becomes the trivial 1-stage config of a general engine.

2. **Blocking is a first-class, optional stage — the difference between a toy and a
   library.** The O(n²) → sub-quadratic transition is *the* scalability lever
   [1,2,3]. Expose a `Blocker`/`Indexer` protocol (`records → candidate pairs`)
   with built-ins (standard, sorted-neighborhood, q-gram, LSH/MinHash, ANN/embedding)
   and a null blocker (all-pairs) as the default for small inputs. `equate`'s
   "sparse similarity matrix" is exactly a materialized candidate set — unify the
   two: **the blocker decides which cells of the similarity matrix get computed.**

3. **Separate `compare` (produce a comparison vector) from `classify` (decide).**
   A `Comparison` protocol maps a pair to a feature vector via per-field similarity
   functions; a separate `Classifier`/`Scorer` maps that vector to a
   score/decision. This is the SSOT that lets edit distance, TF-IDF cosine,
   embeddings, and learned models coexist. Ship a registry of similarity functions
   (Levenshtein, Jaro-Winkler, Jaccard/cosine, phonetic, numeric/date/geo,
   embedding-cosine) as small, independently-testable callables.

4. **Make the *decision policy* pluggable and support the three-way outcome.**
   Support deterministic rules, threshold-on-score, **Fellegi–Sunter (with
   unsupervised EM)**, and pluggable supervised/active-learning classifiers behind
   one `Classifier` interface. Crucially, allow a **possible-match / abstain**
   outcome (not just match/non-match) [4] so downstream code can route uncertain
   pairs to review — a differentiator most fuzzy-join tools lack.

5. **Treat clustering + canonicalization as a distinct, swappable back-end.** Do
   not hard-wire transitive closure. Offer `ClusterResolver` strategies
   (connected-components/union-find as default, plus correlation clustering,
   hierarchical, MCL) and a separate `Canonicalizer` (merge policy: majority,
   most-complete, source-trust, custom reducer) [1,7]. Expose the
   deduplication-vs-linkage setting (one set vs. two/many sets, and the many-to-one
   "link-to-KB" case) as a *matching-constraint parameter*, not separate code
   paths — including `equate`'s existing "never reuse a value" one-to-one
   constraint as one such policy (it is bipartite assignment / stable matching).

6. **Optional-dependency / strategy boundaries** (progressive disclosure —
   heavy machinery only when asked):
   - **Core (zero heavy deps):** exact + basic string similarities, all-pairs or
     simple blocking, threshold/greedy matching. Works out of the box.
   - **`[ml]` extra:** scikit-learn-backed classifiers, EM/Fellegi–Sunter,
     active-learning loop (à la dedupe [11]).
   - **`[embed]` extra:** vectorizers + ANN blocking (fastText/SBERT + FAISS/HNSW).
   - **`[deep]` extra:** Transformer matchers (Ditto-style) [16].
   - **`[scale]` extra:** out-of-core / SQL-backend execution (Splink's design of
     pushing comparisons into DuckDB/Spark [10] is the model to emulate for large
     inputs).
   Each extra plugs into the *same* stage interfaces, so a user upgrades capability
   without rewriting their pipeline.

7. **Build evaluation in, at both granularities.** Ship `pairwise_prf`, `bcubed`,
   and blocking metrics (**pair completeness** + **reduction ratio**) [1,17]. A
   matching framework that cannot tell a user their blocking recall or their
   cluster-level F1 is not usable for real ER. Provide a `ground_truth` adapter and
   a one-call `evaluate(pipeline, labels)`.

8. **Adopt the field's vocabulary as the public API surface**, with a synonyms map
   in the docs. Name things `entity_resolution` / `link` / `deduplicate` /
   `block` / `compare` / `classify` / `cluster` / `canonicalize`, and document the
   cross-community glossary (§Glossary) so users arriving from statistics, NLP, or
   data-engineering recognize the operation. This is directly load-bearing for the
   synthesis step this corpus feeds.

**One-line summary for the synthesis:** ER says the general matching operation is a
*pipeline of independently-swappable strategies* over a candidate set, and the
biggest architectural wins are (a) making blocking/indexing a first-class optional
stage that decides which similarities get computed, (b) splitting compare from
classify, and (c) treating cluster+canonicalize as a distinct back-end with an
explicit match-constraint (dedup vs. linkage vs. link-to-KB, one-to-one vs.
many-to-one).

---

## Glossary (canonical terminology)

- **Entity Resolution (ER):** identifying and grouping records that refer to the
  same real-world entity without a shared key; umbrella term for the family below.
- **Record linkage:** ER across two (or more) sources — the classic
  statistics/epidemiology framing; often implies the Fellegi–Sunter model.
- **Deduplication / duplicate detection:** ER *within* a single source.
- **Data matching / fuzzy matching:** operational/industry name for approximate
  ER, especially in data engineering.
- **Merge/purge:** early database term for dedup+consolidation; birthplace of the
  Sorted Neighborhood Method.
- **Reference reconciliation:** ER over relational/RDF references, exploiting
  relationships.
- **Coreference resolution:** the NLP sibling — grouping textual *mentions* of the
  same entity.
- **Entity linking / disambiguation:** many-to-one ER against a knowledge base.
- **Blocking (indexing):** partitioning records into blocks by a **blocking key**
  so only intra-block pairs are compared; the O(n²) → sub-quadratic step.
- **Blocking key / predicate:** the function assigning records to blocks.
- **Meta-blocking / block processing:** pruning redundant/superfluous comparisons
  from a (redundancy-positive) blocking via a blocking graph.
- **Comparison (feature) vector:** the per-attribute similarity/agreement values
  computed for a candidate pair — the input to classification.
- **Fellegi–Sunter model:** probabilistic record linkage scoring pairs by a
  log-likelihood ratio built from **m** (agreement-given-match) and **u**
  (agreement-given-non-match) probabilities.
- **m-probability / u-probability:** the two conditional agreement probabilities
  underlying the F–S match weight; their ratio `m/u` is a per-field Bayes factor.
- **Match / non-match / possible-match:** the three F–S decision regions;
  "possible-match" routes to clerical/human review.
- **Transitive closure:** propagating pairwise matches so that A≈B, B≈C ⇒ A≈C;
  simplest clustering (connected components), risky (can collapse entities).
- **Canonicalization (merging / fusion / golden record):** producing one
  representative record per resolved entity cluster.
- **Pairwise metrics:** precision/recall/F1 over matching *pairs*.
- **B³ (B-Cubed):** per-record cluster-level precision/recall/F1 over the final
  partition.
- **Pair completeness / reduction ratio:** recall of, and pruning achieved by, the
  blocking stage.
- **Schema-aware vs. schema-agnostic:** whether the method uses known attribute
  correspondences or treats records as attribute-free token bags.

---

## References

[1] Christophides V, Efthymiou V, Palpanas T, Papadakis G, Stefanidis K. *An
Overview of End-to-End Entity Resolution for Big Data.* ACM Computing Surveys
53(6), Article 127, 2020.
[https://dl.acm.org/doi/10.1145/3418896](https://dl.acm.org/doi/10.1145/3418896)
(open access:
[hal.science/hal-02955445](https://hal.science/hal-02955445/document))

[2] Papadakis G, Skoutas D, Thanos E, Palpanas T. *Blocking and Filtering
Techniques for Entity Resolution: A Survey.* ACM Computing Surveys 53(2), Article
31, 2020.
[https://dl.acm.org/doi/abs/10.1145/3377455](https://dl.acm.org/doi/abs/10.1145/3377455)

[3] Elmagarmid AK, Ipeirotis PG, Verykios VS. *Duplicate Record Detection: A
Survey.* IEEE Transactions on Knowledge and Data Engineering 19(1):1–16, 2007.
[https://www.cs.purdue.edu/homes/ake/pub/TKDE-0240-0605-1.pdf](https://www.cs.purdue.edu/homes/ake/pub/TKDE-0240-0605-1.pdf)

[4] Fellegi IP, Sunter AB. *A Theory for Record Linkage.* Journal of the American
Statistical Association 64(328):1183–1210, 1969.
[https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501049](https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501049)
(PDF:
[www2.stat.duke.edu](https://www2.stat.duke.edu/~rcs46/linkage/presentations/01-baiLi_FelleigSunter1969.pdf))

[5] Getoor L, Machanavajjhala A. *Entity Resolution: Theory, Practice & Open
Challenges.* Proceedings of the VLDB Endowment 5(12):2018–2019, 2012 (tutorial).
[http://vldb.org/pvldb/vol5/p2018_lisegetoor_vldb2012.pdf](http://vldb.org/pvldb/vol5/p2018_lisegetoor_vldb2012.pdf)

[6] Papadakis G, Ioannou E, Thanos E, Palpanas T. *The Four Generations of Entity
Resolution.* Synthesis Lectures on Data Management, Morgan & Claypool, 2021.
[https://dblp.org/rec/series/synthesis/2021Papadakis.html](https://dblp.org/rec/series/synthesis/2021Papadakis.html)
(companion site:
[entityresolution4g.com](https://entityresolution4g.com/))

[7] Binette O, Steorts RC. *(Almost) All of Entity Resolution.* Science Advances
8(12):eabi8021, 2022 (arXiv:2008.04443).
[https://arxiv.org/abs/2008.04443](https://arxiv.org/abs/2008.04443)
(PMC:
[pmc.ncbi.nlm.nih.gov/articles/PMC11636688](https://pmc.ncbi.nlm.nih.gov/articles/PMC11636688/))

[8] Naumann F, Herschel M. *An Introduction to Duplicate Detection.* Synthesis
Lectures on Data Management, Morgan & Claypool, 2010 (background; Sorted
Neighborhood, merge/purge lineage). DOI listing:
[https://link.springer.com/book/10.1007/978-3-031-01835-0](https://link.springer.com/book/10.1007/978-3-031-01835-0)

[9] Robin Linacre et al. *Splink: Free Software for Probabilistic Record Linkage
at Scale* (project + Fellegi–Sunter theory guide). Ministry of Justice, UK.
[https://github.com/moj-analytical-services/splink](https://github.com/moj-analytical-services/splink)

[10] Splink documentation — *The Fellegi–Sunter Model* (m/u probabilities,
match-weight Bayes factor, EM training; DuckDB/Spark/Athena backends).
[https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html](https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html)

[11] Gregg F, Eder D, et al. *dedupe — a Python library for fuzzy matching,
deduplication and entity resolution* (L2-regularized logistic regression; learned
predicate blocking; active learning). Docs:
[https://docs.dedupe.io/](https://docs.dedupe.io/) · Source:
[https://github.com/dedupeio/dedupe](https://github.com/dedupeio/dedupe)

[12] de Bruin J. *Python Record Linkage Toolkit* (clean→index→compare→classify→
evaluate; Jaro/Jaro-Winkler/Levenshtein/q-gram/cosine; LogisticRegression, SVM,
K-means, ECM/EM).
[https://recordlinkage.readthedocs.io/](https://recordlinkage.readthedocs.io/)
· Source:
[https://github.com/J535D165/recordlinkage](https://github.com/J535D165/recordlinkage)

[13] Goyal S, et al. *Zingg — scalable ML entity resolution & MDM on Spark*
(active-learning training, smart blocking, golden record; Databricks/Snowflake/
Glue).
[https://github.com/zinggAI/zingg](https://github.com/zinggAI/zingg)

[14] Papadakis G, et al. *pyJedAI: A Library with Resolution-Related Structures and
Procedures* (schema-agnostic blocking + meta-blocking + clustering). INFORMS
Journal on Computing, 2024.
[https://pyjedai.readthedocs.io/](https://pyjedai.readthedocs.io/)
(DOI: [10.1287/ijoc.2023.0410](https://doi.org/10.1287/ijoc.2023.0410))

[15] Mudgal S, Li H, Rekatsinas T, Doan A, Park Y, Krishnan G, Deep R, Arcaute E,
Raghavendra V. *Deep Learning for Entity Matching: A Design Space Exploration*
(DeepMatcher). ACM SIGMOD 2018:19–34.
[https://dl.acm.org/doi/10.1145/3183713.3196926](https://dl.acm.org/doi/10.1145/3183713.3196926)

[16] Li Y, Li J, Suhara Y, Doan A, Tan W-C. *Deep Entity Matching with Pre-Trained
Language Models* (Ditto). Proceedings of the VLDB Endowment 14(1):50–60, 2020
(arXiv:2004.00584).
[https://arxiv.org/abs/2004.00584](https://arxiv.org/abs/2004.00584)
(ACM: [10.14778/3421424.3421431](https://dl.acm.org/doi/10.14778/3421424.3421431))

[17] Binette O, Reiter JP, Steorts RC, et al. *How to Evaluate Entity Resolution
Systems: An Entity-Centric Framework with Application to Inventor Name
Disambiguation.* 2024 (arXiv:2404.05622) — pairwise vs. B³ vs. cluster metrics,
over/under-clustering error, sampling caveats.
[https://arxiv.org/abs/2404.05622](https://arxiv.org/abs/2404.05622)

[18] Konda P, Das S, Suganthan GC P, Doan A, Ardalan A, Ballard JR, Li H, Panahi
F, Zhang H, Naughton J, Prasad S, Krishnan G, Deep R, Raghavendra V. *Magellan:
Toward Building Entity Matching Management Systems.* Proceedings of the VLDB
Endowment 9(12):1197–1208, 2016.
[https://dl.acm.org/doi/10.14778/2994509.2994535](https://dl.acm.org/doi/10.14778/2994509.2994535)
(PDF:
[cs.wisc.edu/~anhai/papers/magellan-tr.pdf](http://www.cs.wisc.edu/~anhai/papers/magellan-tr.pdf))
