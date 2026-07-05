# Blocking, Filtering & Scalable Candidate Generation

**Abstract.** Any framework that *matches* two collections of objects faces a
combinatorial wall: comparing every object in one collection against every
object in another is O(n·m) (O(n²) for deduplication within one collection),
which is intractable past a few tens of thousands of records. The universal
remedy is a *candidate-generation* stage that cheaply proposes a small set of
promising pairs so that the expensive similarity/matcher only runs on those.
This document surveys the two-phase paradigm and its concrete algorithms:
classical **blocking** (standard, phonetic, q-gram, suffix, token, sorted
neighborhood, canopy), **block processing / meta-blocking**, **filtering /
set-similarity joins** (prefix, length, positional, suffix filtering; PPJoin),
**locality-sensitive hashing** (MinHash for Jaccard, SimHash for cosine),
**approximate nearest-neighbor** indexes (NSW/HNSW, IVF, PQ, IVF-PQ, ScaNN),
and modern **learned / embedding-based (dense) blocking** (DeepBlocker,
UniBlocker). It standardizes the vocabulary across the database, information-
retrieval, and machine-learning communities (which name the same ideas
differently), fixes the recall-vs-efficiency metrics (pair completeness,
reduction ratio, pairs quality), and closes with concrete abstractions and
optional-dependency boundaries for the `equate` matching framework.

---

## 1. The scalability problem and the two-phase paradigm

Matching (a.k.a. entity resolution, record linkage, deduplication, fuzzy
join) asks which objects across one or two collections *correspond* — refer to
the same underlying entity, or are "close enough" under some notion of
similarity. The naive solution computes a similarity for every pair. For two
collections of sizes n and m this is the **Cartesian product** of size n·m;
for self-matching one collection of size n it is n·(n−1)/2 ≈ O(n²). At
n = m = 10⁵ that is 10¹⁰ comparisons — and each comparison may itself run an
expensive learned matcher. This is the bottleneck that *all* scalable matching
systems attack [1,2].

The near-universal architecture is a **two-stage pipeline**:

1. **Candidate generation** (cheap, high-recall, approximate): produce a set of
   **candidate pairs** C ⊆ A × B that is far smaller than the Cartesian product
   yet still contains (almost) all true matches. This is the subject of this
   document. Its two dominant framings are:
   - **Blocking / indexing** — *positive* selection: group objects likely to
     match into **blocks**, and only compare within blocks.
   - **Filtering** — *negative* selection: quickly *discard* pairs that are
     provably below a similarity threshold (set-similarity joins, bounds).
2. **Matching / verification** (expensive, high-precision): run the real
   similarity function / classifier on each candidate pair in C.

The survey by Papadakis, Skoutas, Thanos & Palpanas, *"Blocking and Filtering
Techniques for Entity Resolution: A Survey"* (ACM Computing Surveys 53(2),
Article 31, 2020) [1] is the authoritative reference for this stage; the
companion end-to-end survey by Christophides, Efthymiou, Palpanas, Papadakis &
Stefanidis (ACM CSUR 53(6), 2020) [2] situates it in the whole pipeline. Both
frame candidate generation as the step that turns quadratic ER into something
that scales to "Big Data."

**Terminology across communities (synonyms to normalize).** The same idea is
named differently by different fields — a synthesis must pick canonical terms:

| Concept | Database / ER term | IR / ML / vector-search term |
|---|---|---|
| Reduce comparisons by grouping | blocking, indexing | bucketing, partitioning |
| A proposed pair to verify | candidate pair, comparison | candidate, retrieved neighbor |
| Cheap approximate retrieval | blocking / filtering | approximate nearest-neighbor (ANN), retrieval |
| Signature that collides for similar items | blocking key, LSH signature | hash bucket, sketch, fingerprint |
| Reduction in work vs all-pairs | reduction ratio | (implicit; speedup / QPS) |
| Fraction of true matches kept | pair completeness | recall |

---

## 2. Evaluating candidate generation: the recall–efficiency trade-off

A candidate generator is judged on two axes — *how much work it saves* and *how
many true matches it keeps*. The standard metrics (from [1,2]) are defined
against the set of true duplicate/matching pairs D and the generated candidate
set C (with brute-force set being the full Cartesian product):

- **Pair Completeness (PC)** = |D ∩ C| / |D|. This *is recall*: the fraction of
  true matches that survive blocking. Because pairs discarded here can never be
  recovered by the matcher, PC upper-bounds the whole system's recall — so
  blocking is deliberately tuned to keep PC very high (≈1.0) [1].
- **Reduction Ratio (RR)** = 1 − |C| / (|A|·|B|). How much the comparison space
  shrank relative to brute force. RR → 1 means near-total pruning.
- **Pairs Quality (PQ)** = |D ∩ C| / |C|. This *is precision* of the candidate
  set: the fraction of generated pairs that are real matches. Low PQ means the
  matcher wastes work on non-matches [1].
- **Blocking cardinality / comparison count** (|C|, or average blocks per
  entity, or block-size distribution): the raw quantity the matcher must
  process; drives absolute runtime and memory.

The fundamental tension: **PC (recall) trades against RR and PQ (efficiency)**.
Coarser blocks / lower thresholds raise PC but generate more pairs; aggressive
pruning raises RR/PQ but risks dropping true matches. Because the downstream
matcher can still reject false positives but can *never* rescue a missed pair,
the received wisdom is to **prioritize PC (recall) in blocking and push
precision downstream** [1,2]. The ANN community expresses the identical
trade-off as a **recall vs queries-per-second (QPS)** Pareto curve, the basis
of the ANN-Benchmarks methodology [15].

---

## 3. Classical blocking

### 3.1 Standard (key-based) blocking and blocking keys

The oldest and simplest method. A **Blocking Key Definition (BKD)** is a
function that maps each object to one or more **blocking keys**; objects sharing
a key land in the same **block**, and comparisons happen only within blocks
[1,2]. Example keys: "first 3 letters of surname + birth-year", "zipcode",
"normalized publisher name". Cost drops from n² to the sum of squared block
sizes, Σ|bᵢ|².

- **Schema-based (schema-aware) blocking**: keys are defined over known
  attributes; requires clean, aligned schemas and human/learned key design.
- **Disjoint vs redundancy-positive**: if each object gets one key, blocks are
  disjoint (a partition); if multiple keys, blocks *overlap*
  (**redundancy-positive**), and the number of blocks two objects share becomes
  a similarity signal exploited by meta-blocking (§4) [1,14].

Failure modes: a poor key with skewed distribution produces one giant block
(no reduction) or splits true matches across blocks (recall loss). Errors and
variation in the key attribute directly cause missed pairs — motivating
error-tolerant keys below.

### 3.2 Error-tolerant keys: phonetic, q-gram, suffix

- **Phonetic blocking** (Soundex, Metaphone, NYSIID, Double Metaphone): keys
  are phonetic codes so that spelling variants of names collide ("Catherine" /
  "Katharine"). Robust to typos in names; language/domain specific [2].
- **Q-gram (n-gram) blocking**: keys are (sub)sets of character q-grams; two
  strings sharing enough q-grams co-block. Tolerates edit-distance noise;
  tunable via q and the number of q-grams required [1].
- **Suffix-array blocking**: index suffixes of the key string of length ≥ a
  minimum; entities sharing a suffix co-block. Bounds maximum block size,
  giving good reduction on names/titles [1].

### 3.3 Token blocking (schema-agnostic)

When schemas are dirty, heterogeneous, or unknown (Web data, JSON, RDF),
**token blocking** dispenses with schema entirely: tokenize *all* attribute
values, and create one block per distinct token, containing every entity whose
(any) value contains that token (kept only if shared by ≥2 entities) [1,2].
Every token is a blocking key, so blocks heavily overlap (redundancy-positive).
This is the default *Block Building* step in JedAI/pyJedAI [16]. It has very
high PC but low PQ, and is almost always followed by block processing /
meta-blocking (§4) to prune the resulting explosion of pairs. Variants:
**Attribute-Clustering blocking** (cluster similar attributes first) and
**TF-IDF-weighted / high-value token blocking** to drop uninformative
stop-word-like tokens [1].

### 3.4 Sorted Neighborhood Method (SNM)

Introduced by Hernández & Stolfo for the *merge/purge* problem [13]. Assign each
record a **sorting key**, sort all records, then slide a fixed-size **window** of
w records over the sorted list, comparing only pairs co-occurring in a window.
Complexity: O(n log n) to sort + O(w·n) comparisons — roughly linear in n for
fixed w. The window exploits the intuition that near-duplicates sort near each
other.

- **Trade-off**: w too small → misses matches (low PC); w too large → many
  useless pairs (low PQ). The fixed window is the core weakness.
- **Multi-pass SNM**: run several passes with *different* sorting keys and union
  the candidate pairs — dramatically improves PC at modest window size [13].
- **Extensions**: incrementally/adaptively sized windows, and MapReduce/parallel
  multi-pass SNM for distributed settings.

### 3.5 Canopy clustering

McCallum, Nigam & Ungar's canopy method (KDD 2000) [12] uses a **cheap,
approximate distance** to form overlapping clusters ("canopies") before any
expensive comparison. Pick a random point as a canopy center; add all points
within a loose threshold **T₁** to that canopy; any point within a tighter
threshold **T₂ < T₁** is removed from the pool of future centers (density
control). Repeat until the pool empties. Expensive comparisons then run only
between points **sharing a canopy**. The canopies overlap (redundancy-positive),
and the cheap metric (e.g., inverted-index token overlap, TF-IDF) is what makes
it scale to high-dimensional data. Canopy blocking remains one of the most cited
scaling techniques for record linkage [1,12].

---

## 4. Block processing and meta-blocking

Schema-agnostic block building (token blocking) sacrifices precision for recall,
so a **block-processing** layer restores efficiency *without* re-reading the raw
data — it works purely on the block collection. Papadakis et al.'s
**meta-blocking** line [1,14] is the reference framework. Two families:

**Block cleaning (block-level).** Discard or shrink whole blocks before any
comparison:
- **Block filtering**: remove each entity from its largest (least informative)
  blocks — huge blocks contribute mostly non-matches. A simple, very effective
  cheap pruner [1].
- **Block purging / size-based cleaning**: drop oversized blocks whose sizes
  exceed a threshold (they are dominated by stop-word-like tokens) [1].
- **Comparison propagation**: process blocks in an order and skip any pair
  already compared in an earlier block — eliminates *redundant* comparisons in
  overlapping blocks with **zero** recall loss (raises PQ and RR at no PC cost)
  [1].

**Meta-blocking (comparison-level, graph-based).** Convert the block collection
into a weighted **blocking graph**: nodes are entities, an edge connects two
entities that co-occur in ≥1 block, and the **edge weight** estimates match
likelihood from block co-occurrence. Then prune edges; survivors are the
candidate pairs [1,14]. Weighting schemes:
- **CBS** (Common Blocks Scheme): number of shared blocks.
- **ARCS** (Aggregate Reciprocal Comparisons Scheme): sum of 1/|block| over
  shared blocks — rewards co-occurrence in *small* (discriminative) blocks.
- **ECBS** (Enhanced CBS) and **JS** (Jaccard Scheme): normalize CBS by the
  entities' total block counts.

Pruning strategies over the graph: **Weighted Edge Pruning** (drop edges below
the average weight), **Cardinality Edge Pruning** (keep the globally top-K
edges), and their **node-centric** variants (Weighted/Cardinality Node Pruning,
keeping each node's best edges) [1,14]. **Supervised meta-blocking** learns the
edge classifier from labeled pairs; **GSM** generalizes this for scalability;
**BLAST** [19] injects loose schema-awareness (attribute statistics / entropy)
into the weights. pyJedAI packages block building → block cleaning →
meta-blocking as composable steps [16].

---

## 5. Filtering and set-similarity joins

**Filtering** is the *negative* dual of blocking: instead of grouping likely
matches, it derives cheap **upper bounds** on similarity that let it *prove* a
pair is below the threshold and skip it. This is the domain of **set-similarity
joins** — given sets (token/q-gram sets) and a threshold t on Jaccard/cosine/
overlap/dice, return all pairs with similarity ≥ t, exactly and efficiently
[1,9]. The workhorse is **prefix filtering** and its refinements:

- **Prefix filtering**: impose a global total order on all tokens (e.g., by
  ascending document frequency). For a set x and threshold t, if two sets are
  similar enough, their *prefixes* under this order must share at least one
  token. Indexing only a short prefix of each set (length |x| − ⌈t·|x|⌉ + 1 for
  overlap-style thresholds) and generating candidates from shared prefix tokens
  prunes the vast majority of dissimilar pairs. Basis of the **AllPairs**
  algorithm [1,9].
- **Length filtering**: two sets whose sizes differ too much cannot reach
  threshold t (|x|·t ≤ |y| ≤ |x|/t) — skip by size alone [9].
- **Positional filtering (PPJoin)**: Xiao, Wang, Lin & Yu [9] track the
  *positions* of shared prefix tokens to derive a tighter overlap upper bound,
  eliminating more candidates than plain prefix filtering.
- **Suffix filtering (PPJoin+)**: a divide-and-conquer bound on the *suffix*
  (post-prefix) portion via a pivot token, pruning further [9].
- **AdaptJoin**: adaptively lengthens prefixes per set to trade indexing cost
  against candidate-set size.

These are **exact** methods (no recall loss for the given threshold) — their
recall/precision guarantee is with respect to the similarity threshold, unlike
LSH/ANN which are approximate. Empirical comparisons (Mann, Augsten & Bouros,
PVLDB 2016 [20]) show that for many workloads a well-implemented AllPairs with
length+prefix filtering is competitive with the more elaborate variants, so the
"best" filter is workload-dependent.

---

## 6. Locality-Sensitive Hashing (LSH)

**LSH** is a family of hash functions engineered so that **similar items collide
with high probability and dissimilar items rarely do** — turning
similarity-search into hash-bucket lookup, with *sub-linear* query cost. The key
idea: use several hash functions from an LSH family, bucket items by their hash
values, and treat co-bucketed items as candidate pairs. LSH gives *approximate*
recall (tunable false-negative/false-positive rates) — the opposite guarantee
from exact filtering. Indyk, Broder & Charikar received the 2012 ACM
Kanellakis Award for founding LSH.

### 6.1 MinHash — LSH for Jaccard similarity

Broder's **MinHash** (min-wise independent permutations, 1997/1998) [5]
estimates the **Jaccard similarity** J(A,B) = |A∩B|/|A∪B| of two token/shingle
sets. Apply k random hash permutations to a set; the **minimum** hashed element
under a permutation is the MinHash, and P(minhash(A)=minhash(B)) = J(A,B). A
signature of k MinHashes estimates J with standard error ≈ 1/√k (e.g., k=128–256
is typical). Originally used at AltaVista to dedupe ~30M web pages via shingling.

**Banding (the LSH amplification trick).** Split the k-value signature into **b
bands of r rows** (k = b·r). Two items are candidates if they are identical in
*any whole band*. The probability two items with Jaccard s become candidates is
1 − (1 − sʳ)ᵇ — an S-curve with a tunable threshold ≈ (1/b)^(1/r). Choosing b
and r sets the recall/precision knee. This yields sub-linear candidate lookup.

### 6.2 SimHash — LSH for cosine similarity

Charikar's **SimHash** (STOC 2002) [6] is the LSH family for **cosine
similarity / angular distance**. Draw random hyperplanes through the origin; each
hyperplane contributes one bit (sign of the dot product of the item vector with
a random Gaussian vector). The resulting binary **fingerprint** has the property
that the **Hamming distance** between two fingerprints is proportional to the
*angle* between the original vectors: P(bit agrees) = 1 − θ/π. Near-duplicates
get near-identical fingerprints, enabling fast Hamming-distance near-dup
detection (used at scale by Google for web-page dedup). SimHash operates on
dense weighted feature vectors (e.g., TF-IDF), whereas MinHash operates on sets.

### 6.3 Practical LSH in Python: `datasketch`

`datasketch` (ekzhu) [11] is the de-facto Python library: `MinHash`,
`MinHashLSH` (banded Jaccard index), `MinHashLSHForest` (top-k via variable-
length prefixes), `MinHashLSHEnsemble` (containment/asymmetric set search),
`WeightedMinHash`, plus `HyperLogLog` and an HNSW index. It supports Redis /
Cassandra storage backends for indexes exceeding memory. Results are approximate
(false positives and false negatives are expected and tunable). This is the
natural optional dependency for a Jaccard/token-set blocking strategy.

---

## 7. Approximate Nearest-Neighbor (ANN) search — dense-vector blocking

When objects are represented as **dense embedding vectors** (from a neural
encoder), candidate generation becomes **k-nearest-neighbor retrieval** in
vector space: for each query object, retrieve its top-k nearest vectors as
candidates. Exact kNN is O(n·d) per query; **ANN indexes** trade a little recall
for orders-of-magnitude speedup. Three algorithmic families dominate, all
surveyed/benchmarked by ANN-Benchmarks [15] on the **recall@k vs QPS** Pareto
frontier.

### 7.1 Graph-based: NSW → HNSW

**HNSW** (Hierarchical Navigable Small World; Malkov & Yashunin, IEEE TPAMI
2020) [7] builds a multi-layer proximity graph: each node links to its nearest
neighbors, upper layers hold exponentially fewer nodes as "express lanes," and
search greedily descends from the top layer. It achieves **≈ O(log n)** search
complexity, top-tier recall/QPS, and supports incremental inserts. Key params:
**M** (neighbors per node — memory/recall), **efConstruction** (build-time
candidate list), **efSearch** (query-time candidate list — recall/latency).
Costs: high memory (stores the full graph + vectors) and slower build. On
ANN-Benchmarks, HNSW is consistently on or near the Pareto-optimal front [15].
Python: `hnswlib` [17], and inside FAISS as `IndexHNSW`.

### 7.2 Quantization & inverted files: IVF, PQ, IVF-PQ

- **IVF (Inverted File)**: k-means-cluster the vectors into `nlist` cells; at
  query time probe only the `nprobe` nearest cells. Cuts the search set by
  ~nlist/nprobe. Recall depends on nprobe (more cells = higher recall, slower).
- **PQ (Product Quantization)**: compress each vector by splitting it into `m`
  sub-vectors and quantizing each to one of 2^`nbits` centroids — a d-dim float
  vector becomes m bytes. Enables **billion-scale in-RAM** search and fast
  approximate distances via lookup tables, at some accuracy loss. **OPQ**
  (Optimized PQ) rotates the space first for better codebooks.
- **IVF-PQ**: combine coarse IVF partitioning with PQ-compressed residuals — the
  standard recipe for very large corpora that do not fit uncompressed in RAM
  [10]. Trades memory and recall for scale.

### 7.3 ScaNN — anisotropic vector quantization

Google's **ScaNN** (Guo et al., ICML 2020) [8,18] introduces an **anisotropic
quantization loss** that penalizes quantization error *parallel* to the query
direction more than orthogonal error — because for **maximum inner-product
search (MIPS)** the parallel component dominates the score. This "score-aware"
loss yields more accurate top-k inner products than reconstruction-error PQ. On
the glove-100-angular ANN-Benchmark, ScaNN reported ≈ 2× the QPS of the next
library at equal recall [8,18]. It powers Google's Vertex Matching Engine.

### 7.4 Libraries and the benchmark

- **FAISS** (Meta) [10]: the reference C++/Python library — Flat (exact), IVF,
  PQ, IVF-PQ, HNSW, OPQ, GPU acceleration; the *toolbox* most systems build on.
- **hnswlib** [17]: lightweight header-only HNSW, easy Python bindings.
- **ScaNN** [18]: Google's MIPS-optimized library (TensorFlow/NumPy interfaces).
- **Annoy** (Spotify): tree-based (random projection forests); simple,
  memory-mapped, but generally Pareto-dominated by HNSW on ANN-Benchmarks [15].
- **ANN-Benchmarks** (Aumüller, Bernhardsson & Faithfull, Information Systems
  2020) [15]: the standard evaluation harness — plots recall vs QPS on public
  datasets (SIFT1M, GloVe-100, DEEP1B, GIST), also measuring build time and
  memory. Practical rule of thumb: **HNSW when recall is priority and the index
  fits in RAM; IVF-PQ when the corpus is too big for RAM; LSH when constant
  inserts / theoretical guarantees matter** [15].

---

## 8. Learned / embedding-based (dense) blocking

The above ANN machinery only helps if objects have *good* vector
representations, which couples candidate generation tightly to **featurization**.
Two regimes:

- **Sparse blocking**: bag-of-tokens / q-grams / TF-IDF → set-similarity joins
  or MinHash-LSH. Cheap, interpretable, no training, strong on textual overlap.
- **Dense blocking**: encode each object with a neural model into a dense
  embedding, then ANN-retrieve nearest neighbors as candidates. Captures
  *semantic* similarity (synonyms, paraphrase, cross-lingual) that token overlap
  misses [3,4].

**DeepBlocker** (Thirumuruganathan et al., *"Deep Learning for Blocking in
Entity Matching: A Design Space Exploration,"* PVLDB 14(11), 2021) [3] maps out
a design space of deep-learning blockers, finding that **self-supervised**
tuple embeddings (no labeled data) followed by vector nearest-neighbor search
match or beat classical blocking on recall while staying practical. It
establishes that **self-supervised + ANN** is a strong, label-free dense
blocking recipe (code at `qcri/DeepBlocker`).

**UniBlocker / Universal Dense Blocking** (Wang et al., arXiv:2404.14831, 2024)
[4] pushes toward a *domain-independent* blocker: self-supervised contrastive
pre-training on a generic tabular corpus yields embeddings that transfer to new
domains **without** per-domain fine-tuning, reported as comparable and
*complementary* to state-of-the-art sparse blocking (i.e., combining sparse +
dense candidates improves recall). This reinforces a design stance:
**sparse and dense blocking are complementary; a framework should let users run
and union both.**

---

## 9. Choosing a strategy — a practical map

| Object representation | Similarity notion | Recommended candidate generator | Optional dep |
|---|---|---|---|
| Clean, aligned attributes | equality on a key | Standard blocking (multi-key) | none |
| Names with typos | phonetic / edit distance | Soundex / q-gram / suffix blocking | none / `jellyfish` |
| Token/q-gram sets, threshold t | Jaccard/overlap, **exact** | Prefix + length + positional filtering (PPJoin) | custom / `py_stringmatching` |
| Token sets, approximate | Jaccard, **approx**, huge n | MinHash-LSH (banded) | `datasketch` |
| TF-IDF / weighted vectors | cosine, near-dup | SimHash | `datasketch`-style |
| Records sorted by a key | positional locality | Sorted Neighborhood (multi-pass) | none |
| Dense embeddings, RAM-fit | cosine / inner product | HNSW | `hnswlib` / `faiss` |
| Dense embeddings, billion-scale | cosine / inner product | IVF-PQ / ScaNN | `faiss` / `scann` |
| Dirty, schema-heterogeneous | many/unknown | Token blocking → meta-blocking | `pyjedai`-style |
| Semantic / cross-lingual | learned | Dense (DeepBlocker/UniBlocker) + ANN | encoder + `faiss` |

Blocking is never one algorithm: production ER runs **workflows** (block build →
block clean → meta-block → optionally union with a dense blocker) [1,2,16].

### Glossary of canonical terms

- **Blocking / indexing** — grouping objects so only intra-group pairs are
  compared (positive candidate selection).
- **Filtering** — proving pairs are below threshold and discarding them
  (negative candidate selection; set-similarity joins).
- **Candidate pair (comparison)** — a proposed pair passed to the matcher.
- **Blocking key** — the value(s) that place an object into block(s).
- **Block** — a set of objects sharing a key; may be disjoint or overlapping
  (**redundancy-positive**).
- **Sorted neighborhood** — sort by a key, slide a window, compare within it.
- **Canopy clustering** — cheap-metric overlapping pre-clusters (T₁ loose, T₂
  tight); exact comparison only within shared canopies.
- **Meta-blocking** — restructuring/pruning a block collection via a weighted
  blocking graph to raise precision without re-reading data.
- **LSH** — hash family where similar items collide with high probability.
- **MinHash** — LSH estimator of Jaccard similarity (min of hashed set elements).
- **SimHash** — LSH for cosine/angular similarity (random-hyperplane sign bits).
- **ANN** — approximate nearest-neighbor search over vectors.
- **HNSW** — hierarchical navigable small-world graph ANN index (≈ O(log n)).
- **IVF-PQ** — inverted-file coarse partitioning + product-quantization
  compression for large-scale ANN.
- **ScaNN** — Google's anisotropic-quantization MIPS/ANN library.
- **Set-similarity join** — exact join returning all pairs with set similarity ≥ t.
- **Prefix filtering** — index only a token prefix under a global order; shared
  prefix is necessary for similarity ≥ t.
- **Reduction Ratio (RR)** — 1 − |C|/(|A|·|B|); comparison-space shrinkage.
- **Pair Completeness (PC)** — recall of the candidate set (|D∩C|/|D|).
- **Pairs Quality (PQ)** — precision of the candidate set (|D∩C|/|C|).

---

## 10. Design implications for `equate`

`equate` is a *general* matching framework, so candidate generation must be a
**first-class, swappable stage** — not baked into the matcher. Concretely:

1. **A `CandidateGenerator` (Blocker) strategy interface.** Define one small
   protocol — e.g. `candidate_pairs(A, B) -> Iterable[tuple[IdA, IdB]]` (and a
   self-join form `candidate_pairs(A)`). Every technique in §§3–8 is an
   implementation. Return a **lazy iterator/generator** of pairs, never a
   materialized n×m matrix — this is the whole point (and aligns with the
   iterables skill: stream candidates, don't build them). Provide a trivial
   `AllPairsGenerator` as the correctness baseline / fallback.

2. **Separate "keying" from "grouping."** Standard/token/phonetic/q-gram
   blocking are all *the same algorithm* parameterized by a **key function**
   `object -> Iterable[key]`. Model that as an injected `key_fn` (dependency
   injection, open-closed), so users get standard blocking, phonetic blocking,
   and token blocking by swapping a function — no new class. A `BlockBuilder`
   consumes a `key_fn` and yields blocks; a generic "blocks → candidate pairs"
   adapter (with **comparison propagation** to dedupe overlapping-block pairs)
   is shared by all of them.

3. **An `Index` / retrieval protocol for the ANN & LSH families.** MinHash-LSH,
   SimHash, HNSW, IVF-PQ, ScaNN all fit a common `build(vectors) → query(vec, k)
   → neighbor ids` shape. Wrap each backend behind this protocol so the blocking
   layer is backend-agnostic; the choice of FAISS vs hnswlib vs datasketch vs
   ScaNN becomes configuration, not code.

4. **Optional-dependency boundaries (progressive disclosure).** Keep the core
   pure-Python and dependency-free (all-pairs, standard/token/phonetic blocking,
   sorted neighborhood, canopy, prefix/length filtering can be implemented with
   the stdlib). Gate the heavy backends behind extras:
   `equate[lsh]` → `datasketch`; `equate[ann]` → `faiss` / `hnswlib`;
   `equate[scann]` → `scann`; `equate[dense]` → an embedding encoder. Follow the
   package-UX rule: import lazily, and on `ImportError` raise an informative
   error naming the extra and the install command (a `check_requirements`-style
   helper). Sensible default = token/standard blocking with no extra deps.

5. **Couple blocking to featurization, but keep them decoupled as objects.**
   Dense blocking needs an embedding of each object; sparse blocking needs a
   token/q-gram set; SimHash needs a weighted vector. Expose a
   `Featurizer` / `to_signature` step separate from the generator, so the *same*
   ANN index serves any embedding, and users can plug DeepBlocker/UniBlocker-
   style encoders. Let sparse and dense generators be **composable/unionable**
   (they are complementary [4]).

6. **Meta-blocking as a pluggable post-processor over candidate streams.** Since
   overlapping-block methods over-generate, offer an optional
   `refine(candidates) -> candidates` stage implementing block filtering,
   comparison propagation, and graph-based edge pruning (CBS/ARCS/JS weights,
   weighted/cardinality pruning). This keeps recall-first blocking cheap and
   precision recovery modular.

7. **Make PC / RR / PQ first-class, measurable outputs.** Ship an evaluation
   utility that, given ground-truth pairs, reports **pair completeness (recall),
   reduction ratio, and pairs quality** for any generator — so users can tune
   the recall/efficiency knob empirically (the ANN world's recall-vs-QPS curve).
   This turns "which blocker/threshold?" into a measured decision, not a guess.

8. **Expose the recall/efficiency knob uniformly.** Every generator has one
   dial: window size w (SNM), canopy T₁/T₂, LSH bands/rows or Jaccard threshold,
   ANN efSearch/nprobe/k. Surface a normalized `recall_target` or explicit
   per-strategy params (keyword-only, with smart defaults) so the trade-off is
   discoverable and consistent across strategies.

9. **Design for streaming / incremental and blocking (self) vs matching (two
   collections) symmetrically.** Support both dedup (one collection) and
   record-linkage (two collections) via the same interface, and prefer indexes
   that allow incremental inserts (HNSW, LSH) for growing datasets.

---

## References

[1] G. Papadakis, D. Skoutas, E. Thanos, T. Palpanas. *Blocking and Filtering
Techniques for Entity Resolution: A Survey.* ACM Computing Surveys 53(2),
Article 31, 2020. [arxiv.org/abs/1905.06167](https://arxiv.org/abs/1905.06167)

[2] V. Christophides, V. Efthymiou, T. Palpanas, G. Papadakis, K. Stefanidis.
*An Overview of End-to-End Entity Resolution for Big Data.* ACM Computing
Surveys 53(6), Article 127, 2020.
[dl.acm.org/doi/10.1145/3418896](https://dl.acm.org/doi/10.1145/3418896)

[3] S. Thirumuruganathan, H. Li, N. Tang, M. Ouzzani, Y. Govind, D. Paulsen,
G. Fung, A. Doan. *Deep Learning for Blocking in Entity Matching: A Design Space
Exploration* (DeepBlocker). PVLDB 14(11), 2021.
[dl.acm.org/doi/10.14778/3476249.3476294](https://dl.acm.org/doi/10.14778/3476249.3476294)
· code: [github.com/qcri/DeepBlocker](https://github.com/qcri/DeepBlocker)

[4] T. Wang, et al. *Towards Universal Dense Blocking for Entity Resolution*
(UniBlocker). arXiv:2404.14831, 2024.
[arxiv.org/abs/2404.14831](https://arxiv.org/abs/2404.14831)

[5] A. Z. Broder, M. Charikar, A. M. Frieze, M. Mitzenmacher. *Min-Wise
Independent Permutations.* STOC 1998 / JCSS 2000 (foundational MinHash; orig.
Broder, *On the Resemblance and Containment of Documents*, SEQUENCES 1997).
[cs.princeton.edu/courses/archive/spr04/cos598B/bib/BroderCFM-minwise.pdf](https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/BroderCFM-minwise.pdf)

[6] M. S. Charikar. *Similarity Estimation Techniques from Rounding Algorithms*
(SimHash). STOC 2002, pp. 380–388.
[cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf](https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf)

[7] Yu. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate Nearest
Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI
42(4), 2020 (arXiv:1603.09320).
[arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)

[8] R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, S. Kumar.
*Accelerating Large-Scale Inference with Anisotropic Vector Quantization*
(ScaNN). ICML 2020 (arXiv:1908.10396).
[arxiv.org/abs/1908.10396](https://arxiv.org/abs/1908.10396)

[9] C. Xiao, W. Wang, X. Lin, J. X. Yu. *Efficient Similarity Joins for Near
Duplicate Detection* (PPJoin / PPJoin+). WWW 2008; ACM TODS 36(3), 2011.
[rutgers-db.github.io/cs541-fall19/paper/prefixfilter.pdf](https://rutgers-db.github.io/cs541-fall19/paper/prefixfilter.pdf)

[10] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré,
M. Lomeli, L. Hosseini, H. Jégou. *The Faiss Library.* arXiv:2401.08281, 2024.
[arxiv.org/abs/2401.08281](https://arxiv.org/abs/2401.08281) · code:
[github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

[11] E. Zhu. *datasketch: MinHash, LSH, LSH Forest, Weighted MinHash,
HyperLogLog, HNSW.* Docs:
[ekzhu.com/datasketch](https://ekzhu.com/datasketch/) · code:
[github.com/ekzhu/datasketch](https://github.com/ekzhu/datasketch)

[12] A. McCallum, K. Nigam, L. H. Ungar. *Efficient Clustering of
High-Dimensional Data Sets with Application to Reference Matching* (Canopy
clustering). ACM SIGKDD (KDD) 2000.
[semanticscholar.org/paper/8856b09c032ed4f10ef8367a8f7088fbb891ec2b](https://www.semanticscholar.org/paper/Efficient-clustering-of-high-dimensional-data-sets-McCallum-Nigam/8856b09c032ed4f10ef8367a8f7088fbb891ec2b)

[13] M. A. Hernández, S. J. Stolfo. *The Merge/Purge Problem for Large
Databases* (Sorted Neighborhood Method). ACM SIGMOD 1995.
[semanticscholar.org/paper/8c6ce827ed6821f915895506005fdcf6a8c5b707](https://www.semanticscholar.org/paper/The-merge-purge-problem-for-large-databases-Hern%C3%A1ndez-Stolfo/8c6ce827ed6821f915895506005fdcf6a8c5b707)

[14] G. Papadakis, G. Koutrika, T. Palpanas, W. Nejdl. *Meta-Blocking: Taking
Entity Resolution to the Next Level.* IEEE TKDE 26(8), 2014.
[helios2.mi.parisdescartes.fr/~themisp/publications/tkde13-metablocking.pdf](https://helios2.mi.parisdescartes.fr/~themisp/publications/tkde13-metablocking.pdf)

[15] M. Aumüller, E. Bernhardsson, A. Faithfull. *ANN-Benchmarks: A Benchmarking
Tool for Approximate Nearest Neighbor Algorithms.* Information Systems 87, 2020
(arXiv:1807.05614). [arxiv.org/abs/1807.05614](https://arxiv.org/abs/1807.05614)
· site: [ann-benchmarks.com](https://ann-benchmarks.com/)

[16] AI-team-UoA. *pyJedAI: end-to-end Entity Resolution workflows* (block
building, block cleaning, meta-blocking). Code:
[github.com/AI-team-UoA/pyJedAI](https://github.com/AI-team-UoA/pyJedAI) · docs:
[pyjedai.readthedocs.io](https://pyjedai.readthedocs.io/en/latest/intro.html)

[17] Yu. Malkov et al. *hnswlib: header-only C++/Python HNSW library.*
[github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib)

[18] Google Research. *ScaNN (Scalable Nearest Neighbors).* Code:
[github.com/google-research/google-research/tree/master/scann](https://github.com/google-research/google-research/tree/master/scann)
· blog:
[research.google/blog/announcing-scann-efficient-vector-similarity-search](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/)

[19] G. Simonini, S. Bergamaschi, H. V. Jagadish. *BLAST: a Loosely Schema-aware
Meta-blocking Approach for Entity Resolution.* PVLDB 9(12), 2016.
[vldb.org/pvldb/vol9/p1173-simonini.pdf](http://www.vldb.org/pvldb/vol9/p1173-simonini.pdf)

[20] W. Mann, N. Augsten, P. Bouros. *An Empirical Evaluation of Set Similarity
Join Techniques.* PVLDB 9(9), 2016.
[vldb.org/pvldb/vol9/p636-mann.pdf](http://www.vldb.org/pvldb/vol9/p636-mann.pdf)
