# Sequence & Graph/Structure Matching: Alignment and Network-Alignment Matcher Families

> Round-2 gap-filling research note for the redesign of **equate**, a general
> framework for matching collections of objects. The round-1 corpus
> ([`00`](00-taxonomy-and-terminology.md)–[`10`](10-design-implications-for-equate.md))
> largely assumed *flat* objects reduced to a vector or a string, scored pairwise,
> then decided by a bipartite **linear** assignment ([`03`](03-assignment-and-graph-matching.md)).
> This note covers the matcher families needed when the objects are themselves
> **structured** — time series / audio (sequences) and graphs/networks — where the
> comparison and the correspondence are computed by an *internal* alignment or a
> *quadratic* assignment, not a dot product.

## Abstract

When the objects being matched are sequences (audio, time series, biosequences,
event logs) or graphs (molecules, knowledge graphs, social networks, ASTs), the
`compare` and `match` stages fuse: scoring two objects *means* solving an
alignment between their internal elements, and the alignment path *is* a matching.
This note surveys two such families — **sequence alignment** (Dynamic Time Warping
and its soft/constrained/subsequence/fast variants; Needleman-Wunsch global and
Smith-Waterman local biosequence alignment; edit distance with traceback) and
**graph/structure matching** ((sub)graph isomorphism via VF2, graph edit distance
and its approximations, maximum common subgraph, Weisfeiler-Lehman graph kernels)
plus **network alignment** (aligning nodes across two graphs: IsoRank, FINAL,
REGAL, CONE-Align, GNN/knowledge-graph entity alignment). It gives concrete
algorithms, complexities, tradeoffs, and Python libraries (dtaidistance, tslearn,
fastdtw, Biopython, parasail, networkx, GraKeL, pygmtools, OpenEA), and maps each
family onto equate's `featurize → compare → match` triad and `Matcher` abstraction.
The load-bearing insight: **matching is self-similar** — matching two structured
objects recurses into matching their sub-parts, so equate should model
alignment-based comparators and quadratic (network-alignment) matchers as
first-class, optional-dependency strategy families alongside the flat LAP core.

---

## 1. Why structured objects break the flat pipeline (and how they recurse)

The round-1 triad ([`00`](00-taxonomy-and-terminology.md)) is:

```
objects ──featurize──▶ representations ──compare──▶ score field ──match──▶ matched tuples
```

For flat objects, `compare` is a *generic* metric over precomputed representations
(cosine, Jaccard) — the **featurize-then-compare** route, which is indexable — or an
*opaque* pairwise scorer `h(a, b)` — the **direct-compare** route, which can only
re-rank a candidate set ([`00`](00-taxonomy-and-terminology.md#11-the-core-triad),
[`05`](05-comparison-and-similarity-functions.md)). Structured objects add two
twists that this note is about:

1. **The comparator is itself a matcher.** To score two sequences you *align* their
   samples; to score two graphs you find a node correspondence. The score is a
   by-product of an internal optimization, and that optimization returns an
   **alignment/correspondence** you often want to keep. DTW, Needleman-Wunsch, graph
   edit distance and maximum common subgraph are all **opaque direct comparators**
   in equate's taxonomy — but ones that *also emit an internal matching*.

2. **Matching is self-similar (recursive).** Matching two *collections* of
   sequences = (compare each pair by aligning their elements) → (LAP over the score
   matrix). Matching two *graphs* = matching their *nodes*. The same
   `featurize → compare → match` shape appears **one level down**, over the objects'
   sub-parts. equate's naming thesis — "`equate` is equality relaxed"
   ([`00`](00-taxonomy-and-terminology.md#4-the-equivalence-relation-framing-why-equate-is-not-equality))
   — extends cleanly: relax equality *of structured objects* and you get *structural
   alignment*.

There is a third, orthogonal route worth flagging up front because it rescues
indexability: **structural featurization**. A **graph kernel** (§3.4) or a shingle/
embedding of a sequence turns a structured object into a *flat vector*, which then
uses the ordinary featurize-then-compare path and is blockable/ANN-indexable
([`02`](02-blocking-and-scalable-candidate-generation.md),
[`04`](04-featurization-and-representation.md)). So structured matching splits into:

| Route | Mechanism | Indexable? | Returns internal alignment? | Examples |
|---|---|---|---|---|
| **Alignment comparator** | internal DP / search per pair | no (direct/re-rank only) | yes | DTW, NW/SW, GED, MCS, VF2 |
| **Structural featurizer** | reduce object → vector, then generic metric | yes | no | WL kernel, graphlets, n-gram/shingle, learned graph/seq embedding |
| **Quadratic matcher** | jointly optimize a node correspondence across two graphs | n/a (operates on the pair) | *is* the alignment | IsoRank, FINAL, REGAL, CONE-Align, GNN entity alignment |

---

## 2. Sequence alignment as matchers

Sequences are ordered element streams `x = (x_1, …, x_n)`, `y = (y_1, …, y_m)`. All
the methods below share one skeleton — a **dynamic-programming (DP) accumulated-cost
matrix** with a **traceback** — and differ in step rules, local costs, and boundary
conditions. This is the sequence analogue of the assignment layer in
[`03`](03-assignment-and-graph-matching.md): a global optimum over an exponential
space computed in polynomial time.

### 2.1 The DP alignment template (edit distance with traceback)

**Edit (Levenshtein) distance** is the archetype: the minimum number of
insertions/deletions/substitutions turning `x` into `y`, via a DP recurrence over an
`(n+1)×(m+1)` matrix in **O(nm)** time and space; the **traceback** from the
bottom-right cell recovers the concrete edit script (the alignment). This is exactly
the direct-comparator `h(a,b)` equate already uses through `difflib.SequenceMatcher`
([`05`](05-comparison-and-similarity-functions.md)); `rapidfuzz`/`python-Levenshtein`
are the fast wrappers ([`09`](09-python-ecosystem-landscape.md)). Every method in
this section is a *reparameterization* of this template: change the local cost from
0/1 to a real-valued (dis)similarity, change which moves are legal, and change where
traceback starts/ends.

### 2.2 Dynamic Time Warping (DTW)

**DTW** aligns two *real-valued* time series that may be locally stretched or
compressed in time, minimizing the total cost of a monotone **warping path**
through the local-cost matrix `c(i,j) = ‖x_i − y_j‖`, subject to boundary
(align endpoints), monotonicity, and step-size constraints [1]. Classic **O(nm)**
time and space. Unlike Euclidean distance it handles series of *different lengths*
and phase shifts, which is why it dominates speech, gesture, sensor, and audio
alignment [1]. Caveats to expose: DTW is **not a metric** (it violates the triangle
inequality), and unconstrained warping can produce "pathological" alignments [1].

Two families of refinements matter for a general framework:

- **Global path constraints** bound how far the warping path may stray from the
  diagonal, both to *speed up* (skip cells) and to *regularize* against pathological
  warping: the **Sakoe-Chiba band** (a fixed-width band around the diagonal) [2] and
  the **Itakura parallelogram** (slope-limited) are the two classics; the band is
  usually the better default [2]. A band of width `w` reduces cost to **O(nw)**.
- **Relaxed boundary conditions** give **subsequence / open-ended DTW**: drop the
  requirement that the *whole* of one series be consumed, so a short query is aligned
  to its **best-matching sub-span** of a long series (open-begin, open-end, or both).
  This is the sequence analogue of *partial matching*
  ([`00`](00-taxonomy-and-terminology.md#35-partial-matching-unmatched-items-allowed))
  and the biosequence analogue of local alignment (§2.4). Müller's DTW chapter is the
  standard tutorial reference for the constrained and subsequence variants [1].

**FastDTW.** Salvador & Chan's **FastDTW** [3] approximates DTW in **O(n)** time and
space via a multilevel scheme: solve at a coarse resolution, project the warp path
up, and refine within a radius. It is heavily cited and widely wrapped (`fastdtw`
on PyPI [4]). **Important design caveat:** Wu & Keogh [5] show empirically that in
realistic settings FastDTW is *slower* than a **constrained exact DTW** (a
Sakoe-Chiba-banded `O(nw)` solve), because FastDTW's overhead and radius dominate
until series are extremely long. The practical lesson for equate: prefer a **banded
exact DTW** by default and treat FastDTW as an opt-in for very long series, rather
than reaching for it reflexively.

### 2.3 Soft-DTW

**Soft-DTW** (Cuturi & Blondel, ICML 2017) [6] replaces DTW's hard `min` over
alignments with a **soft-minimum** (log-sum-exp with temperature `γ`) over *all*
alignment paths. Consequences, each with framework relevance:

- **Differentiable everywhere**, with value and gradient computable in **O(nm)** [6].
  This makes it a **loss function** for end-to-end learning (e.g. training a model to
  predict a time series, or learning DTW **barycenters**/averages), which hard DTW
  cannot be.
- It is a *smoothed discrepancy*, **not a distance**: soft-DTW can be **negative**
  and `softDTW(x, x) ≠ 0`; a "divergence" correction restores those properties. Do
  not present it as a metric.
- `γ → 0` recovers hard DTW; larger `γ` gives smoother, more averaging behavior.

Soft-DTW is the natural fit when equate needs a **graded/soft** sequence
correspondence or a differentiable objective — the sequence-level parallel to the
soft/optimal-transport matchers in [`03`](03-assignment-and-graph-matching.md).
It is shipped in **tslearn** (`SoftDTWLossPyTorch`, soft-DTW barycenters,
`TimeSeriesKMeans(metric="softdtw")`) [7] and in Blondel's reference implementation.

### 2.4 Biosequence alignment: Needleman-Wunsch, Smith-Waterman, semi-global

For *symbolic* sequences (DNA/protein, but equally any token stream) the biosequence
tradition contributes two canonical, still-load-bearing algorithms — both the DP
template with a **substitution score matrix** and **gap penalties** instead of unit
edit costs:

- **Needleman-Wunsch (1970)** — **global** alignment: align the *entireties* of both
  sequences end to end, maximizing total score in **O(nm)** [8]. Traceback starts at
  the bottom-right corner. This is the "align the whole of A to the whole of B" case.
- **Smith-Waterman (1981)** — **local** alignment: find the highest-scoring *aligned
  sub-region* of each sequence, in **O(nm)** [9]. The DP matrix is floored at zero and
  traceback starts at the global maximum cell and stops at a zero — so it returns the
  best-matching *fragment pair*, the symbolic analogue of subsequence DTW (§2.2).
- **Semi-global / "glocal"** variants relax gap penalties only at the ends (fitting a
  short sequence into a long one without end-gap penalties) — the biosequence version
  of open-ended matching.
- **Affine gap penalties** (Gotoh, 1982) charge gap-open and gap-extend separately,
  still in **O(nm)**; this is the standard in practice and worth exposing as a
  scoring option rather than a separate algorithm.

The **substitution matrix + gap model is the injectable comparator** here: the *same*
DP engine produces edit distance, DNA alignment, or a domain-specific token alignment
purely by swapping the local scoring scheme — a clean strategy seam.

### 2.5 Complexity & library cheat-sheet (sequences)

| Method | Objective | Complexity | Emits alignment? | Primary Python |
|---|---|---|---|---|
| Edit / Levenshtein | min edit ops | O(nm) | yes (traceback) | `difflib`, `rapidfuzz`, `Levenshtein` [09] |
| **DTW** (unconstrained) | min warping-path cost | O(nm) | yes (warp path) | **`dtaidistance`** [10], `tslearn` [7] |
| DTW (Sakoe-Chiba band) | constrained warp | O(nw) | yes | `dtaidistance` (`window=`) [10] |
| Subsequence / open-ended DTW | best sub-span match | O(nm) | yes | `dtaidistance` (`subsequence`), `tslearn` |
| **FastDTW** | approx DTW | O(n) (but see [5]) | yes | `fastdtw` [4] |
| **Soft-DTW** | soft-min, differentiable | O(nm) | soft (all paths) | **`tslearn`** [7], `soft-dtw` [6] |
| **Needleman-Wunsch** | global score | O(nm) | yes | **Biopython** `Bio.Align.PairwiseAligner` [11], `parasail` [12] |
| **Smith-Waterman** | local score | O(nm) | yes | Biopython [11], **`parasail`** (SIMD) [12] |

Notes: **dtaidistance** [10] is a fast C/Cython DTW (banded, subsequence, DTW
barycenter, `dtw_ndim` for multivariate) with a pure-Python fallback — a good
lightweight default. **tslearn** [7] adds soft-DTW, LB_Keogh lower bounds, and
DTW-based k-means/kNN (scikit-learn-style API). **parasail** [12] is a SIMD-vectorized
C library (with Python bindings) implementing global/semi-global/local alignment at
high throughput; **Biopython** [11] provides the readable reference implementations
(`PairwiseAligner`; the older `pairwise2` is deprecated).

### 2.6 Mapping sequences onto equate's abstractions

- **A sequence-alignment scorer is a `Comparator` of the `direct(h)` kind**
  ([`10`](10-design-implications-for-equate.md#2-the-featurize--compare--match-separation-as-injectable-stages)):
  `compare='dtw'`, `compare='softdtw'`, `compare='needleman_wunsch'`. It is *not*
  featurize-then-compare, so it is **not ANN-indexable** and can only re-rank a
  candidate set produced by a cheaper blocker (e.g. LB_Keogh envelope blocking, or a
  coarse feature vector). This is exactly the doc-00 direct-vs-featurized distinction,
  instantiated for sequences.
- **The scorer should optionally return its internal alignment** (warp path / edit
  script / aligned fragment), not just the scalar, because downstream visualization,
  explanation ([`08`](08-interactive-active-learning-and-hitl.md)), and
  element-level matching need it. Return a small dataclass `Alignment(score, path)`.
- **Collection-level matching is unchanged**: build the score matrix `S[i,j] =
  dtw(A_i, B_j)` (a *dis*similarity, so `sense='minimize'`), then hand it to the flat
  LAP/greedy/soft matchers of [`03`](03-assignment-and-graph-matching.md). The
  structured part lives entirely in the comparator; the assignment layer is reused.
- **Soft-DTW is the sequence entry to the "soft matching" interface** parallel to
  optimal transport ([`03`](03-assignment-and-graph-matching.md#6-soft--graded-matching-via-optimal-transport-sinkhorn)).

---

## 3. Graph & structure matching (within a pair)

Now the objects are graphs `G = (V, E)` with optional node/edge labels. "How similar
are these two graphs, and which of their nodes correspond?" splits into an
NP-hard exact family and tractable approximations/kernels.

### 3.1 (Sub)graph isomorphism — VF2 / VF3

**Graph isomorphism** asks whether two graphs are structurally identical (a
bijection preserving adjacency); **subgraph isomorphism** asks whether one graph
appears inside another. Subgraph isomorphism is **NP-complete**; graph isomorphism's
complexity is famously *quasi-polynomial* (Babai 2016) but not known to be in P. In
practice the **VF2 algorithm** (Cordella, Foggia, Sansone, Vento, IEEE TPAMI 2004)
[13] is the workhorse: a state-space tree search with feasibility rules that prune
aggressively and use **O(V)** memory, handling large graphs well; the successor
**VF3** [14] scales further on dense/large instances. In Python, **networkx**
exposes VF2 via `is_isomorphic`, `GraphMatcher`/`DiGraphMatcher`,
`subgraph_isomorphisms_iter`, and label-aware node/edge match predicates [13].
Isomorphism is the **exact / boolean** end of graph matching — the structural
analogue of `==` — useful for exact dedup of structured keys and as a fast
pre-filter, but too brittle for fuzzy correspondence on its own.

### 3.2 Graph edit distance (GED) and its approximations

**Graph edit distance** generalizes string edit distance to graphs: the minimum-cost
sequence of node/edge insertions, deletions, and substitutions transforming `G_1`
into `G_2` [15]. It is the most general, most interpretable graph *dissimilarity* —
and computing it exactly is **NP-hard**, with exact solvers (A*-based) exponential in
the worst case. The practical toolkit:

- **Exact / anytime**: networkx `graph_edit_distance` (exact but exponential) and
  `optimize_graph_edit_distance` / `optimize_edit_paths`, which yield *successively
  tighter approximations* and accept custom node/edge cost callbacks [15].
- **Bipartite (assignment-based) approximation** — Riesen & Bunke [16]: reduce GED to
  a **linear assignment problem** over node-plus-local-structure costs, solved by the
  Hungarian/Jonker-Volgenant algorithm in **O(V³)**, giving an *upper bound* on true
  GED. This is a direct bridge back to [`03`](03-assignment-and-graph-matching.md):
  approximate GED **is** a LAP over enriched node costs. Bipartite GED is the standard
  scalable default; libraries like GMatch4py and pygmtools (§3.5) implement it.

GED with custom cost functions is the right choice when you need an **interpretable,
tunable** graph dissimilarity and a **node correspondence** together.

### 3.3 Maximum common subgraph (MCS)

The **maximum common subgraph** is the largest subgraph (by nodes/edges) shared
between two graphs; graph similarity can be derived from its size, and it yields a
partial node correspondence. MCS is **NP-hard** [17]. Modern exact solvers —
notably **McSplit** (McCreesh, Prosser, Trimble, IJCAI 2017) [17], a branch-and-bound
with a compact partitioning data structure — push tractability to graphs of a few
tens of nodes; beyond that, heuristics/portfolios are used. MCS is the tool of choice
in **cheminformatics** (aligning molecular scaffolds) and anywhere the *shared
substructure itself* (not just a score) is the deliverable. It is closely related to
GED (both express structural overlap) and to maximum-clique on the product graph.

### 3.4 Weisfeiler-Lehman kernels and graph kernels (the featurizer route)

Graph **kernels** sidestep NP-hardness by mapping each graph to an (implicit or
explicit) **feature vector**, then comparing vectors — the *featurize-then-compare*
route, which restores **indexability and blocking**
([`02`](02-blocking-and-scalable-candidate-generation.md),
[`04`](04-featurization-and-representation.md)). The dominant family is the
**Weisfeiler-Lehman (WL) subtree kernel** (Shervashidze et al., JMLR 2011) [18]:
iteratively **relabel** each node by hashing the multiset of its neighbors' labels
(the 1-dimensional WL / color-refinement test of isomorphism), and use the histogram
of labels across `h` iterations as the feature map. Runtime is **linear in the number
of edges and in `h`** [18] — extremely scalable — and WL is the conceptual ancestor
of modern **message-passing GNNs** (a GNN is at most as expressive as the WL test).
Other kernels trade expressivity for cost: **shortest-path**, **random-walk**,
**graphlet-sampling**, **subtree** kernels. **GraKeL** (Siglidis et al., JMLR 2020)
[19] is the scikit-learn-compatible library implementing WL, shortest-path,
random-walk, graphlet, and ~15 more, so a graph classifier/matcher is a `Pipeline`.

**Design payoff:** a WL/graphlet kernel is a **`Featurizer` that turns a graph into a
vector**, letting equate reuse cosine/ANN/LSH blocking for *graphs* exactly as for
text — no special-casing in the match layer. This is the structured-featurizer route
of §1, and it is the scalable default for *scoring many graph pairs*.

### 3.5 Python: networkx, GraKeL, pygmtools

- **networkx** — VF2 isomorphism, exact and approximate GED, general graph
  plumbing; pure Python, already a light dep in equate's `util.py` pattern.
- **GraKeL** [19] — graph *kernels* (featurizer route), scikit-learn API.
- **pygmtools** (Wang et al., JMLR 2024) [20] — a dedicated **graph-matching** toolkit:
  linear (Sinkhorn, Hungarian) and **quadratic** assignment solvers (spectral matching,
  RRWM random-walk, IPFP), plus **learning-based neural** matchers, over NumPy/PyTorch/
  Jittor/Paddle backends. This is where the *quadratic assignment problem* (QAP) —
  the true model of "match nodes so that edges are preserved" — is solved, bridging
  §3 (within-pair graph matching) and §4 (network alignment).

---

## 4. Network alignment (node correspondence across two graphs)

**Network alignment** is the graph analogue of entity resolution: given two graphs
`G_1, G_2` (often with node/edge attributes), find a correspondence between their
**nodes** that preserves structure — "who is the same node across the two networks."
Applications: cross-social-network user linking, protein-interaction-network
comparison across species, knowledge-graph fusion, and ontology/schema alignment
([`07`](07-schema-and-ontology-matching.md)). Recent surveys (Saxena & Chandra,
IJCAI 2024 [21]) organize the field into the families below.

### 4.1 The quadratic core: QAP, and why LAP is not enough

The formal objective is a **Quadratic Assignment Problem (QAP)**: find a permutation
`P` maximizing edge agreement `‖A_1 − P A_2 Pᵀ‖` (plus a node-attribute term),
where `A_i` are adjacency matrices. QAP is **NP-hard** — strictly harder than the
**linear** assignment problem (LAP) at the heart of
[`03`](03-assignment-and-graph-matching.md), because the cost of matching `u↔u'`
*depends on how their neighbors are matched*. Every method below is a tractable
relaxation or heuristic for this quadratic objective. **This is the key theoretical
bridge**: flat matching (doc 03) is LAP; structural matching is QAP, and QAP
relaxations typically *reduce to* one or more LAP/Sinkhorn solves in an inner loop.

### 4.2 Spectral / consistency methods — IsoRank, FINAL

- **IsoRank** (Singh, Xu, Berger, PNAS 2008) [22] — the pioneering global aligner.
  Intuition: *two nodes match well if their neighbors match well.* It computes a
  cross-network node-similarity matrix as the stationary distribution of a
  PageRank-style random walk on the **tensor-product graph**, blending topology with
  a prior sequence/attribute similarity, then extracts an alignment (greedy or LAP)
  from that matrix. Won the RECOMB Test-of-Time award; still the reference baseline.
- **FINAL** (Zhang & Tong, KDD 2016) [23] — recasts IsoRank's consistency as an
  optimization and generalizes it to **attributed** networks (node/edge attributes),
  with closed-form and iterative solvers. It *reduces to IsoRank when attributes are
  absent*, making it the natural attributed generalization.

These are **topology + optional attributes** methods that output a full
cross-network similarity matrix — from which any doc-03 matcher (greedy, LAP,
top-k) extracts the final node correspondence.

### 4.3 Embedding-based methods — REGAL / xNetMF, CONE-Align

Instead of a random walk, learn **node embeddings** and match in embedding space:

- **REGAL** (Heimann, Shen, Safavi, Koutra, CIKM 2018) [24] with its **xNetMF**
  embedding: represent each node by its *structural identity* (degree-histogram
  signatures over `k`-hop neighborhoods) plus attributes, factorize the resulting
  cross-network similarity implicitly (Nyström low-rank), then align by nearest
  neighbor in embedding space (greedy / KD-tree). Fast and **unsupervised** — no seed
  alignments needed.
- **CONE-Align** (Chen, Heimann, Vahedian, Koutra, CIKM 2020) [25] — first learn
  *within-network* proximity-preserving embeddings **independently** per graph, then
  **align the two embedding subspaces** (Wasserstein-Procrustes) before matching
  nodes. Notably more **robust to structural noise** than REGAL. It is a clean
  three-stage pipeline (embed → subspace-align → node-match) that mirrors equate's own
  stage decomposition.

Embedding-based aligners are essentially **featurize (structural node embedding) →
compare (embedding similarity) → match (LAP/greedy)** applied *inside* one graph
pair — the triad recursing one level down, over nodes.

### 4.4 GNN and knowledge-graph entity alignment — RREA, OpenEA

For **knowledge graphs** (multi-relational, attribute-rich), **entity alignment (EA)**
matches equivalent entities across two KGs, typically **semi-supervised** from seed
pairs. The modern approach is GNN embedding: encode each KG with a
(relation-aware) GNN so that seed-aligned entities are close, then match by embedding
distance (nearest-neighbor or LAP, with a **cross-domain-similarity/CSLS** correction
for hubness). **RREA** (Relational Reflection Entity Alignment, Mao et al., CIKM 2020)
[26] is a strong, widely-cited representative: it applies per-relation **orthogonal
reflection** transformations that produce relation-specific entity embeddings while
preserving norms and relative distances. **OpenEA** (Sun et al., PVLDB 2020) [27] is
the standard **benchmark and library**: cross-lingual (EN-FR, EN-DE) and cross-KG
(DBpedia-Wikidata, DBpedia-YAGO) datasets at 15K/100K scales with density variants,
plus reference implementations of a dozen EA methods — the go-to for evaluation.
Surveys of embedding-based KG-EA give the taxonomy and caveats (test-set leakage,
name-bias) [28]. Entity alignment is the KG-flavored instance of the same
featurize→compare→match recursion, and connects directly to
[`06`](06-deep-learning-and-llm-entity-matching.md) (learned matchers) and
[`07`](07-schema-and-ontology-matching.md) (ontology matching).

### 4.5 The optimal-transport / Gromov-Wasserstein connection

Matching two graphs with **no shared node feature space** is precisely what
**Gromov-Wasserstein (GW) optimal transport** solves — align by comparing *intra-graph*
distance structures — already flagged in
[`03`](03-assignment-and-graph-matching.md#6-soft--graded-matching-via-optimal-transport-sinkhorn)
(`ot.gromov.gromov_wasserstein` in POT). A growing line of network-alignment work
(GW learning; joint OT-and-embedding) sits at this intersection, and it is the
**soft** counterpart to the hard node-permutation methods above. equate's soft-matching
interface should therefore reach into graphs via GW, not only vectors.

### 4.6 Python: pygmtools, PLANETALIGN, OpenEA

- **pygmtools** [20] — QAP solvers (spectral, RRWM, IPFP) + neural matchers; the
  general engine for two-graph and multigraph matching.
- **PLANETALIGN** (arXiv 2025) [29] — a recent comprehensive Python **benchmarking**
  library bundling IsoRank/FINAL/REGAL/CONE-Align and more behind a common API — the
  best single place to see the family compared and to borrow interface design.
- **OpenEA** [27] — KG entity-alignment datasets + methods.

### 4.7 Network alignment cheat-sheet

| Family | Representative | Signal used | Supervision | Inner solver | Python |
|---|---|---|---|---|---|
| Spectral / consistency | IsoRank [22] | topology (+prior) | none | random walk → greedy/LAP | networkx-buildable, PLANETALIGN [29] |
| Consistency, attributed | FINAL [23] | topology + attrs | none | fixed-point | PLANETALIGN [29] |
| Embedding (structural) | REGAL/xNetMF [24] | structural identity | none | NN in embedding space | GemsLab REGAL, PLANETALIGN |
| Embedding (subspace-aligned) | CONE-Align [25] | proximity embeddings | none | Wasserstein-Procrustes + match | GemsLab, PLANETALIGN |
| GNN / KG entity alignment | RREA [26] | relations + attrs | seed pairs | GNN embed → NN/CSLS | OpenEA [27] |
| Quadratic solvers | RRWM/IPFP/spectral | topology + node affinity | none | QAP relaxation | **pygmtools** [20] |
| Soft / cross-space | Gromov-Wasserstein | intra-graph distances | none | entropic GW | POT (doc 03) |

---

## 5. Cross-cutting map to the equate abstractions

| Structured method | equate stage it instantiates | Return type | Optional-dependency boundary |
|---|---|---|---|
| DTW / soft-DTW / NW / SW | `Comparator` (`direct` kind) + `Alignment` | dissimilarity + warp/edit path | `equate[timeseries]`→dtaidistance/tslearn; `equate[bio]`→Biopython/parasail |
| Edit distance | `Comparator` (`direct`) | distance + script | core (`difflib`) / `equate[fuzzy]`→rapidfuzz |
| WL / graphlet kernel | **`Featurizer`** (graph→vector) | vector (indexable) | `equate[graph-kernels]`→GraKeL |
| VF2 isomorphism | `Comparator` (boolean/exact) | bool + node map | `equate[graph]`→networkx |
| Graph edit distance | `Comparator` (`direct`) → LAP inside | distance + node map | `equate[graph]`→networkx; bipartite GED = doc-03 LAP |
| Maximum common subgraph | `Comparator` (`direct`) | shared subgraph + partial map | `equate[graph-matching]`→pygmtools/GMatch4py |
| IsoRank/FINAL/REGAL/CONE-Align | **quadratic `Matcher`** (over two graphs) | node correspondence | `equate[network-align]`→pygmtools/PLANETALIGN |
| KG entity alignment (RREA) | quadratic `Matcher` (seeded) | entity correspondence | `equate[kg]`→OpenEA (heavy: torch) |
| Gromov-Wasserstein | **soft** quadratic `Matcher` | transport plan | `equate[ot]`→POT (doc 03) |

Two structural observations for the API:

1. **The `Matcher` protocol needs a second arity.** Today
   `matcher(scores) → pairs` consumes a *precomputed score matrix*
   ([`03`](03-assignment-and-graph-matching.md#11-the-optimization-layer-and-why-it-deserves-a-clean-seam)).
   Network aligners consume **two structured objects (+ optional node features)** and
   *produce* the scores and the correspondence jointly — they cannot be expressed as
   `matcher(scores)`. equate should recognize a distinct
   **`QuadraticMatcher(G1, G2, *, node_affinity=None, seeds=None) → correspondence`**
   family alongside the flat `Matcher`, with the flat matcher usable as the *inner*
   extraction step (greedy/LAP over the aligner's similarity matrix).

2. **Alignment comparators want to emit their internal matching.** A `Comparator`
   that internally solves a DP/search should optionally return a structured
   `Alignment` (path + score), so explanations and element-level correspondence are
   available without recomputation ([`08`](08-interactive-active-learning-and-hitl.md)).

---

## 6. Glossary

- **Dynamic Time Warping (DTW).** Elastic alignment of two real-valued sequences via
  a minimum-cost monotone warping path; O(nm), not a metric.
- **Sakoe-Chiba band / Itakura parallelogram.** Global constraints limiting warp-path
  deviation from the diagonal — speed and anti-pathology regularization.
- **Subsequence / open-ended DTW.** DTW with relaxed endpoint boundaries: align a
  query to its best sub-span of a longer series (sequence "partial matching").
- **FastDTW.** O(n) multilevel DTW approximation; empirically often slower than a
  banded exact DTW [5].
- **Soft-DTW.** Differentiable soft-min over all DTW alignments (Cuturi & Blondel
  2017); a learnable loss, not a distance (can be negative).
- **Edit / Levenshtein distance.** Min-cost insert/delete/substitute to transform one
  string into another; the DP-with-traceback template.
- **Needleman-Wunsch.** Global biosequence alignment (whole vs whole), O(nm).
- **Smith-Waterman.** Local biosequence alignment (best sub-region pair), O(nm).
- **Substitution matrix / affine gap.** The injectable scoring scheme (per-symbol
  costs, gap-open/gap-extend) shared by all biosequence DP variants.
- **Graph isomorphism / subgraph isomorphism.** Exact structural identity / containment;
  subgraph case NP-complete.
- **VF2 / VF3.** Practical state-space (sub)graph isomorphism algorithms; O(V) memory.
- **Graph edit distance (GED).** Min-cost node/edge edits transforming one graph into
  another; NP-hard; **bipartite approximation** reduces it to a LAP (Riesen-Bunke).
- **Maximum common subgraph (MCS).** Largest shared subgraph of two graphs; NP-hard;
  McSplit is a leading exact solver.
- **Graph kernel / Weisfeiler-Lehman kernel.** Map a graph to a feature vector (WL:
  iterative neighbor-label hashing) for indexable, scalable graph comparison; WL is
  linear in edges × iterations and bounds GNN expressivity.
- **Network alignment.** Cross-graph **node** correspondence preserving structure; the
  graph analogue of entity resolution; formally a **QAP**.
- **Quadratic Assignment Problem (QAP).** Assignment whose costs depend on *pairs* of
  assignments (edge preservation); NP-hard; the model network alignment relaxes.
- **IsoRank / FINAL.** Consistency/spectral aligners ("match if neighbors match");
  FINAL is the attributed generalization.
- **REGAL / CONE-Align.** Unsupervised embedding-based aligners (structural-identity
  embeddings; subspace-aligned proximity embeddings).
- **Knowledge-graph entity alignment (EA).** Seed-supervised, GNN-embedding matching of
  equivalent entities across KGs (e.g. RREA); benchmarked by **OpenEA**.
- **Gromov-Wasserstein.** Optimal transport across *incomparable* spaces using only
  intra-set distances; the soft, cross-space network-alignment objective.

---

## 7. Design implications for equate

1. **Add an alignment-comparator category (`direct` scorers that emit a path).**
   Register `compare='dtw' | 'softdtw' | 'needleman_wunsch' | 'smith_waterman' |
   'edit'` returning either a scalar or a small `Alignment(score, path)` dataclass.
   These are **opaque direct comparators** ([`00`](00-taxonomy-and-terminology.md#11-the-core-triad),
   [`05`](05-comparison-and-similarity-functions.md)): document that they are
   **not ANN-indexable** and can only re-rank a candidate set. Default sequence
   dissimilarity should be a **Sakoe-Chiba-banded exact DTW** (via `dtaidistance`),
   *not* FastDTW, per Wu & Keogh [5].

2. **Keep the collection-level matcher untouched.** A structured comparator produces
   `S[i,j]`; the flat greedy/LAP/soft matchers of
   [`03`](03-assignment-and-graph-matching.md) consume it. Structured matching is a
   *comparator* concern, so no change to the assignment layer — with one flag:
   alignment scores are **dissimilarities**, so route them through the SSOT
   `_to_cost(scores, sense='minimize')` helper
   ([`10`](10-design-implications-for-equate.md#1-the-core-canonical-case-to-optimize-first)).

3. **Model graph kernels as `Featurizer`s, not matchers.** WL/graphlet/shortest-path
   kernels (GraKeL) turn a graph into a vector, restoring the **indexable
   featurize-then-compare path** and letting equate reuse cosine + ANN/LSH blocking
   ([`02`](02-blocking-and-scalable-candidate-generation.md),
   [`04`](04-featurization-and-representation.md)) for graphs. This is the scalable
   default for *scoring many graph pairs*; reserve GED/MCS/VF2 for small candidate
   sets after blocking.

4. **Introduce a distinct `QuadraticMatcher` protocol** for network alignment:
   `align(G1, G2, *, node_affinity=None, seeds=None) → correspondence (+ scores)`.
   It cannot be expressed as `matcher(scores)` because it *produces* the scores
   jointly with the correspondence. Ship IsoRank as a readable in-house baseline;
   wrap **pygmtools** [20] (QAP solvers) and defer to **PLANETALIGN** [29] /
   **OpenEA** [27] for the embedding/GNN families. Let the flat LAP/greedy matcher be
   the reusable *inner* extraction step over the aligner's similarity matrix — SSOT.

5. **Unify soft structured matching under the existing OT interface.** Soft-DTW
   (sequences) and Gromov-Wasserstein (graphs) are the structured members of the
   soft-matching family already scoped in
   [`03`](03-assignment-and-graph-matching.md#6-soft--graded-matching-via-optimal-transport-sinkhorn).
   Expose them through the same `soft_match(...) → plan` + `harden(plan)` seam, so
   sequences, graphs, and vectors share one soft path.

6. **Draw hard optional-dependency boundaries; keep the core `numpy`/`scipy`-light.**
   Suggested extras with lazy imports and `check_requirements`-style errors
   ([`09`](09-python-ecosystem-landscape.md),
   [`10`](10-design-implications-for-equate.md#7-wrap-vs-reimplement-per-layer)):
   `equate[timeseries]`→dtaidistance/tslearn; `equate[bio]`→Biopython/parasail;
   `equate[graph]`→networkx (VF2, GED); `equate[graph-kernels]`→GraKeL;
   `equate[graph-matching]`→pygmtools; `equate[network-align]`→PLANETALIGN;
   `equate[kg]`→OpenEA (heavy: torch); `equate[ot]`→POT. **Reimplement almost
   nothing computational** here — own the orchestration and the protocol seams only.

7. **Surface the self-similar recursion as a first-class design principle.** "Match a
   collection of sequences/graphs" = "compare each pair by matching their sub-parts"
   → "assign at the collection level." The `featurize → compare → match` triad appears
   at every level; a `structured_comparator = build_from(inner_featurize,
   inner_compare, inner_match)` factory makes DTW-vs-NW-vs-GED a *configuration* of the
   inner scoring/assignment rather than three bespoke code paths — the strongest
   expression of equate's "equality relaxed, recursively"
   thesis ([`00`](00-taxonomy-and-terminology.md#4-the-equivalence-relation-framing-why-equate-is-not-equality)).

8. **Be numerically and semantically honest in the API.** Document: DTW/soft-DTW are
   **not metrics** (no triangle inequality; soft-DTW can be negative); GED, MCS,
   subgraph isomorphism, and QAP are **NP-hard** (so exact solvers do not scale —
   guide users to banded/approximate/kernel variants); FastDTW is an approximation that
   is *often slower* than banded exact DTW [5]; KG entity-alignment benchmarks have
   known leakage/name-bias caveats [28]. Users should choose with eyes open.

Net: equate should treat structured matching as **two new strategy families layered on
the existing seams** — *alignment comparators* (sequences and within-pair graph
matching, feeding the unchanged flat assignment layer) and *quadratic matchers*
(network alignment, a new protocol arity) — with graph kernels reclaimed as
featurizers, soft variants folded into the OT interface, and every heavy engine
behind a lazy optional-dependency boundary.

---

## References

1. Müller M. Dynamic Time Warping. In: *Information Retrieval for Music and Motion*, ch. 4, pp. 69-84. Springer, 2007. [https://link.springer.com/book/10.1007/978-3-540-74048-3](https://link.springer.com/book/10.1007/978-3-540-74048-3) · [FMP tutorial notebook](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html)
2. Sakoe H, Chiba S. Dynamic Programming Algorithm Optimization for Spoken Word Recognition. *IEEE Trans. Acoustics, Speech, and Signal Processing* 26(1):43-49, 1978. [https://ieeexplore.ieee.org/document/1163055](https://ieeexplore.ieee.org/document/1163055)
3. Salvador S, Chan P. Toward Accurate Dynamic Time Warping in Linear Time and Space (FastDTW). *Intelligent Data Analysis* 11(5):561-580, 2007. [https://cs.fit.edu/~pkc/papers/tdm04.pdf](https://cs.fit.edu/~pkc/papers/tdm04.pdf)
4. `fastdtw` — Python implementation of FastDTW. [https://pypi.org/project/fastdtw/](https://pypi.org/project/fastdtw/)
5. Wu R, Keogh EJ. FastDTW is Approximate and Generally Slower than the Algorithm it Approximates. *IEEE Trans. Knowledge and Data Engineering* 34(8), 2022. [https://arxiv.org/abs/2003.11246](https://arxiv.org/abs/2003.11246)
6. Cuturi M, Blondel M. Soft-DTW: a Differentiable Loss Function for Time-Series. *ICML* 2017 (PMLR 70:894-903). [https://arxiv.org/abs/1703.01541](https://arxiv.org/abs/1703.01541) · [PMLR](https://proceedings.mlr.press/v70/cuturi17a.html) · [code](https://github.com/mblondel/soft-dtw)
7. Tavenard R, Faouzi J, Vandewiele G, et al. Tslearn, A Machine Learning Toolkit for Time Series Data. *JMLR* 21(118):1-6, 2020. [https://jmlr.org/papers/v21/20-091.html](https://jmlr.org/papers/v21/20-091.html) · [GitHub](https://github.com/tslearn-team/tslearn)
8. Needleman SB, Wunsch CD. A General Method Applicable to the Search for Similarities in the Amino Acid Sequence of Two Proteins. *Journal of Molecular Biology* 48(3):443-453, 1970. [https://doi.org/10.1016/0022-2836(70)90057-4](https://doi.org/10.1016/0022-2836%2870%2990057-4)
9. Smith TF, Waterman MS. Identification of Common Molecular Subsequences. *Journal of Molecular Biology* 147(1):195-197, 1981. [https://doi.org/10.1016/0022-2836(81)90087-5](https://doi.org/10.1016/0022-2836%2881%2990087-5)
10. Meert W, et al. DTAIDistance: Time Series Distances (fast DTW in C). [https://dtaidistance.readthedocs.io/](https://dtaidistance.readthedocs.io/) · [GitHub](https://github.com/wannesm/dtaidistance)
11. Cock PJA, Antao T, Chang JT, et al. Biopython: Freely Available Python Tools for Computational Molecular Biology and Bioinformatics. *Bioinformatics* 25(11):1422-1423, 2009. [https://doi.org/10.1093/bioinformatics/btp163](https://doi.org/10.1093/bioinformatics/btp163) · [PairwiseAligner docs](https://biopython.org/docs/dev/Tutorial/chapter_pairwise2.html)
12. Daily J. Parasail: SIMD C Library for Global, Semi-Global, and Local Pairwise Sequence Alignments. *BMC Bioinformatics* 17:81, 2016. [https://doi.org/10.1186/s12859-016-0930-z](https://doi.org/10.1186/s12859-016-0930-z) · [parasail-python](https://github.com/jeffdaily/parasail-python)
13. Cordella LP, Foggia P, Sansone C, Vento M. A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs (VF2). *IEEE Trans. Pattern Analysis and Machine Intelligence* 26(10):1367-1372, 2004. [https://ieeexplore.ieee.org/document/1323804](https://ieeexplore.ieee.org/document/1323804) · [NetworkX VF2](https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html)
14. Carletti V, Foggia P, Saggese A, Vento M. Introducing VF3: A New Algorithm for Subgraph Isomorphism. *GbRPR* 2017. [https://link.springer.com/chapter/10.1007/978-3-319-58961-9_12](https://link.springer.com/chapter/10.1007/978-3-319-58961-9_12)
15. NetworkX. Graph edit distance: `graph_edit_distance`, `optimize_graph_edit_distance`, `optimize_edit_paths`. [https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.optimize_graph_edit_distance.html](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.optimize_graph_edit_distance.html)
16. Riesen K, Bunke H. Approximate Graph Edit Distance Computation by Means of Bipartite Graph Matching. *Image and Vision Computing* 27(7):950-959, 2009. [https://doi.org/10.1016/j.imavis.2008.04.004](https://doi.org/10.1016/j.imavis.2008.04.004)
17. McCreesh C, Prosser P, Trimble J. A Partitioning Algorithm for Maximum Common Subgraph Problems (McSplit). *IJCAI* 2017. [https://www.ijcai.org/proceedings/2017/0099.pdf](https://www.ijcai.org/proceedings/2017/0099.pdf)
18. Shervashidze N, Schweitzer P, van Leeuwen EJ, Mehlhorn K, Borgwardt KM. Weisfeiler-Lehman Graph Kernels. *JMLR* 12:2539-2561, 2011. [https://jmlr.org/papers/v12/shervashidze11a.html](https://jmlr.org/papers/v12/shervashidze11a.html)
19. Siglidis G, Nikolentzos G, Limnios S, Giatsidis C, Skianis K, Vazirgiannis M. GraKeL: A Graph Kernel Library in Python. *JMLR* 21(54):1-5, 2020. [https://jmlr.org/papers/volume21/18-370/18-370.pdf](https://jmlr.org/papers/volume21/18-370/18-370.pdf) · [GitHub](https://github.com/ysig/GraKeL)
20. Wang R, Guo Z, Pan W, et al. Pygmtools: A Python Graph Matching Toolkit. *JMLR* 25(33):1-7, 2024. [https://jmlr.org/papers/v25/23-0572.html](https://jmlr.org/papers/v25/23-0572.html) · [docs](https://pygmtools.readthedocs.io/)
21. Saxena S, Chandra J. A Survey on Network Alignment: Approaches, Applications and Future Directions. *IJCAI* 2024. [https://www.ijcai.org/proceedings/2024/908](https://www.ijcai.org/proceedings/2024/908)
22. Singh R, Xu J, Berger B. Global Alignment of Multiple Protein Interaction Networks with Application to Functional Orthology Detection (IsoRank). *PNAS* 105(35):12763-12768, 2008. [https://www.pnas.org/doi/10.1073/pnas.0806627105](https://www.pnas.org/doi/10.1073/pnas.0806627105)
23. Zhang S, Tong H. FINAL: Fast Attributed Network Alignment. *ACM SIGKDD (KDD)* 2016. [https://dl.acm.org/doi/10.1145/2939672.2939766](https://dl.acm.org/doi/10.1145/2939672.2939766) · [extended (TKDE)](https://tonghanghang.org/pdfs/tkde18_final.pdf)
24. Heimann M, Shen H, Safavi T, Koutra D. REGAL: Representation Learning-based Graph Alignment. *ACM CIKM* 2018. [https://arxiv.org/abs/1802.06257](https://arxiv.org/abs/1802.06257) · [code](https://github.com/GemsLab/REGAL)
25. Chen X, Heimann M, Vahedian F, Koutra D. CONE-Align: Consistent Network Alignment with Proximity-Preserving Node Embedding. *ACM CIKM* 2020. [https://arxiv.org/abs/2005.04725](https://arxiv.org/abs/2005.04725) · [code](https://github.com/GemsLab/CONE-Align)
26. Mao X, Wang W, Xu H, Wu Y, Lan M. Relational Reflection Entity Alignment (RREA). *ACM CIKM* 2020. [https://arxiv.org/abs/2008.07962](https://arxiv.org/abs/2008.07962)
27. Sun Z, Zhang Q, Hu W, Wang C, Chen M, Akrami F, Li C. A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs (OpenEA). *PVLDB* 13(11):2326-2340, 2020. [https://www.vldb.org/pvldb/vol13/p2326-sun.pdf](https://www.vldb.org/pvldb/vol13/p2326-sun.pdf) · [code](https://github.com/nju-websoft/OpenEA)
28. Zhu R, et al. A Survey: Knowledge Graph Entity Alignment Research Based on Graph Embedding. *Artificial Intelligence Review* 57, 2024. [https://link.springer.com/article/10.1007/s10462-024-10866-4](https://link.springer.com/article/10.1007/s10462-024-10866-4)
29. Xiong Z, et al. PLANETALIGN: A Comprehensive Python Library for Benchmarking Network Alignment. *arXiv* 2505.21366, 2025. [https://arxiv.org/abs/2505.21366](https://arxiv.org/abs/2505.21366)
