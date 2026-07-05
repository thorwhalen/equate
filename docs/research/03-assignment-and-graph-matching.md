# Assignment, Bipartite Matching & the Optimization Layer

> Research note for the redesign of **equate**, a general framework for matching
> collections of objects. This document covers the *decision* layer: given a
> matrix (or stream) of pairwise scores/costs, how do we choose *which* pairs to
> match? It surveys the algorithmic core — bipartite matching, the linear
> assignment problem, stable matching, k-best enumeration, optimal transport,
> generalized/many-to-many assignment, and online re-optimization — with concrete
> algorithms, complexities, Python libraries, and their tradeoffs, and closes with
> design implications for equate's optimization layer.

## Abstract

A matching framework separates cleanly into two layers: a **scoring layer** that
produces a (possibly sparse, possibly rectangular) matrix of pairwise affinities,
and an **optimization layer** that turns that matrix into a concrete set of
matched pairs. This note is about the second layer. The central object is the
**linear assignment problem (LAP)** — find the one-to-one correspondence of
minimum total cost — solvable in polynomial time by the Hungarian/Kuhn-Munkres
and (faster in practice) Jonker-Volgenant algorithms, or via min-cost flow and
auction methods. Around this core sit important variants that change the *objective
and semantics*: maximum-cardinality matching (Hopcroft-Karp), maximum-weight
matching (Edmonds' blossom), **stable** matching (Gale-Shapley, which optimizes
*stability*, not total cost), **k-best** enumeration (Murty, essential for
interactive re-optimization), **soft/fractional** matching via optimal transport
(Sinkhorn), and **many-to-many / generalized** assignment (NP-hard). The key
design lesson for equate is that "matching" is not one algorithm but a family of
strategies with different objectives, return types, and complexity/precision
tradeoffs, best exposed behind a small `Matcher` protocol with explicit
minimize/maximize sense, sparse routing, and optional-dependency boundaries.

---

## 1. The optimization layer and why it deserves a clean seam

Almost every matching task decomposes into three stages:

1. **Candidate generation & scoring** — compute an affinity/cost `score(a, b)`
   for pairs `(a ∈ A, b ∈ B)`. The result is a dense or sparse matrix `S`.
   (Covered by other notes in this corpus.)
2. **Assignment / decision** — choose a subset `M ⊆ A × B` optimizing some global
   objective under constraints (one-to-one, capacities, thresholds). **This note.**
3. **Post-processing** — apply thresholds, drop weak matches, rank alternatives,
   present k-best options for human-in-the-loop editing.

equate already reflects this split: every matcher in `equate/util.py`
(`greedy_matching`, `hungarian_matching`, `maximal_matching`,
`stable_marriage_matching`, `kuhn_munkres_matching`) takes a `similarity_matrix`
and yields `(row_index, col_index)` pairs. That uniform signature is the correct
Strategy seam — the redesign should formalize and extend it rather than replace
it.

Two framing distinctions organize everything below:

- **Cost vs. score (sense).** LAP solvers minimize cost by convention; equate
  scores are usually *similarities* to maximize. Converting between them
  (`cost = max(S) − S`, or `cost = −S`) is a recurring source of subtle bugs when
  done inconsistently per-matcher. This must be a single, explicit `sense`
  parameter.
- **Hard vs. soft matching.** Most algorithms return a *hard* 0/1 assignment
  (each pair matched or not). Optimal-transport methods return a *soft*,
  fractional **transport plan** (a doubly-stochastic-like matrix). These are
  different return types and need different interfaces.

---

## 2. Bipartite matching foundations

The two collections `A` (rows) and `B` (columns) form a **bipartite graph**;
edges carry weights = scores. Two classical objectives, both polynomial:

### 2.1 Maximum-cardinality matching — Hopcroft-Karp

Ignore weights; maximize the *number* of matched pairs such that no vertex is
reused. The **Hopcroft-Karp algorithm** [1] runs in **O(|E|·√|V|)** by augmenting
along many shortest augmenting paths per phase — the best known bound for
unweighted bipartite matching. In Python: `networkx.algorithms.bipartite.matching.hopcroft_karp_matching`
[2]. Use it when you only care *whether* a feasible one-to-one pairing exists
(e.g. after a hard threshold produces a 0/1 "compatible?" mask) and want the
largest such pairing.

### 2.2 Maximum-weight matching — Edmonds' blossom / Galil

Maximize the *sum of weights* of matched edges. For **general** graphs
(not just bipartite) this is solved by Edmonds' **blossom** algorithm with a
primal-dual method; NetworkX's `max_weight_matching` implements the variant from
Galil's 1986 survey [3] in **O(V³)** [4]. Important numerical caveat from the docs
[4]: with **integer** weights the algorithm is exact; with **floating-point**
weights it "could return a slightly suboptimal matching due to numeric precision
errors." A `maxcardinality=True` flag forces the maximum-weight matching *among
those of maximum cardinality*. equate's `maximal_matching` uses exactly this.

For **bipartite** maximum-weight *perfect/full* matching specifically, this is
equivalent to the linear assignment problem (§3) after a sign flip, and the
dedicated LAP solvers below are far faster than the general blossom algorithm.

**Terminology flag.** "Maximum-weight matching," "maximum-weight *perfect*
matching," and "minimum-*cost* perfect matching" are the same problem up to
negation and a completeness constraint; papers and libraries use all three.

---

## 3. The Linear Assignment Problem (LAP)

**Definition.** Given an `n × n` cost matrix `C`, find a permutation `π`
minimizing `Σᵢ C[i, π(i)]` — a one-to-one assignment of every row to a distinct
column of minimum total cost. Also called the **linear sum assignment problem
(LSAP)**, or just "the assignment problem." It is the canonical, exactly-solvable
core of hard one-to-one matching. The authoritative reference is Burkard,
Dell'Amico & Martello, *Assignment Problems* (SIAM, revised reprint 2012) [5],
which treats LSAP and its many variants (bottleneck, quadratic, multi-index)
comprehensively.

### 3.1 Hungarian / Kuhn-Munkres

The **Hungarian algorithm** (Kuhn 1955 [6], building on König and Egerváry;
Munkres 1957 [7] gave the polynomial matrix form — hence **Kuhn-Munkres** and
"Munkres algorithm" as synonyms) solves LAP exactly via a primal-dual
labeling/augmenting-path method in **O(n³)**. It is the textbook algorithm and the
mental model most people carry, but naive matrix implementations (e.g. the pure-
Python `munkres` package) are slow in practice.

### 3.2 Jonker-Volgenant (JV / LAPJV) — the practical default

The **Jonker-Volgenant algorithm** (1987 [8]) is a shortest-augmenting-path method
with a clever dual-variable initialization. It has the **same O(n³)** worst-case
bound but is **substantially faster in practice** — often an order of magnitude —
and is the de facto standard for real workloads.

**SciPy uses JV.** `scipy.optimize.linear_sum_assignment` is documented as a
"modified Jonker-Volgenant algorithm with no initialization" and cites Crouse's
2016 rectangular-assignment paper [9, 10]. It **supports rectangular** cost
matrices (more rows than columns ⇒ not every row need be assigned), returning
`(row_ind, col_ind)` arrays. This is exactly what equate's `hungarian_matching`
wraps today (the function name is a slight misnomer — SciPy is JV, not classic
Hungarian).

### 3.3 Auction algorithm (Bertsekas)

Bertsekas' **auction algorithm** [11] is a distinct, distributed relaxation
method: unassigned "persons" iteratively *bid* for "objects," raising prices until
an ε-optimal assignment emerges. It parallelizes well and shines on large sparse
or structured instances and in distributed/streaming settings; it underlies much
multi-target-tracking software. Bertsekas has continued to extend it (new
auction algorithms, 2023 [12]).

### 3.4 Min-cost flow formulation

LAP is a special case of **minimum-cost flow**: a bipartite transportation network
with unit supplies/demands. Solving it as min-cost flow (e.g. via
`networkx.min_cost_flow`, or SSP/network-simplex) is more general — it directly
extends to **capacities** (many-to-many, §7) and imbalanced supply/demand — at
some constant-factor cost versus a specialized LAP solver. This is the natural
bridge from one-to-one assignment to generalized assignment.

### 3.5 Sparse and rectangular LAP

- **Sparse cost matrices** (most pairs forbidden/irrelevant): use
  `scipy.sparse.csgraph.min_weight_full_bipartite_matching` [10], or `lapmod`
  from `lap`/`lapx` (§6). Building a dense `n × m` matrix or an explicit NetworkX
  graph over all pairs — as equate's current `maximal_matching` and
  `kuhn_munkres_matching` do — is O(n·m) in memory and should be avoided for large
  inputs.
- **Rectangular** (|A| ≠ |B|): SciPy handles it natively; `lapx.lapjv` handles it
  via an `extend_cost` padding parameter [13].

### 3.6 Python LAP-solver landscape and benchmarks

The `LAP-solvers` benchmark repository [14] compares seven implementations on
matrices from 8×8 up to 16 384×16 384. Representative timings at the largest size:

| Solver | Language / method | 16 384² time | Notes |
|---|---|---|---|
| `lapjv.lapjv` | C++ JV, AVX2-optimized | ~14.6 s | fastest overall [14] |
| `lap.lapjv` / `lapx` | C++ JV (1987), + `lapmod` sparse | ~50.8 s | maintained fork = `lapx` [13] |
| `scipy.optimize.linear_sum_assignment` (v1.4+) | modified JV | ~52.8 s | core dep, rectangular, well-maintained [9] |
| `lapsolver.solve_dense` | C++ shortest-augmenting-path | ~77.9 s | |
| `hungarian` | C++ | good < 100×100 | slower past 256×256 [14] |
| `munkres.Munkres` | pure Python | slowest | avoid for anything large [14] |

**Takeaways for equate:** (a) all are worst-case O(n³) but real-world speed varies
by **>10×** [14]; (b) SciPy JV is the right *default* — it is a core scientific
dependency, supports rectangular inputs, and is competitive; (c) `lap`/`lapx` and
`lapjv` are worthwhile *optional* accelerators for very large or sparse problems;
(d) `munkres` (pure-Python) should never be a default.

---

## 4. Stable matching (Gale-Shapley) — a *different objective*

**Stable matching** answers a different question from LAP. Given each element's
*ranked preference list* over the other side, the **Gale-Shapley** deferred-
acceptance algorithm (1962 [15]) produces a **stable** matching: no unmatched pair
mutually prefers each other over their assigned partners. It runs in **O(n²)**.

Two properties are essential to communicate to users, because they are surprising:

1. **Stability ≠ optimality of total score.** Gale-Shapley does **not** minimize
   total cost or maximize total similarity. It guarantees *no blocking pair*. A
   stable matching can have a worse total score than the LAP-optimal one, and
   vice-versa. Choosing Gale-Shapley vs. LAP is a *semantic* choice about what
   "best matching" means, not a speed/quality tradeoff.
2. **Side-optimality asymmetry.** The proposing side gets its **best** achievable
   stable partner (proposer-optimal); the other side gets its **worst**
   (receiver-pessimal) [16]. equate's `stable_marriage_matching` inherits this
   asymmetry — worth documenting, and worth exposing *which* side proposes.

Stable matching is the right model when the inputs are genuinely two-sided
*preferences* (residents↔hospitals, students↔schools) rather than symmetric
similarity scores. For symmetric affinity, LAP or max-weight matching is usually
the intended semantics. (equate's current implementation derives preference lists
from a similarity matrix, which is a reasonable convenience but blurs this
distinction — flag it.)

---

## 5. k-best assignments (Murty) and interactive re-optimization

A single optimal assignment is often not enough. **Murty's algorithm** (1968 [17])
enumerates assignments in **increasing order of cost** — the 1st-best, 2nd-best,
…, k-th-best — by recursively partitioning the solution space (force/forbid
specific pairs) and re-solving a small LAP in each partition. Generating one more
solution costs at most solving `(n−1)` assignment subproblems of sizes `2…n` [17];
modern implementations add lower-bound pruning to skip hopeless subproblems.
Any LAP solver (Hungarian, JV) can be plugged in as the inner subroutine. Python/R
implementations exist (e.g. the `muRty` package [18]); it is also a staple of
multi-target-tracking / data-association libraries.

**Why this matters for a *framework*.** k-best is the enabling primitive for
**interactive, human-in-the-loop re-optimization**:

- Present the top-k candidate matchings (or top-k partners per row) so a user can
  pick among near-optimal alternatives rather than accept one opaque answer.
- On a user **edit** ("force `a₃ ↔ b₇`", "forbid `a₁ ↔ b₂`"), re-solve under the
  new constraints — Murty's force/forbid partitioning is *exactly* this operation,
  so the top-k can be recomputed cheaply by re-using the retained partition tree
  instead of solving from scratch.
- Retain top-k as a cache; warm-start the next solve from it.

This makes Murty-style enumeration a first-class capability equate should expose,
not an afterthought.

---

## 6. Soft / graded matching via Optimal Transport (Sinkhorn)

When a *hard* one-to-one assignment is too rigid — mass should be *split*, or the
two sides have differing "amounts" — the right generalization is **Optimal
Transport (OT)**, a.k.a. the **Wasserstein / Earth Mover's Distance (EMD) /
Monge-Kantorovich** problem. OT finds a **transport plan** `P` (a nonnegative
matrix whose row/column sums match prescribed marginals) minimizing `⟨P, C⟩`.
LAP is exactly the special case where both marginals are uniform and `P` is forced
to be a permutation matrix.

- **Exact OT** is a linear program solvable by the network simplex; `ot.emd` /
  `ot.emd2` in **POT** (Python Optimal Transport) [19, 20]. Documented complexity
  ~**O(n³ log n)** but exploits solution sparsity.
- **Entropic-regularized OT (Sinkhorn)** — Cuturi 2013 [21] — adds an entropy term
  and solves via the **Sinkhorn-Knopp** matrix-scaling iteration (alternating
  row/column normalization). It is **near-O(n²)** per iteration [20], GPU-friendly,
  differentiable, and *orders of magnitude faster* than exact solvers, at the cost
  of a **soft, fractional** plan (a regularization parameter ε trades sharpness for
  speed/stability). POT provides `ot.sinkhorn`/`ot.sinkhorn2` plus numerically
  stabilized variants (`sinkhorn_log`, `sinkhorn_stabilized`, `sinkhorn_epsilon_scaling`,
  `greenkhorn`, `screenkhorn`) [20].
- **Beyond balanced OT:** `ot.sinkhorn_unbalanced` (relaxed marginals — allows
  creation/destruction of mass, good for partial correspondence),
  `ot.partial.partial_wasserstein` (match only a fraction), and
  `ot.gromov.gromov_wasserstein` (match across *incomparable* spaces using only
  intra-set distances — relevant for matching two collections with no shared
  feature space) [20].
- **Backends:** POT runs on NumPy, PyTorch, JAX, TensorFlow, and CuPy, enabling
  GPU and autodiff [20]. The standard survey/reference is Peyré & Cuturi,
  *Computational Optimal Transport* (2019) [22].

**Framework relevance.** Sinkhorn/OT is the natural home for **graded** or
**probabilistic** matching (return `P[i, j]` as a soft correspondence weight),
**partial** matching, and matching **unequal-mass** or **cross-modal** collections.
Its return type (a plan matrix) differs from hard matchers (a pair list) and needs
a distinct-but-parallel interface. A hard assignment can always be recovered from
a soft plan by rounding (e.g. argmax per row, or LAP on `−P`).

---

## 7. Many-to-many and generalized assignment

Relaxing one-to-one opens a spectrum:

- **Capacitated / many-to-many matching:** each row may match up to `u_i`
  columns and each column up to `v_j` rows. Still polynomial — model as
  **min-cost flow** with node capacities (§3.4).
- **Generalized Assignment Problem (GAP):** assign each job to exactly one agent
  subject to agent **capacity/knapsack** constraints, minimizing cost. GAP is
  **NP-hard** [23]; solved in practice by branch-and-bound, Lagrangian relaxation,
  or ILP solvers (PuLP/OR-Tools), or approximated. Recent work recasts GAP via
  network-flow formulations for larger exact instances [24].
- **Stable many-to-many / hospitals-residents:** the capacitated generalization of
  Gale-Shapley, with its own stability theory and recent results on popularity and
  perfect matchings in the many-to-many setting [25].

**Design consequence:** the moment capacities or side-constraints enter, you leave
the "call one LAP solver" world and need either a flow model or an ILP escape
hatch. equate should recognize this boundary and route to an optional solver
(NetworkX flow, OR-Tools) rather than pretend a single LAP call suffices.

---

## 8. Online / incremental assignment

In streaming or interactive settings the cost matrix changes over time (new
rows/columns arrive; scores are edited). Options:

- **Re-solve from scratch** each time (simplest; fine for small/medium `n`).
- **Auction algorithms** (§3.3) support **warm-starting** from previous prices,
  making incremental re-solves cheap — a major reason they dominate real-time
  data-association/tracking.
- **Murty force/forbid** (§5) gives principled incremental re-optimization under
  user edits.
- **Online Sinkhorn** variants compute OT distances from sample streams.

For equate's interactive use cases (a user curating a join/mapping), the practical
recipe is: keep the current optimal + top-k, accept force/forbid edits, and
re-solve incrementally rather than recompute blindly.

---

## 9. Complexity & selection cheat-sheet

| Problem / objective | Canonical algorithm | Complexity | Primary Python |
|---|---|---|---|
| Max-cardinality bipartite (unweighted) | Hopcroft-Karp [1] | O(E·√V) | `networkx.hopcroft_karp_matching` [2] |
| Max-weight matching (general graph) | Edmonds blossom / Galil [3] | O(V³) | `networkx.max_weight_matching` [4] |
| **LAP** (min-cost 1-to-1), dense | Hungarian [6,7] / **JV** [8] | O(n³) | **`scipy.linear_sum_assignment`** [9] |
| LAP, large/fast | JV (AVX2) | O(n³), fastest const | `lapjv`, `lap`/`lapx` [13,14] |
| LAP, sparse | JV sparse / bipartite | ~O(n³) sparse-aware | `scipy.sparse.csgraph.min_weight_full_bipartite_matching`, `lapmod` [10,13] |
| LAP via flow (+capacities) | Min-cost flow | poly | `networkx.min_cost_flow` |
| Stable matching | Gale-Shapley [15] | O(n²) | (small; roll-your-own / `matching` pkg) |
| k-best assignments | Murty [17] | k × LAP | `muRty`-style [18] |
| Soft/fractional (OT) | Sinkhorn [21] / EMD | ~O(n²)/it; O(n³ log n) | **POT** `ot.sinkhorn`/`ot.emd` [19,20] |
| Generalized assignment (capacities) | B&B / ILP / flow | **NP-hard** [23] | OR-Tools / PuLP; flow [24] |

---

## 10. Glossary of canonical terminology (with synonyms)

- **Linear Assignment Problem (LAP / LSAP / "the assignment problem").** Min-cost
  one-to-one assignment on a cost matrix. The exactly-solvable core.
- **Hungarian algorithm = Kuhn-Munkres = Munkres algorithm.** The classic O(n³)
  primal-dual LAP solver (Kuhn 1955, Munkres 1957).
- **Jonker-Volgenant (JV / LAPJV).** Shortest-augmenting-path LAP solver, same
  O(n³) bound, much faster in practice; what SciPy actually uses.
- **Maximum-weight matching.** Maximize summed edge weights; general-graph version
  = Edmonds' **blossom** algorithm. Equivalent to min-cost perfect matching up to
  sign.
- **Hopcroft-Karp.** O(E·√V) maximum-*cardinality* (unweighted) bipartite matching.
- **Min-cost flow.** Network-flow generalization of LAP; adds capacities/imbalance.
- **Stable matching = stable marriage.** Matching with no blocking pair, from
  ranked preferences; solved by **Gale-Shapley** (deferred acceptance). Optimizes
  *stability*, not total score.
- **Murty's algorithm (k-best).** Enumerates the k lowest-cost assignments in order,
  via force/forbid partitioning; enables re-optimization.
- **Optimal Transport (OT) = Wasserstein / EMD / Monge-Kantorovich.** Min-cost
  transport plan between two mass distributions; LAP is the permutation-matrix
  special case.
- **Sinkhorn = entropic-regularized OT = Sinkhorn-Knopp scaling.** Fast, soft,
  differentiable OT approximation (Cuturi 2013).
- **Generalized Assignment Problem (GAP).** Capacity-constrained assignment;
  NP-hard.
- **Data association.** The tracking community's name for repeatedly solving LAP /
  Murty over frames — same math, different vocabulary.

---

## 11. Design implications for equate

equate's current `matcher(similarity_matrix) -> Iterable[(i, j)]` convention is
the right foundation. Concrete recommendations:

1. **Formalize a `Matcher` protocol as the Strategy seam.** A structural
   `typing.Protocol`:
   `def matcher(scores, *, sense: Literal['maximize','minimize'] = 'maximize', **opts) -> Iterable[tuple[int, int]]`.
   Every algorithm (greedy, JV/Hungarian, Hopcroft-Karp, max-weight, Gale-Shapley,
   Murty, OT-rounded) implements it. Expose a name→matcher **registry** so callers
   select by string and users can register custom strategies (open-closed).

2. **Centralize the score↔cost conversion (SSOT).** Today each matcher flips sign
   differently (`hungarian_matching` does `max − S`; `kuhn_munkres_matching`
   negates; `stable_marriage_matching` does `1 − S`). This is a latent bug and a
   correctness/semantics hazard. Make a single explicit `sense` parameter and one
   `_to_cost(scores, sense)` helper used everywhere. Do not silently assume
   similarities in `[0, 1]`.

3. **Make SciPy JV the default hard-matcher; gate the rest as optional extras.**
   SciPy is the sensible core dependency (rectangular support, fast, maintained).
   Treat `networkx` (Hopcroft-Karp, max-weight, flow), `lap`/`lapx`/`lapjv`
   (large/sparse acceleration), and `POT` (optimal transport) as **optional
   dependencies** behind lazy imports with a `check_requirements`-style helper that
   gives an actionable install hint. Follow the codebase's existing lazy-`import
   networkx` pattern, but wrap it in a friendly error.

4. **Route sparse/large inputs to sparse solvers.** Detect `scipy.sparse` input (as
   `ensure_sparse` already hints) and dispatch to
   `min_weight_full_bipartite_matching` or `lapmod` instead of densifying or
   building a full NetworkX graph over all `n·m` pairs. The current
   `maximal_matching`/`kuhn_munkres_matching` graph construction is O(n·m) and
   should be reserved for small dense cases.

5. **Model the objective, not just the algorithm.** Surface the *semantic* choice
   explicitly: `optimal` (LAP total-cost), `stable` (Gale-Shapley — document
   proposer-optimality and that it does **not** optimize total score), `greedy`
   (fast, order-dependent, myopic), `max_cardinality` (most pairs), `soft` (OT
   plan). Users pick meaning first, implementation second.

6. **First-class k-best / interactive re-optimization.** Add a matcher that yields
   *ranked* solutions (`k_best_matchings(scores, k, *, sense)`), implemented as
   Murty wrapping the chosen inner LAP solver. Support **force/forbid**
   constraints (masking specific cells) as the primitive for user edits, and allow
   retaining the top-k / partition tree to re-solve incrementally on edit rather
   than from scratch.

7. **Support partial matching & thresholds as constraints, not post-hoc filters.**
   A `minimum_score` (already in `match_greedily`) and "not every element must be
   matched" should be modeled inside the optimizer (rectangular padding, dummy
   nodes, or forbidden edges) so the global optimum respects them — not applied
   after the fact, which can leave a globally worse matching.

8. **Give soft matching its own parallel interface.** OT/Sinkhorn returns a
   **plan matrix**, not a pair list. Provide `soft_match(scores, *, reg=..., mode=
   'sinkhorn'|'emd'|'unbalanced'|'partial') -> plan` and a `harden(plan)` helper
   (argmax or LAP on `−plan`) so soft and hard live side by side and interconvert.
   This is where "graded correspondence," partial, and cross-modal (Gromov-
   Wasserstein) matching belong.

9. **Provide an escape hatch for capacitated / generalized assignment.** When
   capacities or side-constraints appear (many-to-many, GAP), route to a min-cost
   **flow** model (NetworkX) or an ILP backend (OR-Tools) as an optional strategy,
   and document that GAP is NP-hard so expectations about scale are calibrated.

10. **Keep numerical honesty in the API.** Document the float-weight caveat of
    `max_weight_matching` [4], the ε-approximation nature of Sinkhorn, and the
    order-dependence of greedy — so users choose with eyes open rather than
    trusting an opaque "best."

Net: equate's optimization layer should be a small, well-documented **strategy
family** — one protocol, a sense-aware cost normalizer, SciPy-JV as the batteries-
included default, optional accelerators/solvers behind clean dependency
boundaries, and first-class support for k-best re-optimization and soft (OT)
matching as distinct-but-parallel capabilities.

---

## References

[1] Hopcroft JE, Karp RM. *An n^{5/2} Algorithm for Maximum Matchings in Bipartite
Graphs.* SIAM Journal on Computing, 2(4):225–231, 1973.
[Wikipedia summary](https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm)

[2] NetworkX. *hopcroft_karp_matching* (bipartite maximum-cardinality matching).
[docs](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.matching.hopcroft_karp_matching.html)

[3] Galil Z. *Efficient Algorithms for Finding Maximum Matching in Graphs.* ACM
Computing Surveys, 18(1):23–38, 1986.
[ACM](https://dl.acm.org/doi/10.1145/6462.6502)

[4] NetworkX. *max_weight_matching* (Edmonds blossom, O(V³); integer-exact,
float-approx). [docs](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html)

[5] Burkard R, Dell'Amico M, Martello S. *Assignment Problems, Revised Reprint.*
SIAM, 2012.
[SIAM/Google Books](https://books.google.com/books/about/Assignment_Problems_Revised_Reprint.html?id=dyQR2ElvTTUC)

[6] Kuhn HW. *The Hungarian Method for the Assignment Problem.* Naval Research
Logistics Quarterly, 2:83–97, 1955.
[Wiley](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109)

[7] Munkres J. *Algorithms for the Assignment and Transportation Problems.*
Journal of the SIAM, 5(1):32–38, 1957.
[SIAM](https://epubs.siam.org/doi/10.1137/0105003)

[8] Jonker R, Volgenant A. *A Shortest Augmenting Path Algorithm for Dense and
Sparse Linear Assignment Problems.* Computing, 38(4):325–340, 1987.
[Springer](https://link.springer.com/article/10.1007/BF02278710)

[9] SciPy. *scipy.optimize.linear_sum_assignment* (modified Jonker-Volgenant;
rectangular support). [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)

[10] Crouse DF. *On Implementing 2D Rectangular Assignment Algorithms.* IEEE
Transactions on Aerospace and Electronic Systems, 52(4):1679–1696, 2016.
[IEEE / DOI](https://doi.org/10.1109/TAES.2016.140952)

[11] Bertsekas DP. *The Auction Algorithm: A Distributed Relaxation Method for the
Assignment Problem.* Annals of Operations Research, 14:105–123, 1988.
[Springer](https://link.springer.com/article/10.1007/BF02186476)

[12] Bertsekas DP. *New Auction Algorithms for the Assignment Problem and
Extensions.* 2023.
[MIT/arXiv](https://arxiv.org/pdf/2310.03159)

[13] lapx (maintained fork of `lap`). *lapjv / lapmod* Jonker-Volgenant solvers
(dense/sparse, rectangular via extend_cost). [PyPI](https://pypi.org/project/lapx/)

[14] Habte B. *LAP-solvers: Benchmarking Linear Assignment Problem Solvers*
(scipy, munkres, hungarian, lap, lapjv, lapsolver, laptools).
[GitHub](https://github.com/berhane/LAP-solvers)

[15] Gale D, Shapley LS. *College Admissions and the Stability of Marriage.*
American Mathematical Monthly, 69(1):9–15, 1962.
[JSTOR/DOI](https://doi.org/10.1080/00029890.1962.11989827)

[16] Kleinberg J, Tardos É. *Algorithm Design*, Ch. 1: Stable Matching
(proposer-optimal / receiver-pessimal, O(n²)).
[Princeton notes](https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pearson/01StableMatching-2x2.pdf)

[17] Murty KG. *An Algorithm for Ranking All the Assignments in Order of Increasing
Cost.* Operations Research, 16(3):682–687, 1968.
[INFORMS](https://pubsonline.informs.org/doi/abs/10.1287/opre.16.3.682)

[18] Bijelić A. *muRty: Murty's Algorithm for k-Best Assignments* (R package).
[CRAN](https://cran.r-project.org/package=muRty) · [GitHub](https://github.com/arg0naut91/muRty)

[19] Flamary R, Courty N, et al. *POT: Python Optimal Transport.* Journal of
Machine Learning Research, 22(78):1–8, 2021.
[JMLR](https://jmlr.org/papers/v22/20-451.html)

[20] POT — Python Optimal Transport documentation (emd, sinkhorn, unbalanced,
partial, Gromov-Wasserstein; NumPy/Torch/JAX/TF/CuPy backends).
[quickstart](https://pythonot.github.io/quickstart.html) · [site](https://pythonot.github.io/)

[21] Cuturi M. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.*
Advances in Neural Information Processing Systems (NeurIPS) 26, 2013.
[NeurIPS](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)

[22] Peyré G, Cuturi M. *Computational Optimal Transport.* Foundations and Trends
in Machine Learning, 11(5–6):355–607, 2019.
[arXiv](https://arxiv.org/abs/1803.00567) · [DOI](https://doi.org/10.1561/2200000073)

[23] *Generalized Assignment Problem* (NP-hard; branch-and-bound / relaxations
overview). [ScienceDirect topic](https://www.sciencedirect.com/topics/mathematics/generalized-assignment-problem)

[24] Hu ... et al. *A Network Flow Algorithm for Solving Generalized Assignment
Problem.* Mathematical Problems in Engineering, 2021.
[Wiley](https://onlinelibrary.wiley.com/doi/10.1155/2021/5803092)

[25] *Perfect Matchings and Popularity in the Many-to-Many Setting.* arXiv, 2024.
[arXiv](https://arxiv.org/pdf/2411.00384)
