# Human-in-the-Loop, Active Learning & Interactive Re-optimization for Matching

> Research note 08 of the `equate` redesign corpus. Topic: keeping humans in the
> matching loop — cheaply *training* a matcher (active learning), *verifying* and
> *explaining* its output (review workflows), and *re-optimizing* an assignment
> when a human confirms or overrides one match (incremental / interactive
> re-solution).

## Abstract

Matching collections of objects (fuzzy correspondence, not exact equality) is
rarely fully automatable: the last few percent of accuracy, the resolution of
genuinely ambiguous cases, and the trust needed to deploy all require human
judgment. This note surveys three intertwined bodies of work that put humans in
the loop economically: (1) **active learning** for entity resolution/matching —
uncertainty sampling, query-by-committee, and the systems that operationalize
them (dedupe, Magellan, SystemER, DTAL, DIAL, and the Meduri et al. benchmark);
(2) **crowdsourced ER** and human verification/explanation workflows (CrowdER,
Corleone, Falcon, risk-analysis-driven review); and (3) **incremental and
interactive re-optimization** — retaining top-*k* scored candidates per item so
that a single human edit re-solves the rest, via *k*-best assignment (Murty),
constrained re-assignment (fixed/forbidden edges), and constraint propagation in
constrained/correlation clustering. It closes with concrete abstraction,
extension-point, and optional-dependency recommendations for `equate` as a
matching *framework*.

---

## 1. Framing: three distinct human-in-the-loop roles

A matching framework touches humans in three orthogonal ways. Conflating them is
the most common design mistake; `equate` should keep them as separate,
independently optional loops.

| Loop | Human role | Question answered | Canonical machinery |
|------|-----------|-------------------|---------------------|
| **Train loop** | *labeler* | "Which pairs should I label so the matcher learns fastest?" | Active learning (uncertainty, QBC) |
| **Review loop** | *verifier* | "Which of the matcher's decisions are risky enough to check?" | Confidence/risk ranking, explanation, match-review UI |
| **Re-opt loop** | *editor* | "I fixed one match — what else must change?" | Incremental ER, *k*-best reuse, constraint propagation |

The end-to-end ER survey of Christophides, Efthymiou, Palpanas, Papadakis &
Stefanidis (ACM Computing Surveys, 2020) [1] is the standard map of the pipeline
these loops attach to: **blocking → block-processing → matching → clustering**,
with explicit sections on budget-aware, incremental, crowdsourced, and
deep-learning ER. Humans can enter at every stage, but the three loops above are
where they add the most leverage per unit of attention.

### 1.1 Terminology hazard: which "matching" community?

The same ideas appear under different names. This note treats **entity resolution
(ER)**, **record linkage**, **deduplication**, **entity matching (EM)**, and
**fuzzy matching** as the same problem viewed from the database, statistics,
data-cleaning, and ML communities respectively [1]. Where a term is
community-specific it is flagged inline.

---

## 2. Glossary of canonical terms (with cross-community synonyms)

- **Active learning (AL)** — a learning setting where the algorithm chooses which
  unlabeled examples a human should label, to reach target accuracy with far
  fewer labels than random labeling. Canonical survey: Settles, *Active Learning
  Literature Survey* (2009) [2].
- **Pool-based active learning** — the dominant AL mode for matching: a large
  fixed *pool* of unlabeled candidate pairs exists; each round the strategy ranks
  the whole pool and queries the top one(s). (Contrast: *stream-based* and
  *membership-query synthesis*.) [2]
- **Uncertainty sampling** — query the example the current model is least sure
  about. Three standard scores: **least-confidence** (`1 − max_c p(c|x)`),
  **margin** (`p(top1) − p(top2)`, query smallest margin), **entropy** (`−Σ p
  log p`, query largest). For binary match/non-match these largely coincide near
  `p = 0.5` [2].
- **Query-by-committee (QBC)** — train a *committee* of models on the labeled set
  (via bootstrap/bagging or different hypotheses); query the pair on which they
  most **disagree**. Disagreement measured by **vote entropy**, **consensus
  (soft-vote) entropy**, or **max KL-divergence to the consensus**. Origin:
  Seung, Opper & Sompolinsky (COLT 1992) [3].
- **Human-in-the-loop (HITL)** vs **human-on-the-loop (HOTL)** — HITL: a human is
  a *required step* inside each decision cycle (e.g., labels every queried pair).
  HOTL: the machine runs autonomously and the human *supervises/intervenes*.
  "Hands-off" crowdsourcing (Corleone/Falcon [5,6]) means *no developer* in the
  loop — the *crowd* still labels, so it is HITL for the crowd, HOTL for the
  engineer.
- **Match review / verification** — a human confirming, rejecting, or correcting
  proposed matches. Synonyms: adjudication, clerical review (classic
  Fellegi–Sunter record linkage), curation.
- **Incremental ER** (a.k.a. **online / dynamic / evolving ER**) — updating an
  existing resolution when data or rules change, *without* recomputing from
  scratch. Includes cluster **split/merge** in response to insert/delete/update
  (Gruenheid et al. [17]) and rule evolution (Whang & Garcia-Molina [18]).
- **Progressive / pay-as-you-go ER** — emit high-confidence matches *early*,
  under a compute budget, before the full resolution completes (SPER [20]).
- **Constraint propagation** — enforcing user-supplied **must-link** (same entity)
  / **cannot-link** (different entity) constraints and their logical
  consequences (transitive closure of must-link; anti-transitive spread of
  cannot-link) throughout a clustering.
- **k-best reuse** — pre-computing (or enumerating on demand) the top-*k* ranked
  global assignments/matchings so that after a human fixes one edge, the next
  consistent global solution is read off the ranking rather than recomputed
  (Murty's algorithm [14,15]).
- **Crowdsourced ER** — outsourcing match/non-match judgments to many
  non-expert workers, with redundancy and quality control (CrowdER [4]).
- **Confidence / calibration** — a matcher's `[0,1]` score treated as a
  probability of a true match; only *calibrated* scores can rank a review queue
  correctly (Platt scaling / isotonic regression).
- **Explanation** — a human-comprehensible reason for a match decision:
  learned symbolic *rules* (SystemER [12]), feature/attribute attributions, or
  interpretable *risk features* flagging likely mislabels (Chen et al. [13]).

---

## 3. Active learning for entity matching (training the matcher cheaply)

### 3.1 The pool-based loop

The canonical loop, instantiated for matching:

1. **Block** the Cartesian product `A × B` down to a candidate-pair *pool* `P`
   (blocking is essential: `|A×B|` is quadratic; the pool must be sub-quadratic
   before AL is affordable) [1].
2. **Seed** with a few labeled pairs (often found by heuristics, since matches
   are rare — see §3.3 skew).
3. **Train** a matcher `M` on the labeled set `L`.
4. **Score & rank** every pair in `P \ L` by a query strategy.
5. **Query** the human on the top pair(s); append labels to `L`.
6. **Refit** `M` (weights *and*, in some systems, blocking predicates); repeat
   until a **stopping criterion** (budget exhausted, uncertainty below a
   threshold, or F1 on a held-out probe plateaus) [2,7].

### 3.2 Query strategies and their tradeoffs

| Strategy | Cost per round | Needs | Note for matching |
|----------|----------------|-------|-------------------|
| Random | O(1) | — | Baseline; wasteful under match/non-match skew |
| Uncertainty (LC/margin/entropy) | O(\|P\|·infer) | one probabilistic model | Cheapest informative strategy; default in most ER tools |
| Query-by-committee | O(C·\|P\|·infer) | C models | Robust to a single model's blind spots; C× cost |
| Density/diversity-weighted | + O(\|P\|²) or ANN | similarity structure | Avoids querying redundant near-duplicate pairs |
| Expected error / variance reduction | very high | retrain per candidate | Theoretically strongest, rarely tractable at ER scale |

The **Meduri, Popa, Sen & Sarwat benchmark** (SIGMOD 2020) [7] is the most
authoritative empirical comparison for EM specifically. Key reported findings:
active learning with *far fewer* labels matches or beats fully-supervised
baselines on several product/publication datasets; their optimizations yield
~**9% F1** improvement and cut **example-selection latency by up to 10×** without
quality loss. It systematically crosses *learners* (e.g., tree ensembles, linear
models, deep nets) with *selectors* (uncertainty, QBC), which is exactly the
strategy-matrix an `equate` framework should expose.

### 3.3 ER-specific complications (why generic AL isn't enough)

- **Extreme class skew.** True matches are a tiny fraction of `A×B`; random or
  naive uncertainty sampling drowns in non-matches. Blocking + seeding with
  high-similarity pairs mitigates this [1,7].
- **Blocker/matcher coupling.** The matcher is only as good as the pool the
  blocker produces; a pair the blocker discards can never be labeled or matched.
  Modern systems learn *both* jointly (DIAL [9]).
- **Batch labeling.** Refitting after every single label is slow; batch-mode AL
  queries *b* diverse-yet-uncertain pairs per round, trading some sample
  efficiency for far fewer refits.
- **Transitivity.** A label on `(a,b)` and `(b,c)` constrains `(a,c)`; good ER-AL
  propagates these (see §6.4) rather than spending a query.

### 3.4 Concrete systems and libraries

- **dedupe** (Python) [10] — the reference open-source ER-by-active-learning
  tool. Its labeling session *is* uncertainty sampling: as the user labels pairs,
  dedupe relearns feature weights (a regularized logistic model), recomputes
  distances, regenerates blocking **predicates**, and re-proposes the most
  uncertain pairs; a final step does **connected-components / hierarchical
  clustering** on the pairwise scores. **deduplipy** [11] is a lighter
  active-learning dedupe with a similar interactive labeling loop.
- **Magellan / `py_entitymatching`** (AnHai Doan's group) [19] — a "how-to" EM
  *management system*. Its guided workflow: load tables → block → label a sample →
  iteratively train **and debug** a learning-based matcher → report estimated
  accuracy. Emphasis on *matcher debugging* (why did this pair mismatch?) is a
  distinctive review/explanation contribution.
- **SystemER** (Qian, Popa & Sen, VLDB 2019) [12] — HITL learning of
  *explainable* ER, where the learned artifact is a set of **human-comprehensible
  rules**; active learning keeps the label count small while a domain expert can
  *verify and edit* the learned rules. This is the cleanest example of coupling
  AL with explanation.
- **DTAL** (Kasai, Qian, Gurajada, Li & Popa, ACL 2019) [8] — Deep Transfer +
  Active Learning: a neural matcher with uncertainty sampling plus adversarial
  transfer from a high-resource dataset; reaches F1 ≈ 97.7 with ~300 labels on
  DBLP-ACM.
- **DIAL** (Jain, Sarawagi & Sen, VLDB 2022) [9] — Deep **Indexed** Active
  Learning: an **Index-By-Committee** framework where committee members are
  transformer encoders; it maintains an **ANN index over learned embeddings** so
  the blocker (maximize recall) and matcher (maximize accuracy) are learned
  jointly and the informative-pair search avoids the quadratic scan. Reports
  gains in precision, recall, *and* running time over DTAL.
- **LLMs as labelers / matchers** — a 2023–2025 shift: an LLM can serve as the
  *oracle* (labeling AL-queried pairs cheaply, sometimes with a weak/strong LLM
  cascade) or as a few-shot matcher requiring near-zero training labels
  [25]. Low-resource work such as the "Battleship" approach [26] further shrinks
  the label budget. Relevant to `equate` as a *pluggable oracle*, not a core dep.
- **Generic AL toolkits** (community-standard, framework-agnostic): **modAL**
  [16] (built on scikit-learn; ships least-confidence/margin/entropy and
  committee vote-entropy/consensus-entropy/max-disagreement), **scikit-activeml**
  [24], and **libact**. These are the natural *optional* strategy providers.

### 3.5 Complexity summary

- Per AL round: **O(|P| · inference)** for uncertainty, **O(C · |P| · inference)**
  for a C-member committee; refit cost depends on the learner. The dominant
  practical cost is *number of refits* (→ batch mode) and *pool size* (→
  blocking / ANN indexing as in DIAL) [7,9].

---

## 4. Crowdsourced entity resolution

When one expert is a bottleneck, distribute judgments to a crowd — with
redundancy and quality control.

- **CrowdER** (Wang, Kraska, Franklin & Feng, VLDB 2012) [4] — the seminal
  **hybrid human-machine** design: machines do a cheap coarse pass to prune
  obvious non-matches, humans verify only the **likely** pairs. Introduces
  **cluster-based** HITs (show a worker a small group, ask which are the same)
  that are far cheaper than pairwise HITs, plus task-generation to minimize the
  number of HITs.
- **Corleone** (Gokhale, Das, Doan, Naughton, Rampalli, Shavlik & Zhu, SIGMOD
  2014) [5] — **hands-off crowdsourcing**: the crowd drives *all* EM steps
  (blocking, matching, accuracy estimation) with **no developer** in the loop. It
  uses crowd-labeled data to train and refine matchers and to estimate quality.
  Limitation: does not scale to large tables.
- **Falcon** (Das, Gokhale, Doan et al., SIGMOD 2017) [6] — scales Corleone via
  **RDBMS-style query planning** over a Hadoop cluster, interleaving machine and
  crowd operators, and using tricks like **masking machine time behind crowd
  time**; scales hands-off crowdsourced EM to millions of tuples for cloud EM
  services.
- **Fine-grained crowdsourced ER** (Nie et al., *Applied Sciences*, 2025) [22] —
  recent work replacing yes/no pairwise HITs with easier fine-grained tasks that
  exploit intra-record connections, reducing worker error and cost.

**Design lesson for a framework:** the "crowd" is just a *many-worker,
noisy-oracle* specialization of the review/label loop. Redundancy, majority
vote, worker-quality weighting, and **transitive-closure** post-processing of
crowd answers should be *pluggable oracle adapters*, not baked into the core.

---

## 5. Human verification, review workflows & explanation

Even a good automatic matcher needs a **review loop** to earn trust and to catch
its own errors. The key question is *triage*: humans can review only a fraction
of decisions, so which ones?

### 5.1 Confidence-/risk-driven review triage

- **Calibrated confidence.** A raw classifier score is not a match probability;
  calibration (Platt scaling, isotonic regression) is a prerequisite for using
  the score to rank a review queue. Pairs near the decision boundary are the
  natural first review candidates (the same signal uncertainty sampling uses).
- **Risk analysis** (Chen, Chen, Hou, Duan, Li & Li, SIGMOD 2020) [13] — instead
  of raw confidence, learn a **risk model** over automatically generated,
  *interpretable* **risk features** that predicts which machine-labeled pairs are
  likely **mislabeled**, then rank pairs by risk of error. This directly answers
  "which results should a human review?" better than boundary distance alone, and
  the risk features double as explanations.

### 5.2 Explanation to support review

- **Symbolic rules** (SystemER [12]) — the matcher *is* rules a human can read,
  verify, and edit.
- **Attribute/feature attributions** — for learned matchers, surface which
  similarity features drove the score (Magellan's debugging emphasis [19]).
- **Provenance** — showing the two records side-by-side with aligned/mismatched
  fields is the minimal viable "explanation" for a match-review UI.

### 5.3 Review UI patterns (from the HITL-ER literature)

Present intermediate outputs for accept/correct; propagate accepted edits as
additional context to the next round; use datagrids / side-by-side diffs /
result timelines so a reviewer can compare alternative hypotheses and branch
feedback [12,13]. These are UI concerns, but they impose a **data contract** on
the core (see §7): every proposed match needs a stable id, a score, an
explanation payload, and an edit hook.

---

## 6. Incremental & interactive re-optimization

This is the least standardized but most `equate`-relevant area: when matching is
a **global assignment** (not independent per-pair decisions), one human edit can
have ripple effects. The framework's job is to re-solve *cheaply and locally*,
reusing prior computation.

### 6.1 Incremental ER (data/rules change)

- **Incremental Record Linkage** (Gruenheid, Dong & Srivastava, VLDB 2014) [17] —
  maintains a clustering under record insert/delete/update by *locally* merging
  and, crucially, **splitting** existing clusters, instead of re-resolving the
  whole dataset. Establishes the core primitive: touch only the affected
  connected component.
- **Entity Resolution with Evolving Rules** (Whang & Garcia-Molina, VLDB 2010)
  [18] — when the *matching rule* changes (e.g., becomes stricter), which
  existing clusters must be re-evaluated? Introduces the **inconsistency**
  problem that a human's edit-as-rule can trigger.
- **Progressive / SPER** (Karapiperis, Papadakis, Palpanas & Verykios, 2025) [20]
  — replaces the O(|E| log |E|) sort of candidate pairs with **stochastic
  bipartite maximization** (Bernoulli trials with probability ∝ similarity),
  achieving **O(|E|)** and 3–6× speedups, emitting matches early under a budget.
  Relevant because interactive review wants a *stream* of scored matches, not a
  batch delivered at the end.

### 6.2 The core interactive primitive: top-*k* scored candidates + edit → re-solve

The task brief's central pattern: **retain, per item, its top-*k* candidate
matches with scores**, so that when a user confirms/rejects/edits one match, the
assignment **re-solves for the rest**. Three complementary mechanisms:

**(a) Constrained re-assignment (fixed / forbidden edges).** Model global
one-to-one matching as the **linear assignment problem (LAP)** on a cost matrix;
solve with **Hungarian / Kuhn–Munkres in O(n³)** (`scipy.optimize.
linear_sum_assignment`). A human **confirm** = *force* an edge (fix it in, or
delete its row/column); a human **reject** = *forbid* an edge (set cost = +∞).
Re-solving the reduced problem yields the new optimum consistent with the edit.
Warm-starting from the previous dual solution makes the re-solve much cheaper
than a cold O(n³).

**(b) *k*-best assignment enumeration (Murty).** Precompute the top-*k* ranked
*global* assignments once; after an edit, walk down the ranking to the
best solution consistent with the constraint, instead of recomputing.
**Murty's algorithm** (Operations Research, 1968) [14] enumerates assignments in
increasing-cost order by repeatedly *partitioning* the solution space
(forcing/forbidding edges) and re-solving each partition — exactly the
fixed/forbidden mechanism of (a). Naive cost is ~O(k · n⁴); the standard
optimizations of **Miller, Stone & Cox** (IEEE TAES, 1997) [15] — inheriting dual
variables and partial solutions across partitions, sorting subproblems by lower
bound — bring it to ~**O(k · n³)**. (Origin: multi-target tracking / motion
correspondence, a matching problem structurally identical to ER assignment.)

**(c) Constraint propagation over clusters.** When matching is *many-to-many* or
*clustering* (an entity = a cluster of records) rather than 1-to-1, edits are
**must-link / cannot-link** constraints. Enforcing them:
- **must-link** is transitive → take the transitive closure (union-find);
- **cannot-link** spreads anti-transitively and can force cluster **splits**;
- re-cluster only the affected component (as in incremental ER §6.1).

### 6.3 Constrained / correlation clustering with human feedback

- **Constrained clustering** (semi-supervised) integrates must-link/cannot-link
  into the objective; recent taxonomy: Springer *AI Review* 2024 [27]. **Active
  constrained clustering** chooses *which* pairwise constraint to ask the human
  for next — uncertainty sampling for clustering.
- **Correlation clustering** is the natural ER objective (cluster to agree with
  ±1 pairwise match signals); must-link/cannot-link are hard constraints on the
  partition.
- **Interactive Correlation Clustering with Existential Cluster Constraints**
  (Angell, Monath, Yadav & McCallum, ICML 2022) [21] — richer feedback than
  pairwise: a user asserts a cluster should (or should not) *have* an item with
  certain features; the algorithm re-solves the correlation clustering under
  these existential constraints. A template for expressive, incremental human
  edits beyond single-edge fixes.

### 6.4 Putting it together — the re-opt loop contract

An edit is an **event** carrying a constraint (`force`, `forbid`, `must-link`,
`cannot-link`, or `existential`). The re-opt engine (i) appends it to a
**ConstraintSet**, (ii) identifies the **affected sub-problem** (block / connected
component), and (iii) **re-solves only that sub-problem** using warm-started
assignment or *k*-best walk, then (iv) emits the delta. Locality (only re-solve
what changed) is what makes interactive re-optimization feel instantaneous.

---

## 7. Design implications for `equate`

The overarching principle: **three optional loops, each behind a small protocol,
sharing one candidate/score store as the single source of truth.** Core stays
dependency-light; strategies, oracles, and heavy solvers are optional extras.

### 7.1 Core abstractions (protocols, `collections.abc`/`Protocol`-style)

- **`Matcher`** — `score(pair) -> float in [0,1]` (calibrated), `fit(labeled)`,
  optional `refit(new_labels)` for cheap incremental updates, and optional
  `explain(pair) -> Explanation`. Keep it a *strategy*: a linear/logistic default,
  with tree-ensemble, deep, or LLM matchers as optional plug-ins.
- **`Oracle` (a.k.a. `Labeler`)** — `label(pairs) -> labels`. **Dependency
  injection point** that unifies: a CLI/notebook prompt, a gold-standard file, an
  **LLM** labeler, or a **crowd** connector (with redundancy + majority vote +
  worker weighting as an `Oracle` decorator). The AL loop must not know which it
  is.
- **`QueryStrategy`** — `rank(pool, matcher_state) -> ordered pairs`. Ship
  `random`, `uncertainty(kind='margin'|'least_confidence'|'entropy')`, and
  `query_by_committee(committee, disagreement=...)`. **Default = margin
  uncertainty** (cheapest informative). Optionally delegate to `modAL` /
  `scikit-activeml` behind this protocol rather than reimplementing.
- **`CandidateStore`** — the SSOT: per item, its **top-*k* candidates with
  scores** (and provenance/explanation). Enables both review triage and *k*-best
  reuse. Persist it (`dol` storage / `cache_this`) so a session resumes.
- **`ConstraintSet`** — first-class, append-only log of human edits (`force`,
  `forbid`, `must_link`, `cannot_link`, `existential`). Consumed by both the
  assignment solver and the clusterer.
- **`AssignmentSolver`** — `solve(cost, constraints) -> matching`, default
  `scipy.optimize.linear_sum_assignment` with warm-start under constraints;
  optional `k_best(cost, k) -> ranked matchings` (Murty, optionally via an
  optional dep or a small bundled implementation). This is where §6.2 lives.
- **`Clusterer`** — connected-components / correlation / constrained clustering
  with `must_link`/`cannot_link` support and **local re-clustering** of an
  affected component (§6.1, §6.3).

### 7.2 The three loops as composable services

- **`ActiveLearningSession`** — orchestrates seed → train → `QueryStrategy.rank`
  → `Oracle.label` → `Matcher.refit`; configurable **batch size** and
  **stopping criterion** (budget, uncertainty threshold, plateau). A
  `functools.cached_property` matcher, persisted labeled set.
- **`ReviewQueue`** — ranks decisions for human verification by **calibrated
  confidence** or a pluggable **`RiskModel`** (§5.1 [13]); every item exposes an
  `Explanation` and an **edit hook** that emits a constraint event.
- **`InteractiveReoptimizer`** — subscribes to edit events, appends to
  `ConstraintSet`, finds the **affected sub-problem**, and re-solves *locally* via
  `AssignmentSolver`/`Clusterer`, emitting a delta. Warm-start + locality are the
  performance contract.

### 7.3 Strategy / optional-dependency boundaries

Keep the core (blocking + a linear matcher + LAP solver via scipy) installable
with minimal deps. Gate the rest behind extras:

- `equate[active]` → modAL / scikit-activeml strategy adapters.
- `equate[llm]` → LLM oracle/matcher adapter (weak/strong cascade optional).
- `equate[crowd]` → crowd-platform oracle adapters (CrowdER-style redundancy).
- `equate[deep]` → transformer matchers / DIAL-style indexed AL (heavy).
- `equate[kbest]` → optimized Murty *k*-best (if not bundled).

Follow `check_requirements`-style dynamic guidance so a user who calls a
strategy whose optional dep is missing gets an informative install hint, not an
ImportError.

### 7.4 Cross-cutting contracts

- **Calibrated scores everywhere.** Both the review queue and re-optimization
  need scores that behave like probabilities; make calibration a first-class,
  swappable step of `Matcher`.
- **Everything streams.** Emit matches progressively with confidence (§6.1 [20])
  so review can start before a full pass completes.
- **Edits are events, re-solves are local.** An edit never triggers a global
  recompute by default; it re-solves the touched block/component (§6.1, §6.4).
- **Explanation is a payload, not a printout.** Return structured
  attributions/rules/risk-features so any UI (CLI, notebook widget, web) can
  render them — matching the user's declarative/schema-based UI preference.

---

## References

[1] Christophides V, Efthymiou V, Palpanas T, Papadakis G, Stefanidis K. *An
Overview of End-to-End Entity Resolution for Big Data.* ACM Computing Surveys,
53(6), 2020. [dl.acm.org/doi/10.1145/3418896](https://dl.acm.org/doi/10.1145/3418896)
(preprint: [arxiv.org/abs/1905.06397](https://arxiv.org/abs/1905.06397))

[2] Settles B. *Active Learning Literature Survey.* University of
Wisconsin–Madison, Computer Sciences TR 1648, 2009.
[burrsettles.com/pub/settles.activelearning.pdf](https://burrsettles.com/pub/settles.activelearning.pdf)

[3] Seung HS, Opper M, Sompolinsky H. *Query by Committee.* COLT 1992.
[dl.acm.org/doi/10.1145/130385.130417](https://dl.acm.org/doi/10.1145/130385.130417)

[4] Wang J, Kraska T, Franklin MJ, Feng J. *CrowdER: Crowdsourcing Entity
Resolution.* PVLDB 5(11):1483–1494, 2012.
[vldb.org/pvldb/vol5/p1483_jiannanwang_vldb2012.pdf](https://www.vldb.org/pvldb/vol5/p1483_jiannanwang_vldb2012.pdf)
(preprint: [arxiv.org/abs/1208.1927](https://arxiv.org/abs/1208.1927))

[5] Gokhale C, Das S, Doan A, Naughton JF, Rampalli N, Shavlik J, Zhu X.
*Corleone: Hands-off Crowdsourcing for Entity Matching.* SIGMOD 2014.
[dl.acm.org/doi/10.1145/2588555.2588576](https://dl.acm.org/doi/10.1145/2588555.2588576)

[6] Das S, Gokhale C, et al. (Doan A). *Falcon: Scaling Up Hands-Off Crowdsourced
Entity Matching to Build Cloud Services.* SIGMOD 2017.
[dl.acm.org/doi/10.1145/3035918.3035960](https://dl.acm.org/doi/10.1145/3035918.3035960)
(TR: [pages.cs.wisc.edu/~anhai/papers/falcon-tr.pdf](https://pages.cs.wisc.edu/~anhai/papers/falcon-tr.pdf))

[7] Meduri V, Popa L, Sen P, Sarwat M. *A Comprehensive Benchmark Framework for
Active Learning Methods in Entity Matching.* SIGMOD 2020.
[arxiv.org/abs/2003.13114](https://arxiv.org/abs/2003.13114)

[8] Kasai J, Qian K, Gurajada S, Li Y, Popa L. *Low-resource Deep Entity
Resolution with Transfer and Active Learning (DTAL).* ACL 2019.
[aclanthology.org/P19-1586](https://aclanthology.org/P19-1586/)
(preprint: [arxiv.org/abs/1906.08042](https://arxiv.org/abs/1906.08042))

[9] Jain A, Sarawagi S, Sen P. *Deep Indexed Active Learning for Matching
Heterogeneous Entity Representations (DIAL).* PVLDB 15(1), 2022.
[arxiv.org/abs/2104.03986](https://arxiv.org/abs/2104.03986)

[10] Gregg F, Eder D, et al. *dedupe — a Python library for accurate and scalable
fuzzy matching, record deduplication and entity-resolution.*
[github.com/dedupeio/dedupe](https://github.com/dedupeio/dedupe)

[11] Hermans F. *DedupliPy — deduplication/entity resolution using active
learning.* [github.com/fritshermans/deduplipy](https://github.com/fritshermans/deduplipy)

[12] Qian K, Popa L, Sen P. *SystemER: A Human-in-the-loop System for Explainable
Entity Resolution.* PVLDB 12(12):1794–1797, 2019.
[vldb.org/pvldb/vol12/p1794-qian.pdf](https://www.vldb.org/pvldb/vol12/p1794-qian.pdf)

[13] Chen Z, Chen Q, Hou B, Duan T, Li Z, Li G. *Towards Interpretable and
Learnable Risk Analysis for Entity Resolution.* SIGMOD 2020.
[arxiv.org/abs/1912.02947](https://arxiv.org/abs/1912.02947)

[14] Murty KG. *An Algorithm for Ranking All the Assignments in Order of
Increasing Cost.* Operations Research 16(3):682–687, 1968.
[doi.org/10.1287/opre.16.3.682](https://doi.org/10.1287/opre.16.3.682)

[15] Miller ML, Stone HS, Cox IJ. *Optimizing Murty's Ranked Assignment Method.*
IEEE Trans. Aerospace and Electronic Systems 33(3), 1997.
[semanticscholar.org/paper/7d8418e5910943155075fd3acc3167f0429ce215](https://www.semanticscholar.org/paper/Optimizing-Murty's-ranked-assignment-method-Miller-Stone/7d8418e5910943155075fd3acc3167f0429ce215)

[16] Danka T, Horvath P. *modAL: A modular active learning framework for Python.*
[github.com/modAL-python/modAL](https://github.com/modAL-python/modAL)
(docs: [modal-python.readthedocs.io](https://modal-python.readthedocs.io/))

[17] Gruenheid A, Dong XL, Srivastava D. *Incremental Record Linkage.* PVLDB
7(9):697–708, 2014.
[vldb.org/pvldb/vol7/p697-gruenheid.pdf](http://www.vldb.org/pvldb/vol7/p697-gruenheid.pdf)

[18] Whang SE, Garcia-Molina H. *Entity Resolution with Evolving Rules.* PVLDB
3(1):1326–1337, 2010. [vldb.org/pvldb/vol3/R117.pdf](https://vldb.org/pvldb/vol3/R117.pdf)

[19] Konda P, et al. (Doan A). *Magellan: Toward Building Entity Matching
Management Systems* / `py_entitymatching`. PVLDB 2016; CACM 2020.
[github.com/anhaidgroup/py_entitymatching](https://github.com/anhaidgroup/py_entitymatching)

[20] Karapiperis D, Papadakis G, Palpanas T, Verykios VS. *SPER: Accelerating
Progressive Entity Resolution via Stochastic Bipartite Maximization.* 2025.
[arxiv.org/abs/2512.23491](https://arxiv.org/abs/2512.23491)

[21] Angell R, Monath N, Yadav N, McCallum A. *Interactive Correlation Clustering
with Existential Cluster Constraints.* ICML 2022.
[proceedings.mlr.press/v162/angell22a.html](https://proceedings.mlr.press/v162/angell22a.html)

[22] Nie T, Mao H, Liu X, Yu S. *Fine-Grained Tasks for Crowdsourced Entity
Resolution.* Applied Sciences 15(1):4, 2025.
[mdpi.com/2076-3417/15/1/4](https://www.mdpi.com/2076-3417/15/1/4)

[23] *Enhancing Entity Resolution with a Hybrid Active Machine Learning
Framework.* Information Systems, 2024.
[sciencedirect.com/science/article/abs/pii/S0306437924000681](https://www.sciencedirect.com/science/article/abs/pii/S0306437924000681)

[24] Kottke D, et al. *scikit-activeml — active learning on top of scikit-learn.*
[github.com/scikit-activeml/scikit-activeml](https://github.com/scikit-activeml/scikit-activeml)

[25] Peeters R, Bizer C. *Entity Matching using Large Language Models.* 2023–2024.
[arxiv.org/abs/2310.11244](https://arxiv.org/abs/2310.11244)

[26] Genossar B, Shraga R, Gal A. *The Battleship Approach to the Low-Resource
Entity Matching Problem.* 2023. [arxiv.org/abs/2311.15685](https://arxiv.org/abs/2311.15685)

[27] *Semi-supervised Constrained Clustering: An In-depth Overview, Ranked
Taxonomy and Future Research Directions.* Artificial Intelligence Review, 2024.
[link.springer.com/article/10.1007/s10462-024-11103-8](https://link.springer.com/article/10.1007/s10462-024-11103-8)
