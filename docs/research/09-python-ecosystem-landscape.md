# Python Ecosystem Landscape for Matching: What to Wrap vs. Reimplement

*Research note for the `equate` redesign — one of a corpus in `docs/research/`.*

## Abstract

A general matching framework should own its *orchestration and abstractions*, not
its algorithms: nearly every hard sub-problem — fast string metrics, text
embeddings, approximate nearest-neighbor (ANN) search, MinHash/LSH blocking, the
linear assignment problem (LAP), optimal transport (OT), and schema/join
discovery — already has a mature, well-maintained Python library behind it. This
note surveys ~30 packages across six layers of the canonical matching pipeline
(schema alignment → blocking/indexing → representation → comparison/scoring →
assignment/classification → clustering), recording for each *what it does, which
layer it serves, its license, maintenance signals (latest release ~mid-2026), its
API style, and a concrete wrap-vs-reimplement recommendation*. The dominant
finding: `equate` should reimplement almost nothing computational and instead
define thin, uniform *strategy protocols* per layer, wrap best-in-class libraries
behind them as **optional dependencies** (extras), keep all *hard* dependencies
permissively licensed, and treat copyleft or out-of-process engines (Zingg on
Spark) as pluggable backends never bundled. It closes with the abstractions,
extension points, and optional-dependency boundaries this suggests.

---

## 1. Framing: the pipeline as layers, and where each package sits

The sibling notes in this corpus converge on a layered decomposition of matching
[1,2]. This survey maps the Python ecosystem onto those same layers so the
wrap-vs-reimplement decision is made *per seam*, not per package:

| Layer | Responsibility | Representative packages |
|---|---|---|
| **L0. Schema / join discovery** | Which columns/attributes correspond across sources | Valentine |
| **L1. Blocking / indexing** | Reduce the O(nm) candidate space to a tractable candidate set | datasketch (MinHash-LSH), faiss, hnswlib, annoy, scann, pynndescent; blocking modules inside recordlinkage / dedupe / py_entitymatching |
| **L2. Representation / featurization** | Turn raw objects into comparable vectors/tokens | sentence-transformers, transformers, openai, fastText |
| **L3. Comparison / scoring** | Compute pairwise affinity/cost `score(a,b)` | rapidfuzz, thefuzz, jellyfish, textdistance, py_stringmatching; cosine over L2 vectors |
| **L4. Assignment / classification** | Turn scores into decisions: 1-1 assignment, soft/partial transport, match/non-match | scipy `linear_sum_assignment`, lap/lapx, lapsolver, POT, networkx; Fellegi-Sunter classifiers in Splink/dedupe/recordlinkage |
| **L5. Clustering / canonicalization** | Resolve pairwise links into entity clusters / golden records | networkx (connected components), igraph (inside Splink), dedupe |
| **End-to-end** | Bundled pipelines spanning L1–L5 | dedupe, Splink, recordlinkage, py_entitymatching (Magellan), Zingg |

A key vocabulary caution for the synthesis step: **"blocking" (ER community) =
"candidate generation" / "indexing" (DB community) = "ANN retrieval / filtering"
(vector-search community)** — the same L1 operation under three names. Likewise
**"matching"** is overloaded: at L3 it means *scoring a pair*, at L4 it means
*choosing an assignment*, and end-to-end it means *the whole task*. Doc 03 in this
corpus reserves "matcher" for L4; this note follows that convention.

---

## 2. Glossary of the canonical terms

- **dedupe** — an MIT-licensed, single-machine Python library for deduplication
  and record linkage that uses **active learning** (interactive pair labeling) to
  train a Fellegi-Sunter-style classifier plus learned blocking predicates [3].
- **Splink** — an MIT-licensed probabilistic linkage engine (UK Ministry of
  Justice) implementing the **Fellegi-Sunter** model with **EM**-estimated `m`/`u`
  match weights, executed on SQL backends (DuckDB, Spark, Athena, Postgres) for
  scale [4].
- **recordlinkage** — a BSD-3, pandas-based *toolkit* exposing modular `Index`
  (blocking), `Compare`, and `Classify` stages; the most "library-like" of the ER
  frameworks [5].
- **py_entitymatching** (the **Magellan** project, UW-Madison) — a BSD academic
  library for end-to-end entity matching on pandas + scikit-learn, emphasizing a
  guided *how-to* workflow (blocking → feature generation → ML matcher →
  debugging) [6].
- **Zingg** — an AGPL-3.0 ML entity-resolution/MDM engine that runs **on Apache
  Spark** for cluster-scale linkage; distributed as a JVM application with a thin
  Python client [7].
- **rapidfuzz** — an MIT C++/SIMD library of fuzzy string metrics (Levenshtein,
  Damerau-Levenshtein, Jaro, Jaro-Winkler, Indel) and batch utilities
  (`process.cdist`); a faster, permissively licensed drop-in for FuzzyWuzzy's API
  [8].
- **jellyfish** — an MIT library (now implemented in **Rust**) of approximate
  string metrics *and* **phonetic** encoders (Soundex, Metaphone, NYSIIS, Match
  Rating) [10].
- **faiss** — Meta's MIT library for billion-scale similarity search and
  clustering of dense vectors: **IVF** (inverted file) partitioning + **PQ**
  (product quantization) compression, flat/HNSW indexes, CPU/GPU [17].
- **hnswlib** — a small Apache-2.0 header-only C++/Python implementation of the
  **HNSW** graph ANN index (logarithmic-ish search, high recall, incremental
  add/mark-delete) [18].
- **annoy** — Spotify's Apache-2.0 ANN library using **random-projection forests**
  with **memory-mapped**, read-only, process-shareable indexes [19].
- **datasketch** — an MIT library of probabilistic sketches: **MinHash** (Jaccard
  estimation), **MinHash-LSH** / LSH-Forest / LSH-Ensemble for sub-linear blocking,
  and HyperLogLog [22].
- **scipy `linear_sum_assignment`** — SciPy's exact **LAP** solver (a modified
  Jonker-Volgenant / shortest-augmenting-path algorithm, Crouse 2016), the
  ubiquitous BSD default for optimal 1-1 assignment [24].
- **lap** (and its maintained fork **lapx**) — BSD/MIT LAP solvers implementing
  **LAPJV** (dense Jonker-Volgenant) and **LAPMOD** (sparse), often faster than
  SciPy on large/sparse cost matrices [25].
- **POT** (Python Optimal Transport) — an MIT library for **optimal transport**:
  exact EMD (network simplex), entropic **Sinkhorn**, unbalanced/partial OT, and
  Gromov-Wasserstein; the L4 tool for *soft/fractional/many-to-many* matching
  [28,29].
- **sentence-transformers** — an Apache-2.0 framework for computing sentence/text
  **embeddings** with bi-encoders and re-ranking with cross-encoders, plus helpers
  (`semantic_search`, `paraphrase_mining`) [13].

---

## 3. Master comparison table

Maintenance dates are the latest PyPI release observed at the time of writing
(mid-2026); "maturity" folds in that plus community size and API stability.

| Package | Layer | What it does | License | Latest (≈date) | API style | equate: **wrap vs reimplement** |
|---|---|---|---|---|---|---|
| **dedupe** | E2E (L1–L5) | Active-learning dedup/linkage, learned blocking + FS classifier | MIT | 3.0.3 (2024) | Stateful `Dedupe`/`RecordLink` objects, `.train()`, requires labeling | **Wrap** (optional backend) — don't reimplement active learning |
| **Splink** | E2E (L1–L5) | Probabilistic FS linkage on SQL engines, scalable | MIT | 4.0.x (2025) | Linker object + settings dict; SQL backend injected | **Wrap** as scalable backend; big dep tree |
| **recordlinkage** | E2E (L1–L4) | Modular pandas toolkit: Index/Compare/Classify | BSD-3 | 0.16 (Jul 2023) | Composable classes over pandas MultiIndex | **Reference design**; wrap optionally; slower maintenance |
| **py_entitymatching** (Magellan) | E2E (L1–L5) | Guided EM workflow, ML matchers, debugging | BSD | 0.4.2 (Feb 2024) | pandas + sklearn, many free functions | **Learn from**; wrap sparingly; heavy, dated deps |
| **Zingg** | E2E (L1–L5) | Spark-scale ML ER / MDM | **AGPL-3.0** | 0.6.x py client (2026) | JVM app + thin Python client; needs Spark | **Out-of-process backend only**; never bundle (copyleft) |
| **rapidfuzz** | L3 | Fast fuzzy string metrics + batch `cdist` | MIT | 3.14.x (2025–26) | Pure functions; `fuzz`, `distance`, `process` | **Wrap as default fast backend** |
| **thefuzz** | L3 | FuzzyWuzzy-compatible wrappers over rapidfuzz | MIT | 0.22.1 (Jan 2024) | `fuzz.ratio`, `process.extract` | **Skip** — just use rapidfuzz directly |
| **jellyfish** | L3 | Edit + **phonetic** string metrics (Rust) | MIT | 1.2.x (2025) | Pure functions | **Wrap** for phonetics/Soundex |
| **textdistance** | L3 | 30+ distance algorithms, pure Python | MIT | 4.6.3 (Jul 2024) | Uniform callable objects | **Wrap** for breadth/prototyping; slow |
| **py_stringmatching** | L3 | Tokenizers + set/hybrid similarity (Jaccard, Monge-Elkan, Soft-TF-IDF) | BSD | 0.4.7 (2026) | Tokenizer + measure objects | **Wrap** for token/set & hybrid measures |
| **sentence-transformers** | L2 | Bi-/cross-encoder text embeddings + rerank | Apache-2.0 | 5.x (2025–26) | `SentenceTransformer(...).encode()` | **Wrap** as default embedder; heavy (torch) |
| **transformers** | L2 | Model framework (custom/cross-encoder, Ditto-style EM) | Apache-2.0 | 5.x (2025–26) | `AutoModel`/`pipeline` | **Wrap** for advanced/custom matchers |
| **openai** | L2 | Hosted embeddings + LLM-as-judge | Apache-2.0 (SDK) | 2.x (2025–26) | Async/sync client | **Wrap** optionally; network+cost, no local compute |
| **fastText** | L2 | Subword word vectors, language ID | MIT | 0.9.3 (2024) | `load_model`, `get_word_vector` | **Wrap** for multilingual/OOV tokens; build finicky, stalled |
| **faiss** (`faiss-cpu`) | L1 | Billion-scale ANN + kNN + clustering | MIT | 1.13.x (Dec 2025) | Index objects (`IndexFlat`, `IndexIVFPQ`) | **Wrap** as scalable ANN backend |
| **hnswlib** | L1 | HNSW graph ANN, incremental | Apache-2.0 | 0.8.0 (Dec 2023) | `Index(...).add_items/knn_query` | **Wrap** as default in-memory ANN |
| **annoy** | L1 | RP-forest ANN, mmap, static | Apache-2.0 | 1.17.3 (Jun 2023) | `AnnoyIndex.build/get_nns` | **Wrap** for mmap/low-RAM ANN |
| **scann** | L1 | Anisotropic-VQ ANN (SOTA recall/QPS) | Apache-2.0 | 1.4.x (2025) | Builder API | **Wrap optionally**; **Linux-only wheels** |
| **pynndescent** | L1 | NN-descent kNN graph, arbitrary metrics | BSD-2 | 0.6.0 (2026) | `NNDescent(...).query` | **Wrap** for custom-metric ANN; Numba warmup |
| **datasketch** | L1 | MinHash / LSH / LSH-Forest / LSH-Ensemble | MIT | 2.0.x (2024–25) | `MinHash`, `MinHashLSH` objects | **Wrap** as set-similarity blocker |
| **scipy** `linear_sum_assignment` | L4 | Exact dense LAP (JV variant) | BSD-3 | 1.18.x (2026) | One function, dense matrix | **Wrap as default matcher** (already used) |
| **lap / lapx** | L4 | LAPJV/LAPMOD (dense & sparse) LAP | BSD-2 / MIT | 0.5.x / 0.9.x (2026) | `lapjv(cost)` | **Wrap optionally** for large/sparse speed |
| **lapsolver** | L4 | Fast dense LAP (C++) | MIT | 1.1.0 (2020) | `solve_dense` | Optional; **stale**, prefer scipy/lapx |
| **POT** | L4 | Optimal transport (EMD, Sinkhorn, GW) | MIT | 0.9.x (2025) | `ot.emd`, `ot.sinkhorn` | **Wrap** for soft/partial/many-many matching |
| **networkx** | L1/L4/L5 | Bipartite/blossom matching, min-cost flow, components | BSD-3 | 3.6.x (2026) | Graph objects + algorithms | **Wrap** for graph matching & clustering |
| **Valentine** | L0 | Schema-matching methods + eval suite | Apache-2.0 | 1.0.0 (2026) | `valentine_match(df1, df2, matcher)` | **Wrap** as join/schema-discovery backend |

---

## 4. Layer-by-layer package notes

### 4.1 End-to-end ER frameworks (L1–L5)

These bundle the whole pipeline. For `equate` they are **reference designs and
optional heavy-duty backends**, not things to imitate wholesale — `equate` is a
*framework* whose value is composability, while these are *applications* with
opinionated, stateful workflows.

- **dedupe** (MIT, 3.0.3, 2024) [3]. Single-machine dedup + record linkage. Its
  distinguishing feature is **active learning**: the library proposes borderline
  record pairs, a human labels match/distinct, and it trains a regularized
  logistic (Fellegi-Sunter-flavored) model plus a set of **learned blocking
  predicates** that it composes into an efficient blocker. API is a stateful
  `Dedupe` / `RecordLink` object (`.prepare_training`, `.train`, `.partition`).
  *Recommendation:* wrap as an optional `equate[dedupe]` backend for the
  "supervised, interactive" matching mode; do **not** reimplement predicate
  learning or the labeling loop.

- **Splink** (MIT, 4.0.x, 2025) [4]. The most scalable open-source probabilistic
  linker. Implements the **Fellegi-Sunter** model with **EM**-trained match
  weights and term-frequency adjustments, and — crucially — pushes computation
  down to a **SQL backend** (DuckDB in-process by default; Spark/Athena/Postgres
  for scale) via `sqlglot`. Rich diagnostics (waterfall charts, match-weight
  visualizations). *Recommendation:* wrap as the `equate[splink]` "probabilistic,
  scalable" backend. Note the non-trivial dependency tree (duckdb, altair,
  igraph, sqlglot); keep it optional.

- **recordlinkage** (BSD-3, 0.16, **Jul 2023**) [5]. The most *library-shaped*
  framework and the closest philosophical cousin to `equate`: cleanly separated
  `Index` (full, blocking, sorted-neighbourhood), `Compare` (string/numeric/
  date/geo comparers), and `Classify` (Fellegi-Sunter, ECM unsupervised, plus
  supervised sklearn classifiers) stages composed over a pandas `MultiIndex` of
  candidate pairs. *Recommendation:* treat as the **primary structural reference**
  for `equate`'s comparer/classifier seams; wrap optionally. Maintenance has
  slowed (last release mid-2023), which argues *against* depending on it hard.

- **py_entitymatching / Magellan** (BSD, 0.4.2, Feb 2024) [6]. UW-Madison's
  academic EM system, notable for framing EM as a *guided how-to workflow* (blocker
  selection → feature auto-generation from `py_stringmatching` measures → ML
  matcher selection with cross-validation → error debugging). API is a large flat
  namespace of functions over pandas + scikit-learn. *Recommendation:* mine it for
  the **feature-generation catalog** and workflow ideas; wrap sparingly — it pins
  older scientific-stack versions and is heavy.

- **Zingg** (**AGPL-3.0**, py client 0.6.x, 2026) [7]. Spark-native ML entity
  resolution/MDM for cluster scale; the JVM engine does blocking (LSH-style),
  pairwise ML, and connected-components clustering, driven by a thin Python client
  that submits Spark jobs. *Recommendation:* integrate **only as an
  out-of-process backend behind a driver interface** — never a Python dependency.
  Its **AGPL** license makes bundling a copyleft hazard; a subprocess/service
  boundary keeps `equate` itself permissive.

**Community-vocabulary note.** All four open frameworks descend from the same
**Fellegi-Sunter** probabilistic core [1]; they differ mainly in *execution engine*
(pandas vs. SQL vs. Spark) and *labeling strategy* (unsupervised EM vs. active
learning vs. supervised). Modern research has moved to **deep/LM entity matching**
(e.g., Ditto: BERT-family, cast EM as sequence-pair classification, reporting up to
~29% F1 improvement over prior SOTA on standard benchmarks) [31] — a capability
`equate` can reach *not* by wrapping these frameworks but by composing L2
(transformers) + L3 (cross-encoder scoring).

### 4.2 String similarity (L3)

- **rapidfuzz** (MIT, 3.14.x) [8]. The performance leader: C++/SIMD
  implementations of Levenshtein, Damerau-Levenshtein, Indel, Jaro, Jaro-Winkler,
  plus `fuzz.*` composite ratios and, importantly, **batch `process.cdist`** that
  builds an entire score matrix efficiently. MIT-licensed (the point of its
  existence: FuzzyWuzzy was GPL). *Recommendation:* the **default fast L3 backend**
  behind `equate`'s scorer protocol, with `difflib.SequenceMatcher` (already used
  in `match_greedily`) as the zero-dependency fallback.

- **thefuzz** (MIT, 0.22.1) [9]. The maintained successor to FuzzyWuzzy; it is now
  a thin wrapper *over rapidfuzz*. *Recommendation:* **skip** — depend on rapidfuzz
  directly rather than the compatibility shim.

- **jellyfish** (MIT, 1.2.x, Rust) [10]. Edit metrics *and* the main reason to
  include it: **phonetic** encoders (Soundex, Metaphone, NYSIIS, Match Rating
  Approach) for name matching. *Recommendation:* wrap as `equate`'s phonetic
  provider; rapidfuzz covers the edit metrics faster.

- **textdistance** (MIT, 4.6.3) [11]. Breadth over speed: ~30 algorithms
  (edit-based, token-based, sequence-based, phonetic, compression-based) behind a
  uniform callable interface, pure-Python by default (optional C accelerators).
  *Recommendation:* wrap as an "algorithm zoo" for prototyping/rare metrics; not a
  hot-path default.

- **py_stringmatching** (BSD, 0.4.7) [12]. The measure library underlying Magellan:
  **tokenizers** (whitespace, qgram, delimiter) composed with **set-based and
  hybrid** similarities (Jaccard, Cosine, Dice, Overlap Coefficient,
  **Monge-Elkan**, **Soft-TF-IDF**, Generalized Jaccard). *Recommendation:* wrap
  for token/set and hybrid measures, which rapidfuzz/jellyfish do not cover.

**Coverage map (so `equate` can expose one metric registry):** character-edit →
rapidfuzz; phonetic → jellyfish; token-set & hybrid → py_stringmatching;
long-tail/experimental → textdistance; zero-dep fallback → stdlib `difflib`.

### 4.3 Representation / embeddings (L2)

- **sentence-transformers** (Apache-2.0, 5.x) [13]. The default for turning text
  into vectors: `model.encode(list_of_texts)` → normalized embeddings for cosine
  scoring, plus **cross-encoders** (joint pair scoring, higher accuracy, no vector
  cache) and utilities `semantic_search` / `paraphrase_mining` that internally do
  L1+L3. Heavy (pulls torch + transformers). *Recommendation:* the default
  `equate[embed]` backend behind an `Embedder` protocol; keep optional.

- **transformers** (Apache-2.0, 5.x) [14]. The general model framework; the route
  to *custom* matchers — a Ditto-style fine-tuned cross-encoder, or any HF model.
  *Recommendation:* wrap for the "advanced/custom scorer" extension point, not for
  the common path.

- **openai** (Apache-2.0 SDK, 2.x) [16]. Hosted embeddings (`text-embedding-3-*`)
  and **LLM-as-judge / LLM-as-matcher** for zero-shot pairwise decisions. Trades
  local compute for network latency, cost, and a hard API dependency.
  *Recommendation:* wrap behind the same `Embedder` (and a future `LLMJudge`)
  protocol so it is one interchangeable strategy among many.

- **fastText** (MIT, 0.9.3, 2024) [15]. Subword (character n-gram) word vectors —
  strong for **out-of-vocabulary** tokens and **multilingual** settings — plus a
  fast language identifier. The upstream repo is effectively archived and the pip
  build can be finicky (C++ toolchain). *Recommendation:* wrap optionally for
  token-level multilingual embeddings; not a default.

### 4.4 ANN & blocking (L1)

The L1 job is to avoid the O(nm) all-pairs blow-up. Two families: **vector ANN**
(for dense-embedding retrieval) and **sketch/LSH** (for set/Jaccard blocking). The
ANN-Benchmarks project is the standard reference for recall-vs-throughput
tradeoffs across these libraries [23].

- **faiss** (`faiss-cpu`, MIT, 1.13.x, Dec 2025) [17]. The heavyweight: **IVF**
  (inverted-file coarse quantization) + **PQ** (product quantization) for
  compressed, billion-scale indexes, plus flat and HNSW indexes, k-means, and GPU
  support. Best when the corpus is large or memory-constrained (PQ compresses
  vectors 8–64×). API is C++-ish index objects. *Recommendation:* wrap as the
  scalable ANN backend (`equate[faiss]`).

- **hnswlib** (Apache-2.0, 0.8.0, Dec 2023) [18]. A small, focused **HNSW** graph
  index: near-logarithmic search, high recall at modest parameters, **incremental
  `add_items`** and mark-delete, tiny dependency footprint (just numpy). Higher
  memory than PQ, no compression. *Recommendation:* the **default in-memory ANN**
  when the embedding path is used — simplest to install, excellent
  recall/latency.

- **annoy** (Apache-2.0, 1.17.3, Jun 2023) [19]. **Random-projection forests**
  with **memory-mapped**, read-only indexes shareable across processes — ideal for
  low-RAM/serverless serving. Index is **static** (no adds after `build`), and
  recall/latency is typically below HNSW/ScaNN [23]. *Recommendation:* wrap for the
  mmap/low-memory niche.

- **scann** (Apache-2.0, 1.4.x, 2025) [20]. Google's **anisotropic vector
  quantization** (a score-aware quantization loss) — frequently top of
  ANN-Benchmarks for recall-at-throughput. Caveat: **Linux-only wheels** (x86_64
  AVX/FMA or aarch64), Python 3.9–3.13. *Recommendation:* wrap as an optional
  high-performance backend, gated on platform detection.

- **pynndescent** (BSD-2, 0.6.0, 2026) [21]. Numba-JIT implementation of
  **NN-Descent** kNN-graph construction; supports **arbitrary distance metrics**
  (not just L2/inner-product), which the others largely do not, and is the ANN
  engine inside UMAP. First-call JIT warmup cost. *Recommendation:* wrap when the
  affinity is a custom/non-Euclidean metric.

- **datasketch** (MIT, 2.0.x) [22]. The **set-similarity blocker**: **MinHash**
  signatures estimate Jaccard cheaply, and **MinHash-LSH** buckets similar sets for
  sub-linear candidate retrieval; LSH-**Forest** (top-k) and LSH-**Ensemble**
  (containment) variants extend it, and HyperLogLog does cardinality. This is the
  classic **blocking** primitive [1,2]. *Recommendation:* wrap as `equate`'s
  token/set-based blocker, complementing vector ANN.

**When to use which (blocking strategy = a pluggable policy):** dense semantic
similarity → hnswlib/faiss/scann; token-set/Jaccard (e.g., addresses, titles) →
datasketch MinHash-LSH; custom metric → pynndescent; classic ER blocking keys
(sorted-neighbourhood, standard blocking) → recordlinkage's `Index` or a small
`equate`-native predicate blocker.

### 4.5 Assignment & graph (L4/L5)

This is the "optimization layer" of doc 03; `equate` already uses
`scipy.optimize.linear_sum_assignment` in `util.py`. The ecosystem here is
mature, permissive, and should be **wrapped, never reimplemented**.

- **scipy `linear_sum_assignment`** (BSD-3, 1.18.x) [24]. Exact **LAP** via a
  modified Jonker-Volgenant / shortest-augmenting-path method (Crouse 2016), worst
  case O(n³), handles rectangular cost matrices and a `maximize` flag. The sensible
  **default matcher**. Dense only.

- **lap / lapx** (BSD-2 / MIT, 0.5.x / 0.9.x) [25]. **LAPJV** (dense
  Jonker-Volgenant) and **LAPMOD** (sparse, core-oriented) — often faster than
  SciPy on large matrices, and LAPMOD exploits sparsity (helpful when the cost
  matrix comes pre-pruned by a blocker, i.e., mostly infinities). `lapx` is the
  actively maintained fork with modern wheels (Python 3.7–3.14). *Recommendation:*
  optional speed backend selected when the score matrix is large/sparse.

- **lapsolver** (MIT, 1.1.0, **2020**) [26]. Fast dense C++ LAP with a minimal
  `solve_dense` API. **Stale** (no release since 2020). *Recommendation:* deprioritize
  in favor of scipy/lapx.

- **POT — Python Optimal Transport** (MIT, 0.9.x) [28]. The tool for *soft,
  fractional, many-to-many* matching: exact **EMD** (network simplex), entropic
  **Sinkhorn** [29] (O(n²) per iteration, GPU/torch backends), **unbalanced** and
  **partial** OT (unequal masses / not-everything-matches), and
  **Gromov-Wasserstein** (matching across *incomparable* spaces via intra-set
  distances). *Recommendation:* wrap as `equate`'s "distributional / soft" matcher
  — a genuinely different L4 semantics than hard 1-1 LAP.

- **networkx** (BSD-3, 3.6.x) [27]. General graph algorithms relevant to matching:
  **Hopcroft-Karp** bipartite maximum-cardinality matching
  (`bipartite.maximum_matching`), **Edmonds' blossom** maximum-weight matching on
  general graphs (`max_weight_matching`), **min-cost flow**, and — for L5 —
  **connected components** to turn pairwise links into entity clusters. Note it
  does *not* ship an LAP-optimized or a stable-matching (Gale-Shapley) solver, so
  those remain scipy/lapx (LAP) or an `equate`-native routine (stable).
  *Recommendation:* wrap for graph-matching objectives and clustering.

### 4.6 Schema / join discovery (L0)

- **Valentine** (Apache-2.0, 1.0.0, 2026) [30]. Both an experiment suite and a
  usable library for **schema matching / column correspondence** — exactly the
  `equate` use case of "find the columns to match (join keys) by comparing how
  well the columns' values match." Bundles five method families: **COMA** (schema
  + instance based), **Cupid** (structural), **Similarity Flooding** (graph
  propagation), **Distribution-Based** (value-distribution clustering), and a
  **Jaccard/Levenshtein** value-overlap baseline. API: `valentine_match(df1, df2,
  matcher, df1_name, df2_name)` → ranked column-pair scores. *Recommendation:*
  wrap as `equate`'s L0 join/schema-discovery backend behind a `SchemaMatcher`
  protocol; its baseline matcher also validates `equate`'s own value-overlap
  approach.

---

## 5. Cross-cutting observations

**License hygiene.** The overwhelming majority of these libraries are permissive
(MIT/BSD/Apache-2.0) and safe as optional dependencies. The one salient exception
is **Zingg (AGPL-3.0)**, which must be kept behind an out-of-process boundary and
never bundled. FuzzyWuzzy's old GPL is why **rapidfuzz/thefuzz (MIT)** exist —
`equate` should standardize on rapidfuzz. `equate`'s **hard** dependencies should
be limited to permissive, lightweight packages (numpy, scipy, and its existing
stdlib fallbacks); everything heavy or copyleft-adjacent goes behind extras.

**Install weight is a first-class design axis.** Backends cluster sharply:
featherweight (stdlib `difflib`, jellyfish, rapidfuzz, datasketch, hnswlib, annoy,
lap/lapx, networkx, scipy) vs. heavyweight (sentence-transformers/transformers →
torch; faiss; Splink → duckdb+igraph+altair; Zingg → Spark/JVM). A framework that
makes the light path work with zero heavy deps, and lets users opt into weight,
will have far better UX than the monolithic E2E tools.

**API-style spectrum.** Three idioms recur: (a) **pure functions** (rapidfuzz,
jellyfish, scipy, POT) — trivial to wrap behind callables; (b) **builder/index
objects** (faiss, hnswlib, annoy, datasketch, sentence-transformers) — need a thin
lifecycle adapter (build → add → query); (c) **stateful pipelines** (dedupe,
Splink, Magellan) — best treated as opaque backends invoked through a coarse
`fit`/`predict` facade. `equate`'s adapters should normalize all three so no
library type leaks into user code.

**Maintenance signals.** Actively developed (2025–26 releases): rapidfuzz,
jellyfish, sentence-transformers, transformers, faiss, POT, lapx, scipy, networkx,
Valentine, py_stringmatching, Splink, Zingg, pynndescent, scann. Slower / at-risk:
recordlinkage (last release Jul 2023), hnswlib (Dec 2023, but stable and
small-surface), annoy (Jun 2023, mature/stable), fastText (0.9.3, upstream
archived), **lapsolver (2020, effectively unmaintained)**. Prefer actively
maintained backends for *defaults*; the stable-but-quiet ones (hnswlib, annoy) are
fine because their scope is small and frozen.

---

## 6. Design implications for equate

1. **A layered protocol architecture — one small ABC/Protocol per seam.** Define
   `SchemaMatcher` (L0), `Blocker` (L1, "given items, yield candidate pairs"),
   `Embedder` (L2, "items → vectors"), `Scorer`/`Comparer` (L3, "pair → score"),
   and `Matcher` (L4, "score matrix → chosen pairs") plus a `Clusterer` (L5). Every
   surveyed library slots behind exactly one of these. This mirrors doc 03's
   scoring/optimization split and keeps `equate` composable where the E2E tools are
   monolithic.

2. **A single `ScoreMatrix` SSOT that flows L3 → L4.** Standardize on one
   sparse-friendly matrix abstraction (dense ndarray or `scipy.sparse`) carrying an
   explicit **sense** (minimize cost vs. maximize similarity) and row/col labels.
   This is the contract between scorers and matchers; it lets a blocker prune it to
   sparse form and route to LAPMOD/`lapx`, and it is exactly what `equate.util`
   already gestures at with `ensure_sparse` + `linear_sum_assignment`.

3. **Optional-dependency boundaries as `pip` extras, with permissive hard deps.**
   Suggested extras: `equate[fuzzy]`→rapidfuzz, `equate[phonetic]`→jellyfish,
   `equate[strsim]`→py_stringmatching/textdistance, `equate[embed]`→
   sentence-transformers, `equate[llm]`→openai/transformers, `equate[ann]`→
   hnswlib (+`equate[faiss]`, `equate[scann]`, `equate[annoy]` for alternatives),
   `equate[lsh]`→datasketch, `equate[assign]`→lapx, `equate[ot]`→POT,
   `equate[graph]`→networkx, `equate[schema]`→valentine, `equate[er]`→
   splink/dedupe/recordlinkage. Keep only numpy/scipy (and stdlib fallbacks) as
   hard deps.

4. **Zero-dependency defaults, upgraded by capability detection.** Every seam must
   have a stdlib/numpy/scipy fallback so a bare `pip install equate` works
   (difflib for L3, brute-force cosine for L1, `scipy.linear_sum_assignment` for
   L4 — all already present). A `check_requirements`-style detector should probe
   which backends are installed, **auto-select the fastest available** (rapidfuzz >
   difflib; hnswlib/faiss > brute force; lapx > scipy on large sparse), and emit an
   actionable message ("install `equate[ann]` for large inputs") rather than
   failing.

5. **Adapter pattern — never leak a backend's types.** Wrap builder/index objects
   (faiss/hnswlib/annoy/datasketch/sentence-transformers) behind a uniform
   `build → add → query` lifecycle, and stateful pipelines (dedupe/Splink/Magellan)
   behind a coarse `fit`/`predict` facade. Users see `equate` types only.

6. **A strategy registry (open-closed) keyed by name.** Expose named strategies
   (`scorer="rapidfuzz.jaro_winkler"`, `matcher="lap"`, `blocker="minhash_lsh"`,
   `embedder="sbert:all-MiniLM-L6-v2"`) resolved through a registry so users can
   register custom callables without subclassing — honoring "simple things simple,
   complex things possible."

7. **Model the matcher *family*, not one algorithm** (from doc 03): hard 1-1 LAP
   (scipy/lapx), greedy/threshold (already in `equate`), bipartite/blossom
   (networkx), and **soft/partial/many-to-many optimal transport** (POT) are
   distinct L4 *semantics* with different return types. Make the `Matcher` protocol
   explicit about cardinality (1-1, 1-many, many-many, fractional) and sense.

8. **Copyleft and heavyweight engines only out-of-process.** Zingg (AGPL, Spark)
   and, where desired, Splink-on-Spark belong behind a `Backend`/driver interface
   invoked as a subprocess or service — never a Python import — so `equate` stays
   permissively licensed and light.

9. **Reimplement almost nothing computational; own the orchestration.** Do **not**
   reimplement: string metrics, phonetics, embeddings, ANN, MinHash-LSH, LAP,
   optimal transport, or schema matching — mature wrappers exist for all. **Do**
   own: the layer protocols, the `ScoreMatrix` SSOT, the blocking-key/predicate DSL,
   capability detection, the strategy registry, and the existing native matchers
   (`match_greedily`, threshold assignment). `equate`'s differentiation is the
   *uniform, progressively-disclosed composition* of this ecosystem — the thing no
   single library above provides.

---

## References

[1] Christophides V, Efthymiou V, Palpanas T, Papadakis G, Stefanidis K. *An
Overview of End-to-End Entity Resolution for Big Data.* ACM Computing Surveys
53(6), 2020.
[ACM](https://dl.acm.org/doi/abs/10.1145/3418896)

[2] Papadakis G, Skoutas D, Thanos E, Palpanas T. *Blocking and Filtering
Techniques for Entity Resolution: A Survey.* ACM Computing Surveys 53(2), 2020.
[Ghent U. biblio](https://biblio.ugent.be/publication/01J90Y07Y2GTA6QWMTRZVB1PCB)

[3] Gregg F, Eder D. *dedupe: A Python library for accurate and scalable data
deduplication and entity-resolution.*
[GitHub](https://github.com/dedupeio/dedupe) ·
[PyPI](https://pypi.org/project/dedupe/)

[4] Linacre R, et al. *Splink: Fast, accurate and scalable probabilistic data
linkage* (UK Ministry of Justice).
[GitHub](https://github.com/moj-analytical-services/splink) ·
[docs](https://moj-analytical-services.github.io/splink/)

[5] de Bruin J. *Python Record Linkage Toolkit.*
[docs](https://recordlinkage.readthedocs.io/) ·
[GitHub](https://github.com/J535D165/recordlinkage)

[6] Konda P, Das S, Suganthan G C P, Doan A, et al. *Magellan: Toward Building
Entity Matching Management Systems.* PVLDB 9(12):1197-1208, 2016.
[VLDB/ACM](https://dl.acm.org/doi/10.14778/2994509.2994535) ·
[py_entitymatching](https://github.com/anhaidgroup/py_entitymatching)

[7] Goyal S, et al. *Zingg: Scalable identity resolution, entity resolution and
deduplication using ML.*
[GitHub](https://github.com/zinggAI/zingg) · [PyPI](https://pypi.org/project/zingg/)

[8] Bachmann M. *RapidFuzz: rapid fuzzy string matching.*
[GitHub](https://github.com/rapidfuzz/RapidFuzz) ·
[PyPI](https://pypi.org/project/rapidfuzz/)

[9] SeatGeek. *TheFuzz (maintained successor to FuzzyWuzzy).*
[GitHub](https://github.com/seatgeek/thefuzz) ·
[PyPI](https://pypi.org/project/thefuzz/)

[10] Turk J. *Jellyfish: approximate and phonetic string matching (Rust).*
[GitHub](https://github.com/jamesturk/jellyfish) ·
[PyPI](https://pypi.org/project/jellyfish/)

[11] Orsinium (Gram M). *TextDistance: compute distance between sequences.*
[GitHub](https://github.com/life4/textdistance) ·
[PyPI](https://pypi.org/project/textdistance/)

[12] AnHai Group. *py_stringmatching: a comprehensive set of string tokenizers and
similarity measures.*
[GitHub](https://github.com/anhaidgroup/py_stringmatching) ·
[PyPI](https://pypi.org/project/py-stringmatching/)

[13] Reimers N, Gurevych I. *Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks.* EMNLP-IJCNLP 2019.
[arXiv](https://arxiv.org/abs/1908.10084) · [SBERT](https://www.sbert.net/)

[14] Wolf T, et al. *Transformers: State-of-the-Art Natural Language Processing.*
EMNLP 2020 (System Demos).
[arXiv](https://arxiv.org/abs/1910.03771) ·
[GitHub](https://github.com/huggingface/transformers)

[15] Bojanowski P, Grave E, Joulin A, Mikolov T. *Enriching Word Vectors with
Subword Information* (fastText). TACL 5, 2017.
[arXiv](https://arxiv.org/abs/1607.04606) ·
[fastText](https://github.com/facebookresearch/fastText)

[16] OpenAI. *Embeddings guide (text-embedding-3) and Python SDK.*
[docs](https://platform.openai.com/docs/guides/embeddings) ·
[SDK](https://github.com/openai/openai-python)

[17] Johnson J, Douze M, Jégou H. *Billion-scale similarity search with GPUs*
(FAISS). IEEE Transactions on Big Data, 2019.
[arXiv](https://arxiv.org/abs/1702.08734) ·
[GitHub](https://github.com/facebookresearch/faiss)

[18] Malkov Yu A, Yashunin D A. *Efficient and robust approximate nearest neighbor
search using Hierarchical Navigable Small World graphs.* IEEE TPAMI 42(4), 2018.
[arXiv](https://arxiv.org/abs/1603.09320) ·
[hnswlib](https://github.com/nmslib/hnswlib)

[19] Bernhardsson E. *Annoy: Approximate Nearest Neighbors in C++/Python optimized
for memory usage and loading/saving to disk* (Spotify).
[GitHub](https://github.com/spotify/annoy)

[20] Guo R, Sun P, Lindgren E, Geng Q, Simcha D, Chern F, Kumar S. *Accelerating
Large-Scale Inference with Anisotropic Vector Quantization* (ScaNN). ICML 2020.
[arXiv](https://arxiv.org/abs/1908.10396) ·
[ScaNN](https://github.com/google-research/google-research/tree/master/scann)

[21] Dong W, Charikar M, Li K. *Efficient K-Nearest Neighbor Graph Construction for
Generic Similarity Measures* (NN-Descent). WWW 2011.
[ACM](https://dl.acm.org/doi/10.1145/1963405.1963487) ·
[pynndescent](https://github.com/lmcinnes/pynndescent)

[22] Broder A Z. *On the resemblance and containment of documents* (MinHash).
Compression and Complexity of Sequences, 1997.
[IEEE](https://ieeexplore.ieee.org/document/666900) ·
[datasketch](https://github.com/ekzhu/datasketch)

[23] Aumüller M, Bernhardsson E, Faithfull A. *ANN-Benchmarks: A benchmarking tool
for approximate nearest neighbor algorithms.* Information Systems 87, 2020.
[arXiv](https://arxiv.org/abs/1807.05614) · [site](https://ann-benchmarks.com/)

[24] Crouse D F. *On implementing 2D rectangular assignment algorithms.* IEEE
Trans. Aerospace and Electronic Systems, 2016 (SciPy's `linear_sum_assignment`).
[SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)

[25] Jonker R, Volgenant A. *A shortest augmenting path algorithm for dense and
sparse linear assignment problems.* Computing 38, 1987.
[Springer](https://link.springer.com/article/10.1007/BF02278710) ·
[lap](https://github.com/gatagat/lap) · [lapx](https://github.com/rathaROG/lapx)

[26] Heindl C. *py-lapsolver: fast linear assignment problem solvers.*
[GitHub](https://github.com/cheind/py-lapsolver) ·
[PyPI](https://pypi.org/project/lapsolver/)

[27] Hagberg A, Schult D, Swart P. *Exploring network structure, dynamics, and
function using NetworkX.* Proc. SciPy 2008.
[NetworkX](https://networkx.org/) ·
[matching docs](https://networkx.org/documentation/stable/reference/algorithms/matching.html)

[28] Flamary R, Courty N, et al. *POT: Python Optimal Transport.* JMLR 22(78):1-8,
2021.
[JMLR](https://jmlr.org/papers/v22/20-451.html) · [site](https://pythonot.github.io/)

[29] Cuturi M. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.*
NeurIPS 26, 2013.
[NeurIPS](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)

[30] Koutras C, Siachamis G, Ionescu A, Psarakis K, et al. *Valentine: Evaluating
Matching Techniques for Dataset Discovery.* IEEE ICDE 2021.
[IEEE](https://ieeexplore.ieee.org/document/9458921) ·
[GitHub](https://github.com/delftdata/valentine)

[31] Li Y, Li J, Suhara Y, Doan A, Tan W-C. *Deep Entity Matching with Pre-Trained
Language Models* (Ditto). PVLDB 14(1):50-60, 2020.
[arXiv](https://arxiv.org/abs/2004.00584) ·
[VLDB/ACM](https://dl.acm.org/doi/10.14778/3421424.3421431)
