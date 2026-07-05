# `equate` research corpus

Grounded, cited reference material for the **equate redesign** — a general framework
for **matching**: given collections of objects with no reliable shared identifier,
decide which correspond and emit that correspondence as a set of (optionally scored)
tuples. Two research rounds (19 docs, ~11k lines) map the problem *and* solution
space and decide the architecture. All references are Vancouver-style with working
links; round-1's 12 load-bearing claims were independently fact-checked (all
supported).

## Read in this order

1. **[`00-taxonomy-and-terminology.md`](00-taxonomy-and-terminology.md)** — start
   here. The canonical decomposition **featurize → compare → match/resolve**, the
   cross-community synonyms table (ER / record linkage / data matching / bipartite
   matching / similarity search / schema matching), the dimensions matching problems
   vary along, the equivalence-relation framing (why `equate` ≠ equality), and a
   consolidated glossary.
2. **[`10-design-implications-for-equate.md`](10-design-implications-for-equate.md)** —
   the architecture: the core `match(A,B)` case, injectable stage protocols, optional-
   extras strategy, scalability & interactivity paths, qh/zodal attach points, a full
   module sketch, and a wrap-vs-reimplement table.
3. **[`11-design-decisions-and-open-questions.md`](11-design-decisions-and-open-questions.md)** —
   the **decision register**. Resolves 10 core design tensions (D1–D10) into one
   coherent design + the resolved score & data-model contract, and lists 12 tracked
   open questions. **The roadmap epic draws its sub-issues from here.**

## Facet docs

**Round 1 — the pipeline stages & the field**

| Doc | Covers |
|---|---|
| [`01`](01-entity-resolution-record-linkage.md) | Entity resolution / record linkage: the end-to-end pipeline, Fellegi-Sunter, clustering, evaluation |
| [`02`](02-blocking-and-scalable-candidate-generation.md) | Blocking, sorted-neighborhood, meta-blocking, LSH/MinHash, ANN (HNSW/IVF-PQ), PC/RR/PQ metrics |
| [`03`](03-assignment-and-graph-matching.md) | Assignment & bipartite matching: Hungarian/JV, stable (Gale-Shapley), Murty k-best, optimal transport |
| [`04`](04-featurization-and-representation.md) | Featurization: feature vectors/embeddings (text/image/audio), the key-function contract, distance metrics |
| [`05`](05-comparison-and-similarity-functions.md) | Comparators: edit/token/phonetic/hybrid string sim, numeric/geo, comparison vectors, calibration |
| [`06`](06-deep-learning-and-llm-entity-matching.md) | Learned matchers: DeepMatcher, Ditto, bi- vs cross-encoders, LLM matching, cascades |
| [`07`](07-schema-and-ontology-matching.md) | Schema/ontology/join-key matching (the tabular-completion angle): COMA, Valentine, LSH-Ensemble |
| [`08`](08-interactive-active-learning-and-hitl.md) | Human-in-the-loop: active learning, review, interactive re-optimization with top-k memory |
| [`09`](09-python-ecosystem-landscape.md) | Python ecosystem: dedupe/Splink/recordlinkage/Magellan/rapidfuzz/faiss/POT — wrap vs reimplement |

**Round 2 — gap-filling (from the round-1 completeness critic)**

| Doc | Covers |
|---|---|
| [`12`](12-sequence-and-graph-structure-matching.md) | Sequence alignment (DTW/soft-DTW, Needleman-Wunsch/Smith-Waterman) & graph/network alignment (GED, VF2, WL kernels, IsoRank/REGAL) as matcher families |
| [`13`](13-llm-and-modern-embedding-matching.md) | LLM matching (Jellyfish, AnyMatch, ComEM, structured output, cascades) & modern embedders (E5, BGE-M3, GTE, Nomic, Voyage, text-embedding-3, MRL) |
| [`14`](14-evaluation-benchmarks-and-methodology.md) | Evaluation: pairwise & cluster metrics (B³/CEAF/ARI), blocking metrics, benchmark catalog, leakage, honest methodology |
| [`15`](15-collective-incremental-and-bayesian-er.md) | Collective/relational ER, incremental/streaming/temporal ER, Bayesian ER (Sadinle, Steorts d-blink), multi-source linkage |
| [`16`](16-data-fusion-and-canonicalization.md) | Data fusion, truth discovery (TruthFinder, Dawid-Skene), canonicalization/survivorship, MDM (the golden-record step) |
| [`17`](17-vector-databases-and-scale-out.md) | Vector-DB backends (pgvector/Qdrant/Milvus/LanceDB/…) & scale-out (Spark/Dask/DuckDB/GPU) for blocking/matching |
| [`18`](18-frontiers-pprl-fairness-geospatial.md) | Responsible/specialized frontiers: privacy-preserving record linkage, fairness/bias, geospatial & map matching, uncertainty |

## How this maps to the build

The redesign is tracked as a GitHub **epic issue** with linked sub-issues on
`thorwhalen/equate`; each sub-issue names the doc(s) and decision(s) (`Dn`) it
implements. The agent-facing north-star is the **`equate-dev-architecture`** dev skill
(repo `skills/`, indexed from `.claude/CLAUDE.md`); adding a strategy follows
**`equate-dev-add-strategy`**.
