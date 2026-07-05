# Vector Databases & Scale-Out Execution for Blocking/Matching

**Abstract.** Doc [`02`](02-blocking-and-scalable-candidate-generation.md) covered
the *algorithms* of candidate generation (blocking, LSH, ANN indexes); this
document covers the *infrastructure and product layer* that runs them at scale.
It maps the boundary between **raw ANN libraries** (FAISS, hnswlib, ScaNN — an
index you own inside your process) and **vector databases** (pgvector, Qdrant,
Milvus, Weaviate, LanceDB, Chroma, Pinecone, Vespa — a persistent service that
adds CRUD, metadata filtering, hybrid sparse+dense search, replication, and an
API), surveys the on-disk index families that make billion-scale search
affordable (DiskANN/Vamana, IVF-PQ, CAGRA on GPU), and covers **scale-out
execution engines** for the matcher itself (DuckDB in-process, Spark via Splink /
Dedupe / Zingg, Dask out-of-core, and GPU acceleration via RAPIDS cuDF / cuVS /
faiss-GPU). It closes with an explicit **wrap-vs-embed** verdict for `equate`'s
blocking layer and a **`dol`-style storage/index interface** so that scale is
*delegated to a backend* rather than reimplemented, keeping the core small.

---

## 1. Scope: the layer *beneath* candidate generation

Blocking and ANN search are *algorithms*; running them over hundreds of millions
of records, keeping the index warm, filtering by metadata, updating it as data
arrives, and doing so within a memory and cost budget are *systems* concerns.
This document is about that systems layer. It answers two operational questions a
general matching framework must delegate rather than solve:

1. **Where do the vectors/signatures live, and what serves the neighbor
   queries?** — the *vector-database vs raw-ANN-library* decision (§§2–5).
2. **What executes the matching workload when it does not fit in one process's
   RAM?** — the *scale-out execution* decision: in-process analytical engines,
   distributed clusters, out-of-core streaming, and GPUs (§6).

These are product/infrastructure choices. The thesis of the corpus
([`09`](09-python-ecosystem-landscape.md),
[`10`](10-design-implications-for-equate.md)) — *own the orchestration, wrap the
heavy machinery* — applies with full force here: `equate` should define narrow
interfaces and let a swappable backend carry the scale.

---

## 2. Raw ANN libraries vs vector databases — the boundary

A **raw ANN library** (FAISS [1], hnswlib, ScaNN — all covered in
[`02`](02-blocking-and-scalable-candidate-generation.md) §7) is a *data
structure*: you hold vectors in memory, `build()` an index, `query(vec, k)`, and
you own everything else — persistence, concurrency, incremental updates,
metadata filtering, sharding, backups. It is the fastest and lightest option and
often the *right* one for batch blocking, where you build once and query once.

A **vector database (VDBMS)** is a *product* wrapped around such an index. It
adds the operational surface that a library deliberately omits [2,3,4]:

| Capability | Raw ANN library | Vector database |
|---|---|---|
| kNN / ANN query | ✅ (the whole point) | ✅ |
| Persistence & durability | you write files | ✅ managed |
| Incremental insert/update/delete | limited (HNSW yes, IVF awkward) | ✅ (with tombstones/compaction) |
| **Metadata / payload filtering** | ✗ (do it yourself) | ✅ (filtered ANN) |
| **Hybrid sparse+dense (BM25) search** | ✗ | ✅ (most products) |
| Concurrency / transactions / API | ✗ | ✅ (gRPC/REST/SQL) |
| Sharding, replication, HA | ✗ | ✅ |
| Multi-tenancy, auth, quotas | ✗ | ✅ |
| Cost model | your RAM | server / managed / serverless |

The practical rule: **use a raw library for one-shot batch blocking that fits in
RAM; use a vector database when the index must persist, update online, be
filtered by attributes, be queried by a service, or exceed a single machine.**
For `equate` this is decisive — the framework should be able to sit on *either*,
behind one interface (§8–9). Crucially, **every mainstream vector database uses
the same underlying index algorithms** (HNSW, IVF, PQ, DiskANN, CAGRA) — the
product differences are operational, not algorithmic. Choosing a backend is a
deployment decision, not a recall decision.

---

## 3. The index algorithms inside every backend

[`02`](02-blocking-and-scalable-candidate-generation.md) §7 covered the
in-memory families (HNSW, IVF, PQ, IVF-PQ, ScaNN). Two additional families
matter specifically at the *product/scale* layer: **on-disk graphs** and
**GPU graphs**. Together with quantization, they are what let a database serve
billions of vectors under a fixed RAM/cost budget.

### 3.1 On-disk ANN: DiskANN / Vamana

Keeping HNSW's full graph plus uncompressed vectors in RAM is the dominant cost
of in-memory ANN. **DiskANN** (Subramanya et al., NeurIPS 2019) [5] breaks this
by putting the bulk of the index on **SSD** and keeping only a *compressed*
(product-quantized) representation in RAM for navigation. Its graph index,
**Vamana**, is built with a tunable pruning parameter α that yields a graph with
long-range edges and a small search radius, so that a query touches few disk
pages. Reported result: a **billion-point** SIFT1B index served on a *single*
64 GB machine with an SSD at **> 5000 QPS, < 3 ms mean latency, 95%+ recall@1**
[5] — where in-memory IVF/PQ methods of comparable footprint plateau near 50%
recall. DiskANN (and its streaming/updatable variant) is now the on-disk index
inside Milvus [8], Weaviate, Azure, and — via **pgvectorscale**'s
`StreamingDiskANN` — Postgres [17]. The tradeoff vs HNSW-in-RAM: much lower
memory/cost, at the price of SSD I/O per query (higher tail latency).

### 3.2 GPU ANN: CAGRA and cuVS

For build-heavy or high-throughput workloads, GPUs win. NVIDIA's **cuVS**
(formerly RAFT) provides GPU implementations of IVF-Flat, IVF-PQ, and **CAGRA**
(a graph index built from the ground up for GPUs, loosely NSG-like) [6,7,15].
cuVS has been **integrated into FAISS** so that `GpuIndexIVFFlat`,
`GpuIndexIVFPQ`, and a new `GpuIndexCagra` can use the cuVS kernels [7], and into
**Milvus** as a GPU index option [8]. The headline benefit is **index build
time** (often an order of magnitude faster than CPU HNSW) and very high query
throughput; the cost is GPU memory and the need for CUDA hardware. A CAGRA graph
can also be *converted to an HNSW graph* for CPU serving — build on GPU, serve on
CPU. This matters for blocking because blocking is frequently a *rebuild-often*
workload (new batch → new index).

### 3.3 Quantization — the memory lever shared by all backends

Every backend exposes some compression to trade recall for RAM/throughput:

- **Scalar quantization (SQ)**: `float32 → int8`, ~4× smaller, small recall loss.
- **Product quantization (PQ)** ([`02`](02-blocking-and-scalable-candidate-generation.md) §7.2):
  split into sub-vectors, quantize each to a codebook; 8–32× smaller.
- **Binary quantization (BQ)**: 1 bit per dimension, Hamming distance; up to 32×
  smaller and extremely fast, used as a *coarse* first stage before re-ranking
  full-precision vectors. Qdrant, Milvus, and pgvectorscale (Statistical Binary
  Quantization [17]) all ship a binary-quantize-then-rerank path.
- **Half precision**: `float16`/`bfloat16` (pgvector's `halfvec` [9]) — 2×
  smaller, and in Postgres it doubles the indexable dimensionality (§4.1).

The universal pattern is **coarse-and-fine**: search a compressed index for many
candidates, then re-rank the shortlist with full-precision vectors — the same
two-stage philosophy as blocking-then-matching, one level down.

---

## 4. Vector-DB products as blocking backends

All of the following can serve as `equate`'s candidate generator: featurize
objects → upsert vectors (+ metadata) → `search(query_vec, k, filter)` returns
candidate ids. They differ in *deployment shape* and *operational tradeoffs*, not
in the core algorithm.

### 4.1 pgvector (+ pgvectorscale) — vectors *inside Postgres*

**pgvector** [9] is a Postgres extension adding a `vector` type and ANN indexes.
The compelling property: your vectors live **in the same transactional database
as your records**, so a "block + join + filter" is a single SQL query — no
separate service, no data sync. Key facts (v0.8):

- **Index types**: `HNSW` (params `m`=16, `ef_construction`=64 defaults; query
  `ef_search`=40) and `IVFFlat` (`lists`, `probes`=1). HNSW has better
  speed/recall; IVFFlat builds faster and uses less memory but needs data present
  to train [9].
- **Vector types**: `vector` (float32, ≤16,000 dims stored), `halfvec` (float16),
  `bit` (binary, ≤64,000 dims), `sparsevec` (≤16,000 nonzeros). **Indexing
  gotcha**: the full-precision `vector` type can only be *indexed* up to **2,000
  dimensions** (an 8 KB Postgres page limit); `halfvec` roughly doubles that to
  ~4,000, so high-dim embeddings (e.g. 3,072-d) must use `halfvec` to be indexed
  [9,10].
- **Distance operators**: L2 `<->`, inner product `<#>`, cosine `<=>`, L1 `<+>`,
  Hamming `<~>`, Jaccard `<%>` [9].
- **Hybrid search**: pair the vector index with Postgres full-text search
  (`tsvector`/BM25-style ranking) and combine — hybrid in one engine.
- **Iterative index scans** (v0.8+): keep scanning the index until *k* results
  survive the metadata filter, fixing the classic "filter throws away the ANN
  shortlist" problem [9].

**pgvectorscale** [17] adds `StreamingDiskANN` (on-disk, updatable) + binary
quantization + label-based filtered search, claiming (on 50 M embeddings) 28×
lower p95 latency and 16× higher QPS than Pinecone's storage-optimized tier at
75% lower self-hosted cost. Both are permissively licensed (PostgreSQL license).
**For `equate`, pgvector is the "no new infrastructure" blocking backend** —
ideal when data already lives in Postgres.

### 4.2 Qdrant — Rust, filtering-first

**Qdrant** [11] is a Rust vector DB whose distinguishing feature is
**filterable/"filtrable" HNSW**: rather than filter *after* search (which can
empty the shortlist) or *before* (which is slow), it folds the metadata
predicate **into the HNSW graph traversal** so the walk only visits points
satisfying the filter (the approach the literature calls ACORN-style filtered
ANN). It supports scalar/product/**binary** quantization, `mmap` on-disk storage
for out-of-RAM collections, **named vectors** (multiple embeddings per point),
and a **Query API** with server-side multi-stage `prefetch` + **fusion (RRF and
DBSF)** for hybrid sparse+dense and late-interaction (ColBERT) re-ranking [11].
Strong default choice for *filtered blocking* (block within a partition — e.g.
same country/type — then ANN).

### 4.3 Milvus — distributed, cloud-native, largest scale

**Milvus 2.x** [8] is a distributed, cloud-native system that **separates storage
from compute**: data is chunked into immutable **segments** in object storage;
query nodes, data nodes, and index nodes scale independently. It supports the
widest index menu — FLAT (exact), IVF variants, HNSW, SCANN, **DiskANN**, and
**GPU (cuVS/CAGRA)** — plus `mmap`, scalar filtering, and hybrid search. It is
the safe pick for **billion-scale**, multi-tenant, elastically-scaled workloads,
at the cost of operational complexity (etcd, object store, message queue, several
node types). **Milvus Lite** offers an embedded mode for prototyping. For
`equate`, Milvus is the "we already run a cluster and need billions" backend, not
the default.

### 4.4 Weaviate — modular, hybrid-native

**Weaviate** [12] is an HNSW-based vector DB with first-class **hybrid search**:
its `hybrid` query takes an **`alpha`** knob interpolating between pure BM25
keyword (`alpha=0`) and pure dense vector (`alpha=1`), fusing the two ranked
lists via **Reciprocal Rank Fusion** or relative-score fusion [12,13]. It offers
product/binary quantization, on-disk options, and modular "vectorizer" plugins
that embed text at ingest. Attractive when the matching signal is *both* lexical
and semantic (names + descriptions).

### 4.5 LanceDB — embedded, on-disk, data-lake-native

**LanceDB** [14] is an *embedded* (in-process, "SQLite-for-vectors") retrieval
library built on the **Lance** columnar format — an Arrow-native, versioned,
multimodal file format optimized for **fast random access** on object storage
(S3) [14,16]. Its **disk-based, compute-storage-separated** architecture targets
billion-scale search at a fraction of in-RAM cost, and it stores vectors,
scalars, full-text indexes, and blobs *in one table*. No server to run — you
`pip install` and point it at a directory or S3 bucket. **This is the natural fit
for `equate`'s local/offline default at scale**: it gives on-disk ANN + metadata
+ full-text without standing up a service, and its `dol`-like table semantics map
cleanly onto a storage interface (§9).

### 4.6 Chroma — easiest to start, RAM-bound

**Chroma** is the minimal-friction option: `pip install chromadb`, a few lines to
a working semantic index, HNSW under the hood. The tradeoff is that it is
essentially **in-memory** (every vector in RAM), so it does not scale to the tens
of millions the way DiskANN/Lance backends do. Good as a zero-config *prototype*
blocker; not a scale answer.

### 4.7 Pinecone — managed serverless

**Pinecone Serverless** (2024 rewrite) [18] is a fully-managed VDBMS that
**separates reads, writes, and storage**, with **blob storage as the source of
truth**. Records are organized into immutable **slabs** (an LSM-tree-like design:
writes land in an in-memory memtable, flush to object storage as L0 slabs, then
compact into larger, index-bearing slabs), giving on-demand compute over cheap
storage and a claimed 10–100× cost reduction vs the prior pod model [18]. No
infrastructure to run, usage-based pricing, but proprietary and network-bound.
For `equate` it is a *hosted* backend behind the same interface — never an import.

### 4.8 Vespa — one engine for text + tensors + ANN

**Vespa** [19] is a mature big-data serving engine where **text (BM25),
attributes, tensors, and HNSW ANN live in one index**, with a rich ranking DSL
that composes lexical and vector scores in a **multi-phase** pipeline, plus
multi-vector HNSW and real-time updates at scale. It is the heaviest-weight but
most *complete* hybrid/ranking engine — appropriate when matching is really a
learned ranking problem over many signals.

### 4.9 Operational tradeoffs at a glance

| Backend | Shape | Storage | Filtering | Hybrid | Scale sweet spot | License |
|---|---|---|---|---|---|---|
| pgvector | Postgres ext | in-DB, disk | SQL `WHERE` + iterative scan | FTS + vector | ≤ ~10 M (more w/ pgvectorscale DiskANN) | PostgreSQL |
| Qdrant | server (Rust) | mmap/disk | **filterable HNSW** | RRF/DBSF, ColBERT | 10 M–1 B | Apache-2.0 |
| Milvus | distributed | object store + segments | scalar + mmap | yes | **1 B+** | Apache-2.0 |
| Weaviate | server (Go) | disk options | pre/post | **alpha + RRF** | 10 M–1 B | BSD-3 |
| LanceDB | **embedded** | Lance on disk/S3 | yes | FTS + vector | 1 M–1 B (on-disk, low cost) | Apache-2.0 |
| Chroma | embedded/server | **RAM** | metadata | limited | ≤ few M (prototype) | Apache-2.0 |
| Pinecone | **managed serverless** | blob (slabs) | metadata | sparse+dense | elastic, hosted | proprietary |
| Vespa | server (JVM) | disk | rich | **multi-phase** | 1 B+ | Apache-2.0 |

Decision axes for `equate`: **embedded vs server vs serverless** (does the user
want a process, a service, or a bill?); **in-memory vs on-disk** (RAM budget vs
tail latency); **self-hosted vs managed** (ops vs money); and **permissive vs
copyleft/proprietary** — matters because `equate` must stay permissively licensed
and must never *import* an AGPL engine (cf. Zingg, §6.2, and
[`10`](10-design-implications-for-equate.md) §7).

---

## 5. Metadata filtering and hybrid search — why the product layer matters for blocking

Two VDBMS features are not conveniences but *core blocking primitives*.

### 5.1 Filtered ANN = "block + retrieve" in one call

A **filtered ANN query** ("nearest neighbors *where* `country='FR' AND type='org'`")
is *exactly* the classic pattern of **partition-then-match**: the metadata
predicate is a cheap disjoint block, the ANN is dense candidate generation within
it. Doing both in one indexed call is far better than the two naive extremes:

- **Post-filtering** (ANN then drop non-matching) can return **fewer than k** (or
  zero) results when the filter is selective — the ANN shortlist is wasted.
- **Pre-filtering** (scan the subset then brute-force) loses the index.

The modern answer is **filter-aware graph traversal**: Qdrant's filterable HNSW
[11], Milvus's filtered search [8], and pgvector's **iterative index scans** [9]
all keep searching until *k* filtered results survive. **For `equate`, this means
a `Blocker` can express structured blocking keys as filter predicates and dense
similarity as the ANN, unifying [`02`](02-blocking-and-scalable-candidate-generation.md)'s
schema-based blocking and dense blocking into one backend call.**

### 5.2 Hybrid sparse+dense search = "union sparse and dense blockers"

[`02`](02-blocking-and-scalable-candidate-generation.md) §8 argued that sparse
(token/TF-IDF/BM25) and dense (embedding) blocking are **complementary** and
should be unioned. Hybrid search is that union productized. BM25 nails exact/rare
tokens (product codes, IDs, rare names); dense catches paraphrase/synonymy.
Because their score scales differ and are not linearly separable, systems fuse by
**rank**, not score: **Reciprocal Rank Fusion (RRF)** — score(d) = Σ 1/(k + rankᵢ(d))
with constant k≈60 (Cormack, Clarke & Buettcher, SIGIR 2009) — is the de-facto
standard [12,13,11], with **Distribution-Based Score Fusion (DBSF)** and
Weaviate's `alpha` linear interpolation as alternatives. Qdrant's `prefetch`
pipeline additionally allows a cheap first stage (sparse or low-dim dense) feeding
an expensive re-ranker (ColBERT late interaction) [11] — a productized version of
the **cascade** ([`06`](06-deep-learning-and-llm-entity-matching.md)). **For
`equate`, hybrid retrieval is the backend-native way to run "sparse ∪ dense
blocking + re-rank" without hand-wiring two indexes.**

---

## 6. Scale-out execution for the matcher itself

Blocking bounds the *candidate* count; the matcher still runs comparisons,
clustering, and assignment over what may be a huge sparse graph. When that
exceeds one process, four execution strategies apply.

### 6.1 In-process analytical engines: DuckDB (the "scale-up" default)

**DuckDB** [20] is an *embedded, in-process* OLAP database — "SQLite for
analytics" — with a **vectorized, columnar, larger-than-RAM (spilling)** engine.
It parallelizes across cores, streams data that does not fit in memory, and runs
inside your Python process with zero server. It is the backend that makes
**Splink** [21,22] link **millions of records on a laptop in ~1 minute** and
deduplicate **7 M records in ~2 minutes** [23] — the entire blocking + comparison
+ Fellegi-Sunter scoring pipeline is expressed as SQL that DuckDB executes
out-of-core. **The key lesson for `equate`: "scale-out" often means "scale-up
with a vectorized out-of-core engine" first.** Pushing candidate generation and
comparison down into DuckDB SQL (join on blocking keys, compute similarities as
SQL functions) reaches tens of millions of records on one machine before any
cluster is needed. DuckDB can also query Postgres, Parquet, and Arrow in place,
and has community extensions for vector similarity — making it a strong
*execution* substrate under `equate`'s blocking + comparison stages.

### 6.2 Distributed clusters: Spark (Splink, Dedupe, Zingg)

When even out-of-core single-node is not enough (hundreds of millions+), the
answer is a **distributed dataframe engine**, overwhelmingly **Apache Spark**.
The pattern is identical across tools: **blocking becomes a distributed join/
shuffle on blocking keys, comparison becomes a per-partition UDF/map, clustering
becomes a distributed connected-components (GraphFrames)**.

- **Splink** [21,22] is the exemplar of the **multi-backend** design: the *same*
  probabilistic linkage model runs on **DuckDB** (default, fastest, up to ~few
  million on a laptop / tens of millions on a big box), **Spark** (very large,
  distributed, fewer OOM risks), **AWS Athena** (10 M+ on AWS), **SQLite**, and
  **PostgreSQL** — because the logic is compiled to **SQL** and the backend
  executes it. This is the single most important architectural idea for `equate`
  to borrow (§8): *express the workload abstractly, dispatch to a backend.*
- **Dedupe** [24] runs in-process with a PostgreSQL example for big data and
  documented PySpark patterns for scaling its blocking/scoring.
- **Zingg** [25] is Spark-native ER (blocking model + ML matching + distributed
  clustering) reading Snowflake/S3/RDBMS; it explicitly learns a blocking model so
  it compares ~0.05–1% of the pair space. **It is AGPL-3.0**, so `equate` may
  *shell out* to it as an out-of-process job but must **never import it**
  ([`10`](10-design-implications-for-equate.md) §7).

### 6.3 Dask and out-of-core (the Pythonic middle path)

**Dask** [26] parallelizes NumPy/pandas/scikit-learn across cores and machines
with familiar APIs, and streams **larger-than-memory** partitions (out-of-core).
For `equate` it is the lightest way to fan blocking and pairwise scoring across a
partitioned dataset without a JVM cluster: partition A and B, block within/across
partitions, `map_partitions` the comparator, and reduce. It is the natural
`equate[dask]` execution backend for users who want horizontal scale but live in
the PyData stack, sitting between single-node DuckDB and full Spark.

### 6.4 GPU acceleration: RAPIDS cuDF, cuVS, faiss-GPU

GPUs accelerate two distinct parts of matching:

- **Candidate generation / ANN**: **cuVS** (IVF-Flat, IVF-PQ, CAGRA) and
  **faiss-GPU** [1,6,7] build and query indexes orders of magnitude faster than
  CPU — decisive for *rebuild-often* blocking and for scoring millions of
  candidate similarities as batched matrix ops.
- **Featurization & comparison at scale**: **RAPIDS cuDF** [27] is a
  GPU DataFrame (pandas-like) with GPU string ops, joins, and text utilities
  (`nvtext`, including edit-distance kernels), reporting large speedups (100×+)
  over pandas on text-heavy pipelines. Blocking-key joins and vectorized string
  comparison move onto the GPU with minimal code change.

GPUs shine when the workload is **batchable and compute-bound** (dense
comparison, index build, embedding); they help less with irregular graph
clustering. Gate behind `equate[gpu]`; never a hard dependency.

### 6.5 Execution strategy decision

| Data scale | Recommended execution | `equate` extra |
|---|---|---|
| ≤ ~1 M pairs after blocking | pure Python / NumPy / SciPy (default) | core |
| ~1 M – tens of M records | **DuckDB** SQL, out-of-core, one machine | `equate[duckdb]` |
| PyData stack, > RAM, multi-core | **Dask** partitions | `equate[dask]` |
| 100 M+ records, cluster available | **Spark** (Splink-style SQL dispatch) | `equate[spark]` |
| GPU available, batch/index-heavy | **cuVS / faiss-GPU / cuDF** | `equate[gpu]` |

---

## 7. Benchmarks — choosing on evidence, not vibes

Two benchmark families answer two different questions, and conflating them is a
common mistake:

- **ANN-Benchmarks** (Aumüller, Bernhardsson & Faithfull) [28] evaluates **raw
  index algorithms/libraries** on the **recall@k vs QPS** Pareto frontier over
  standard datasets (SIFT1M, GloVe, DEEP1B), plus build time and memory. Use it to
  pick an *algorithm* (HNSW vs IVF-PQ vs ScaNN). It does **not** test ingestion,
  filtering, updates, or system stability.
- **VectorDBBench / VDBBench** (Zilliz) [29] evaluates **whole database
  products** — adding **filtered search** (int- and label-based predicates at
  varying selectivity), **streaming ingestion recall**, **concurrent load**,
  **resource consumption**, and **cost**. Use it to pick a *product*.

The universal caveat: benchmarks use synthetic/standard embeddings and workloads;
**the recall/latency you get depends on your vectors, your filters, and your
update pattern**, so any framework should let users *measure* pair completeness /
reduction ratio / pairs quality on *their* data
([`02`](02-blocking-and-scalable-candidate-generation.md) §2) rather than trust a
leaderboard. The recall–QPS knee (efSearch/nprobe/ef) is per-dataset.

---

## 8. Wrap vs embed — the verdict for `equate`'s blocking layer

**Verdict: WRAP, do not embed. `equate` should define a small storage/index
interface and ship thin adapters to backends behind optional extras; it should
implement *no* production index, database, or execution engine itself.** The
reasoning, drawn from the whole corpus:

1. **The algorithms are commodities and already optimal.** HNSW, IVF-PQ, DiskANN,
   CAGRA are implemented, tuned, and benchmarked to death in FAISS/cuVS and every
   VDBMS [1,5,6,7,28]. Reimplementing them would be strictly worse and a
   maintenance sink ([`10`](10-design-implications-for-equate.md) §7: "reimplement
   almost nothing computational").
2. **The operational surface is enormous and out of scope.** Persistence, filtered
   ANN, hybrid fusion, sharding, compaction, HA — these *are* the product. A
   matching *framework* should consume them, not build them.
3. **No single backend fits all users.** The right choice spans pgvector (no new
   infra), LanceDB (embedded at scale), Qdrant/Milvus (server), Pinecone (managed).
   Only an *interface* accommodates that spread — the exact lesson of Splink's
   multi-backend SQL design [21].
4. **Licensing.** Embedding would force a dependency choice; wrapping keeps
   `equate` permissive and keeps AGPL/proprietary engines (Zingg, Pinecone) at
   arm's length ([`10`](10-design-implications-for-equate.md) §7).
5. **Progressive disclosure.** The default must be zero-dependency (in-memory
   brute force / stdlib blocking from
   [`02`](02-blocking-and-scalable-candidate-generation.md)); scale is an opt-in
   backend swap, never a required install (user's CLAUDE.md: "no heavy library is
   ever a hard dependency").

The one thing `equate` *does* own: the **orchestration and the interface** — the
`dol`-style storage/index abstraction below, capability detection, and the glue
that turns "featurize → block → compare → match" into backend calls.

---

## 9. Design implications for `equate`

### 9.1 A `dol`-style vector store: index-as-Mapping

Model the vector/signature store as a `dol`
`MutableMapping[Key, Vector]` **plus** a search method, so that persistence is a
storage concern (swap in-memory ↔ LanceDB ↔ Qdrant by swapping the mapping) and
*scale is delegated to the backend* (aligns with the user's `python-storage`
skill and the `dol` facade philosophy). Two orthogonal roles:

```python
from typing import Protocol, Iterable, Mapping, Sequence
from collections.abc import MutableMapping

Key = str            # object id
Vector = Sequence[float]
Filter = Mapping     # backend-native metadata predicate (structured blocking key)

class VectorStore(MutableMapping[Key, Vector], Protocol):
    """CRUD over vectors keyed by object id — a dol store. Persistence,
    on-disk/in-memory, local/remote are all *backend* concerns behind this face.
    `store[key] = vector` upserts; `del store[key]` deletes; iteration lists ids.
    Optional parallel stores hold metadata and sparse/text signatures."""

class VectorIndex(Protocol):
    """The retrieval role layered over a VectorStore."""
    def search(
        self, query: Vector, k: int, *,
        filter: Filter | None = None,          # §5.1 filtered ANN = structured block
        sparse: Vector | None = None,          # §5.2 hybrid: sparse side
        fusion: str = 'rrf',                   # 'rrf' | 'dbsf' | 'alpha'
    ) -> list[tuple[Key, float]]: ...
    def build(self, *, metric: str = 'cosine', **index_params) -> None: ...
```

Every backend in §§2–4 satisfies this: an in-memory dict + FAISS/hnswlib
(default), LanceDB (embedded on-disk), pgvector (SQL), Qdrant/Milvus/Weaviate/
Pinecone/Vespa (service). Because `VectorStore` *is* a `MutableMapping`, users get
`dol` niceties (caching via `cache_this`, wrapping, key transforms) for free, and
`equate` never sees a backend-specific type
([`10`](10-design-implications-for-equate.md) §6: "adapters so no library type
leaks").

### 9.2 The `Blocker` is *backed by* an index, and delegates scale

`equate`'s `Blocker` protocol from
[`10`](10-design-implications-for-equate.md) §3 —
`candidate_pairs(A, B) -> Iterable[(i, j)]` — gets one concrete `IndexBlocker`
implementation that featurizes both sides, upserts into a `VectorStore`, and
emits pairs from `VectorIndex.search`. **Scale then lives entirely in the injected
backend**: swap the in-memory store for LanceDB and the same blocker handles
100 M records on disk; swap for Qdrant and it runs as a service. The blocker code
does not change — only the injected `VectorStore`/`VectorIndex`.

### 9.3 Filtered + hybrid retrieval, surfaced through the blocker

Expose §5 as first-class blocker options, not backend trivia:

- A **structured blocking key** (from
  [`02`](02-blocking-and-scalable-candidate-generation.md) §3) compiles to the
  backend's `filter` predicate → *filtered ANN* runs "block + retrieve" in one
  call (§5.1). A pure-Python fallback filters candidates in the process when the
  backend has no native filtering.
- **sparse ∪ dense** blocking is a `hybrid=True` flag that passes both a sparse and
  a dense query and fuses by RRF (§5.2) — realizing
  [`02`](02-blocking-and-scalable-candidate-generation.md) §8's "run and union
  both" as a single backend feature where available, and as an in-process
  rank-fusion of two separate index queries where not.

### 9.4 An execution backend seam (separate from the index seam)

Distinct from *where vectors live* is *what executes the comparison/clustering
workload*. Define an `ExecutionBackend` (or reuse a dataframe abstraction) with a
default that is plain Python/NumPy and optional dispatch to DuckDB SQL, Dask
partitions, or Spark — mirroring **Splink's compile-to-SQL, dispatch-to-backend**
design [21] (§6). The matcher expresses blocking-join + pairwise-compare + connected-
components abstractly; the backend runs it in-process, out-of-core, or
distributed. GPU (cuVS/cuDF/faiss-GPU) is a further backend for the batchable
stages (§6.4).

### 9.5 Optional-dependency boundaries (progressive disclosure)

Keep the core dependency-free (in-memory brute-force index + stdlib blocking).
Gate everything else behind extras with **lazy imports** and
`check_requirements`-style errors naming the extra and install command:

| Extra | Wraps | Role |
|---|---|---|
| `equate[ann]` | faiss / hnswlib | in-process ANN index (default scale step) |
| `equate[lancedb]` | lancedb | embedded on-disk vector store at scale |
| `equate[pgvector]` | psycopg + pgvector | vectors inside Postgres |
| `equate[qdrant]` / `[milvus]` / `[weaviate]` | client SDKs | server backends |
| `equate[pinecone]` | pinecone client | managed serverless |
| `equate[duckdb]` | duckdb | out-of-core single-node execution |
| `equate[dask]` | dask | out-of-core / distributed PyData execution |
| `equate[spark]` | pyspark | distributed execution (Splink-style) |
| `equate[gpu]` | cuvs / cudf / faiss-gpu | GPU index + featurization/compare |

Never import AGPL/proprietary engines (Zingg, Pinecone server) — shell out or use
their client SDK only. Copyleft engines stay strictly out-of-process
([`10`](10-design-implications-for-equate.md) §7).

### 9.6 Capability detection + measured selection

Auto-select the **fastest installed** backend (in-memory FAISS if present,
LanceDB if a path/S3 URI is given, a running Qdrant/Milvus if configured) and emit
actionable install hints otherwise — the registry/capability pattern of
[`10`](10-design-implications-for-equate.md) §6. Pair it with the block-quality
metrics (PC/RR/PQ) and an ANN recall/QPS probe (§7) so backend/param choice is an
*empirical* decision on the user's own data, not a guess. Expose one normalized
`recall_target` dial mapped to each backend's native knob (efSearch, nprobe,
lists/probes) — the open question flagged in
[`02`](02-blocking-and-scalable-candidate-generation.md) and
[`10`](10-design-implications-for-equate.md) §9.

---

## Glossary

- **Vector database (VDBMS)** — a persistent service/library that stores vectors
  with metadata and serves ANN queries, adding CRUD, filtering, hybrid search,
  replication, and an API on top of an ANN index.
- **pgvector** — Postgres extension adding a `vector` type + HNSW/IVFFlat indexes;
  keeps vectors in the transactional DB. **pgvectorscale** adds on-disk
  `StreamingDiskANN`.
- **Qdrant** — Rust vector DB known for filterable-HNSW (metadata filtering fused
  into graph traversal) and server-side RRF/DBSF hybrid fusion.
- **Milvus** — distributed, cloud-native vector DB (storage/compute separated,
  segment-based) for billion-scale; widest index menu incl. GPU (cuVS).
- **Weaviate** — HNSW vector DB with `alpha`-weighted BM25+dense hybrid search.
- **LanceDB** — embedded, on-disk, data-lake-native vector store on the **Lance**
  columnar (Arrow) format; billion-scale at low cost, no server.
- **Chroma** — minimal, mostly in-memory embedded vector DB; prototype scale.
- **Pinecone** — managed serverless VDBMS; slabs/memtable over blob storage.
- **Vespa** — big-data serving engine unifying BM25, tensors, and HNSW ANN with a
  multi-phase ranking DSL.
- **Hybrid search** — combining sparse (BM25/keyword) and dense (embedding)
  retrieval, fused by rank (**RRF**) or score (**DBSF**, `alpha`).
- **HNSW** — hierarchical navigable small-world graph ANN index (in-memory).
- **DiskANN / Vamana** — on-disk graph ANN; billion points on one SSD box.
- **CAGRA** — GPU-native graph ANN index (in cuVS/FAISS).
- **Filtered ANN** — nearest-neighbor search constrained by a metadata predicate;
  productized "block + retrieve."
- **Quantization** — scalar/product/binary/half compression trading recall for
  RAM and speed; the coarse stage before full-precision re-ranking.
- **Spark** — distributed dataframe engine; ER blocking = shuffle-join on keys.
- **Dask** — Python-native parallel/out-of-core dataframe & array engine.
- **DuckDB** — embedded, in-process, vectorized, out-of-core OLAP engine;
  Splink's default backend.
- **GPU blocking** — running index build/query and featurization/comparison on
  GPUs via cuVS, faiss-GPU, and RAPIDS **cuDF**.
- **RAPIDS** — NVIDIA's GPU data-science stack (cuDF dataframes, cuVS vector
  search).
- **Out-of-core** — processing data larger than RAM by streaming/spilling to disk.

---

## References

[1] Douze M, Guzhva A, Deng C, Johnson J, et al. *The Faiss Library.*
arXiv:2401.08281, 2024. [https://arxiv.org/abs/2401.08281](https://arxiv.org/abs/2401.08281)
· code [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

[2] Papadakis G, Skoutas D, Thanos E, Palpanas T. *Blocking and Filtering
Techniques for Entity Resolution: A Survey.* ACM Computing Surveys 53(2), 2020.
[https://arxiv.org/abs/1905.06167](https://arxiv.org/abs/1905.06167)

[3] Zilliz. *VectorDBBench: An Open-Source VectorDB Benchmark Tool / VDBBench 1.0.*
[https://zilliz.com/vdbbench-leaderboard](https://zilliz.com/vdbbench-leaderboard)

[4] Firecrawl. *Best Vector Databases: A Complete Comparison Guide.*
[https://www.firecrawl.dev/blog/best-vector-databases](https://www.firecrawl.dev/blog/best-vector-databases)

[5] Subramanya SJ, Devvrit, Kadekodi R, Krishnaswamy R, Simhadri HV. *DiskANN:
Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS
2019.
[https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/)
· pdf [https://suhasjs.github.io/files/diskann_neurips19.pdf](https://suhasjs.github.io/files/diskann_neurips19.pdf)

[6] Ootomo H, Naruse A, Nolet C, Wang R, Feher T, Wang Y. *CAGRA: Highly Parallel
Graph Construction and Approximate Nearest Neighbor Search for GPUs.*
arXiv:2308.15136, 2023.
[https://arxiv.org/abs/2308.15136](https://arxiv.org/abs/2308.15136)

[7] Facebook Research. *GPU Faiss with cuVS* (wiki) and NVIDIA **cuVS**.
[https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS](https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS)
· [https://github.com/rapidsai/cuvs](https://github.com/rapidsai/cuvs)
· [https://rapids.ai/cuvs/](https://rapids.ai/cuvs/)

[8] Milvus / Zilliz. *What is Milvus* (docs overview) and *Milvus on GPU with
NVIDIA RAPIDS cuVS.*
[https://milvus.io/docs/overview.md](https://milvus.io/docs/overview.md)
· [https://zilliz.com/blog/milvus-on-gpu-with-nvidia-rapids-cuvs](https://zilliz.com/blog/milvus-on-gpu-with-nvidia-rapids-cuvs)

[9] pgvector. *Open-source vector similarity search for Postgres.*
[https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)

[10] pgvector issue #799 / Nile. *Why the indexing dimension limit is 2000 (8 KB
page limit); use halfvec for higher dims.*
[https://github.com/pgvector/pgvector/issues/799](https://github.com/pgvector/pgvector/issues/799)
· [https://www.thenile.dev/blog/pgvector_myth_debunking](https://www.thenile.dev/blog/pgvector_myth_debunking)

[11] Qdrant. *Hybrid Search Revamped — Building with Qdrant's Query API* (prefetch,
RRF/DBSF, filterable HNSW, ColBERT re-ranking).
[https://qdrant.tech/articles/hybrid-search/](https://qdrant.tech/articles/hybrid-search/)

[12] Weaviate. *Hybrid Search Explained* (alpha, RRF, relative-score fusion).
[https://weaviate.io/blog/hybrid-search-explained](https://weaviate.io/blog/hybrid-search-explained)

[13] Cormack GV, Clarke CLA, Buettcher S. *Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods.* SIGIR 2009 (RRF, k≈60);
overview at
[https://weaviate.io/blog/hybrid-search-explained](https://weaviate.io/blog/hybrid-search-explained)

[14] LanceDB. *Developer-friendly embedded multimodal retrieval; Lance columnar
format.*
[https://github.com/lancedb/lancedb](https://github.com/lancedb/lancedb)
· [https://www.lancedb.com/](https://www.lancedb.com/)
· format [https://docs.lancedb.com/lance](https://docs.lancedb.com/lance)

[15] NVIDIA. *cuVS — GPU vector search and clustering (IVF-Flat, IVF-PQ, CAGRA).*
[https://rapids.ai/cuvs/](https://rapids.ai/cuvs/)

[16] LanceDB. *Lance v2: A New Columnar Container Format.*
[https://blog.lancedb.com/lance-v2/](https://blog.lancedb.com/lance-v2/)

[17] Timescale. *pgvectorscale — StreamingDiskANN + Statistical Binary
Quantization + label-based filtered search for pgvector.*
[https://github.com/timescale/pgvectorscale](https://github.com/timescale/pgvectorscale)

[18] Pinecone. *Reimagining the vector database to enable knowledgeable AI*
(serverless: slabs, memtable, blob storage) and *Serverless Architecture* docs.
[https://www.pinecone.io/blog/serverless-architecture/](https://www.pinecone.io/blog/serverless-architecture/)
· [https://docs.pinecone.io/reference/architecture/serverless-architecture](https://docs.pinecone.io/reference/architecture/serverless-architecture)

[19] Vespa. *Hybrid Text Search Tutorial* and *Multi-Vector HNSW Indexing.*
[https://docs.vespa.ai/en/learn/tutorials/hybrid-search.html](https://docs.vespa.ai/en/learn/tutorials/hybrid-search.html)
· [https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/)

[20] DuckDB. *An in-process, vectorized, larger-than-memory analytical (OLAP)
database.* [https://duckdb.org/](https://duckdb.org/)

[21] Splink (MoJ Analytical Services). *Fast, accurate, scalable probabilistic
data linkage with multiple SQL backends* — Backends overview.
[https://github.com/moj-analytical-services/splink](https://github.com/moj-analytical-services/splink)
· [https://moj-analytical-services.github.io/splink/topic_guides/splink_fundamentals/backends/backends.html](https://moj-analytical-services.github.io/splink/topic_guides/splink_fundamentals/backends/backends.html)

[22] Linacre R. *Fuzzy Matching and Deduplicating Hundreds of Millions of Records
with Splink.*
[https://www.robinlinacre.com/introducing_splink/](https://www.robinlinacre.com/introducing_splink/)

[23] Linacre R. *Deduplicating 7 Million Records in Two Minutes with Splink.*
[https://medium.com/data-science-collective/deduplicating-7-million-records-in-two-minutes-with-splink-4b1a87035a85](https://medium.com/data-science-collective/deduplicating-7-million-records-in-two-minutes-with-splink-4b1a87035a85)

[24] Dedupe.io. *A Python library for fuzzy matching, deduplication, and
entity resolution* (PostgreSQL / big-data / Spark examples).
[https://github.com/dedupeio/dedupe](https://github.com/dedupeio/dedupe)

[25] Zingg AI. *Scalable entity resolution / data mastering on Spark* (AGPL-3.0;
learns a blocking model, ~0.05–1% of pair space).
[https://github.com/zinggAI/zingg](https://github.com/zinggAI/zingg)

[26] Dask. *Parallel and out-of-core computing for the PyData stack.*
[https://www.dask.org/](https://www.dask.org/)

[27] RAPIDS cuDF (NVIDIA). *GPU DataFrame with GPU string/text ops (`nvtext`,
edit distance).*
[https://github.com/rapidsai/cudf](https://github.com/rapidsai/cudf)
· [https://developer.nvidia.com/blog/nlp-and-text-precessing-with-rapids-now-simpler-and-faster/](https://developer.nvidia.com/blog/nlp-and-text-precessing-with-rapids-now-simpler-and-faster/)

[28] Aumüller M, Bernhardsson E, Faithfull A. *ANN-Benchmarks: A Benchmarking Tool
for Approximate Nearest Neighbor Algorithms.* Information Systems 87, 2020
(arXiv:1807.05614). [https://arxiv.org/abs/1807.05614](https://arxiv.org/abs/1807.05614)
· [https://ann-benchmarks.com/](https://ann-benchmarks.com/)

[29] Zilliz. *VectorDBBench — benchmark for vector databases (filtering, ingestion,
cost, stability).*
[https://github.com/zilliztech/VectorDBBench](https://github.com/zilliztech/VectorDBBench)
