# Schema, Ontology & Join-Key Matching: The Tabular-Completion Angle

**Abstract.** This document surveys *schema-level* matching — establishing correspondences between the **attributes/columns** of data sources rather than between individual instances — and situates it against `equate`'s goal of automatic table completion (detecting join keys, finding near-duplicate columns, and aligning/completing tables from alternate sources). It covers the classical schema-matching taxonomy (Rahm & Bernstein) and canonical algorithms (Cupid, Similarity Flooding, COMA), the modern data-lake discovery stack (LSH Ensemble, JOSIE, Aurum, Table Union Search, D3L, Starmie, SANTOS), instance-based column matching via value-set and distribution similarity, and the Valentine benchmark that unifies their evaluation. The organizing lens throughout is `equate`'s **featurize → compare → match** decomposition: treat *columns as the objects*, *value-sets / distributions / embeddings as the features*, a similarity function as the comparator, and an assignment solver as the matcher.

---

## 1. Why schema-level matching is the right altitude for `equate.completion`

`equate` already matches *instances* (strings, filenames) by a three-stage pipeline: `similarity_matrix` (a `featurize` step, `text_to_vect`, composed with a `compare` step, `similarity_func`) followed by a `matcher` (Hungarian, greedy, stable-marriage, etc.). The `completion.py` docstring names four tabular tasks — (1) detect join keys, (2) compare columns with flexible (non-equality) similarity, (3) find near-duplicate columns, (4) align rows via fuzzy cell matching. Tasks (1)–(3) are **not** instance matching; they are *schema matching*: the objects being matched are **columns**, and the features are **summaries of each column's values** (its value-set, its value distribution, its header, or a learned embedding). This is precisely the sub-field surveyed here, and it maps cleanly onto `equate`'s existing decomposition — the only thing that changes is *what an "object" is* and *how it is featurized*.

Two distinct communities converge on this problem, with different vocabulary (see the glossary, §2):

- **Schema matching / data integration** (databases, since ~2001): given two schemas *S* and *T*, output a set of correspondences between their elements. The output is an *alignment* (a match on a bipartite graph of attributes) — the same object `equate`'s matchers already produce.
- **Data discovery / data lakes** (since ~2016): given a query table and a *corpus* of thousands-to-millions of tables, retrieve tables that are **joinable** (share values in some column → horizontal extension) or **unionable** (share a domain/type → vertical extension). This is *search*, not pairwise matching, so it adds an **indexing** concern (LSH, inverted indexes, HNSW) that pairwise matching does not have.

For `equate`, the first framing supplies the *comparison and assignment* logic; the second supplies the *scalability* story (how to avoid an O(n·m) all-pairs similarity matrix when there are many columns/tables).

---

## 2. Glossary of canonical terminology (with cross-community synonyms)

- **Schema matching**: producing correspondences between the *elements* (attributes/columns, or nested XML/JSON nodes) of two schemas. Output = a set of matched element pairs, optionally with confidence scores. [1]
- **Attribute matching / column matching**: schema matching specialized to the flat relational case (columns of one table ↔ columns of another). The dominant framing in tabular contexts and in `equate`.
- **Schema mapping** (distinct from *matching*): the executable transformation (e.g., a SQL/relational-algebra expression) that moves data from source to target *given* a match. Matching is discovery; mapping is the query. `equate.completion` needs matching first.
- **Element-level vs structure-level** matchers [1]: element-level considers each schema element in isolation (name, data type, value-set); structure-level exploits the *graph/tree* of relationships (a column matches better if its siblings/parents match). Cupid and Similarity Flooding are structure-level; a Jaccard-on-values matcher is element-level.
- **Schema-level vs instance-level** matchers [1]: schema-level uses only metadata (names, types, constraints); instance-level ("value-based") inspects the actual data values. `equate`'s completion tasks are naturally *instance-level* because headers are often missing or unreliable.
- **Linguistic vs constraint-based** matchers [1]: linguistic = name/description similarity (tokenization, synonyms, edit distance, embeddings); constraint-based = keys, data types, value ranges, cardinalities.
- **Individual vs combining** matchers [1]: an individual matcher yields one similarity signal; a *hybrid* matcher hard-codes several signals into one algorithm; a *composite* matcher (COMA) runs independent matchers and *combines* their matrices via aggregation + selection strategies.
- **Join-key discovery / joinability**: finding a column *c* in a candidate table whose values *overlap* the values of a query column, so the tables can be equi-joined. Measured by **set overlap** / **set containment**. [8][9]
- **Table union search (TUS) / unionability**: finding tables whose columns are drawn from the *same domain/type* as the query's, so rows can be vertically concatenated. Measured by domain/distribution/semantic similarity, not raw value overlap. [10][11][12]
- **Near-duplicate columns**: two columns (possibly in the same or different tables) whose value-sets or distributions are near-identical; a special high-similarity case of column matching, useful for dedup and for detecting redundant join keys.
- **Set containment** vs **Jaccard similarity**: containment `|Q ∩ X| / |Q|` is *asymmetric* and robust when |Q| ≪ |X|; Jaccard `|Q ∩ X| / |Q ∪ X|` is symmetric but degrades badly when domain sizes differ — the key reason LSH Ensemble prefers containment for web-scale join search. [8]
- **Data discovery**: the umbrella task of finding relevant tables/columns in a large corpus (data lake / enterprise) for a downstream task; joinability and unionability are its two workhorses. [15][16]
- **Ontology matching / alignment**: the semantic-web analogue of schema matching, over richer graphs (classes, properties, individuals, subsumption). Benchmarked by the OAEI. [13][14]

**Synonym map (flag for the synthesis step):** *schema matching* ≈ *attribute/column matching* ≈ *element correspondence discovery*; *alignment* (ontology community) ≈ *match result* (DB community); *joinability* ≈ *set-overlap search*; *unionability* ≈ *domain/type compatibility*; *matcher* is overloaded — in Rahm–Bernstein it is a *similarity producer*, whereas in `equate` a *matcher* is the *assignment solver*. The synthesis doc should standardize on **featurizer / comparator / assigner** to disambiguate.

---

## 3. The classical schema-matching taxonomy (Rahm & Bernstein, 2001)

The foundational reference is Rahm & Bernstein, *"A survey of approaches to automatic schema matching"*, VLDB Journal 10(4):334–350, 2001 [1]. It remains the standard taxonomy and its axes (schema/instance × element/structure × linguistic/constraint × individual/combining) are still used to *categorize* the modern methods below. Its central design lesson — **no single matcher is robust; combine several** — directly motivates a strategy-pattern architecture. The 2011 retrospective *"Generic Schema Matching, Ten Years Later"* [6] reaffirms that composite, reusable matcher frameworks (COMA-style) were the most durable contribution.

### 3.1 Cupid (Madhavan, Bernstein & Rahm, VLDB 2001) [2]

A **hybrid, structure-level** matcher. Two phases: (i) *linguistic matching* of element names (tokenization, abbreviation/synonym normalization, data-type compatibility) yielding a `lsim` matrix; (ii) *structural matching* over the schema tree via a bottom-up/top-down `ssim` propagation with a deliberate **bias toward leaf nodes** (where most schema content lives) — two non-leaves are similar if their leaf sets are similar. Final `wsim = w·lsim + (1−w)·ssim`, thresholded. Strength: handles nested/hierarchical schemas and shared types. Weakness: needs reasonable names and a real tree structure — degrades on flat, unlabeled columns (the common `equate` case).

### 3.2 Similarity Flooding (Melnik, Garcia-Molina & Rahm, ICDE 2002) [3]

A **generic, structure-level** graph-matching algorithm based on a **fixpoint iteration**. Both schemas are converted to directed labeled graphs; a *pairwise connectivity graph* (PCG) is built over candidate node-pairs, and an *induced propagation graph* assigns propagation coefficients. Initial similarities (e.g., string similarity of names) are then **"flooded"**: `σ^{k+1}(a,b) = σ^k(a,b) + Σ σ^k(neighbors) · propagation_weight`, normalized each round until convergence. Intuition: *nodes are similar when their neighbors are similar.* A filter then extracts a 1:1 (or constrained) match — often via a stable-marriage / assignment step. Complexity: each iteration is linear in the PCG edge count; convergence is typically fast. Strength: fully generic, works on any graph (schemas, taxonomies, even provenance). Weakness: needs a meaningful graph topology and a decent initial map; pure flat columns give it little to propagate over.

### 3.3 COMA / COMA++ (Do & Rahm, VLDB 2002; Aumüller et al., SIGMOD 2005) [4][5]

The canonical **composite** matcher and the most architecturally influential for a *framework* like `equate`. COMA provides an **extensible library of individual matchers** plus a **combination framework**: run *k* matchers → get *k* similarity cubes → **aggregate** (max / average / weighted / min) → **select** (threshold, top-*k*, max-delta) → optionally iterate. It also supports **match reuse** (remember prior alignments) and, in COMA++, a *generic* internal graph model that uniformly handles **relational schemas, XML Schema, and OWL ontologies**. The key takeaway for `equate`: *the winning design is not a better single similarity — it is a clean pipeline that composes many similarities and many selection strategies.* This is the same insight `equate`'s `similarity_matrix` + pluggable `matcher` split already embodies; COMA extends it with a **combination layer between featurize/compare and match**.

---

## 4. Instance-based column matching: value-sets and distributions

When headers are missing or untrustworthy (the norm for `equate.completion`), the column's *values* are the only signal. Three families, each a different **featurizer**:

- **Value-set / element-level.** Represent a column as the *set* of its distinct values; compare by **Jaccard**, **overlap coefficient**, **Tversky index**, or containment. Valentine's `JaccardDistanceMatcher` combines set similarity with a per-value string distance (Levenshtein / Jaro-Winkler) so that *near*-equal values still count as intersecting [7]. This is the direct generalization of `equate`'s existing edit-distance instance matcher up to the column level. Best for **joinability** and **near-duplicate** detection.
- **Distribution-based / statistical.** Represent a column by a *distribution* over its values (or over value features: length, character histograms, numeric quantiles) and compare distributions (e.g., Earth Mover's / KS / cosine over histograms). Valentine implements a **Distribution-Based** matcher (after Zhang et al.'s "automatic discovery of attributes in relational databases") that clusters columns by value-distribution similarity — robust to *disjoint but same-domain* columns (e.g., two customer tables with no overlapping IDs but identical value shape). Best for **unionability**.
- **Learned embeddings.** Represent a column by an aggregated *embedding* of its values/context. **EmbDI** (Cappuzzo et al., SIGMOD 2020) [17] builds a tripartite graph over tokens/attributes/rows, does random walks, and trains word2vec to get unsupervised relational embeddings usable for both schema matching and entity resolution. **SemProp / Seeping Semantics** (Fernandez et al., ICDE 2018) [18] uses word embeddings + "coherent groups" to link columns *semantically* (and to external ontologies) even without value overlap. **Starmie** (Fan et al., VLDB 2023) [12] trains a contextual column encoder by contrastive learning and uses cosine of column embeddings as the unionability score.

For `equate` this trio is important: it is exactly the same `featurize` slot with three swappable implementations, all feeding the *same* similarity-matrix-then-matcher back end.

---

## 5. Data-lake discovery: joinability and unionability at scale

Pairwise matching (an O(n·m) similarity matrix) is fine for two tables but not for a corpus. The discovery literature contributes **index structures** that turn "compare all pairs" into "search top-*k*". These are the scalability strategies `equate` should expose as *optional* back ends.

### 5.1 Joinable-table discovery (set overlap / containment)

- **LSH Ensemble** (Zhu, Nargesian, Pu & Miller, VLDB 2016) [8]. Formalizes **domain search** using **Jaccard set containment** `|Q ∩ X| / |Q|` as relevance. Uses **MinHash** sketches + **domain partitioning** to cope with extreme skew in value-set sizes; proves an optimal partitioning exists and that *equi-depth* partitioning approximates it for power-law corpora. Evaluated on **262M+ domains** (Canadian Open Data + WebTables). This is the go-to *approximate, sub-linear* join-key index. (Reference implementation: `datasketch`/`lshensemble`.)
- **JOSIE** (Zhu, Deng, Nargesian & Miller, SIGMOD 2019) [9]. **Exact** top-*k* **overlap set similarity search**. Models each column as a set and join candidacy as set intersection; a cost model minimizes the combined cost of *set reads* and *inverted-index probes* (prefix/position filtering). Scales to sets with tens of millions of elements and dictionaries of hundreds of millions of distinct values — where prior overlap-search methods (built for short strings) fail. Use when *exact* overlap ranking matters.

### 5.2 Unionable-table discovery (domain/type/semantic compatibility)

- **Table Union Search on Open Data** (Nargesian, Zhu, Pu & Miller, VLDB 2018) [10]. Defines **unionability** via an *ensemble of three domain-similarity tests* — **set-based** (value overlap), **semantic** (shared ontology classes), and **natural-language** (word-embedding similarity of values) — combined by a statistical model, indexed with LSH for top-*k* retrieval. The seminal TUS formulation.
- **SANTOS** (Khatiwada et al., SIGMOD / PACMMOD 2023) [11]. Adds **relationship semantics**: unionability should consider not just each column's type but the *binary relationships between column pairs* in a table. Uses either an external knowledge base or a **synthesized KB mined from the data lake itself**, improving precision on tables that share column types but mean different things.
- **Starmie** (Fan, Wang, Li, Zhang & Miller, VLDB 2023) [12]. State-of-the-art *learned* TUS: unsupervised **contrastive** multi-column pre-training of a column encoder; cosine of column embeddings as unionability; **HNSW** index for retrieval. Reports gains of ~**6.8 points in MAP/recall** over prior best, a **~400× speedup over LSH** and **~3000× over linear scan** for query processing.

### 5.3 Whole-corpus discovery systems (put it together)

- **Aurum** (Fernandez et al., ICDE 2018) [15]. Builds and maintains an **Enterprise Knowledge Graph (EKG)** whose **nodes are columns/attributes** and **edges are similarity relationships** (content overlap, PK–FK candidates, semantic links). A **one-pass, two-step** profiling pipeline uses **LSH + TF-IDF signatures** so it scales without hammering source systems; a query language (SRQL) then navigates the graph. Aurum is the reference architecture for "a graph of columns linked by fuzzy similarity" — a useful mental model for what `equate.completion` produces when run over many tables.
- **D3L** (Bogatu, Fernandes, Paton & Konstantinou, ICDE 2020) [16]. Multi-signal dataset discovery: builds **hash-based (LSH) indexes over five column features** — value overlap, header/attribute-name similarity, value *format/regex* similarity, numeric-distribution similarity, and word-embedding similarity — mapping all into a uniform distance space and combining them. Notable for treating *format* and *distribution* as first-class, embedding-free features (cheap and robust).

---

## 6. Ontology / entity alignment (the semantic-web sibling)

Ontology matching finds correspondences between semantically related entities (classes, properties, individuals) across ontologies, where correspondences can be equivalence *or* subsumption/disjointness — richer than DB schema matching. The standard monograph is Euzenat & Shvaiko, *Ontology Matching* (Springer, 2nd ed. 2013) [13]; the survey *"Ontology matching: state of the art and future challenges"* (Shvaiko & Euzenat, IEEE TKDE 2013) [14] catalogs element-level (string, language, linguistic-resource) and structure-level (graph, taxonomic, model-based) techniques — a taxonomy parallel to Rahm–Bernstein but for graphs. The **Ontology Alignment Evaluation Initiative (OAEI)**, running yearly since 2005, is the OAEI benchmark analogue of Valentine for this community. For `equate` the relevance is conceptual: (a) matching can return *typed* relations (not just 1:1 equivalence), and (b) *background knowledge* (a KB, WordNet, embeddings) is a legitimate, pluggable similarity source — the same idea SANTOS and SemProp import into tabular matching.

---

## 7. The Valentine benchmark (Koutras et al., ICDE 2021)

*"Valentine: Evaluating Matching Techniques for Dataset Discovery"*, ICDE 2021 [7], with the scalable demo *"Valentine in Action"*, VLDB 2021 [19], is the most comprehensive unified evaluation of **column/schema matching for tabular dataset discovery**, and the single most useful reference for `equate` because it (a) fixes a vocabulary of *matching problems*, (b) re-implements the classic methods behind one API (open-source: `github.com/delftdata/valentine`), and (c) reports which method wins where.

**Four dataset-relatedness scenarios it fabricates and evaluates** [7][19]:

1. **Unionable** — same schema/domain, disjoint rows (vertical concat).
2. **View-Unionable** — unionable only after a projection/transformation aligns the schemas.
3. **Joinable** — share values in a key column (horizontal join).
4. **Semantically-Joinable** — columns related by *meaning* rather than literal value overlap (needs semantic similarity).

**Methods benchmarked, mapped to the Rahm–Bernstein axes** [7]:

| Method | Category | Featurizer (in `equate` terms) |
|---|---|---|
| COMA (schema / instance / hybrid variants) | composite; schema + instance | names + TF-IDF over values, combined |
| Cupid | schema-level, structure | tokenized names + tree structure |
| Similarity Flooding | structure-level, graph | name-similarity flooded over schema graph |
| Distribution-Based | instance-level, statistical | value-distribution clustering |
| Jaccard-Levenshtein | instance-level, element | value-set overlap w/ fuzzy equality |
| EmbDI | instance-level, learned | relational graph embeddings |
| SemProp | instance-level, semantic | word embeddings + ontology links |

**Metrics**: Precision, Recall, F1, plus rank-aware variants (Precision@n%, Recall at ground-truth size) — because a matcher outputs a *ranked* set of correspondences, not a single answer. **Headline findings** [7][19]: no single matcher dominates across all four scenarios; **COMA's instance/hybrid variants are the most robust overall**; **instance-based** signals win on **joinable** problems while **schema/name-based** signals win on **unionable** ones; and *all* methods degrade sharply under noise, missing values, and header corruption. A 2025 critical re-evaluation [21] further warns that popular TUS benchmarks can be "gamed" by trivial features, underscoring that **evaluation methodology matters as much as the algorithm** — a caution `equate` should heed by shipping honest, reproducible example benchmarks.

**LLM-era note.** Recent work applies LLMs to schema matching — an experimental study at VLDB-TaDA 2024 [20] and *Magneto* (VLDB 2025) [22], which *combines* small embedding models (cheap candidate retrieval) with large LLMs (precise re-ranking). The consistent pattern: LLMs help most on the **semantic** cases (semantically-joinable, cryptic headers) but are unnecessary and costly for value-overlap joinability — reinforcing a *strategy* design where the expensive semantic matcher is one optional plug-in among cheaper ones.

---

## 8. Design implications for `equate`

The literature converges on an architecture strikingly close to `equate`'s existing `featurize → compare → match`, plus three additions: a **combination layer**, a **column-as-object abstraction**, and an **indexed candidate-generation** back end for scale. Concrete recommendations:

### 8.1 Make "the object" polymorphic: lift the pipeline from cells to columns

Introduce an explicit `Featurizer` protocol (a callable `objects -> feature array/objects`) so the *same* `similarity_matrix` + `matcher` machinery matches **columns** as readily as strings. `completion.py` should define column featurizers — `value_set`, `value_distribution`, `header_text`, `embedding` — that all return something the existing `similarity_func` can consume. This is the SSOT/strategy realization of "columns as objects, value distributions as features."

### 8.2 Standardize a three-slot strategy interface (Featurizer × Comparator × Assigner)

- **Featurizer** (a.k.a. `obj_to_vect`/`text_to_vect`): `value_set`, `value_distribution`, `format/regex profile`, `tfidf`, `embedding`. Default: cheap value-set + tfidf. Heavy learned encoders (Starmie/EmbDI-style) behind an **optional dependency**.
- **Comparator** (`similarity_func`): Jaccard, **overlap/containment** (asymmetric — must be first-class, per LSH Ensemble [8]), Tversky, cosine, distribution divergence. Default: cosine (as today), but expose containment for join-key work.
- **Assigner** (`matcher`): already implemented (Hungarian, greedy, stable-marriage, maximal, Kuhn–Munkres). Keep. Add a **threshold/top-k selector** for the *many-to-many* and *ranked* cases the discovery literature needs (matching is not always 1:1).

### 8.3 Add a COMA-style combination layer (the single most validated idea)

Between `compare` and `match`, insert an optional **ensemble** step: run several featurize+compare pairs → get several similarity matrices → **aggregate** (weighted mean / max / min) → **select**. Rahm–Bernstein [1], COMA [4], D3L [16], and Valentine [7] all conclude that *combining* signals beats any single one. Expose `aggregate` and `select` as injectable functions with sensible defaults — this is pure open/closed design and needs no heavy deps.

### 8.4 Provide task-shaped facades mirroring Valentine's four scenarios

Ship progressive-disclosure entry points that pre-wire the strategies:

- `find_join_keys(left, right)` → containment/overlap comparator + top-k selector (joinable).
- `find_near_duplicate_columns(table)` → high-threshold self-similarity (near-dup).
- `find_unionable_columns(left, right)` → distribution/embedding featurizer (unionable / semantically-joinable).
- `complete_table(target, sources)` → join-key discovery, then row alignment via the existing cell-level matcher.

Each is a thin wrapper over the same core — "simple things simple," while the raw strategy knobs stay reachable.

### 8.5 Optional indexed back ends for scale (strategy boundary = optional dependency)

The all-pairs `similarity_matrix` is the correct default for two tables. For *many* columns/tables, expose an alternative **candidate-generation** strategy that only fills sparse entries:

- **MinHash-LSH / LSH Ensemble** for approximate containment-based join-key search over large corpora [8] (via optional `datasketch`).
- **HNSW / ANN index** over column embeddings for unionability at scale [12] (optional `hnswlib`/`faiss`).
- **Inverted-index overlap search** (JOSIE-style) [9] when exact top-k overlap is required.

Keep these strictly behind `try/except ImportError` with `check_requirements`-style guidance, so base `equate` stays dependency-light and the heavy machinery is opt-in. The abstraction: a `CandidateGenerator` that yields `(i, j)` pairs to score, of which "score every pair" (dense) is the trivial default.

### 8.6 Semantic/LLM matchers as the most-optional plug-in

Per [20][22], expensive semantic matchers (embeddings, LLM re-rankers) should be the outermost optional layer — invoked only for the *semantically-joinable* / cryptic-header cases where value overlap fails. A `Comparator` that calls an embedding model or LLM slots into the exact same interface; it must never be a hard dependency.

### 8.7 Honesty in evaluation

Adopt Valentine's ranked metrics (Precision/Recall/F1 at ground-truth size) for `equate`'s own examples, and heed [21]: ship reproducible, non-trivial benchmark tables so reported quality is trustworthy rather than gamed by degenerate features.

**Summary abstraction map:**

| Literature concept | `equate` slot |
|---|---|
| element/instance matcher signal | `Featurizer` + `Comparator` pair |
| composite matcher (COMA) | ensemble `aggregate`+`select` layer |
| assignment / 1:1 filter | existing `matcher` (Assigner) |
| joinability (overlap/containment) | containment `Comparator` + top-k selector |
| unionability (domain/distribution) | distribution/embedding `Featurizer` |
| LSH / HNSW / inverted index | optional `CandidateGenerator` back end |
| EKG of columns (Aurum) | the graph `equate` outputs over many columns |

---

## References

[1] E. Rahm, P. A. Bernstein. *A survey of approaches to automatic schema matching.* The VLDB Journal 10(4):334–350, 2001. [link.springer.com](https://link.springer.com/article/10.1007/s007780100057)

[2] J. Madhavan, P. A. Bernstein, E. Rahm. *Generic Schema Matching with Cupid.* VLDB 2001. [microsoft.com (PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-58.pdf)

[3] S. Melnik, H. Garcia-Molina, E. Rahm. *Similarity Flooding: A Versatile Graph Matching Algorithm and its Application to Schema Matching.* ICDE 2002. [ilpubs.stanford.edu](http://ilpubs.stanford.edu:8090/730/)

[4] H. H. Do, E. Rahm. *COMA — A System for Flexible Combination of Schema Matching Approaches.* VLDB 2002. [dbs.uni-leipzig.de (PDF)](https://dbs.uni-leipzig.de/files/research/publications/2002-1/pdf/COMA.pdf)

[5] D. Aumüller, H. H. Do, S. Massmann, E. Rahm. *Schema and ontology matching with COMA++.* ACM SIGMOD 2005. [dl.acm.org](https://dl.acm.org/doi/10.1145/1066157.1066283)

[6] P. A. Bernstein, J. Madhavan, E. Rahm. *Generic Schema Matching, Ten Years Later.* PVLDB 4(11):695–701, 2011. [vldb.org (PDF)](https://vldb.org/pvldb/vol4/p695-bernstein_madhavan_rahm.pdf)

[7] C. Koutras et al. *Valentine: Evaluating Matching Techniques for Dataset Discovery.* IEEE ICDE 2021. [computer.org](https://www.computer.org/csdl/proceedings-article/icde/2021/918400a468/1uGXjgG4T1C) · code: [github.com/delftdata/valentine](https://github.com/delftdata/valentine)

[8] E. Zhu, F. Nargesian, K. Q. Pu, R. J. Miller. *LSH Ensemble: Internet-Scale Domain Search.* PVLDB 9(12), 2016. [arxiv.org/abs/1603.07410](https://arxiv.org/abs/1603.07410) · code: [github.com/ekzhu/lshensemble](https://github.com/ekzhu/lshensemble)

[9] E. Zhu, D. Deng, F. Nargesian, R. J. Miller. *JOSIE: Overlap Set Similarity Search for Finding Joinable Tables in Data Lakes.* ACM SIGMOD 2019. [dl.acm.org (PDF)](https://dl.acm.org/doi/pdf/10.1145/3299869.3300065) · code: [github.com/ekzhu/josie](https://github.com/ekzhu/josie)

[10] F. Nargesian, E. Zhu, K. Q. Pu, R. J. Miller. *Table Union Search on Open Data.* PVLDB 11(7):813–825, 2018. [dblp.org](https://dblp.org/rec/journals/pvldb/NargesianZPM18.html)

[11] A. Khatiwada, G. Fan, R. Shraga, Z. Chen, W. Gatterbauer, R. J. Miller, M. Riedewald. *SANTOS: Relationship-based Semantic Table Union Search.* ACM SIGMOD / PACMMOD 1(1), 2023. [dl.acm.org](https://dl.acm.org/doi/10.1145/3588689) · [arxiv.org/abs/2209.13589](https://arxiv.org/abs/2209.13589)

[12] G. Fan, J. Wang, Y. Li, D. Zhang, R. J. Miller. *Semantics-Aware Dataset Discovery from Data Lakes with Contextualized Column-Based Representation Learning (Starmie).* PVLDB, 2023. [dl.acm.org](https://dl.acm.org/doi/10.14778/3587136.3587146) · [arxiv.org/abs/2210.01922](https://arxiv.org/pdf/2210.01922)

[13] J. Euzenat, P. Shvaiko. *Ontology Matching* (2nd ed.). Springer, 2013. [books.google.com](https://books.google.com/books/about/Ontology_Matching.html?id=nzLABAAAQBAJ)

[14] P. Shvaiko, J. Euzenat. *Ontology Matching: State of the Art and Future Challenges.* IEEE TKDE 25(1):158–176, 2013. [semanticscholar.org](https://www.semanticscholar.org/paper/71219c274777ea42e79180d05a9a377690207e07)

[15] R. C. Fernandez, Z. Abedjan, F. Koko, G. Yuan, S. Madden, M. Stonebraker. *Aurum: A Data Discovery System.* IEEE ICDE 2018. [dspace.mit.edu (PDF)](https://dspace.mit.edu/bitstream/handle/1721.1/137860/icde18-aurum.pdf)

[16] A. Bogatu, A. A. A. Fernandes, N. W. Paton, N. Konstantinou. *Dataset Discovery in Data Lakes (D3L).* IEEE ICDE 2020, pp. 709–720. [dblp.org](https://dblp.org/rec/conf/icde/BogatuFP020.html)

[17] R. Cappuzzo, P. Papotti, S. Thirumuruganathan. *Creating Embeddings of Heterogeneous Relational Datasets for Data Integration Tasks (EmbDI).* ACM SIGMOD 2020, pp. 1335–1349. [dl.acm.org](https://dl.acm.org/doi/10.1145/3318464.3389742) · code: [github.com/rcap107/embdi](https://github.com/rcap107/embdi)

[18] R. C. Fernandez, E. Mansour, A. A. Qahtan, A. Elmagarmid, I. F. Ilyas, S. Madden, M. Ouzzani, M. Stonebraker, N. Tang. *Seeping Semantics: Linking Datasets Using Word Embeddings for Data Discovery (SemProp).* IEEE ICDE 2018, pp. 989–1000. [ieeexplore.ieee.org](https://ieeexplore.ieee.org/document/8509314/)

[19] C. Koutras et al. *Valentine in Action: Matching Tabular Data at Scale.* PVLDB 14(12):2871–2874, 2021. [vldb.org (PDF)](http://vldb.org/pvldb/vol14/p2871-koutras.pdf)

[20] M. Parciak et al. *Schema Matching with Large Language Models: an Experimental Study.* VLDB TaDA Workshop, 2024. [arxiv.org/abs/2407.11852](https://arxiv.org/html/2407.11852v1)

[21] *Something's Fishy in the Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks.* 2025. [arxiv.org/abs/2505.21329](https://arxiv.org/pdf/2505.21329)

[22] Y. Liu et al. *Magneto: Combining Small and Large Language Models for Schema Matching.* PVLDB, 2025. [dl.acm.org](https://dl.acm.org/doi/10.14778/3742728.3742757) · [arxiv.org/abs/2412.08194](https://arxiv.org/pdf/2412.08194)

[23] G. Fan, R. Shraga, R. J. Miller. *Table Discovery in Data Lakes: State-of-the-art and Future Directions.* SIGMOD Companion 2023 (tutorial/survey). [dl.acm.org](https://dl.acm.org/doi/abs/10.1145/3555041.3589409) · [tutorial site](https://northeastern-datalab.github.io/table-discovery-tutorial/)
