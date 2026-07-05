# Featurization & Representation of Objects for Matching

*Research note for the `equate` redesign — part of the `docs/research/` corpus.*

## Abstract

Matching (fuzzy correspondence between collections of objects, as opposed to exact equality) is fundamentally a two-stage problem: first turn heterogeneous objects into *comparable representations*, then apply a *comparison operator* (a distance, similarity, or predicate) over those representations. This document surveys the representation stage. It maps the design space from the trivial *identity representation* through *fixed-length feature vectors* (embeddings) to *structured/schemaless* representations, catalogs the concrete featurizers per modality (text: TF-IDF, word2vec/fastText, SBERT, API embeddings; images: CLIP, CNN/ResNet features, perceptual hashes; audio: wav2vec 2.0, CLAP, Chromaprint), pins each representation to the metrics it admits (cosine, Euclidean, Jaccard, Hamming, edit-distance families) and to the blocking/ANN strategies it enables, and closes with concrete abstraction and optional-dependency boundaries for a matching *framework*. The unifying lens is the Python "key function" pattern — the observation that `key: Object -> Comparable` already generalizes `sorted`, `groupby`, and (extended to vectors + a metric) similarity matching.

---

## 1. The core abstraction: representation precedes comparison

Every matching pipeline factors into a **representation function** `repr: Object -> R` and a **comparison** over `R`. The nature of `R` determines everything downstream — which metrics are legal, whether you can index for sub-quadratic search, and whether comparison is symmetric or a learned function. The current `equate` code already exposes this seam explicitly: `similarity_matrix(keys, values, *, obj_to_vect=None, similarity_func=cosine_similarity)` separates the `obj_to_vect` featurizer from the `similarity_func` metric [see `equate/util.py`]. The redesign should generalize and harden exactly this seam.

There is a spectrum of representations, trading off expressivity, comparability, and cost:

| Representation `R` | Example | Comparison it admits | Indexable? |
|---|---|---|---|
| **Identity** | the object itself | `==`, `<`, `>`, hashing | hash table (exact) |
| **Scalar / tuple key** | `str.lower`, `(last, first)` | `==`, ordering, edit distance | sort / trie |
| **Set / bag** | token set, n-gram shingles | Jaccard, overlap, Dice | MinHash + LSH |
| **Fixed-length vector** | TF-IDF, SBERT, CLIP | cosine, Euclidean, dot | ANN (HNSW, IVF, ScaNN) |
| **Bit-string** | perceptual hash | Hamming | multi-index hashing |
| **Structured / nested** | JSON record, graph, tree | learned/aggregated, tree/graph edit | blocking + per-field |

### 1.1 The identity representation

The degenerate case `repr = identity` recovers exact equality — the object *is* its own key, compared with `==` and stored in a hash set. This is the boundary condition a matching framework must contain as a special case, not treat as a separate mechanism: exact join is fuzzy matching with a discrete metric (`0` if equal, `∞` otherwise). Keeping it inside the same abstraction lets a user start with an exact key and *relax* it (lowercase, then token-set, then embedding) without changing the calling code.

### 1.2 The Python "key function" pattern (the native precedent)

Python already ships a widely-understood contract for "map an object to something comparable": the **key function**. `sorted`, `min`, `max`, `heapq.nlargest`, and `itertools.groupby` all accept `key: Object -> Comparable`, where the returned value must support `<`/`==` [1,2]. `groupby` is precisely *exact matching by a key*: it clusters adjacent items whose keys are equal. The conceptual move behind a matching framework is to **widen the codomain of the key function** from "something orderable" to "something a metric can compare", and to widen the operation from "equality grouping" to "nearest-neighbor / assignment". This gives a mental model users already have:

- `key = str.lower` + `==` → case-insensitive exact match (native `groupby`).
- `key = tfidf_vectorize` + `cosine` → fuzzy text match (`equate.similarity_matrix`).
- `key = clip_encode` + `cosine` → cross-modal image↔text match.

Designing `equate`'s featurizer interface to *look like a key function* (a plain callable, composable, defaulting sensibly) is the single most important ergonomic decision, because it makes "simple things simple" (a lambda) while leaving "complex things possible" (a heavy embedding strategy).

---

## 2. Fixed-length vectors: the embedding view

An **embedding** is a learned map `Object -> R^d` such that geometric proximity in `R^d` approximates semantic similarity. Embeddings are the dominant representation because a single, uniform `d`-vector unlocks a huge, modality-agnostic toolchain: linear algebra, cosine/Euclidean metrics, ANN indexes, clustering, and cross-modal alignment. The price is that structure is *flattened* — a record's field boundaries, a document's paragraph structure, or an image's spatial layout are compressed into opaque coordinates.

Two properties matter for a matching framework:

- **Normalization.** Most modern sentence/image encoders are trained (or recommended for use) with **L2-normalized** vectors, which makes cosine similarity, dot product, and Euclidean distance rank-equivalent [3]. Normalization is a per-representation concern, not a global one — perceptual hashes and TF-IDF have different conventions — so it belongs *with* the featurizer, not in the matcher.
- **Dimensionality is tunable.** **Matryoshka Representation Learning (MRL)** (Kusupati et al., NeurIPS 2022 [4]) trains nested coarse-to-fine prefixes into one vector so that truncating to the first *k* coordinates yields a usable lower-dimensional embedding with no retraining. OpenAI's `text-embedding-3-large` (3072-d) exposes this via a `dimensions` parameter, and can be shortened to 256-d while still beating the older 1536-d `ada-002` on MTEB [5]. Implication: dimension is a *knob*, and a framework should let a strategy declare "I can be truncated to `d`" for cheaper storage/ANN.

---

## 3. Text featurization

### 3.1 Sparse lexical: TF-IDF and n-gram shingles

**TF-IDF** weights each term by its in-document frequency times the inverse of how many documents contain it, producing a high-dimensional **sparse** vector (dimension = vocabulary size) [6]. It is the current default in `equate` (via `grub.SearchStore.tfidf`). Properties:

- **Cheap, transparent, no training data, no GPU.** Excellent for near-duplicate detection, code/SKU/title matching, and any domain where *lexical overlap is the signal*.
- **No semantics**: "car" and "automobile" are orthogonal. Character n-gram TF-IDF (e.g., 3–5 grams) partially fixes typo/morphology robustness and is a strong, dependency-light baseline for entity names.
- Compared with **cosine similarity** (angle between sparse vectors), which is the standard pairing.

### 3.2 Static word embeddings: word2vec, GloVe, fastText

**word2vec** (Mikolov et al., 2013 [7]) and **GloVe** learn a fixed vector per word from co-occurrence. **fastText** (Bojanowski et al., TACL 2017 [8]) augments this with **subword (character n-gram) information**, so it produces vectors for out-of-vocabulary and misspelled words — valuable for noisy entity data. A sentence/record is embedded by *averaging* word vectors (a "bag-of-embeddings"), which loses word order but is fast and needs no fine-tuning. In entity-resolution benchmarks, fastText and BERT variants are the most-tested pre-trained embeddings [9].

### 3.3 Contextual sentence embeddings: SBERT and the MTEB era

**Sentence-BERT (SBERT)** (Reimers & Gurevych, EMNLP 2019 [10]) fine-tunes BERT/RoBERTa in a **siamese** (twin-encoder) structure so that a single forward pass yields a sentence vector whose cosine similarity is semantically meaningful. The headline result: finding the most similar pair in a 10 000-sentence collection drops from ~65 hours with naive BERT cross-encoding to ~5 seconds with SBERT embeddings + cosine, at comparable quality [10]. This is *the* reason embeddings dominate matching — they convert an `O(n²)` pairwise-scoring problem into an `O(n)` encode-then-index problem.

The `sentence-transformers` library [11] is the de-facto Python entry point; it hosts 15 000+ models. Practical defaults:

- `all-MiniLM-L6-v2` — **384-d**, ~5× faster, good quality; the standard "cheap default" [11].
- `all-mpnet-base-v2` — **768-d**, best quality of the classic "all-*" family [11].

Model choice is benchmarked by **MTEB (Massive Text Embedding Benchmark)** (Muennighoff et al., EACL 2023 [12]): 58 datasets across 112 languages spanning **8 task types** (bitext mining, classification, pair classification, clustering, reranking, retrieval, STS, summarization). MTEB is the standardized vocabulary the community uses to talk about embedding quality; a framework should point users at it rather than hard-code a "best" model.

Two recent surveys situate the state of the art: Cao (2024) [13] reviews top-performing MTEB methods and identifies three drivers — larger/cleaner/more-diverse training data, LLM-generated synthetic data, and LLM backbones. Zhang et al. (2025) [14] survey the role of pretrained language models in *general-purpose text embeddings (GPTE)*, organizing methods by fundamental roles (embedding extraction, contrastive training strategies, learning objectives) and advanced roles (multilingual, multimodal, code).

### 3.4 API embeddings: OpenAI, Cohere

Hosted embedding APIs (OpenAI `text-embedding-3-small/large`, Cohere `embed`) trade a network call and per-token cost for zero local compute and strong quality. `text-embedding-3-large` is 3072-d with MRL truncation [5]. For a framework these are *just another `obj_to_vect` strategy* behind an optional dependency — the important design point is that they introduce **latency, batching, rate-limits, and non-determinism** that in-process featurizers do not, so the featurizer interface must tolerate batched/async and cached implementations.

---

## 4. Image featurization

### 4.1 Semantic vectors: CLIP and CNN features

**CLIP** (Contrastive Language-Image Pre-training; Radford et al., ICML 2021 [15]) jointly trains an image encoder and a text encoder on 400M (image, text) pairs with a symmetric contrastive loss, producing a **shared image–text embedding space**. This is uniquely powerful for a matching framework because it makes *cross-modal* matching a special case of vector matching: an image and its caption land near each other, so "match these product photos to these text descriptions" reduces to cosine similarity in one space. CLIP embeddings are the default for zero-shot image similarity and dedup-by-meaning.

**CNN penultimate-layer features** — e.g., the 2048-d pooled activations from a **ResNet-50** (He et al., CVPR 2016 [16]) pre-trained on ImageNet — are the classic pre-CLIP image embedding: cheaper, no text alignment, good for "visually similar" rather than "semantically similar". Both are compared with cosine/Euclidean and indexed with ANN.

### 4.2 Perceptual hashes: near-duplicate detection

**Perceptual hashes** map an image to a short **bit-string** that is stable under resizing, mild compression, and minor edits — the antithesis of cryptographic hashes (where one pixel flips every bit). The `ImageHash` library [17] implements:

- **aHash (average hash)** — downscale to 8×8 grayscale, threshold each of 64 pixels against the mean → 64-bit hash. Fast, weakest robustness.
- **pHash (perceptual hash)** — apply a **DCT**, keep low-frequency coefficients, threshold against their median. More robust to gamma/contrast changes; the usual default.
- **dHash (difference hash)** — hash the sign of horizontal gradients between adjacent pixels. Cheap and surprisingly robust.
- **wHash (wavelet hash)** — thresholded wavelet (Haar) coefficients.
- **colorhash**, **crop-resistant hash** — color-distribution and segmentation-based variants.

All are compared with **Hamming distance** (count of differing bits); a threshold (e.g., ≤ 5 bits on a 64-bit hash) declares a match [17]. Perceptual hashes are the "identity representation, relaxed" for images: extremely cheap, no GPU, ideal for a blocking/pre-filter stage, but they capture *appearance*, not *meaning* (a photo and its semantic paraphrase won't match — use CLIP for that).

---

## 5. Audio featurization

### 5.1 Self-supervised speech vectors: wav2vec 2.0

**wav2vec 2.0** (Baevski et al., NeurIPS 2020 [18]) learns speech representations from raw audio with a multi-layer CNN feature extractor feeding a Transformer, trained by a **contrastive task over masked, quantized latents** — no transcripts needed for pretraining. Its contextual frame embeddings (pooled to an utterance vector) are strong general speech representations for speaker/utterance matching and are the standard self-supervised speech backbone.

### 5.2 Cross-modal audio vectors: CLAP

**CLAP** (Contrastive Language-Audio Pretraining; Elizalde et al., 2022 / ICASSP 2023 [19]) is the audio analog of CLIP: a dual-encoder (audio encoder such as a PANN/CNN or HTSAT + a text encoder such as BERT) trained with contrastive/InfoNCE loss to align audio and text in a **shared space**, with L2-normalized embeddings. It enables open-vocabulary, zero-shot audio classification and text↔audio retrieval over music, speech, and environmental sound. For a matching framework, CLAP is the tool for "match this sound to this description" or clustering sounds by meaning.

### 5.3 Acoustic fingerprints: Chromaprint / AcoustID

**Chromaprint** [20] is the audio counterpart of a perceptual hash — an **acoustic fingerprint** optimized for identifying *near-identical* recordings (the same track, possibly transcoded), not perceptual similarity. It converts audio to 11025 Hz, computes **chroma features** (energy in each of the 12 pitch classes, ~8 frames/sec), and post-processes them into a compact fingerprint (~2.5 KB for a 2-minute track, extracted in <100 ms) [20]. The **AcoustID** service matches fingerprints by similarity to recover track identity. Design lesson mirroring images: use a fingerprint (Chromaprint) for *exact-recording dedup/blocking* and a learned embedding (CLAP) for *semantic* audio matching — two different rungs on the identity→embedding ladder.

---

## 6. Metrics, normalization, and the metric/non-metric distinction

A representation is only half of a comparison; each `R` admits particular metrics, and the framework must keep the (representation, metric) pairing coherent.

### 6.1 The metric families

- **Cosine similarity** — `1 - (a·b)/(‖a‖‖b‖)`; the default for TF-IDF, SBERT, CLIP, CLAP. Scale-invariant (ignores vector magnitude), which is usually desired for text/semantic vectors.
- **Euclidean (L2)** — natural for CNN features and any space where magnitude carries meaning. On L2-normalized vectors, L2 and cosine are rank-equivalent (monotone transforms of each other) [3].
- **Dot / inner product** — used by ANN systems like ScaNN that optimize maximum-inner-product search (MIPS); equals cosine on normalized inputs.
- **Jaccard / Dice / overlap** — for **set** representations (token sets, shingles). Jaccard = |A∩B|/|A∪B|. Estimated at scale with **MinHash** (Broder, 1997 [21]) + LSH (`datasketch` [22]).
- **Hamming distance** — for **bit-strings** (perceptual hashes, binary/quantized embeddings). Cheap (XOR + popcount); indexable with multi-index hashing.
- **Edit-distance family** — **Levenshtein**, Damerau-Levenshtein, Jaro-Winkler, longest-common-subsequence ratio — for **strings** compared *without* vectorization. `equate`'s current default `score_func` uses `difflib.SequenceMatcher.ratio()` (an LCS-style ratio); production code typically uses `RapidFuzz` [23] for speed. These operate on the *identity/scalar* representation directly.

### 6.2 Metric vs non-metric (why it matters for indexing)

A **metric** satisfies non-negativity, identity of indiscernibles, **symmetry**, and the **triangle inequality**. The triangle inequality is what lets metric-tree indexes (ball trees, VP-trees) and many ANN methods prune the search space. Key subtleties a framework must not paper over:

- **Cosine *similarity* is not a metric** (it violates the triangle inequality and is a similarity, not a distance). Its companion **angular distance** `arccos(cos)/π` *is* a proper metric, and `1 - cosine` is commonly used as a "distance" but is only a semimetric.
- **Learned / cross-encoder scores are often non-metric and asymmetric** — a re-ranker that scores `(query, candidate)` need not satisfy symmetry or triangle inequality at all. Such comparators can be used for *scoring* a candidate set but **cannot** back a metric ANN index.

This bifurcation is a first-class design axis: the framework should know whether a comparator is (a) a vector metric (→ ANN-indexable), (b) a set/bit metric (→ LSH/multi-index), or (c) an opaque scoring function (→ blocking + brute-force over a candidate set only).

---

## 7. Representation drives blocking and ANN

Naive matching is `O(n·m)` similarity evaluations. Every real system avoids the quadratic blow-up with **blocking** (a.k.a. candidate generation / indexing): cheaply restrict comparisons to plausible pairs. **The representation dictates which blocking strategy is available** — this is the load-bearing coupling for a matching framework.

- **Vector embeddings → Approximate Nearest Neighbor (ANN).** Sub-linear top-k retrieval in `R^d`:
  - **HNSW** (Malkov & Yashunin, 2016/2020 [24]) — hierarchical navigable small-world graphs; empirically the best latency/recall trade-off for in-memory search; the engine inside FAISS, hnswlib, Milvus, Chroma, pgvector.
  - **FAISS** (Johnson, Douze, Jégou, 2017 [25]) — Facebook's library; IVF (inverted-file coarse quantization) + PQ (product quantization) for billion-scale, GPU-accelerated search; supports cosine/L2/inner-product.
  - **ScaNN** (Guo et al., ICML 2020 [26]) — Google's **anisotropic vector quantization** tuned for maximum-inner-product search.
  - Benchmarked head-to-head by **ANN-Benchmarks** (Aumüller et al., 2020 [27]), which standardizes the recall-vs-throughput curve on datasets like GloVe-100 (angular) and SIFT-128 (Euclidean).
- **Set/shingle representations → MinHash-LSH** [21,22]: hash similar sets into shared buckets so candidate pairs are those colliding in ≥1 band.
- **Bit-string (perceptual-hash) representations → multi-index hashing / Hamming LSH**: block by hash prefixes, then verify by Hamming distance.
- **Sparse TF-IDF → inverted index**: candidates are records sharing a rare token.
- **Structured records → classic rule-based blocking**: block on a key field (e.g., zip code, sorted-neighborhood on a sort key), then score within blocks.

Recent entity-resolution work confirms the direction of travel: pre-trained embeddings + ANN increasingly replace hand-crafted blocking keys, with fastText/BERT the most-studied encoders (Zeakis et al., VLDB 2023 [9]). The framework should therefore treat **"produce a representation"** and **"index that representation for candidate generation"** as two facets of one strategy object, because they must agree.

---

## 8. Structured, nested, and schemaless representations

Not everything should be flattened to a vector. Records (dicts/JSON), trees (parse trees, filesystem paths), and graphs (knowledge-graph nodes) carry structure that matching can exploit per-field:

- **Per-field featurization + aggregation.** Represent a record as a dict of per-field representations (name→char-n-gram TF-IDF, address→geohash, price→scalar), compute per-field similarities, then aggregate (weighted sum, learned Fellegi-Sunter weights, or an ML classifier). This is the classic entity-resolution "comparison vector" and is strictly more expressive than concatenating everything into one embedding.
- **Schemaless / heterogeneous.** When records don't share a schema, a common tactic is to *serialize* the record to text and embed it (the "attribute name: value; …" trick used by LLM-based entity matchers), sacrificing field structure for uniformity.
- **Tree/graph edit distances** exist but are expensive (tree edit distance is polynomial but costly; graph edit distance is NP-hard), so they are candidate-set scorers, not blockers.

The framework consequence: representations compose. A record representation is a *product* of field representations, and the matcher must support **structured keys** (dict of sub-representations, each with its own metric) alongside flat vectors — mirroring how the native `key` pattern already supports tuple keys for multi-field sorting.

---

## 9. Design implications for `equate`

The current `equate` API already has the right *seam* (`obj_to_vect` / `similarity_func` / `matcher` in `match_keys_to_values`). The redesign should elevate that seam into first-class, pluggable strategies with the following boundaries.

### 9.1 Make the representation a first-class, key-function-shaped strategy

Define a minimal **`Featurizer`** protocol — a callable `Iterable[Object] -> Representations` (batched by default; a scalar convenience wrapper handles the single-object case, as `transform_text` already does). Ship it so that:

- A **plain callable / lambda is a valid featurizer** (progressive disclosure: `key=str.lower` must work).
- Featurizers **compose** (`compose(clip_encode, l2_normalize)`; per-field dict featurizers for structured records).
- Each featurizer optionally **declares metadata**: output kind (`vector` / `set` / `bitstring` / `scalar` / `structured`), dimensionality (and whether MRL-truncatable), whether it L2-normalizes, and its default/compatible metric. This metadata is what lets the framework auto-select a legal metric and a legal index.

### 9.2 Separate three concerns the current code partly conflates

`representation (Featurizer)` × `comparison (Metric/Scorer)` × `assignment (Matcher)`. `equate` already has the third (`hungarian_matching`, `greedy_matching`, `stable_marriage_matching`, …). Formalize the first two as sibling strategy families with a **compatibility contract**: a Matcher consumes a similarity/cost matrix or a candidate-generator; a Metric knows if it is metric/symmetric (→ ANN-eligible) or an opaque scorer (→ candidate-set only). Default them so `match_keys_to_values(keys, values)` still "just works" with TF-IDF + cosine + Hungarian.

### 9.3 Optional-dependency boundaries (strategy + dependency injection)

Heavy featurizers must be **optional extras**, never hard imports, following the pattern already used for `networkx` (imported inside `maximal_matching`). Concretely:

- Core install: identity, scalar/lambda keys, edit-distance (`difflib`/optional `rapidfuzz`), and sparse TF-IDF (via `grub`/`scikit-learn`) — no GPU, no large downloads.
- `equate[text-embeddings]` → `sentence-transformers`; `equate[image]` → `torch`/`open_clip`, `imagehash`; `equate[audio]` → CLAP/`transformers`, `pyacoustid`/Chromaprint; `equate[api]` → `openai`/`cohere`; `equate[ann]` → `faiss`/`hnswlib`; `equate[lsh]` → `datasketch`.
- Provide **`check_requirements`-style guidance**: when a strategy's dependency is missing, raise an informative error naming the extra to `pip install` and (for system deps like the Chromaprint `fpcalc` binary) the install command/link — per the package-UX principle of guiding users dynamically. Isolate this raising/logging in a decorator so featurizer code stays clean.
- Registry pattern: a `featurizers` registry (SSOT) mapping names → lazy factories, so users select `"sbert"`, `"clip"`, `"tfidf"` by string and heavy imports fire only on use.

### 9.4 Elevate blocking/ANN to a real stage (not just a full similarity matrix)

`similarity_matrix` materializes the full `O(n·m)` matrix — fine for small inputs, fatal at scale. Introduce a **`CandidateGenerator`** stage keyed to the representation kind: ANN (HNSW/FAISS) for vectors, MinHash-LSH for sets, multi-index Hamming for bit-strings, inverted index for sparse TF-IDF, rule/sort-key blocking for structured records. The full-matrix path becomes the `brute_force` candidate generator — the small-n special case. The Matcher then runs over the (sparse) candidate set. This preserves the current API as the default while making the framework scale.

### 9.5 Contain exact matching as the discrete special case

Model exact/`groupby` matching as `Featurizer = identity` + discrete metric, so users can *relax* a matcher (identity → lowercase → token-set → embedding) by swapping the featurizer only. This keeps the "simple things simple" promise and gives a single conceptual model spanning `dict`-join to cross-modal CLIP matching.

### 9.6 Normalization and dimensionality belong to the featurizer

Because normalization conventions differ per representation, expose them *on the featurizer* (a `normalize=True` default for embedding strategies), not as a global matcher flag. Support MRL truncation (`dimensions=k`) as a featurizer parameter for API/MRL models, so storage/ANN cost is tunable without changing the matcher.

---

## 10. Glossary

- **Feature vector** — a fixed-length numeric vector representing an object; the input to vector metrics and ANN.
- **Embedding** — a *learned* feature vector where geometric proximity approximates semantic similarity. (All embeddings are feature vectors; not all feature vectors are learned embeddings — TF-IDF is a feature vector but not a learned embedding.)
- **Representation** — the general term for whatever an object is mapped to for comparison (vector, set, bit-string, scalar, structured record, or the object itself).
- **Identity representation / identity feature** — using the object itself as its key; comparison is exact equality (`==`).
- **Key function** — a callable `Object -> Comparable` (Python `sorted`/`groupby` contract); generalized here to `Object -> Representation`.
- **Sentence embedding** — a single vector for a whole sentence/passage (vs. per-token vectors).
- **SBERT (Sentence-BERT)** — siamese-fine-tuned BERT producing cosine-comparable sentence embeddings [10]; the `sentence-transformers` library implements it.
- **MTEB** — Massive Text Embedding Benchmark; the standard multi-task leaderboard for text embeddings [12].
- **CLIP** — Contrastive Language-Image Pre-training; a shared image–text embedding space [15].
- **CLAP** — Contrastive Language-Audio Pretraining; a shared audio–text embedding space [19].
- **Perceptual hash (aHash/pHash/dHash/wHash)** — a short bit-string stable under minor image edits, compared by Hamming distance [17].
- **Acoustic fingerprint (Chromaprint)** — a compact audio signature for identifying near-identical recordings, based on chroma features [20].
- **Cosine similarity** — angular closeness of two vectors; scale-invariant; *not* a metric (its complement is a semimetric; angular distance is a true metric).
- **Jaccard similarity** — |A∩B|/|A∪B| for sets; estimated at scale by MinHash [21].
- **Hamming distance** — number of differing positions between two equal-length strings/bit-strings.
- **Edit distance** — minimum single-character edits (Levenshtein) or related string-alignment scores (Jaro-Winkler, LCS ratio) between strings.
- **Metric vs non-metric** — a metric obeys non-negativity, identity, symmetry, and the triangle inequality; non-metric/asymmetric scorers (e.g., cross-encoders) cannot back a metric index.
- **Normalization** — rescaling vectors (usually to unit L2 norm) so cosine ≡ dot ≡ (rank) Euclidean [3].
- **Matryoshka embedding (MRL)** — an embedding whose leading-`k` prefix is itself a usable lower-dimensional embedding, enabling truncation without retraining [4].
- **Blocking / candidate generation** — cheaply narrowing the set of pairs to compare, avoiding the `O(n·m)` all-pairs cost.
- **ANN (Approximate Nearest Neighbor)** — sub-linear top-k vector retrieval (HNSW, FAISS/IVF-PQ, ScaNN) [24,25,26].
- **LSH (Locality-Sensitive Hashing)** — hashing that makes similar items collide; MinHash-LSH for Jaccard, Hamming-LSH for bit-strings.

### Synonyms / vocabulary flags across communities

- *Blocking* (record linkage / entity resolution) ≈ *candidate generation* / *retrieval* / *indexing* (IR / vector search) ≈ *filtering* (dedup).
- *Entity resolution* ≈ *record linkage* ≈ *deduplication* ≈ *entity matching* ≈ *entity alignment* (graphs).
- *Embedding* ≈ *representation* ≈ *encoding* ≈ *latent vector* ≈ *feature vector* (loosely).
- *Similarity* vs *distance*: often used interchangeably in prose but they are complements — a matcher must be told which direction "better" is (the current `equate` code converts via `cost = sim.max() - sim` in `hungarian_matching`).
- *Fingerprint* / *perceptual hash* / *content hash* (media) ≈ *sketch* / *signature* (MinHash) — all are cheap, robust, collision-tunable representations for near-duplicate detection.

---

## References

[1] Python Software Foundation. *Sorting Techniques (HOWTO) — key functions.* Python 3 documentation. [docs.python.org/3/howto/sorting.html](https://docs.python.org/3/howto/sorting.html)

[2] Python Software Foundation. *itertools.groupby — grouping by a key function.* Python 3 documentation. [docs.python.org/3/library/itertools.html#itertools.groupby](https://docs.python.org/3/library/itertools.html#itertools.groupby)

[3] Reimers, N. et al. *Semantic search / normalized embeddings — Sentence-Transformers documentation* (cosine, dot-product, and Euclidean are rank-equivalent on L2-normalized vectors). [sbert.net](https://www.sbert.net/docs/pretrained_models.html)

[4] Kusupati, A., Bhatt, G., Rege, A., et al. *Matryoshka Representation Learning.* NeurIPS 2022. [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147)

[5] OpenAI. *Vector embeddings guide (text-embedding-3, `dimensions` / Matryoshka truncation).* OpenAI API documentation, 2024. [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)

[6] scikit-learn developers. *Text feature extraction — TF-IDF term weighting.* scikit-learn documentation. [scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)

[7] Mikolov, T., Chen, K., Corrado, G., Dean, J. *Efficient Estimation of Word Representations in Vector Space (word2vec).* 2013. [arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)

[8] Bojanowski, P., Grave, E., Joulin, A., Mikolov, T. *Enriching Word Vectors with Subword Information (fastText).* TACL 2017. [arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606)

[9] Zeakis, A., Papadakis, G., Skoutas, D., Koubarakis, M. *Pre-trained Embeddings for Entity Resolution: An Experimental Analysis.* PVLDB / VLDB 2023. [arxiv.org/abs/2304.12329](https://arxiv.org/abs/2304.12329)

[10] Reimers, N., Gurevych, I. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

[11] Hugging Face / UKP Lab. *sentence-transformers: State-of-the-Art Embeddings, Retrieval, and Reranking* (library + pretrained model card: `all-MiniLM-L6-v2` 384-d, `all-mpnet-base-v2` 768-d). [github.com/huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers)

[12] Muennighoff, N., Tazi, N., Magne, L., Reimers, N. *MTEB: Massive Text Embedding Benchmark.* EACL 2023. [arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316)

[13] Cao, H. *Recent Advances in Text Embedding: A Comprehensive Review of Top-Performing Methods on the MTEB Benchmark.* 2024. [arxiv.org/abs/2406.01607](https://arxiv.org/abs/2406.01607)

[14] Zhang, M., Zhang, X., Zhao, X., Huang, S., Hu, B., Zhang, M. *On The Role of Pretrained Language Models in General-Purpose Text Embeddings: A Survey.* 2025. [arxiv.org/abs/2507.20783](https://arxiv.org/abs/2507.20783)

[15] Radford, A., Kim, J.W., Hallacy, C., et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021 (PMLR 139:8748-8763). [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

[16] He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition (ResNet).* CVPR 2016. [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

[17] Buchner, J. *ImageHash: A Python Perceptual Image Hashing Module* (aHash, pHash, dHash, wHash, colorhash, crop-resistant; Hamming-distance comparison). [github.com/JohannesBuchner/imagehash](https://github.com/JohannesBuchner/imagehash)

[18] Baevski, A., Zhou, H., Mohamed, A., Auli, M. *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS 2020. [arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)

[19] Elizalde, B., Deshmukh, S., Al Ismail, M., Wang, H. *CLAP: Learning Audio Concepts From Natural Language Supervision.* ICASSP 2023 (arXiv 2022). [arxiv.org/abs/2206.04769](https://arxiv.org/abs/2206.04769)

[20] Lalinský, L. et al. *Chromaprint / AcoustID acoustic fingerprinting* (chroma-feature fingerprints; ~2.5 KB / 2-min track). AcoustID. [acoustid.org/chromaprint](https://acoustid.org/chromaprint) — algorithm write-up: [oxygene.sk/2011/01/how-does-chromaprint-work](https://oxygene.sk/2011/01/how-does-chromaprint-work/)

[21] Broder, A. *On the Resemblance and Containment of Documents (MinHash).* SEQUENCES 1997. [ieeexplore.ieee.org/document/666900](https://ieeexplore.ieee.org/document/666900)

[22] Zhu, E. *datasketch: MinHash, LSH, and probabilistic data structures for Jaccard-similarity search.* [github.com/ekzhu/datasketch](https://github.com/ekzhu/datasketch)

[23] Bachmann, M. et al. *RapidFuzz: Rapid fuzzy string matching (Levenshtein, Jaro-Winkler, etc.).* [github.com/rapidfuzz/RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)

[24] Malkov, Yu.A., Yashunin, D.A. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (HNSW).* IEEE TPAMI 2020 (arXiv 2016). [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)

[25] Johnson, J., Douze, M., Jégou, H. *Billion-scale Similarity Search with GPUs (FAISS).* 2017 / IEEE Trans. Big Data 2021. [arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734)

[26] Guo, R., Sun, P., Lindgren, E., et al. *Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN).* ICML 2020. [arxiv.org/abs/1908.10396](https://arxiv.org/abs/1908.10396)

[27] Aumüller, M., Bernhardsson, E., Faithfull, A. *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.* Information Systems 2020. [arxiv.org/abs/1807.05614](https://arxiv.org/abs/1807.05614)
