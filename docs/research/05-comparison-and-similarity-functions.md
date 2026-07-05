# Comparison & Similarity Functions

**Abstract.** This document surveys the *pairwise comparator* — the function that maps two (possibly featurized) items to a similarity score or a match/no-match boolean — which is the atomic operation underneath any fuzzy matching framework. It catalogs the major string-similarity families (edit-based, token-based, phonetic, hybrid), the non-string comparators (numeric, date, geo, categorical), and the learned/parameterized comparators (Fellegi–Sunter weights, embeddings, cross-encoders), with concrete algorithms, computational complexity, and tradeoffs. It maps these onto the concrete Python ecosystem (rapidfuzz, thefuzz, jellyfish, textdistance, py_stringmatching, recordlinkage, difflib, Splink) and closes with design implications for `equate` — chief among them the **featurize-then-compare vs direct-pairwise-compare duality**, a normalized comparator protocol, field-comparator composition into a comparison vector, and a strategy/optional-dependency boundary that keeps heavy backends out of the core.

---

## 1. Framing: what a comparator is

The comparator is the pairwise kernel of matching. In its most general form it is a function

```
compare : A × B → S
```

where `A`, `B` are item spaces (often `A == B`) and `S` is a score space — most commonly a bounded real `[0, 1]` (graded similarity), an unbounded real (a *distance* in `[0, ∞)`, or a log-odds *match weight* in `(-∞, ∞)`), or a `bool` (a *predicate*). Everything else in a matching pipeline — blocking, assignment/matcher, thresholding — is downstream of, and parameterized by, this kernel. In `equate`, the comparator is exactly what populates the `similarity_matrix` before a `matcher` (Hungarian, greedy, stable-marriage, etc.) extracts an optimal assignment.

### 1.1 The central duality: featurize-then-compare vs direct pairwise compare

There are two dual ways to obtain a score, and a matching framework should treat both as first-class:

1. **Direct pairwise compare** — `score = f(a, b)`. The comparator sees both raw items at once and computes a joint quantity (e.g. Levenshtein distance, Jaro–Winkler). No intermediate representation is materialized. This is what `equate.match_greedily` does today via `SequenceMatcher(a, b).ratio()`.

2. **Featurize-then-compare** (a.k.a. *key-function* or *embed-then-metric*) — `score = g(φ(a), φ(b))` where `φ` (the **key function** / featurizer / vectorizer / embedder) maps each item independently into a comparable space, and `g` is a fixed vector metric (cosine, Euclidean, dot). This is what `equate.similarity_matrix` does today: a learned TF-IDF `obj_to_vect` followed by `cosine_similarity`.

The duality matters for three engineering reasons:

- **Amortization.** Featurizing is `O(n)` calls to `φ`; direct comparison of all pairs is `O(n·m)` calls to `f`. When `φ` is expensive (a neural embedder) but `g` is cheap and *vectorizable* (a single matrix product yields the whole `n×m` matrix), featurize-then-compare wins by orders of magnitude. When `φ` is degenerate (identity) the two collapse into one.
- **Metric-space leverage.** If `φ` lands items in a metric space, ANN indexes (FAISS, HNSW, ball trees) give sublinear neighbor retrieval — impossible for an arbitrary direct `f`.
- **Interchangeability.** Many "direct" comparators are secretly featurize-then-compare (token-set Jaccard = featurize to a set, then set overlap; cosine over q-gram counts = featurize to a bag, then normalized dot). Exposing `φ` and `g` separately lets users swap either half.

Note the deep symmetry with the *keyed equality* idea in Python's own idioms: `sorted(xs, key=φ)` and `groupby(xs, key=φ)` are exact-match analogues where `φ` is a key function and the comparator is `==`. Fuzzy matching generalizes `==` to a graded `g` while keeping the same `key=φ` slot.

### 1.2 Similarity vs distance, bounded vs raw, graded vs boolean

Four orthogonal axes describe any comparator's output; conflating them is a common source of bugs:

- **Polarity:** *similarity* (higher = closer) vs *distance/cost* (lower = closer). Conversion is not universal: a normalized similarity `s ∈ [0,1]` maps to `d = 1 − s`, but an unbounded distance has no canonical similarity without a decay function.
- **Normalization:** bounded `[0,1]` (comparable across fields and lengths) vs raw (edit counts, haversine kilometers). Length-normalization is essential before combining fields: raw Levenshtein of 2 means something different for "id" vs "internationalization".
- **Granularity:** graded score vs boolean predicate (obtained by thresholding a graded score, or computed natively as in `exact`).
- **Symmetry & identity:** a true *metric* satisfies non-negativity, identity of indiscernibles, symmetry, and the triangle inequality. Many popular comparators are **not** metrics — Jaro–Winkler and `SequenceMatcher.ratio()` are asymmetric-ish/non-triangular; cosine "distance" violates the triangle inequality; Monge–Elkan is asymmetric (`ME(a,b) ≠ ME(b,a)`). Only metrics can safely index with triangle-inequality-based structures (VP-trees, ball trees).

### 1.3 Glossary of canonical terms

- **Similarity function** — `A × B → S`, higher meaning more alike; typically normalized to `[0,1]`.
- **Distance / metric** — dissimilarity in `[0,∞)`; a *metric* also obeys symmetry, identity, and the triangle inequality.
- **Key function / featurizer / `φ`** — item-to-comparable map applied independently to each side (TF-IDF vector, sorted token tuple, phonetic code, embedding). The pivot of featurize-then-compare.
- **Levenshtein distance** — minimum single-character insertions/deletions/substitutions to transform one string into another.
- **Damerau–Levenshtein** — Levenshtein plus *transposition* of two adjacent characters as a unit-cost operation; "OSA" (optimal string alignment) is the restricted variant forbidding editing a substring more than once.
- **Jaro / Jaro–Winkler** — a similarity for short strings (names) based on matching characters within a sliding window and transpositions; Winkler adds a bonus for a shared prefix (up to 4 chars).
- **Jaccard index** — `|A∩B| / |A∪B|` over token/shingle sets (frequency-agnostic).
- **Sørensen–Dice** — `2|A∩B| / (|A|+|B|)`; monotonically related to Jaccard but weights the intersection more.
- **Cosine similarity** — normalized dot product of two term-count/weight vectors; frequency-aware.
- **TF-IDF** — term weighting (term frequency × inverse document frequency) that down-weights common tokens; usually fed into cosine.
- **Soundex / Metaphone / NYSIIS** — phonetic encoders mapping a word to a code so that homophones collide; comparison becomes exact-match on codes.
- **Monge–Elkan** — hybrid measure: average, over tokens of one string, of the best secondary (e.g. Jaro–Winkler) match to a token of the other string.
- **Comparison vector (feature vector)** — the per-field vector of comparator outputs for one candidate pair; the input to a match classifier.
- **Threshold** — decision cutoff turning a graded score into a boolean.
- **Calibration** — post-hoc mapping of raw scores to true probabilities (Platt scaling, isotonic regression) so thresholds are interpretable.
- **Match weight (Fellegi–Sunter)** — `log2(m/u)`: log-ratio of the probability a field agrees given a true match (`m`) to that probability given a non-match (`u`); summed across fields into a match score.

---

## 2. String similarity families

String comparison is the richest and most-implemented area, so it deserves the most granularity. Recent comparative work stresses that **family fit depends on the error model**: character-based methods (Jaro–Winkler, edit distance) reach perfect F1 on typos and phrase reordering but token-based Jaccard collapses (recall ≈ 0.40) on paraphrase [11], while token/set measures are the right tool for long documents where edit distance is both slow and meaningless [4].

### 2.1 Edit-based (sequence-based) measures

These operate on the character sequence and count transformation operations.

| Measure | Definition | Complexity | Notes / tradeoffs |
|---|---|---|---|
| **Levenshtein** | min ins/del/sub | `O(mn)` DP (Wagner–Fischer); `O(mn/w)` bit-parallel (Myers); `O(nd)` (Ukkonen, band `d`) | The canonical edit distance. Provably **no strongly subquadratic** `O(n^{2−ε})` algorithm unless SETH is false [9] — the DP is essentially optimal in the worst case. |
| **Damerau–Levenshtein / OSA** | Levenshtein + adjacent transposition | `O(mn)` | Better for keyboard typos ("teh"→"the"). Full DL vs restricted OSA differ on overlapping edits. |
| **Hamming** | positional mismatches | `O(n)` | Equal-length only; substitutions only. |
| **Indel / LCS distance** | ins+del only (no sub) | `O(mn)`, bit-parallel speedups | rapidfuzz's `ratio` is normalized Indel similarity; related to longest common subsequence (LCS). |
| **Needleman–Wunsch** | global alignment, arbitrary sub/gap costs | `O(mn)` | Bioinformatics origin; generalizes Levenshtein with a cost matrix. |
| **Smith–Waterman** | *local* alignment | `O(mn)` | Finds best-matching substring region; good for embedded matches. |
| **Affine gap (Gotoh)** | gaps priced open + extend | `O(mn)` | Models "runs" of insertions cheaply (abbreviations, missing middle names). |
| **Jaro** | matching chars in window + transpositions | `O(mn)` worst, near-linear typical | Designed for short strings/names. |
| **Jaro–Winkler** | Jaro + shared-prefix bonus | as Jaro | Standard for personal names in record linkage; asymmetric prefix emphasis. |
| **Ratcliff–Obershelp (gestalt)** | `2·M/T`, M = matched chars in recursively found common blocks | `O(n^2)` typical, `O(n^3)` naive worst case | Python's `difflib.SequenceMatcher.ratio()`; "looks right to humans," **order-sensitive** (`ratio('tide','diet')=0.25` vs `ratio('diet','tide')=0.5`) and not a metric [7]. |

Key complexity fact to internalize: edit distance is **quadratic and cannot be meaningfully accelerated in the worst case** (Backurs–Indyk, STOC 2015 [9]). Practical speed comes from (a) bit-parallelism (Myers packs the DP into machine words), (b) banded DP when only small distances matter (Ukkonen), and (c) early-exit under a max-distance cutoff — all of which rapidfuzz implements in C++ with SIMD [1].

### 2.2 Token-based (set / bag) measures

Tokenize each string into a set or bag (words, or character *q-grams* / *shingles*), then compare the collections. Order within the collection is discarded, which makes these robust to word reordering and (for q-grams) to internal typos, at the cost of ignoring sequence.

- **Set measures (frequency-agnostic):** **Jaccard** `|A∩B|/|A∪B|`, **Sørensen–Dice** `2|A∩B|/(|A|+|B|)`, **Overlap coefficient** `|A∩B|/min(|A|,|B|)`, **Tversky index** (asymmetric generalization with `α,β` weights on the two set-differences — Jaccard and Dice are special cases). Cost is `O(|A|+|B|)` with hashing.
- **Bag/vector measures (frequency-aware):** **Cosine** over term-count or **TF-IDF**-weighted vectors; **Tanimoto** (binary cosine). TF-IDF down-weights ubiquitous tokens so that rare, discriminating tokens dominate the score — this is precisely `equate`'s current default (`grub` TF-IDF + `sklearn` cosine) and is well-suited to short multi-word phrases.
- **Tokenization is a first-class knob.** py_stringmatching exposes five tokenizers — alphabetic, alphanumeric, delimiter, whitespace, and **q-gram** [4]. q-gram (character n-gram) tokenization is the bridge that lets *set* measures approximate *edit* behavior (shared 2-/3-grams survive small typos), and it is what powers the `qgram` string comparison in recordlinkage [6].

Guidance from the literature: token/set measures are recommended for **long** strings (documents, essays); edit measures for **short/medium** strings (names, codes) [4][11].

### 2.3 Phonetic encoders

These are **key functions** `φ` that collapse a word to a code capturing pronunciation; comparison then reduces to exact (or near-exact) match on codes — a clean instance of featurize-then-compare.

- **Soundex** — 4-char code (letter + 3 digits); crude, English-biased, still a baseline. In jellyfish and py_stringmatching [2][4].
- **Metaphone / Double Metaphone** — rule-based, handles more English phonetics; Double Metaphone returns primary + alternate codes for ambiguous pronunciations.
- **NYSIIS** — New York State Identification and Intelligence System; better than Soundex for surnames.
- **Match Rating Approach (MRA)** — Western Airlines' codex + a comparison rule that yields a boolean/graded verdict; jellyfish ships both `match_rating_codex` and the comparison [2].
- **Editex** — a *hybrid*: edit distance where substitution cost depends on phonetic letter-group equivalence (in textdistance, py_stringmatching) [3][4].

jellyfish is the go-to Python home for phonetics (Soundex, Metaphone, NYSIIS, MRA) plus Jaro/Jaro–Winkler/Levenshtein/Damerau–Levenshtein/Hamming, with Rust implementations used by default for speed [2].

### 2.4 Hybrid measures

Hybrids compose a *secondary* character-level similarity inside a *primary* token-level aggregation, capturing both token reordering and intra-token typos.

- **Monge–Elkan** — for each token of `a`, take its best secondary-similarity match among tokens of `b`, then average. Default secondary is Jaro–Winkler [5]. **Asymmetric**, and has no natural `[0,1]` normalization (py_stringmatching deliberately omits `get_sim_score` for it, along with Affine Gap, Needleman–Wunsch, Smith–Waterman, and Soft-TF/IDF, because their raw scores don't normalize cleanly) [4][5].
- **Generalized (soft) Jaccard** — Jaccard where two tokens "intersect" if their secondary similarity exceeds a threshold.
- **Soft-TF/IDF** — TF-IDF cosine where token identity is relaxed to secondary-similarity-weighted token matching; strong on entity names with spelling drift.
- **Token-sort / token-set ratios** (rapidfuzz/thefuzz) — `token_sort_ratio` sorts tokens then applies `ratio`; `token_set_ratio` compares the intersection and remainders; `WRatio` orchestrates several of these with heuristics and preprocessing [1]. These are the practical "just works" fuzzy scorers for dirty short text.

---

## 3. Non-string comparators

A general matching framework compares more than strings. Record-linkage practice (recordlinkage, Splink) standardizes these field-type comparators [6]:

- **Exact / categorical** — `1` if equal else `0`; the boolean baseline, right for codes, gender, enumerations.
- **Numeric** — a *decay function* of the absolute difference, mapping distance to a graded `[0,1]`: **step** (threshold), **linear**, **exponential (exp)**, **Gaussian (gauss)**, **squared**. The scale/offset parameters encode "how close is close." Dates are usually handled by converting to a numeric timestamp and applying the same decays [6].
- **Date** — specialized comparators additionally credit *structured* errors (swapped day/month, off-by-one month) that a raw numeric diff would penalize [6].
- **Geospatial** — **haversine** great-circle distance between lat/lon, then the same decay functions as numeric [6].
- **Set / list / interval** — Jaccard/overlap for multi-valued fields; interval overlap (IoU) for ranges and time spans.

The unifying pattern: a raw domain distance (`|Δ|`, kilometers, days) passed through a **parameterized decay** to yield a normalized, combinable similarity. Decay choice and scale are the calibration knobs.

---

## 4. Learned & parameterized comparators

Beyond fixed formulas, the comparator (or its combination across fields) can be *learned*.

- **Parameterized fixed comparators.** Jaro–Winkler's prefix weight, Tversky's `α,β`, affine gap penalties, decay scales — all are hyperparameters that can be tuned to a domain rather than left at defaults.
- **Fellegi–Sunter probabilistic linkage.** The classic learned *combiner*: each field's agreement contributes a **match weight** `log2(m/u)`, where `m = P(field agrees | true match)` and `u = P(field agrees | non-match)`; weights sum to a match score compared against two thresholds `Tμ > Tλ` (match / possible-match / non-match) [10]. Parameters are typically estimated *unsupervised* by **Expectation–Maximization** [10]. Modern Python: **Splink** implements Fellegi–Sunter at scale with EM training, user-defined comparison *levels* (e.g. exact / Jaro–Winkler>0.9 / else), and **term-frequency adjustments** so that agreeing on a rare value ("Zbigniew") counts more than a common one ("John") [8].
- **Supervised similarity.** With labeled pairs, the comparison vector (§5) feeds any classifier (logistic regression, random forest, SVM, gradient boosting). A 2020 study frames data matching as supervised learning over similarity-metric features and finds ensemble classifiers over multiple string metrics outperform any single metric [14].
- **Embedding + vector metric (featurize-then-compare, learned `φ`).** Replace TF-IDF with word2vec/fastText/sentence-transformer embeddings, keep cosine. Captures semantic (not just lexical) similarity; enables ANN blocking. This is the drop-in `obj_to_vect` extension `equate` already anticipates.
- **Cross-encoder / PLM matchers (learned joint `f`).** **Ditto** casts entity matching as BERT sequence-pair classification, achieving up to +29% F1 over prior deep methods with less labeled data [12]; surveys of deep entity matching catalog the design space (attention, contextualization, data augmentation) [13]. These are the most accurate and the most expensive — a `bool`/probability-valued *direct* comparator that cannot be decomposed into `φ` + `g`.

The accuracy/cost ladder runs: exact < single edit/token metric < tuned metric < classifier over comparison vector < embedding cosine < cross-encoder/LLM. A framework should let users climb it without rewriting the pipeline.

---

## 5. From score to decision: thresholds, calibration, comparison vectors

### 5.1 Thresholding and calibration

A graded comparator becomes a decision via a **threshold**. recordlinkage's string comparator makes this explicit: with a `threshold`, scores `≥ threshold` become `1` else `0`; without one it returns the raw float [6]. `difflib.get_close_matches` exposes the same idea as `cutoff=0.6` [7].

Raw scores are rarely true probabilities, so thresholds set on one dataset don't transfer. **Calibration** fixes this: **Platt scaling** (fit a logistic on scores) or **isotonic regression** (monotone non-parametric fit) map raw scores to calibrated `P(match)`, after which a threshold like `0.5` is meaningful and cost-sensitive thresholds (favoring precision or recall) are principled. Fellegi–Sunter's log-odds weights are themselves a calibration-by-construction: the score *is* interpretable as evidence [10].

### 5.2 The comparison vector and multi-field composition

For structured records, each field yields one comparator output; stacking them gives the **comparison vector** (feature vector) for the pair [6]. "A set of informative, discriminating and independent features is important for good classification of record pairs" [6]. Composition strategies:

- **Weighted sum / mean** — simplest; weights hand-set or learned.
- **Fellegi–Sunter sum of match weights** — principled log-odds combination [8][10].
- **Learned classifier** — the vector is `X`, the label is match/non-match; any sklearn estimator applies [14].
- **Max / min / soft-OR** — for "match on *any* strong field" or "match on *all* fields" semantics.

Keeping per-field comparison and cross-field combination as **separate stages** is the key architectural move: it lets the same field comparators feed a threshold, a rule, or a trained model interchangeably.

---

## 6. Python library landscape

| Library | Strengths | Metrics / features | Impl. / license | Notes |
|---|---|---|---|---|
| **rapidfuzz** [1] | Fast, permissive, batch | `fuzz` (ratio, partial_ratio, token_sort/set_ratio, WRatio, QRatio); `distance` (Levenshtein, DamerauLevenshtein, Hamming, Indel, Jaro, JaroWinkler, LCSseq, OSA, Prefix, Postfix); `process` (extract, extractOne, **cdist** batch matrix) | C++ + SIMD, **MIT**, Py≥3.10 | The default recommendation; ~40% faster than peers in a 2025 multilingual benchmark [11]. Drop-in replacement for the GPL `thefuzz`/`fuzzywuzzy`. |
| **thefuzz / fuzzywuzzy** | Familiar API | ratio family (now wraps rapidfuzz internally) | Python, **GPL/MIT-ish** | Legacy; prefer rapidfuzz for new code [1]. |
| **jellyfish** [2] | Phonetics + edit | Levenshtein, Damerau–Levenshtein, Hamming, Jaro, Jaro–Winkler; Soundex, Metaphone, NYSIIS, Match Rating | Rust (default) + Python | Best phonetic coverage; per-call scalar API (no batch matrix). |
| **textdistance** [3] | Breadth, experimentation | 30+ across edit / token / sequence / phonetic / compression / simple families; normalized + raw interfaces | Pure Python + optional C speedups | Great for exploring; not the fastest at scale. |
| **py_stringmatching** [4][5] | Rigor, hybrids, tokenizers | 24 measures incl. Affine Gap, Needleman–Wunsch, Smith–Waterman, Monge–Elkan, Soft-TF/IDF, Generalized Jaccard, Tversky; 5 tokenizers; explicit `get_sim_score` vs raw | Python/Cython | The Magellan project's measure library; principled about which measures normalize to `[0,1]`. |
| **recordlinkage** [6] | Full linkage pipeline | `Compare`: exact, string (jaro, jarowinkler, levenshtein, damerau_levenshtein, qgram, cosine, smith_waterman, lcs), numeric (step/linear/exp/gauss/squared), geo (haversine), date; produces comparison vectors + classifiers | Python (pandas) | Field-comparator + comparison-vector abstraction to emulate. |
| **difflib** [7] | Stdlib, zero-dep | `SequenceMatcher.ratio`, `quick_ratio`, `get_matching_blocks`, `get_close_matches`, autojunk | Pure Python stdlib | Ratcliff–Obershelp; order-sensitive; fine for small inputs, quadratic. |
| **Splink** [8] | Scale, probabilistic | Fellegi–Sunter + EM, comparison levels, term-frequency adjustment, SQL backends | Python + SQL (DuckDB/Spark) | State of the art for probabilistic linkage at scale (adopted by national statistics agencies). |
| **scikit-learn `pairwise`** | Vectorized `φ`+`g` | `cosine_similarity`, `euclidean_distances`, `pairwise_distances`, custom metrics | NumPy/SciPy | The featurize-then-compare backend `equate` already uses. |

---

## 7. Design implications for `equate`

`equate` today hard-wires two comparators (`SequenceMatcher.ratio` for `match_greedily`; TF-IDF+cosine for `similarity_matrix`). The survey above suggests generalizing the comparator into a small, composable, strategy-driven layer while keeping the simple path trivial. Concrete recommendations:

1. **Adopt a `Comparator` protocol as the core extension point.** Define a minimal `Callable[[A, B], float]` protocol (structural, via `typing.Protocol`) with declared, introspectable metadata: `polarity` (similarity|distance), `bounded` (is output in `[0,1]`?), `is_metric`, `is_symmetric`. This metadata lets downstream code (matchers, thresholders) auto-adapt (e.g. convert distance→similarity, refuse triangle-inequality indexing for non-metrics) instead of the caller hard-coding assumptions. Keep `score_func` (as in `match_greedily`) as the sugar over this protocol.

2. **Make the featurize-then-compare vs direct-compare duality first-class.** Provide two constructors that both produce a `Comparator`: `direct(f)` wrapping a joint `f(a, b)`, and `featurized(phi, g=cosine_similarity)` composing a key function `phi` with a vector metric `g`. `similarity_matrix`'s current `obj_to_vect`+`similarity_func` is exactly `featurized(...)` — expose it under this unified vocabulary. This SSOT for "how do I get a score" is the single most important abstraction, because it mirrors both the library taxonomy and Python's own `key=` idiom.

3. **Separate three stages that are currently entangled: score → normalize → decide.** (a) raw comparator, (b) a `normalize`/decay adapter (`1−d`, exp/gauss decay for numeric/geo, min-max) that guarantees a `[0,1]` similarity, (c) an optional `threshold`/`calibrate` stage (fixed cutoff, Platt, isotonic) turning score into boolean/probability. This lets the *same* comparator feed a raw matrix, a boolean predicate, or a calibrated classifier — the recordlinkage `threshold=None → float else 0/1` pattern, generalized [6].

4. **Ship a comparator registry keyed by domain (strategy pattern).** Namespace built-ins by field type: `string.*` (levenshtein, jaro_winkler, jaccard, cosine_tfidf, monge_elkan, ratcliff), `numeric.*` (linear, exp, gauss decays), `date.*`, `geo.haversine`, `categorical.exact`, `set.jaccard`. Users select by name or pass a callable — no monolithic function. Sensible default stays TF-IDF+cosine (already good for the README's package-name use case).

5. **Draw the optional-dependency boundary at the comparator, not the core.** The core (`match_greedily`, `similarity_matrix`, matchers) must run on stdlib + numpy/scipy only. Heavier comparators live behind optional extras and lazy imports, following the existing pattern (`networkx` is already imported inside the matcher functions): `equate[rapidfuzz]` (fast edit/token scorers), `equate[phonetic]` (jellyfish Soundex/Metaphone/NYSIIS), `equate[stringmatching]` (py_stringmatching hybrids), `equate[embeddings]` (sentence-transformers). Register these only when importable; raise an *informative* error naming the extra when a user asks for an unavailable comparator (aligns with the project's `check_requirements` guidance).

6. **Add multi-field comparison-vector composition.** Provide a `FieldComparators` mapping `{field: Comparator}` that, applied to a pair of records, yields a comparison vector; then a pluggable **combiner** (`weighted_sum`, `mean`, `max`, `fellegi_sunter`, or a fitted sklearn estimator) reduces it to a scalar/boolean. This upgrades `equate` from string-pair matching to record matching without disturbing the string path, and mirrors the recordlinkage/Splink architecture [6][8][10].

7. **Exploit batch/vectorized comparison for the similarity matrix.** Where a comparator is featurize-then-compare, compute the whole `n×m` matrix with one vectorized `g` (as today with `cosine_similarity`). Where it is a fast direct scorer, prefer a batch entry point (rapidfuzz `process.cdist` returns the matrix in C++ [1]) over Python-level double loops. Keep the sparse-matrix path (`ensure_sparse`) so blocking can zero out non-candidate pairs before matching.

8. **Cache and memoize expensive `φ`.** Featurizing (TF-IDF fit, embeddings) is the cost center; cache learned `φ` and per-item vectors (`functools.lru_cache` / `dol.cache_this`) so repeated matches against a fixed corpus don't recompute — consistent with the project's stated persistence conventions.

9. **Provide calibration as opt-in.** Expose Platt/isotonic wrappers and (optionally) a Fellegi–Sunter combiner so thresholds are interpretable and transferable across datasets, rather than magic constants. Default to the raw-score path for the "simple things simple" case.

Net effect: a user still writes `dict(match_greedily(keys, values))` for the trivial case, but a power user can assemble `featurized(embedder, cosine) → gauss_decay → isotonic_calibrate`, compose several field comparators into a vector, feed a learned combiner, and hand the resulting similarity matrix to any matcher — all through one uniform comparator vocabulary, with heavy backends isolated behind optional extras.

---

## References

[1] RapidFuzz — Rapid fuzzy string matching in Python (C++/SIMD, MIT). [github.com/rapidfuzz/RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)

[2] jellyfish — Approximate and phonetic matching of strings (Rust + Python). [jamesturk.github.io/jellyfish](https://jamesturk.github.io/jellyfish/)

[3] textdistance — Compute distance between sequences, 30+ algorithms, pure Python + optional C. [github.com/life4/textdistance](https://github.com/life4/textdistance)

[4] py_stringmatching User Manual (Magellan / AnHai Group), v0.4.x. [anhaidgroup.github.io/py_stringmatching](https://anhaidgroup.github.io/py_stringmatching/v0.4.x/)

[5] py_stringmatching — Monge–Elkan similarity measure documentation. [anhaidgroup.github.io/py_stringmatching MongeElkan](https://anhaidgroup.github.io/py_stringmatching/v0.3.x/MongeElkan.html)

[6] Python Record Linkage Toolkit — Comparing (Compare class: string, numeric, geo, date, exact; thresholds; comparison vectors). [recordlinkage.readthedocs.io](https://recordlinkage.readthedocs.io/en/latest/ref-compare.html)

[7] Python Standard Library — `difflib` (SequenceMatcher / Ratcliff–Obershelp, ratio, get_close_matches). [docs.python.org/3/library/difflib](https://docs.python.org/3/library/difflib.html)

[8] Splink — Term frequency adjustments (Fellegi–Sunter, EM, comparison levels), Ministry of Justice Analytical Services. [moj-analytical-services.github.io/splink](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)

[9] Backurs A, Indyk P. Edit Distance Cannot Be Computed in Strongly Subquadratic Time (unless SETH is false). STOC 2015 / arXiv:1412.0348. [arxiv.org/abs/1412.0348](https://arxiv.org/abs/1412.0348)

[10] Grannis SJ et al. The Data-Adaptive Fellegi–Sunter Model for Probabilistic Record Linkage. J Med Internet Res, 2022;24(9):e33775 (m/u probabilities, match weights, EM, thresholds). [jmir.org/2022/9/e33775](https://www.jmir.org/2022/9/e33775)

[11] Elmobark N. A Comparative Analysis of Python Text Matching Libraries: A Multilingual Evaluation of Capabilities, Performance and Resource Utilization. Int. J. Environment, Engineering and Education, 2025;7(1) (rapidfuzz, thefuzz, difflib, Levenshtein, jellyfish across 5 languages). [ijeedu.com article 188](https://ijeedu.com/index.php/ijeedu/article/view/188)

[12] Li Y, Li J, Suhara Y, Doan A, Tan W-C. Deep Entity Matching with Pre-Trained Language Models (Ditto). PVLDB 2021 / arXiv:2004.00584. [arxiv.org/abs/2004.00584](https://arxiv.org/abs/2004.00584)

[13] Barlaug N, Gulla JA. Neural Networks for Entity Matching: A Survey / Deep Entity Matching: Challenges and Opportunities. ACM J. Data and Information Quality, 2021. [dl.acm.org/doi/fullHtml/10.1145/3431816](https://dl.acm.org/doi/fullHtml/10.1145/3431816)

[14] Supervised Machine Learning Techniques for Data Matching Based on Similarity Metrics. arXiv:2007.04001, 2020. [arxiv.org/abs/2007.04001](https://arxiv.org/abs/2007.04001)
