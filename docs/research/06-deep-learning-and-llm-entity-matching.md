# Deep Learning and LLM-based Entity Matching

**Abstract.** This document surveys the shift from feature-engineered, classical
machine-learning entity matching (EM) to *learned* matchers: attribute-level
RNN/attention models (DeepMatcher, DeepER), fine-tuned pre-trained transformers
(Ditto), self-supervised contrastive representation learners (Sudowoodo, EMBER),
deep blocking via dense retrieval (DeepBlocker, Sentence-BERT/DPR-style
dual-encoders), and — since 2022–2025 — in-context prompting of foundation
models (GPT-4, Claude, open LLMs) for matching. It is precise about the
vocabulary that differs across the database, IR, and NLP communities, records
what the benchmark evidence actually shows about *where learned methods beat
classical baselines and where they do not*, and closes with concrete design
implications for `equate` as a general matching framework. The through-line for
`equate` is the **bi-encoder vs. cross-encoder** split — cheap independently
embeddable scorers that support blocking, versus expensive joint pair scorers
that maximise accuracy — and the **blocking-then-matching-then-assignment**
pipeline that organises essentially all modern EM systems.

---

## 1. Scope and vocabulary

### 1.1 What problem is being solved

**Entity matching (EM)** decides whether two data instances (records, strings,
rows, product offers) refer to the *same real-world entity*. It is the core
*matching* step of the broader **entity resolution (ER)** pipeline. The same
task appears across communities under different names, and a synthesis step
should treat these as synonyms:

| Term | Community of origin | Notes |
|---|---|---|
| Entity matching (EM) | Databases | The pairwise same-entity decision |
| Record linkage | Statistics / health / census | Classically probabilistic (Fellegi–Sunter) |
| Entity resolution (ER) | Databases / KG | End-to-end: block → match → cluster |
| Deduplication / merge–purge | Data cleaning | ER within a *single* source |
| Reference reconciliation / coreference | KG / NLP | Overlaps with entity linking |
| Entity alignment | Knowledge graphs | Matching nodes across two KGs |

`equate`'s framing — match a "keys" collection to a "values" collection under a
flexible score, never a hard-equality join — is exactly EM generalised beyond
tables to "dirty things like language, files, socks and whistles". The
`similarity_matrix` + `matcher` architecture is a direct instance of the
field's **blocking → matching → assignment** decomposition (§4).

### 1.2 The two encoder families (the single most load-bearing distinction)

Almost every learned matcher is one of two shapes, and the distinction dictates
cost, blockability, and where it plugs into `equate`:

- **Bi-encoder** (a.k.a. *dual-encoder*, *two-tower*, *Siamese network*):
  each object is embedded **independently** into a vector; a pair score is a
  cheap vector-space function (cosine, dot product). Because embeddings are
  precomputable and index-friendly, bi-encoders enable **blocking / dense
  retrieval** over millions of items via approximate nearest neighbour (ANN)
  search [18][19]. This is *exactly* `equate`'s `obj_to_vect` + `similarity_func`
  path.
- **Cross-encoder** (a.k.a. *joint encoder*, *interaction model*): the pair is
  fed **together** into one model that attends across both sides and emits a
  match probability. More accurate (it can model token-level interactions) but
  **O(n·m)** — you cannot precompute; you must run the model on every candidate
  pair. Ditto [4] and LLM pair-prompts are cross-encoders. This corresponds to a
  `score_func(key, value)` that does *not* factor through independent vectors.

The universal engineering pattern is a **cascade**: a cheap bi-encoder (or
classical blocker) proposes a small candidate set; an expensive cross-encoder
(or LLM) re-ranks only those candidates. Holding this split explicit is the
central recommendation for `equate` (§7).

### 1.3 Glossary of canonical terms

- **DeepMatcher** — the SIGMOD 2018 design-space study and Python library that
  categorised attribute-level DL matchers into SIF / RNN / Attention / Hybrid
  architectures [1].
- **Ditto** — VLDB 2020 system that fine-tunes a pre-trained transformer
  (BERT/DistilBERT/RoBERTa) as a **sequence-pair classifier** for EM [4].
- **Siamese network** — twin networks with shared weights producing comparable
  embeddings; trained with contrastive or triplet loss. Sentence-BERT is the
  canonical text instance [18].
- **Contrastive learning** — self-supervised objective that pulls
  augmentation-related ("positive") pairs together and pushes random
  ("negative") pairs apart in embedding space, learning representations
  *without labels*; basis of Sudowoodo [7] and DeepBlocker's best variant [6].
- **Dense retrieval** — IR term for first-stage candidate retrieval using
  learned embeddings + ANN, as opposed to sparse lexical (BM25) retrieval [19].
  In EM this *is* embedding-based **blocking**.
- **DeepBlocker** — VLDB 2021 design-space study applying DL (self-supervised,
  no labels) to the **blocking** step [6].
- **Sudowoodo** — ICDE 2023 contrastive self-supervised framework unifying
  blocking, matching, cleaning, and column-type detection [7].
- **LLM entity matching** — prompting a foundation model (hosted or open) to
  judge whether two records match, in zero-shot or few-shot mode [9][10][11][12].
- **In-context learning (ICL)** — supplying a few labelled demonstrations *in
  the prompt* (few-shot) instead of updating weights; "0-shot" = none.
- **Foundation model (FM)** — a large model pre-trained on broad data,
  adaptable to many downstream tasks; **LLM** ⊂ FM, and the older **PLM**
  (pre-trained language model: BERT/RoBERTa, ~10⁸ params) ⊂ FM.
- **Blocking-then-matching** — the two-phase pipeline: reduce the O(n·m)
  comparison space to a candidate set (blocking/indexing), then apply an
  expensive matcher to survivors. Standard across essentially all ER systems
  [13][14].

---

## 2. Classical learned matching (the baseline learned methods must beat)

### 2.1 Magellan (2016) — the strong non-deep baseline

**Magellan** [3] (Konda, Doan et al., PVLDB 2016) is the reference *classical*
EM management system: a guided, modular workflow of **blocking → sampling →
labelling → feature engineering → supervised learning** (random forest, SVM,
logistic regression over hand-built similarity features such as Jaccard,
edit distance, TF-IDF cosine per attribute). It matters here because it is the
baseline against which every deep method reports gains, and because on *clean
structured* data it remains remarkably hard to beat (§6). Its architecture —
pluggable blockers, per-attribute feature functions, a swappable learner —
is a good structural precedent for `equate`.

### 2.2 The Magellan/DeepMatcher benchmark suite

The community standard benchmark set, released with Magellan and reused by
almost every subsequent paper, partitions datasets into three regimes that map
onto the "where does DL help" question:

- **Structured** — clean, aligned attributes (DBLP–ACM, DBLP–Scholar,
  Amazon–Google, Walmart–Amazon, Fodors–Zagat, Beer, iTunes–Amazon).
- **Textual** — attributes are long free text (e.g. Abt–Buy).
- **Dirty** — structured data with values injected into wrong columns
  (attribute misplacement), simulating real ETL noise.

These 13-ish datasets remain the lingua franca of EM evaluation, but see §6.3
for a serious critique of their difficulty.

---

## 3. Deep learning matchers (2018–2023)

### 3.1 DeepER and DeepMatcher (2018) — attribute-level neural matching

**DeepER** (Ebraheem et al., 2018) [2] and **DeepMatcher** (Mudgal et al.,
SIGMOD 2018) [1] introduced neural EM. DeepMatcher is the more influential: it
frames a **design space** of attribute-level architectures of increasing
representational power —

1. **SIF** — smoothed inverse-frequency aggregation of word embeddings (a
   weighted bag-of-embeddings), essentially a learned bi-encoder baseline;
2. **RNN** — bidirectional GRU/LSTM over each attribute's token sequence;
3. **Attention** — decomposable attention that soft-aligns tokens across the
   two records' attribute values;
4. **Hybrid** — RNN + attention, the best-performing variant.

Each attribute is encoded, a **similarity/comparison vector** is computed per
attribute, and the per-attribute vectors are aggregated and passed to a
classifier — a design that anticipates `equate`'s per-attribute similarity
then aggregate structure. **Key empirical finding (verify):** DL "does not
outperform current [Magellan] solutions on structured EM, but it can
significantly outperform them on textual and dirty EM" [1]. Cost: fully
supervised, needing thousands of labelled pairs per task.

### 3.2 Ditto (2020) — fine-tuned transformers as a cross-encoder

**Ditto** (Li, Li, Suhara, Doan, Tan, VLDB 2020) [4], journal extension
"Effective entity matching with transformers" (VLDB Journal 2023) [5], reframed
EM as **sequence-pair classification**: serialise each record as
`COL <attr> VAL <value> COL <attr> VAL <value> …`, concatenate the two records
with a `[SEP]`, and fine-tune BERT/DistilBERT/RoBERTa to predict match/no-match
on the `[CLS]` token. It is a **cross-encoder**. Three optimisations:

1. **Domain-knowledge injection** — span-typing / highlighting important tokens
   (numbers, product IDs, last names) so the model attends to them.
2. **Summarisation** — TF-IDF-based truncation to keep only the most
   informative tokens of over-long strings (transformers have a token budget).
3. **Data augmentation** (`invDA`) — label-preserving edits (deleting/shuffling
   spans, swapping the two records) to improve robustness with little data.

**Reported results (verify):** up to **+29% F1** over prior SOTA on benchmark
datasets, and it can reach prior SOTA using **at most half** the labelled data
[4]. Ditto is the enduring "strong supervised deep baseline" that LLM papers
now measure against.

### 3.3 Deep blocking and dense retrieval

Matching is O(1) per pair but the pair space is O(n·m); **blocking** makes ER
tractable. The learned approach is **dense retrieval**: embed every record with
a bi-encoder, index with ANN (FAISS/HNSW), retrieve top-k neighbours as
candidates.

- **Sentence-BERT (SBERT)** (Reimers & Gurevych, EMNLP 2019) [18] — the
  canonical Siamese text encoder: BERT fine-tuned with a triplet/contrastive
  objective so cosine similarity is meaningful. It cut the cost of finding the
  most similar pair in a 10k-sentence collection from ~65 hours (naive BERT
  cross-encoding) to ~5 seconds (embed once, then cosine), the exact bi-encoder
  economics that make blocking possible [18].
- **Dense Passage Retrieval (DPR)** (Karpukhin et al., EMNLP 2020) [19] — the
  dual-encoder that established that learned dense retrieval beats BM25 (by
  9–19% top-20 accuracy), the IR foundation reused for EM blocking.
- **DeepBlocker** (Thirumuruganathan et al., VLDB 2021) [6] — a *design-space*
  study of DL blocking with 8 variants; the best is **self-supervised**
  (no labels): a Sentence-BERT-style encoder trained with **triplet loss**,
  using data augmentation (randomly delete up to ~40% of tokens to form
  positives, random other records as negatives). It significantly beats
  classical rule/hash blocking on recall-vs-candidate-set-size.
- **EMBER** (Suri et al., VLDB 2022) [8] — "no-code context enrichment via
  similarity-based **keyless joins**": builds a transformer-embedding index and
  serves top-k similarity queries; treating a top-1 query as a binary matcher
  gives F1 comparable to or better than prior SOTA. Notable as a *general
  keyless-join operator* — conceptually the closest published system to what
  `equate` aims to be.

### 3.4 Self-supervised / contrastive matching — Sudowoodo (2023)

**Sudowoodo** (Wang, Li, Wang, ICDE 2023) [7] pushes contrastive learning end
to end: a SimCLR-style objective learns similarity-aware representations from an
**unlabelled** corpus of data items, then the *same* representation serves
blocking **and** matching (and data cleaning, and column-type detection) under a
single "is this pair similar?" formulation. It reaches SOTA across supervision
levels — including few-/zero-label regimes — and is the strongest evidence that
a *single learned embedding space* can drive an entire matching framework, which
is precisely the abstraction `equate` centralises in `obj_to_vect`.

---

## 4. The pipeline shape shared by all these systems

Two authoritative surveys frame the field and should anchor the synthesis's
vocabulary:

- **Christophides, Efthymiou, Palpanas, Papadakis, Stefanidis**, "An Overview of
  End-to-End Entity Resolution for Big Data," *ACM Computing Surveys* 53(6),
  2020 [13] — the end-to-end ER reference (blocking, block processing, matching,
  clustering; schema-agnostic vs schema-aware; batch vs incremental).
- **Papadakis, Skoutas, Thanos, Palpanas**, "Blocking and Filtering Techniques
  for Entity Resolution: A Survey," *ACM Computing Surveys* 53(2), 2020 [14] —
  the blocking/indexing reference (token blocking, sorted neighbourhood, meta-
  blocking, filtering).

The canonical stages:

1. **Blocking / indexing / candidate generation** — reduce O(n·m) to a
   manageable candidate set. Classical (token/sorted-neighbourhood) or learned
   (dense retrieval, §3.3). Metric that matters: **pair completeness / recall**
   at a given candidate-set size.
2. **Matching** — score each surviving candidate pair (bi- or cross-encoder,
   classical features, or LLM prompt). Metric: **precision/recall/F1** of the
   match decision.
3. **Assignment / clustering** — turn pairwise scores into a global decision:
   one-to-one assignment (Hungarian), thresholding, or transitive-closure
   clustering for deduplication. This is where **global consistency** lives, and
   where independent pairwise LLM calls fall short (§5.3).

`equate` already implements (1) as a sparse `similarity_matrix`, (2) as
`score_func`/`similarity_func`, and (3) as `matcher` (greedy, optimal, …). The
literature says: keep these three as *first-class, independently swappable*
stages.

---

## 5. LLM-based entity matching (2022–2025)

### 5.1 The pivot: "Can Foundation Models Wrangle Your Data?" (2022)

**Narayan, Chami, Orr, Ré** (VLDB 2022) [9] were the tipping point: with only a
handful of in-context demonstrations, **GPT-3 few-shot beat the fully
fine-tuned Ditto on 4 of 7 EM datasets** — no task-specific training. The stated
failure mode is important and recurs: FMs struggle on **jargon-heavy domains**
(part numbers, specialised catalogues) where the pre-training corpus gives no
semantic grounding. This reframed EM from "train a model" to "prompt a model."

### 5.2 Systematic LLM-for-EM: Peeters, Steiner & Bizer (EDBT 2025)

The most thorough systematisation [10] (arXiv 2023 → EDBT 2025) evaluates hosted
(GPT-3.5/GPT-4) and locally runnable open LLMs across prompt designs. Headline
findings (verify):

- The best LLMs need **zero or only a few** examples to perform **comparably to
  PLMs fine-tuned on thousands** of pairs.
- LLM matchers show **higher robustness to unseen entities** (out-of-
  distribution generalisation) — the main weakness of fine-tuned PLMs, which
  overfit the entities in their training set.
- GPT-4 can emit **structured natural-language explanations** for its decisions
  and even help diagnose labelling errors. On some benchmark settings GPT-4 is
  reported to outperform the best PLM by a wide margin (order of tens of F1
  points); treat the exact figure as setting-dependent.

Prompt-design axes that matter: serialisation (attribute-tagged vs raw), 0-shot
vs few-shot, demonstration **selection** (random vs similarity-retrieved vs
hard-case), and whether the model is asked for a bare label or label +
rationale.

### 5.3 Beyond pairwise: "Match, Compare, or Select?" (ComEM, COLING 2025)

A key limitation of naive LLM-EM is the **pairwise binary paradigm**: judging
each candidate pair independently ignores **global consistency** among records.
Wang et al. [11] taxonomise three LLM strategies:

- **Matching** — the standard binary "do A and B match?" (one call per pair).
- **Comparing** — "which of B₁, B₂ better matches A?" (pairwise preference).
- **Selecting** — "which of {B₁…Bₖ} (if any) matches A?" (set-level, injects
  record interactions and enforces at-most-one-match consistency).

Their **ComEM** compound framework composes strategies across models (cheap
model to compare/prune, strong model to select) for better
effectiveness *and* cost, validated on 8 ER datasets × 10 LLMs. Takeaway for a
framework: **the assignment stage should be able to consume set-level LLM
judgements, not only independent pairwise scores.**

### 5.4 The cost problem and small-model responses

LLM matching is accurate but expensive: a cross-encoder call per candidate pair,
priced per token, with few-shot prompts costing **1.3×–11×** the tokens of
zero-shot [see 10, and the LLM-EM cost literature]. Two responses:

- **AnyMatch** (Zhang et al., 2024) [12] — fine-tune a **small** LM (GPT-2) for
  **zero-shot** EM via transfer learning plus data selection/augmentation
  (AutoML-selected hard pairs, attribute-level examples, label-balance control).
  Result (verify): within **4.4% average F1** of GPT-4-based MatchGPT while using
  ~**4 orders of magnitude fewer parameters** and **~3,899× lower inference
  cost** per 1k tokens. Strong argument that a small fine-tuned model can sit in
  the cheap tier of a cascade.
- **Cascades** (the dominant pattern) — classical/bi-encoder blocking →
  bi-encoder or small-LM scoring → LLM re-rank only on the top-k hardest
  candidates. This is how you get LLM-level accuracy at tractable cost, and it
  is the pattern `equate` should make idiomatic (§7).

---

## 6. Where learned methods beat classical — and where they do not

This is the crux for a *general* framework: do not assume a heavy learned matcher
is always better. The evidence:

### 6.1 Structured & clean data → classical is competitive

DeepMatcher's own finding [1]: on **structured** EM, deep learning does **not**
beat Magellan's feature-engineered classical learner; the win appears only on
**textual** and **dirty** data. For clean, well-aligned, short-attribute tables,
TF-IDF/edit-distance features + a random forest are cheap, interpretable, and
essentially as good. `equate`'s TF-IDF default is a *reasonable* production
choice for this regime, not a placeholder.

### 6.2 Textual, dirty, heterogeneous, or low-label data → learned wins

- **Long free text / dirty (misplaced) attributes:** attention/transformer
  models win because they align tokens without depending on clean schema [1][4].
- **Low-label / zero-label:** self-supervised (Sudowoodo [7], DeepBlocker [6])
  and LLM/ICL methods [9][10] dominate when labels are scarce or absent.
- **Unseen entities / distribution shift:** LLMs generalise better than
  fine-tuned PLMs, which memorise training entities [10].
- **Heterogeneous / generalised matching (GEM):** matching across structured,
  semi-structured, and textual formats — the **Machamp** benchmark [16] —
  needs the flexible serialisation of transformer/LLM matchers; classical
  schema-aware features struggle.

### 6.3 A caution: many "wins" are on easy benchmarks

**A Critical Re-evaluation of Benchmark Datasets for (Deep) Learning-Based
Matching Algorithms** (Papadakis et al., arXiv 2023 → ICDE 2024) [17] shows that
several classic EM benchmarks are **nearly linearly separable** — the gap
between the best linear and best non-linear matcher is small, and the best
learned matcher is already close to the oracle. Implication: reported deep/LLM
gains are sometimes inflated by easy data. `equate` should therefore ship an
**evaluation harness with held-out, unseen-entity splits** and encourage users
to measure whether a heavy matcher actually earns its cost on *their* data.

### 6.4 The modern benchmark landscape (know these names)

- **Magellan / DeepMatcher suite** — the structured/textual/dirty datasets of
  §2.2; still the default, but see §6.3.
- **WDC Products** (Peeters, Der, Bizer, 2023) [15] — a *multi-dimensional*
  product-matching benchmark from 2020 schema.org data (3,259 e-shops; 11,715
  offers; 2,162 entities) that varies three axes independently: amount of
  **corner cases**, **generalisation to unseen entities**, and **development-set
  size**; offers both pairwise and multi-class formulations. Designed to expose
  the generalisation gap §6.2/§6.3 talk about.
- **Machamp** (Wang et al., CIKM 2021) [16] — the **Generalized Entity
  Matching (GEM)** benchmark: 7 tasks matching across structured, semi-
  structured, and textual formats.
- **Alaska** — a product-matching benchmark used in ER challenges (heterogeneous
  schemas, many sources).

---

## 7. Design implications for `equate`

`equate`'s existing shape — `obj_to_vect` + `similarity_func` →
`similarity_matrix` → `matcher` — is already the field's pipeline. The
literature tells us how to *generalise the extension points* so that everything
from edit-distance to an LLM re-ranker slots in without rewrites. Concrete
recommendations:

1. **Make blocking-then-matching-then-assignment three explicit, swappable
   stages.** Formalise the sparse `similarity_matrix` as a `Blocker` Protocol
   (`candidates(keys, values) -> Iterable[(k, v)]` or a sparse matrix) distinct
   from a `Matcher`/scorer stage, distinct from an `Assigner` (greedy /
   Hungarian / threshold / cluster). This mirrors §4 and every surveyed system
   [13][14], and lets users mix a cheap blocker with an expensive matcher.

2. **Elevate the bi-encoder vs. cross-encoder split to a first-class abstraction
   (SSOT for scoring).** Define two Protocols:
   - `Embedder` (`fit(corpus)?; encode(objs) -> vectors`) — the bi-encoder path;
     covers TF-IDF (default), Sentence-BERT, Sudowoodo/DeepBlocker encoders, LLM
     embedding endpoints. Because outputs are vectors, this path *is*
     blockable via ANN.
   - `PairScorer` (`score(key, value) -> float` and a **batched**
     `score_pairs(pairs) -> floats`) — the cross-encoder path; covers Ditto-style
     joint models and LLM pair-prompts that do **not** factor through independent
     vectors.
   `similarity_func` composes an `Embedder` into a `PairScorer`; a cross-encoder
   is a `PairScorer` with no `Embedder`. This single split organises cost,
   blockability, and dependency weight.

3. **Cascade / re-rank as a built-in composition.** Provide a
   `cascade([blocker, cheap_scorer, expensive_scorer], top_k=…)` combinator so
   the idiomatic pipeline is: classical/bi-encoder blocking → bi-encoder or
   small-LM scoring → LLM re-rank on top-k. This is how the field reconciles
   LLM accuracy with cost [11][12]; it should be one call, not user glue.

4. **Keep heavy learners behind optional-dependency extras with dependency
   injection.** Default install stays lightweight (TF-IDF + edit distance, which
   §6.1 shows is genuinely competitive on structured/clean data). Ship
   `equate[deep]` (torch, transformers, sentence-transformers, faiss) and
   `equate[llm]` (LLM client) as extras. Inject the model/client, never
   hard-import it. Use `check_requirements`-style guidance when an extra is
   missing. This is the progressive-disclosure boundary: simple stays simple,
   powerful stays possible.

5. **Support unsupervised/self-supervised *fitting* of embedders.** The `fit`
   hook already implicit in TF-IDF should be a Protocol method, so contrastive
   corpus-fitting à la Sudowoodo/DeepBlocker [6][7] is a drop-in `Embedder` that
   trains on the user's unlabelled data — the strongest low-label strategy.

6. **Add record serialisation to bridge structured records to the string
   primitives.** EM transformers serialise records as
   `COL <attr> VAL <value> …` [4]; adopt a pluggable `serialize(record)` step so
   multi-attribute / semi-structured / heterogeneous (GEM/Machamp [16]) inputs
   reduce to the string inputs `equate` already handles, and so per-attribute
   weighting/highlighting (Ditto's domain-knowledge injection) is expressible.

7. **Let the assigner consume set-level judgements, not only pairwise scores.**
   To support LLM "select-from-candidates" and clustering-for-dedup [11], the
   `Assigner` interface should accept either a score matrix *or* a callable that,
   given a key and its candidate set, returns the chosen match(es). This is where
   **global consistency** (one-to-one, transitive closure) is enforced — keep it
   decoupled from scoring. `equate`'s greedy-vs-optimal `matcher` distinction is
   the right seed; add Hungarian (optimal one-to-one) and connected-components
   clustering (dedup) as named strategies.

8. **Cache and batch aggressively; account for cost.** Embeddings and LLM calls
   are the expensive resource. Provide transparent caching (`dol.cache_this`) on
   `Embedder.encode` and `PairScorer.score_pairs`, batched APIs everywhere, and
   an optional cost/latency accounting hook so a cascade can budget (few-shot
   prompts cost 1.3×–11× zero-shot tokens [10]).

9. **Ship an evaluation harness with hard, unseen-entity splits.** Given §6.3
   [17], provide P/R/F1 on labelled pairs, threshold calibration, and
   *generalisation* splits (unseen entities) so users can empirically decide
   whether a deep/LLM matcher beats the TF-IDF default on *their* data before
   paying for it. Bundle loaders for the standard suites (Magellan/DeepMatcher,
   WDC Products [15], Machamp [16]) as optional test fixtures.

10. **Default to interpretable-and-cheap; make powerful-and-expensive opt-in.**
    The overarching lesson of §6: there is no universally best matcher. `equate`
    should encode "strategy, not policy" — TF-IDF/edit-distance defaults that
    work out of the box, with deep encoders, cross-encoders, and LLM re-rankers
    as documented strategy objects selected per dataset regime.

---

## 8. Summary table of methods

| Method | Year / venue | Encoder type | Supervision | Where it fits `equate` |
|---|---|---|---|---|
| Magellan [3] | 2016 PVLDB | classical features | supervised | Blocker + feature-`PairScorer` precedent |
| DeepMatcher [1] | 2018 SIGMOD | attribute RNN/attention (cross) | supervised (many labels) | `PairScorer` |
| DeepER [2] | 2018 | LSTM + word emb (bi/cross) | supervised | `Embedder`/`PairScorer` |
| Ditto [4][5] | 2020 VLDB / 2023 VLDBJ | transformer cross-encoder | supervised (fewer labels) | `PairScorer` (`equate[deep]`) |
| Sentence-BERT [18] | 2019 EMNLP | Siamese bi-encoder | supervised (NLI) | `Embedder` for blocking |
| DPR [19] | 2020 EMNLP | dual-encoder | supervised | dense-retrieval `Blocker` |
| DeepBlocker [6] | 2021 VLDB | Siamese bi-encoder | **self-supervised** | `Blocker` (`equate[deep]`) |
| EMBER [8] | 2022 VLDB | transformer bi-encoder | weak/self-sup | keyless-join `Blocker`+matcher |
| Sudowoodo [7] | 2023 ICDE | contrastive bi-encoder | **self-supervised** | fittable `Embedder` |
| FM few-shot [9] | 2022 VLDB | LLM cross-encoder | in-context | `PairScorer` (`equate[llm]`) |
| LLM-EM [10] | 2025 EDBT | LLM cross-encoder | 0/few-shot | `PairScorer` (`equate[llm]`) |
| ComEM [11] | 2025 COLING | LLM (match/compare/**select**) | 0/few-shot | set-level `Assigner` + `PairScorer` |
| AnyMatch [12] | 2024 | small LM (GPT-2) | zero-shot (transfer) | cheap-tier `PairScorer` |

---

## References

[1] Mudgal, Li, Rekatsinas, Doan, Park, Krishnan, Deep, Arcaute, Raghavendra.
"Deep Learning for Entity Matching: A Design Space Exploration." *SIGMOD* 2018.
[dl.acm.org/doi/abs/10.1145/3183713.3196926](https://dl.acm.org/doi/abs/10.1145/3183713.3196926)
· [PDF](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)

[2] Ebraheem, Thirumuruganathan, Joty, Ouzzani, Tang. "DeepER — Deep Entity
Resolution." *arXiv:1710.00597* / PVLDB 2018.
[arxiv.org/abs/1710.00597](https://arxiv.org/abs/1710.00597)

[3] Konda, Das, Suganthan G.C., Doan, et al. "Magellan: Toward Building Entity
Matching Management Systems." *PVLDB* 9(12), 2016.
[dl.acm.org/doi/10.14778/2994509.2994535](https://dl.acm.org/doi/10.14778/2994509.2994535)

[4] Li, Li, Suhara, Doan, Tan. "Deep Entity Matching with Pre-Trained Language
Models (Ditto)." *PVLDB* 14(1), 2020.
[dl.acm.org/doi/10.14778/3421424.3421431](https://dl.acm.org/doi/10.14778/3421424.3421431)
· [arxiv.org/abs/2004.00584](https://arxiv.org/abs/2004.00584)

[5] Li, Li, Suhara, Doan, Tan. "Effective Entity Matching with Transformers."
*The VLDB Journal* 32, 2023 (Ditto journal extension).
[link.springer.com/article/10.1007/s00778-023-00779-z](https://link.springer.com/article/10.1007/s00778-023-00779-z)

[6] Thirumuruganathan, Li, Tang, Ouzzani, et al. "Deep Learning for Blocking in
Entity Matching: A Design Space Exploration (DeepBlocker)." *PVLDB* 14(11),
2021.
[dl.acm.org/doi/abs/10.14778/3476249.3476294](https://dl.acm.org/doi/abs/10.14778/3476249.3476294)

[7] Wang, Li, Wang. "Sudowoodo: Contrastive Self-supervised Learning for
Multi-purpose Data Integration and Preparation." *ICDE* 2023.
[arxiv.org/abs/2207.04122](https://arxiv.org/abs/2207.04122)
· [code](https://github.com/megagonlabs/sudowoodo)

[8] Suri, Ilyas, Ré, Rekatsinas. "Ember: No-Code Context Enrichment via
Similarity-Based Keyless Joins." *PVLDB* 15(3), 2022.
[arxiv.org/abs/2106.01501](https://arxiv.org/abs/2106.01501)
· [PDF](http://www.vldb.org/pvldb/vol15/p699-suri.pdf)

[9] Narayan, Chami, Orr, Ré. "Can Foundation Models Wrangle Your Data?" *PVLDB*
16(4), 2022.
[arxiv.org/abs/2205.09911](https://arxiv.org/abs/2205.09911)
· [PDF](https://www.vldb.org/pvldb/vol16/p738-narayan.pdf)

[10] Peeters, Steiner, Bizer. "Entity Matching using Large Language Models."
*EDBT* 2025 (arXiv 2023).
[arxiv.org/abs/2310.11244](https://arxiv.org/abs/2310.11244)

[11] Wang, Li, et al. "Match, Compare, or Select? An Investigation of Large
Language Models for Entity Matching (ComEM)." *COLING* 2025.
[arxiv.org/abs/2405.16884](https://arxiv.org/abs/2405.16884)
· [code](https://github.com/tshu-w/ComEM)

[12] Zhang et al. "AnyMatch — Efficient Zero-Shot Entity Matching with a Small
Language Model." *arXiv:2409.04073*, 2024.
[arxiv.org/abs/2409.04073](https://arxiv.org/abs/2409.04073)

[13] Christophides, Efthymiou, Palpanas, Papadakis, Stefanidis. "An Overview of
End-to-End Entity Resolution for Big Data." *ACM Computing Surveys* 53(6), 2020.
[dl.acm.org/doi/abs/10.1145/3418896](https://dl.acm.org/doi/abs/10.1145/3418896)
· [PDF](https://hal.science/hal-02955445/file/CSUR5306-127_LR.pdf)

[14] Papadakis, Skoutas, Thanos, Palpanas. "Blocking and Filtering Techniques
for Entity Resolution: A Survey." *ACM Computing Surveys* 53(2), 2020.
[dl.acm.org/doi/abs/10.1145/3377455](https://dl.acm.org/doi/abs/10.1145/3377455)

[15] Peeters, Der, Bizer. "WDC Products: A Multi-Dimensional Entity Matching
Benchmark." *arXiv:2301.09521* / EDBT, 2023.
[arxiv.org/abs/2301.09521](https://arxiv.org/abs/2301.09521)
· [dataset](https://webdatacommons.org/largescaleproductcorpus/wdc-products/)

[16] Wang, Li, Wang, et al. "Machamp: A Generalized Entity Matching Benchmark."
*CIKM* 2021.
[arxiv.org/abs/2106.08455](https://arxiv.org/abs/2106.08455)
· [dl.acm.org/doi/10.1145/3459637.3482008](https://dl.acm.org/doi/10.1145/3459637.3482008)

[17] Papadakis, et al. "A Critical Re-evaluation of Benchmark Datasets for
(Deep) Learning-Based Matching Algorithms." *arXiv:2307.01231* / ICDE 2024.
[arxiv.org/abs/2307.01231](https://arxiv.org/abs/2307.01231)

[18] Reimers, Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks." *EMNLP-IJCNLP* 2019.
[arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
· [aclanthology.org/D19-1410](https://aclanthology.org/D19-1410/)

[19] Karpukhin, Oğuz, Min, Lewis, et al. "Dense Passage Retrieval for
Open-Domain Question Answering." *EMNLP* 2020.
[arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
· [aclanthology.org/2020.emnlp-main.550](https://aclanthology.org/2020.emnlp-main.550/)
