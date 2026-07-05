# LLM-Based Matching & Modern Retrieval Embeddings (2024–2025)

**Abstract.** Round-1 of this corpus (docs `04-featurization-and-representation.md`
and `06-deep-learning-and-llm-entity-matching.md`) took the story up to
Sentence-BERT, Ditto, and the first GPT-3-few-shot entity-matching results. This
round-2 note brings both halves current. On the **cross-encoder / generative**
side it surveys the 2023–2025 wave of LLM entity matchers — the Peeters–Bizer
*MatchGPT* line, the instruction-tuned local model **Jellyfish**, the tiny
**AnyMatch** (GPT-2) matcher, compound/agentic systems (**BoostER**, **ComEM**,
**KcMF**) — plus the prompting design space (serialization, few-shot selection,
chain-of-thought, **batch prompting**, retrieval-augmented matching, and
structured-output / function-calling matchers). On the **bi-encoder** side it
maps the modern open and API retrieval embedders that have displaced SBERT as
the default featurizer/blocker — **E5**, **BGE / BGE-M3**, **GTE**,
**Nomic-embed**, **Jina v3**, **OpenAI text-embedding-3**, **Cohere embed v3**,
**Voyage** — with their dimensions, context lengths, **Matryoshka** truncation,
quantization, licensing, and **MTEB / MMTEB** standing. It closes with concrete
recommendations for `equate`'s default text featurizer(s) and the
optional-dependency boundary. The through-line remains the round-1 split:
**cheap independently-embeddable scorers (bi-encoders) drive blocking; expensive
joint scorers (LLMs/cross-encoders) drive final decisions**; a cascade
reconciles the two.

---

## 1. Scope and where this fits

This document is deliberately narrow: it updates two fast-moving fronts and
avoids re-deriving fundamentals covered elsewhere in the corpus.

- **Encoder taxonomy, the blocking→matching→assignment pipeline, and the classic
  systems (DeepMatcher, Ditto, Sudowoodo, the 2022 GPT-3 pivot)** are in
  `06-deep-learning-and-llm-entity-matching.md`. Read it first; this note assumes
  the **bi-encoder vs. cross-encoder** distinction it establishes.
- **Representation kinds, metrics, normalization, ANN/LSH indexing, and the
  Matryoshka concept** are in `04-featurization-and-representation.md`. This note
  extends §3.3–3.4 of that doc (the "MTEB era" and API embeddings) with the
  specific 2024–2025 models.
- **Blocking/candidate generation** is in
  `02-blocking-and-scalable-candidate-generation.md`; **assignment/global
  consistency** in `03-assignment-and-graph-matching.md`; **the Python matching
  ecosystem** in `09-python-ecosystem-landscape.md`; **the consolidated design
  program** in `10-design-implications-for-equate.md`.

Two orthogonal questions organize everything below:

1. **What is the default text *featurizer/blocker* in 2025?** (§3 — the
   bi-encoder tier: modern retrieval embedders.)
2. **When and how should an *LLM* make the final match decision?** (§4 — the
   cross-encoder/generative tier: LLM-EM systems and prompting.)

---

## 2. The two tiers, restated for 2025

| Tier | Shape | Cost per pair | Precomputable? | Blockable? | 2025 exemplars |
|---|---|---|---|---|---|
| **Bi-encoder** (embedder) | encode each object independently → vector; score = cosine/dot | O(1) after O(n) encode | yes (cache vectors) | yes (ANN) | E5, BGE-M3, GTE, Nomic, Jina, text-embedding-3, Cohere v3, Voyage |
| **Cross-encoder / generative** (LLM matcher) | feed the *pair* (or a *batch* of pairs) into one model → decision | O(n·m) forward passes, priced per token | no | no (only re-rank a candidate set) | MatchGPT (GPT-4), Jellyfish, AnyMatch, ComEM, BoostER |

The economics are the same as SBERT-vs-Ditto in doc 06, only the constants
moved: embedders got much stronger (LLM-quality retrieval from a single forward
pass), and the "cross-encoder" became a *generative* model you prompt rather than
a classifier you fine-tune. The dominant production pattern is unchanged and
now more important because LLM calls are expensive: **embed-and-block with a
bi-encoder, then LLM-re-rank only the top-k survivors** (§4.8).

---

## 3. Modern retrieval embedders (the bi-encoder tier)

Between 2022 and 2025 the open embedding field converged on one recipe and then
raced up a shared leaderboard. This section is the practical "which model, and
why" for `equate`'s default featurizer.

### 3.1 The recipe that unified them

Almost every strong 2023–2025 text embedder is a **two-stage contrastive**
model, popularized by **E5** [1]:

1. **Weakly-supervised contrastive pre-training** on *billions* of naturally
   occurring text pairs (title–body, question–answer, post–comment, query–click),
   using the InfoNCE loss with in-batch negatives.
2. **Supervised fine-tuning** on a smaller set of high-quality labeled pairs
   (NLI, MS MARCO, retrieval datasets) with **mined hard negatives** and often
   **cross-encoder knowledge distillation**.

Two later ingredients define the 2024 frontier:

- **LLM-generated synthetic data.** *Improving Text Embeddings with Large
  Language Models* [3] uses a proprietary LLM to synthesize retrieval tasks
  across 93 languages, then fine-tunes a **decoder-only LLM backbone**
  (`e5-mistral-7b-instruct`) with contrastive loss — reaching MTEB ≈ **66.6**
  and topping the leaderboard at release with *only synthetic + minimal public
  data*.
- **Instruction / task prefixes.** Models are told the task ("Represent this
  sentence for retrieval: …"). E5 requires literal `query:` / `passage:`
  prefixes; instruct variants (mE5-instruct, gte-Qwen2-instruct, Jina v3 task
  LoRA) take a task string. **This is an operational gotcha for a framework: the
  featurizer must know its model's required prefix, or retrieval quality
  silently degrades.**

### 3.2 Open-weight families

- **E5** (Microsoft) [1][2][3] — the reference two-stage recipe.
  `e5-large-v2` (335M, **1024-d**, 512 tokens, MIT); `multilingual-e5-large`
  (**1024-d**, ~512 tokens, 94 languages, MIT) [2]; `e5-mistral-7b-instruct`
  (LLM backbone, **4096-d**, long context, MIT) [3]. Excellent, permissively
  licensed, widely deployed defaults.
- **BGE** (BAAI, *FlagEmbedding*) [4] — `bge-large-en-v1.5` (**1024-d**, 512
  tokens, MIT) was a 2023 SOTA general embedder. Its successor **BGE-M3** [4] is
  the standout for a *general* matching framework: a single model that is
  **multi-lingual** (100+ languages), **multi-granular** (up to **8192 tokens**),
  and **multi-functional** — one forward pass emits *three* comparison
  primitives at once:
  - a **dense** vector (1024-d) for semantic cosine matching,
  - **learned sparse** lexical weights (BM25-like but learned, good for exact
    token/SKU overlap), and
  - **multi-vector / ColBERT-style** token embeddings for late-interaction
    re-ranking.
  M3 is effectively "a bi-encoder, a sparse retriever, and a cheap cross-encoder
  in one model," which maps unusually cleanly onto a matcher's blocking + scoring
  stages. MIT licensed.
- **GTE** (Alibaba) [6] — multi-stage contrastive on BERT backbones.
  `gte-large-en-v1.5` (**1024-d**, **8192 tokens**, Apache-2.0);
  `gte-multilingual-base` (8192 tokens). The LLM-backbone **`gte-Qwen2-7B-instruct`**
  (**3584-d**, 32k context, Apache-2.0) **topped the English MTEB leaderboard in
  2024**. Apache-2.0 makes the GTE line attractive where BGE/E5's MIT is fine and
  Jina's non-commercial license is not.
- **Nomic-embed** [7] — `nomic-embed-text-v1.5` (137M, **768-d**, **8192
  tokens**, **Apache-2.0**). The distinguishing claim is **full
  reproducibility**: open weights, open training code, and an open 235M-pair
  data loader. It reports beating `text-embedding-ada-002` and matching/beating
  `text-embedding-3-small` on short- and long-context tasks, and supports
  **Matryoshka** truncation down to 64-d. The best "audit-friendly, cheap,
  self-hostable, commercially safe" default.
- **Jina embeddings v3** [8] — 570M, **1024-d** with **Matryoshka down to 32-d**,
  **8192 tokens**, multilingual. Its novelty is **task-specific LoRA adapters**
  (retrieval-query, retrieval-passage, separation/clustering, classification,
  text-matching) selected at inference — the same base weights specialized per
  matching use. **License caveat: CC-BY-NC-4.0 (non-commercial)** — verify before
  shipping in a commercial product; this is the main reason a framework should
  not hard-default to it.

### 3.3 API embedders (proprietary, zero local compute)

- **OpenAI text-embedding-3** [9] — `-small` (**1536-d**) and `-large`
  (**3072-d**), both **Matryoshka**-trainable via a `dimensions` parameter, ~8191
  token context. Reported MTEB(English) average rose from **61.0**
  (`ada-002`) to **62.3** (`-small`) and **64.6** (`-large`); MIRACL
  (multilingual) from 31.4 to 44.0 / 54.9. `-large` truncated to 256-d still
  beats the older 1536-d `ada-002` on MTEB — the canonical demonstration of
  Matryoshka's value.
- **Cohere embed v3** [10] — English and multilingual (**1024-d**). Notable for
  **compression-aware training**: the model natively emits **int8** and **binary**
  embeddings. Int8 gives ~4× memory savings retaining ~99.99% of search quality;
  binary compresses a 1024-float (4096-byte) vector to **128 bytes** (32×) with
  small quality loss — decisive at billion-vector scale.
- **Voyage** [11] — `voyage-3` (**1024-d**, **32k context**) and
  `voyage-3-large` (**2048-d**, Matryoshka {256, 512, 1024, 2048}, 32k context)
  with binary/int8 quantization. Marketed as beating `text-embedding-3-large`
  across domains at lower dimension and cost; strong on code and long documents.

API embedders are, for `equate`, *just another `Featurizer` behind an optional
extra* — but they introduce **latency, batching, rate limits, per-token cost, and
non-determinism** that in-process models do not, so the featurizer contract must
tolerate batched/async/cached implementations (per doc 04 §3.4 and §9.3).

### 3.4 Cross-cutting knobs a framework must expose

- **Matryoshka Representation Learning (MRL)** [14] — nested prefixes let you
  truncate a vector to the first *k* dims for cheaper storage/ANN with graceful
  quality loss. Now standard: text-embedding-3, Nomic (→64), Jina v3 (→32),
  Voyage. `equate` should let a `Featurizer` declare `truncatable_to` and accept
  a `dimensions=k` parameter (doc 04 §9.6).
- **Quantization** — int8 (≈4×) and binary (≈32×, Hamming-comparable) shrink
  index cost dramatically; Cohere v3 and Voyage train for it, but *any* embedding
  can be post-hoc binarized. This connects the vector tier to the bit-string /
  Hamming machinery in doc 04 §6.1.
- **Max sequence length** — jumped from SBERT's 256–512 tokens to **8192**
  (BGE-M3, GTE, Nomic, Jina) and **32k** (Voyage, LLM-backbone models). Long
  context matters for record serialization: a full multi-attribute record or a
  document no longer needs truncation (contrast Ditto's TF-IDF summarization
  hack, doc 06 §3.2).
- **Multi-functionality** — BGE-M3's dense+sparse+multi-vector output is the
  clearest signal that "one representation, one metric" is too narrow; a general
  featurizer interface should allow a model to expose several comparison
  primitives (doc 04 §8).
- **Instruction/prefix requirement** — see §3.1; must live *with* the featurizer.

### 3.5 The benchmark to point users at: MTEB → MMTEB

- **MTEB** (Muennighoff et al., EACL 2023) [12] — 8 task types (bitext mining,
  classification, pair classification, clustering, reranking, retrieval, STS,
  summarization), the standard vocabulary for embedding quality. Doc 04 §3.3
  introduces it.
- **MMTEB** (Enevoldsen et al., ICLR 2025) [13] — the massively expanded
  successor (500+ tasks, 250+ languages, plus code/long-doc suites); the live HF
  leaderboard now defaults to it.

**How to read it for matching (caveats):** MTEB's *retrieval* and *pair
classification / STS* columns predict blocking and pairwise-scoring quality
better than the headline average. **Do not chase the leaderboard blindly**:
(a) scores drift and models overfit MTEB; (b) matching often hinges on
*domain-specific, jargon-heavy* strings (part numbers, catalog titles) where
general MTEB rank does not transfer — the same jargon failure mode LLMs show
(doc 06 §5.1); (c) dimension/latency/license usually matter more than the last
MTEB point. Point users at MTEB to *choose a class* of model, then evaluate on
*their* data (doc 06 §6.3).

### 3.6 Comparison table

| Model | Params | Dense dim | Max ctx | MRL | License | Note |
|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` (SBERT baseline) | 22M | 384 | 256 | no | Apache-2.0 | round-1 cheap default |
| `e5-large-v2` [1] | 335M | 1024 | 512 | no | MIT | strong, permissive |
| `multilingual-e5-large` [2] | 560M | 1024 | ~512 | no | MIT | 94 languages |
| `e5-mistral-7b-instruct` [3] | 7B | 4096 | 32k | no | MIT | LLM backbone, MTEB≈66.6 |
| `bge-large-en-v1.5` [4] | 335M | 1024 | 512 | no | MIT | 2023 SOTA |
| **`bge-m3`** [4] | 568M | 1024 | **8192** | no | MIT | **dense+sparse+multi-vector**, 100+ langs |
| `gte-large-en-v1.5` [6] | 434M | 1024 | 8192 | no | Apache-2.0 | long context |
| `gte-Qwen2-7B-instruct` [6] | 7B | 3584 | 32k | no | Apache-2.0 | topped MTEB 2024 |
| `nomic-embed-text-v1.5` [7] | 137M | 768 | 8192 | →64 | **Apache-2.0** | fully reproducible |
| `jina-embeddings-v3` [8] | 570M | 1024 | 8192 | →32 | **CC-BY-NC** | task LoRA; non-commercial |
| `text-embedding-3-small` [9] | API | 1536 | ~8191 | yes | proprietary | 62.3 MTEB |
| `text-embedding-3-large` [9] | API | 3072 | ~8191 | yes | proprietary | 64.6 MTEB |
| `Cohere embed v3` [10] | API | 1024 | ~512 | — | proprietary | native int8/binary |
| `voyage-3-large` [11] | API | 2048 | 32k | yes | proprietary | quantized, long-ctx |

*(MTEB numbers are English-MTEB averages at release and drift over time; treat
as ordinal, not exact — §3.5.)*

### 3.7 Recommended `equate` text-featurizer defaults

- **Core install (no heavy deps):** keep **character-n-gram TF-IDF** (doc 04
  §3.1) as the zero-dependency default — genuinely competitive on clean/structured
  name/SKU matching (doc 06 §6.1) and the honest "simple things simple" baseline.
- **`equate[embeddings]` default model:** **`bge-m3`** or **`multilingual-e5-large`**
  via `sentence-transformers`. Rationale: MIT license (commercial-safe), strong
  multilingual retrieval, long context; BGE-M3 additionally exposes sparse +
  multi-vector primitives that fit the blocking/re-rank stages. For a smaller,
  fully-reproducible, Apache-2.0 option, **`nomic-embed-text-v1.5`** (768-d,
  MRL→64).
- **`equate[api]`:** `text-embedding-3-small` (cheap, MRL) as the default hosted
  option; `voyage-3`/Cohere v3 as documented alternatives.
- **Do not hard-default to Jina v3** despite its quality — the CC-BY-NC license
  makes it a poor library default; offer it as an explicit opt-in.

---

## 4. LLM-based entity matching (the cross-encoder / generative tier)

Round-1 ended at "GPT-3 few-shot beats fine-tuned Ditto on some datasets"
(Narayan et al., 2022 [24], doc 06 §5.1) and the systematic Peeters–Steiner–Bizer
study [15]. Since then the field has (a) hardened *how* to prompt, (b) pushed the
model *local and small* for cost, and (c) moved *beyond independent pairwise
calls* toward compound/agentic and set-level matching.

### 4.1 MatchGPT — the Peeters–Bizer line (2023–2025)

**MatchGPT** is the name of the Mannheim group's code/benchmark line for LLM
entity matching. The 2023 note *Using ChatGPT for Entity Matching* [16] showed
ChatGPT robustly matches or beats fine-tuned RoBERTa/Ditto zero-shot. The
extended *Entity Matching using Large Language Models* (EDBT 2025) [15]
systematizes it across hosted (GPT-3.5/GPT-4) and locally-runnable open LLMs.
Load-bearing findings (verify against [15]):

- The **best LLMs need zero or a few examples to match PLMs fine-tuned on
  thousands** of pairs.
- LLM matchers are **more robust to unseen entities** (out-of-distribution) — the
  main weakness of fine-tuned PLMs, which memorize their training entities.
- **There is no single best prompt**; it must be tuned per model×dataset — prompt
  engineering still matters.
- GPT-4 can emit **structured natural-language explanations** and even **diagnose
  labeling errors** from explanations of wrong decisions.

MatchGPT (with GPT-4) is now the reference "strong but expensive" LLM matcher
that cheaper systems benchmark against (e.g., AnyMatch [18]).

### 4.2 Jellyfish — instruction-tuned *local* LLM data preprocessor (EMNLP 2024)

**Jellyfish** (Zhang, Dong, Xiao, Oyamada) [17] instruction-tunes open base
models — **Mistral-7B → Jellyfish-7B, Llama-3-8B → Jellyfish-8B,
OpenOrca-Platypus2-13B → Jellyfish-13B** — on a mix of four data-preprocessing
tasks: **error detection, data imputation, schema matching, and entity matching**.
Construction uses *data configuration, knowledge injection, and reasoning-data
distillation*. Results (verify against [17]): competitive with GPT-3.5/GPT-4 on
EM and imputation, **runs on a single local, low-cost GPU** (data stays
in-house), generalizes to unseen tasks, and barely degrades the base model's
general NLP ability. Jellyfish is the archetype of the **"self-hostable,
privacy-preserving, no-per-token-cost" LLM matcher** — the natural occupant of
`equate[llm-local]`.

### 4.3 AnyMatch — a *tiny* zero-shot matcher (2024)

**AnyMatch** (Zhang et al.) [18] fine-tunes **GPT-2 (124M params, ~260 MB GPU
RAM)** for **zero-shot** EM via transfer learning plus data selection/augmentation
(AutoML-selected hard pairs, generated attribute-level examples, label-balance
control). Reported headline (verify against [18], nine benchmark datasets):
average F1 **within 4.4%** of **MatchGPT-with-GPT-4** while using **~4 orders of
magnitude fewer parameters** at **3,899× lower inference cost** per 1k tokens, and
**>690k tokens/s throughput (≈25×** competing small LLMs**)**. AnyMatch is the
strongest evidence that a *small fine-tuned model* can occupy the cheap tier of a
cascade at near-LLM accuracy — a concrete `PairScorer` for `equate`'s cheap tier.

### 4.4 Beyond independent pairwise calls: compound & set-level matching

Naive LLM-EM judges each candidate pair independently, ignoring **global
consistency** (a query record should match *at most one* entity). Three lines fix
this:

- **ComEM** (*Match, Compare, or Select?*, COLING 2025) [20] — taxonomizes
  **Matching** (binary "do A,B match?"), **Comparing** ("which of B₁,B₂ better
  matches A?"), and **Selecting** ("which of {B₁…Bₖ}, if any, matches A?"). Its
  compound framework composes a cheap model to prune/compare and a strong model
  to select, improving both accuracy and cost across 8 datasets × 10 LLMs. The
  design lesson for `equate` (doc 06 §5.3): **the assigner must be able to consume
  set-level LLM judgements, not only a score matrix.**
- **BoostER** (The Web Conference 2024) [19] — does **not** ask the LLM to match
  every pair. It runs a cheap matcher first, then **optimally selects a small set
  of uncertain matching questions** to pose to the LLM and uses the answers to
  **refine the posterior distribution** over ER results. This is
  active-learning-flavored LLM use (cross-link `08-interactive-active-learning-and-hitl.md`):
  spend LLM calls only where they most reduce uncertainty.
- **KcMF** (2024) [23] — a **fine-tuning-free** unified framework for *both*
  schema matching and entity matching (cross-link `07-schema-and-ontology-matching.md`),
  using **pseudo-code task decomposition** and knowledge statements to curb
  hallucination and task confusion, improving backbone F1 by ~6–18% across
  datasets without any training.

### 4.5 Prompting strategy design space

The prompt *is* the algorithm for an LLM matcher. The axes that measurably move
accuracy and cost:

- **Serialization.** How a record becomes text. The EM convention is Ditto's
  attribute tagging `COL <attr> VAL <value> …` (doc 06 §3.2); alternatives are
  raw concatenation, JSON, or natural-language templating. Serialization choice
  interacts with the model — no universal winner [15].
- **Zero-shot vs. few-shot & demonstration *selection*.** Few-shot helps, but
  *which* demonstrations dominate the gain: random < similarity-retrieved
  (nearest labeled pairs) < hard/covering-based selection [15][21]. Few-shot also
  costs **1.3×–11× the tokens** of zero-shot (doc 06 §5.4) — selection is a
  cost lever, not just accuracy.
- **Chain-of-thought / rationale.** Asking for a reason before the label can
  raise accuracy and yields explanations for auditing/error diagnosis [15], at
  extra output tokens.
- **Batch prompting.** Pack **multiple pairs into one prompt** to amortize the
  shared system prompt/demonstrations by 1/*b*. **BatchER** (Fan et al., ICDE
  2024) [21] reports **4×–7× cost savings** with *higher, more stable* accuracy
  via a **covering-based demonstration selection + question batching** design;
  the general technique is from *Batch Prompting* [22]. **Tradeoff:** cross-pair
  coupling in a batch can raise recall (more predicted matches) but also **false
  positives**, and very large batches degrade reliability — batch size is a
  tunable, not free.

### 4.6 Retrieval-augmented matching (RAG for EM)

Two distinct "RAG for matching" ideas, both directly relevant to `equate`:

1. **Retrieve demonstrations** — for each query pair, retrieve the most similar
   *labeled* examples (via the §3 bi-encoder) to fill the few-shot slots.
   Similarity-retrieved and covering-based demonstrations beat random ones
   [15][21]; this is a bi-encoder feeding the cross-encoder.
2. **Retrieve candidates, then verify** — the cascade itself is retrieval
   augmentation: embed-and-block to a small candidate set, then let the LLM
   verify only survivors. Recent work makes the blocking budget explicit for
   cost-efficient LLM EM (blocking-based RAG for entity matching, 2025) [26]; the
   design point is that **retrieval quality (recall) caps end-to-end quality**,
   so the embedder tier (§3) and the LLM tier co-determine accuracy.

Both reduce to `equate`'s **cascade** combinator (doc 06 §7.3): the embedder is
reused for *blocking* and for *demonstration retrieval*.

### 4.7 Structured output, function calling, and agentic matchers

- **Structured output.** A matcher should return a *parseable* verdict, not prose.
  OpenAI **Structured Outputs** (Aug 2024) [25] and open **constrained decoding**
  (Outlines, XGrammar, Guidance, llama.cpp grammars) force JSON-Schema-conformant
  output via a finite-state mask over tokens — turning an LLM into a reliable
  `{"match": bool, "confidence": float, "reason": str}` emitter. **Tradeoff
  (verify):** heavy format restriction can *slightly degrade* the model's
  reasoning [*Let Me Speak Freely?*, 27], so allow "reason-then-structured" or a
  light schema. For `equate` this makes the LLM `PairScorer` return a **typed,
  confidence-bearing** result rather than fragile string parsing.
- **Function calling / tool use.** The matcher can be given tools (lookup a
  canonical DB, normalize a unit, fetch an attribute) and *call* them mid-decision
  — useful for jargon-heavy domains where the model lacks grounding (the classic
  LLM-EM failure mode, doc 06 §5.1).
- **Agentic / compound-AI matchers.** The frontier composes several models and
  tools — cheap model prunes, strong model selects, tools ground, a verifier
  checks (ComEM [20] is an early compound instance; enterprise "compound AI"
  blueprints generalize it). For a framework the lesson is **composition, not a
  monolith**: expose blocker, scorer, tool-caller, and assigner as swappable
  stages so an "agentic matcher" is just a particular wiring.

### 4.8 Cost / accuracy / latency, and when LLMs beat fine-tuned PLMs

**When an LLM matcher is worth it:**

- **Low/zero labeled data** — no fine-tuning budget; zero/few-shot LLMs match
  PLMs trained on thousands of pairs [15][24].
- **Unseen entities / distribution shift** — LLMs generalize; fine-tuned PLMs
  memorize [15].
- **Heterogeneous / semi-structured / cross-format inputs** — flexible
  serialization handles what schema-aware features cannot (GEM/Machamp, doc 06
  §6.2).
- **Explanation/auditability required** — LLMs justify decisions [15].

**When a cheaper tier wins:**

- **Clean, structured, high-volume** data — char-n-gram TF-IDF / a fine-tuned PLM
  / AnyMatch is far cheaper and essentially as accurate (doc 06 §6.1).
- **Jargon-heavy domains without grounding** — LLMs degrade; add retrieval/tools
  or fall back to lexical features [24].
- **Tight latency/cost budgets at scale** — O(n·m) token-priced calls are fatal
  without blocking.

**The reconciling pattern (make it idiomatic in `equate`):**
`bi-encoder blocking (§3) → cheap scorer (TF-IDF / small PLM / AnyMatch) →
LLM re-rank on the top-k hardest, with batch prompting → set-level assigner`.
Cost knobs stack: blocking cuts *how many* pairs; batching cuts *tokens per pair*;
demonstration selection cuts *few-shot overhead*; small-model tiers cut *price
per call*; structured output cuts *parsing failures*.

### 4.9 LLM-EM systems at a glance

| System | Year / venue | What it is | Cost tier | Fits `equate` as |
|---|---|---|---|---|
| MatchGPT / Peeters–Bizer [15][16] | 2023 / EDBT 2025 | hosted+open LLM prompting study | high (GPT-4) | reference `PairScorer` (`equate[llm]`) |
| Narayan et al. [24] | 2022 VLDB | GPT-3 few-shot pivot | high | `PairScorer` |
| Jellyfish-7B/8B/13B [17] | 2024 EMNLP | instruction-tuned local LLM | medium (self-host) | `PairScorer` (`equate[llm-local]`) |
| AnyMatch [18] | 2024 | GPT-2 zero-shot matcher | **very low** | cheap-tier `PairScorer` |
| ComEM [20] | 2025 COLING | match/compare/**select** compound | mixed | set-level `Assigner` + `PairScorer` |
| BoostER [19] | 2024 WWW | LLM verifies *uncertain* pairs | low (few calls) | active-learning re-ranker |
| KcMF [23] | 2024 | fine-tuning-free SM+EM | medium | unified schema+entity `PairScorer` |
| BatchER [21] | 2024 ICDE | batch prompting for ER | low | prompting strategy for LLM `PairScorer` |

---

## 5. Design implications for `equate`

These extend, and do not repeat, the ten implications in doc 06 §7 and the
featurizer boundaries in doc 04 §9. New or sharpened for the 2025 landscape:

1. **Ship a *modern* default embedder behind `equate[embeddings]`, not SBERT.**
   The round-1 default (`all-MiniLM`/`all-mpnet`) is superseded. Default to a
   **MIT-licensed** long-context multilingual model — **`bge-m3`** or
   **`multilingual-e5-large`** — with **`nomic-embed-text-v1.5`** as the
   Apache-2.0, fully-reproducible small option. Keep char-n-gram TF-IDF as the
   *core* zero-dependency default (§3.7).

2. **Make license, dimension, context, prefix, and truncation *declared
   metadata* on the `Featurizer`.** A model card in code: `license`,
   `dim`, `max_seq_len`, `query_prefix`/`passage_prefix`, `truncatable_to`,
   `normalize`, and `output_kinds` (dense / sparse / multi-vector). This lets
   `equate` (a) refuse a non-commercial model in a commercial context, (b) apply
   the required E5-style prefix automatically, (c) offer `dimensions=k` MRL
   truncation, and (d) auto-pick a legal metric/index (doc 04 §9.1, §9.6). The
   silent-prefix bug (§3.1) is a real footgun a framework should eliminate.

3. **Model *multi-functional* embedders (BGE-M3) as a featurizer that emits
   several comparison primitives.** Allow one `Featurizer` to yield `{dense,
   sparse, multi_vector}`; the blocker consumes dense+sparse (ANN + inverted
   index), an optional re-ranker consumes multi-vector (late interaction). This
   is the cleanest concrete case for the "representations compose / one model,
   several metrics" point in doc 04 §8.

4. **First-class quantization on the vector path.** Support int8 and **binary**
   embeddings (native for Cohere v3 / Voyage, post-hoc for any model) with
   Hamming comparison, bridging the vector tier to the bit-string machinery in
   doc 04 §6.1. At scale this is a 4×–32× index-cost lever the framework should
   expose, not hide.

5. **Elevate the `LLM PairScorer` to a typed, structured-output strategy.**
   The LLM matcher should return a **schema-validated** `MatchDecision`
   (`match: bool`, `confidence: float`, `reason: str | None`) via structured
   output / constrained decoding [25], never raw-string parsing. Provide a
   `reason_then_answer` option given the format-restriction tradeoff [27]. Inject
   the client (`equate[llm]` / `equate[llm-local]`); never hard-import a provider.

6. **Make *prompting strategy* a swappable object on the LLM scorer.**
   Serialization, shot count, **demonstration selector** (random /
   similarity-retrieved via the §3 embedder / covering-based), CoT on/off, and
   **batch size** are all parameters, not hard-coded prompts [15][21][22]. The
   embedder used for blocking is *reused* as the demonstration retriever — one
   `Embedder`, two roles.

7. **Batch prompting as a built-in cost strategy.** Provide a
   `batch_prompt(pairs, batch_size=…)` mode on the LLM scorer that packs pairs and
   parses per-pair verdicts, with the documented recall/false-positive tradeoff
   surfaced as a knob [21]. This is a 4×–7× cost lever that users should not have
   to hand-roll.

8. **Let the assigner consume *set-level* LLM judgements.** To support ComEM-style
   "select-from-candidates" and BoostER-style uncertainty-targeted verification
   [19][20], the `Assigner` must accept either a score matrix *or* a callable
   `(key, candidates) -> chosen_match(es)`, where at-most-one-match consistency is
   enforced (doc 06 §7.7, cross-link doc 03).

9. **Bake the cascade in, with per-tier cost accounting.** The idiomatic pipeline
   (§4.8) — `embed-block → cheap scorer → LLM re-rank(top-k, batched) →
   set assigner` — should be one `cascade([...])` call with a cost/latency hook so
   a user can budget. AnyMatch [18] and Jellyfish [17] are the concrete
   cheap/self-host tiers; MatchGPT [15] the accurate tier.

10. **Optional-dependency map (extend doc 04 §9.3 / doc 06 §7.4):**
    - core: TF-IDF + edit distance (no heavy deps),
    - `equate[embeddings]` → `sentence-transformers` (BGE-M3 / E5 / Nomic),
    - `equate[api]` → `openai` / `cohere` / `voyageai` embedding clients,
    - `equate[ann]` → `faiss` / `hnswlib` (+ binary/int8 indexes),
    - `equate[llm]` → hosted LLM client (structured output),
    - `equate[llm-local]` → `transformers`/`vllm` for Jellyfish/AnyMatch-class
      models.
    Inject clients; guide missing deps with `check_requirements`-style errors
    naming the exact extra.

11. **Evaluate before paying.** Given MTEB drift (§3.5) and easy-benchmark
    inflation (doc 06 §6.3), ship an eval harness with **unseen-entity splits** and
    a **cost column** so users can see whether `bge-m3 + GPT-4-re-rank` actually
    beats `TF-IDF + AnyMatch` on *their* data and budget.

---

## 6. Glossary

- **Bi-encoder / cross-encoder** — see doc 06 §1.2. Bi-encoders (embedders) are
  the §3 tier; LLM matchers are the §4 cross-encoder/generative tier.
- **E5** — Microsoft's two-stage weakly-supervised contrastive embedder family;
  requires `query:`/`passage:` prefixes [1][2][3].
- **BGE / BGE-M3** — BAAI's FlagEmbedding models; **M3** is multi-lingual,
  multi-granular (8192 tokens), multi-functional (dense + learned-sparse +
  ColBERT multi-vector) [4].
- **GTE** — Alibaba's multi-stage-contrastive embedders; `gte-Qwen2-7B-instruct`
  (LLM backbone) topped MTEB in 2024 [6].
- **Nomic-embed** — fully reproducible (open weights + code + data), Apache-2.0,
  8192-token, 768-d with MRL→64 [7].
- **Jina v3** — multilingual, 8192-token, MRL→32, **task-specific LoRA adapters**;
  CC-BY-NC (non-commercial) [8].
- **text-embedding-3** — OpenAI API embedders (`-small` 1536-d, `-large` 3072-d)
  with `dimensions` (Matryoshka) truncation [9].
- **Cohere embed v3** — API embedder with native **int8/binary** compressed
  output [10].
- **Voyage** — API embedders (`voyage-3` 1024-d, `voyage-3-large` 2048-d MRL),
  32k context, quantized [11].
- **Matryoshka (MRL)** — nested-prefix embeddings; truncate to the first *k* dims
  without retraining [14].
- **Quantization (int8/binary)** — low-precision embeddings for 4×–32× smaller
  indexes; binary is Hamming-comparable.
- **MTEB / MMTEB** — the Massive (Multilingual) Text Embedding Benchmark; the
  standard leaderboard for embedder quality [12][13].
- **MatchGPT** — the Peeters–Bizer code/benchmark line for LLM entity matching;
  "MatchGPT-GPT-4" is the reference strong LLM matcher [15][16].
- **Jellyfish** — instruction-tuned local LLM (7B/8B/13B) for data preprocessing
  incl. entity & schema matching; self-hostable [17].
- **AnyMatch** — GPT-2-based (124M) zero-shot entity matcher; ~within 4.4% F1 of
  MatchGPT-GPT-4 at ~3,899× lower cost [18].
- **BoostER** — LLM verifies only *uncertain* candidate pairs and refines the ER
  posterior [19].
- **ComEM** — *Match / Compare / Select* compound LLM-EM; set-level, enforces
  at-most-one-match consistency [20].
- **KcMF** — fine-tuning-free unified schema+entity matching via pseudo-code task
  decomposition [23].
- **Batch prompting** — packing multiple pairs into one prompt to amortize the
  shared prompt (BatchER: 4×–7× cost cut) [21][22].
- **Retrieval-augmented matching (RAG for EM)** — retrieving similar labeled
  demonstrations and/or candidate records to condition the LLM decision [15][26].
- **Structured output / constrained decoding** — forcing JSON-Schema-conformant
  LLM output via a token-level grammar/FSM mask [25].
- **Serialization** — mapping a structured record to text (Ditto's
  `COL/VAL` tagging, JSON, or NL templating) for a text encoder/LLM (doc 06 §3.2).
- **Instruction / task prefix** — the required per-model input prefix or task
  string (E5 `query:`; instruct/LoRA task selectors); a silent-failure footgun if
  omitted (§3.1).

---

## References

[1] Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R.,
Wei, F. *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5).
arXiv:2212.03533, 2022.
[arxiv.org/abs/2212.03533](https://arxiv.org/abs/2212.03533)

[2] Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., Wei, F. *Multilingual
E5 Text Embeddings: A Technical Report.* arXiv:2402.05672, 2024.
[arxiv.org/abs/2402.05672](https://arxiv.org/abs/2402.05672)

[3] Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., Wei, F. *Improving Text
Embeddings with Large Language Models* (e5-mistral-7b-instruct). ACL 2024;
arXiv:2401.00368.
[arxiv.org/abs/2401.00368](https://arxiv.org/abs/2401.00368)
· [aclanthology.org/2024.acl-long.642](https://aclanthology.org/2024.acl-long.642/)

[4] Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z. *BGE M3-Embedding:
Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through
Self-Knowledge Distillation.* arXiv:2402.03216, 2024.
[arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216)
· [model](https://huggingface.co/BAAI/bge-m3)
· [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)

[5] *(reserved — BGE lineage; see [4] and FlagEmbedding.)*

[6] Li, Z., Zhang, X., Zhang, Y., Long, D., Xie, P., Zhang, M. *Towards General
Text Embeddings with Multi-stage Contrastive Learning* (GTE). arXiv:2308.03281,
2023.
[arxiv.org/abs/2308.03281](https://arxiv.org/abs/2308.03281)
· [gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)

[7] Nussbaum, Z., Morris, J.X., Duderstadt, B., Mulyar, A. *Nomic Embed: Training
a Reproducible Long Context Text Embedder.* arXiv:2402.01613, 2024.
[arxiv.org/abs/2402.01613](https://arxiv.org/abs/2402.01613)
· [code](https://github.com/nomic-ai/contrastors)

[8] Sturua, S., Mohr, I., Akram, M.K., Günther, M., Wang, B., Krimmel, M., et al.
*jina-embeddings-v3: Multilingual Embeddings With Task LoRA.* arXiv:2409.10173,
2024.
[arxiv.org/abs/2409.10173](https://arxiv.org/abs/2409.10173)

[9] OpenAI. *New embedding models and API updates* (text-embedding-3-small /
-large; `dimensions` / Matryoshka). 2024.
[openai.com/index/new-embedding-models-and-api-updates](https://openai.com/index/new-embedding-models-and-api-updates/)
· [embeddings guide](https://platform.openai.com/docs/guides/embeddings)

[10] Cohere / Hugging Face. *Cohere embed v3 and Binary/Scalar Embedding
Quantization for Faster & Cheaper Retrieval* (int8 ≈4×, binary ≈32×).
[huggingface.co/blog/embedding-quantization](https://huggingface.co/blog/embedding-quantization)
· [Cohere Embed](https://cohere.com/blog/introducing-embed-v3)

[11] Voyage AI. *voyage-3 & voyage-3-lite* (2024) and *voyage-3-large* (2025):
long-context, Matryoshka, quantized embedding models.
[blog.voyageai.com/2024/09/18/voyage-3](https://blog.voyageai.com/2024/09/18/voyage-3/)
· [voyage-3-large](https://blog.voyageai.com/2025/01/07/voyage-3-large/)

[12] Muennighoff, N., Tazi, N., Magne, L., Reimers, N. *MTEB: Massive Text
Embedding Benchmark.* EACL 2023; arXiv:2210.07316.
[arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316)

[13] Enevoldsen, K., et al. *MMTEB: Massive Multilingual Text Embedding
Benchmark.* ICLR 2025; arXiv:2502.13595.
[arxiv.org/abs/2502.13595](https://arxiv.org/abs/2502.13595)
· [leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

[14] Kusupati, A., Bhatt, G., Rege, A., et al. *Matryoshka Representation
Learning.* NeurIPS 2022; arXiv:2205.13147.
[arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147)

[15] Peeters, R., Steiner, A., Bizer, C. *Entity Matching using Large Language
Models.* EDBT 2025; arXiv:2310.11244.
[arxiv.org/abs/2310.11244](https://arxiv.org/abs/2310.11244)
· [code (MatchGPT)](https://github.com/wbsg-uni-mannheim/MatchGPT)

[16] Peeters, R., Bizer, C. *Using ChatGPT for Entity Matching.* arXiv:2305.03423,
2023.
[arxiv.org/abs/2305.03423](https://arxiv.org/abs/2305.03423)

[17] Zhang, H., Dong, Y., Xiao, C., Oyamada, M. *Jellyfish: Instruction-Tuning
Local Large Language Models for Data Preprocessing.* EMNLP 2024; arXiv:2312.01678.
[arxiv.org/abs/2312.01678](https://arxiv.org/abs/2312.01678)
· [aclanthology.org/2024.emnlp-main.497](https://aclanthology.org/2024.emnlp-main.497/)
· [model](https://huggingface.co/NECOUDBFM/Jellyfish-13B)

[18] Zhang, Z., Groth, P., Calixto, I., Schelter, S. *AnyMatch — Efficient
Zero-Shot Entity Matching with a Small Language Model.* arXiv:2409.04073, 2024.
[arxiv.org/abs/2409.04073](https://arxiv.org/abs/2409.04073)

[19] Li, H., Li, S., Hao, F., Zhang, C.J., Song, Y., Chen, L. *BoostER:
Leveraging Large Language Models for Enhancing Entity Resolution.* Companion Proc.
The Web Conference (WWW) 2024; arXiv:2403.06434.
[arxiv.org/abs/2403.06434](https://arxiv.org/abs/2403.06434)
· [dl.acm.org/doi/10.1145/3589335.3651245](https://dl.acm.org/doi/10.1145/3589335.3651245)

[20] Wang, T., Chen, X., Lin, H., Chen, X., Han, X., Wang, H., Zeng, Z., Sun, L.
*Match, Compare, or Select? An Investigation of Large Language Models for Entity
Matching* (ComEM). COLING 2025; arXiv:2405.16884.
[arxiv.org/abs/2405.16884](https://arxiv.org/abs/2405.16884)
· [code](https://github.com/tshu-w/ComEM)

[21] Fan, M., Han, X., Fan, J., Chai, C., Tang, N., Li, G., Du, X.
*Cost-Effective In-Context Learning for Entity Resolution: A Design Space
Exploration* (BatchER). ICDE 2024; arXiv:2312.03987.
[arxiv.org/abs/2312.03987](https://arxiv.org/abs/2312.03987)

[22] Cheng, Z., Kasai, J., Yu, T. *Batch Prompting: Efficient Inference with Large
Language Model APIs.* EMNLP 2023 (Industry); arXiv:2301.08721.
[arxiv.org/abs/2301.08721](https://arxiv.org/abs/2301.08721)

[23] Xu, Y., Li, H., Chen, K., Shou, L. *KcMF: A Knowledge-compliant Framework for
Schema and Entity Matching with Fine-tuning-free LLMs.* arXiv:2410.12480, 2024.
[arxiv.org/abs/2410.12480](https://arxiv.org/abs/2410.12480)

[24] Narayan, A., Chami, I., Orr, L., Ré, C. *Can Foundation Models Wrangle Your
Data?* PVLDB 16(4), 2022; arXiv:2205.09911.
[arxiv.org/abs/2205.09911](https://arxiv.org/abs/2205.09911)
· [PDF](https://www.vldb.org/pvldb/vol16/p738-narayan.pdf)

[25] OpenAI. *Introducing Structured Outputs in the API* (JSON-Schema constrained
decoding). Aug 2024.
[openai.com/index/introducing-structured-outputs-in-the-api](https://openai.com/index/introducing-structured-outputs-in-the-api/)

[26] *Cost-Efficient RAG for Entity Matching with LLMs: A Blocking-based
Exploration* (blocking-budgeted retrieval-augmented LLM EM), 2025.
[arxiv.org/abs/2602.05708](https://arxiv.org/abs/2602.05708)

[27] Tam, Z.R., et al. *Let Me Speak Freely? A Study on the Impact of Format
Restrictions on Performance of Large Language Models.* arXiv:2408.02442, 2024.
[arxiv.org/abs/2408.02442](https://arxiv.org/abs/2408.02442)
