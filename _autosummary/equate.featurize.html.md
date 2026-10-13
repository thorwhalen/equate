# equate.featurize

The featurize stage (①): map objects to comparable representations.

A *featurizer* is a batch callable `Iterable[object] -> representations`. The default
is a pure numpy/scipy char-n-gram TF-IDF ([`equate.featurize.tfidf`](equate.featurize.tfidf.html.md#module-equate.featurize.tfidf)) — no heavy
dependencies; dense embedders (SBERT, OpenAI, …) are optional extras
([`equate.featurize.text`](equate.featurize.text.html.md#module-equate.featurize.text)). Select a featurizer by name via the [`featurizers`](#equate.featurize.featurizers)
registry, or pass a callable. Corpus-dependent featurizers (TF-IDF) are built by
fitting on a corpus, which [`resolve_featurizer()`](#equate.featurize.resolve_featurizer) handles.

### Module Attributes

| [`featurizers`](#equate.featurize.featurizers)   | the featurize-stage strategy registry (name -> lazy factory)   |
|----------------------------------------------------------------|----------------------------------------------------------------|

### Functions

| [`identity`](#equate.featurize.identity)(objects)                                 | The identity featurizer: use the objects themselves as their representation.                                                                                                                      |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`resolve_featurizer`](#equate.featurize.resolve_featurizer)([spec, corpus])                | Resolve `spec` to a ready-to-use batch featurizer.                                                                                                                                                |
| [`mk_tfidf`](#equate.featurize.mk_tfidf)(\*learn_texts[, ngram_range, lowercase]) | Fit a char-n-gram TF-IDF on the given collection(s) and return the fitted, callable batch featurizer — the pure numpy/scipy replacement for the legacy grub-backed `equate.util.mk_text_to_vect`. |
| [`char_ngrams`](#equate.featurize.char_ngrams)(text[, ngram_range, lowercase])       | Character n-grams of `text` over the inclusive `ngram_range`.                                                                                                                                     |

### Classes

| [`TfidfFeaturizer`](#equate.featurize.TfidfFeaturizer)(\*[, ngram_range, lowercase])   | Fittable char-n-gram TF-IDF featurizer producing L2-normalized sparse vectors.                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`FeaturizerMeta`](#equate.featurize.FeaturizerMeta)([output_kinds, normalize, ...])  | Declared properties of a featurizer (decision register D9), so the framework can pick a legal index/metric, auto-apply a required query/passage prefix, refuse a non-commercial model in a commercial context, and offer Matryoshka truncation. |

### *class* equate.featurize.FeaturizerMeta(output_kinds=('vector',), normalize=False, dim=None, license=None, max_seq_len=None, query_prefix=None, passage_prefix=None, truncatable_to=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Declared properties of a featurizer (decision register D9), so the framework
can pick a legal index/metric, auto-apply a required query/passage prefix, refuse a
non-commercial model in a commercial context, and offer Matryoshka truncation.

- `output_kinds`: what the representation is — `'vector'` / `'set'` /
  `'bitstring'` / `'scalar'` / `'structured'` (a multi-functional embedder may
  emit several, e.g. dense + sparse);
- `normalize`: whether output vectors are already L2-normalized;
- `dim` / `max_seq_len`: dimensionality and max input length, when fixed;
- `license`: SPDX/label, so a commercial context can refuse non-commercial models;
- `query_prefix` / `passage_prefix`: instruction prefixes some embedders require
  (e.g. E5’s `"query: "` / `"passage: "`) — a silent-omission footgun;
- `truncatable_to`: Matryoshka dims the embedding can be safely truncated to.

### *class* equate.featurize.TfidfFeaturizer(, ngram_range=(2, 4), lowercase=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Fittable char-n-gram TF-IDF featurizer producing L2-normalized sparse vectors.

Fit on a corpus to learn the vocabulary and inverse document frequencies, then
transform texts to vectors. A fitted instance is itself a batch featurizer, so
`vecs = featurizer(texts)` works.

#### meta *= FeaturizerMeta(output_kinds=('vector',), normalize=True, dim=None, license='core', max_seq_len=None, query_prefix=None, passage_prefix=None, truncatable_to=())*

declared metadata (see equate.base.FeaturizerMeta)

### equate.featurize.char_ngrams(text, ngram_range=(2, 4), , lowercase=True)

Character n-grams of `text` over the inclusive `ngram_range`.

The whole string is padded with one leading and one trailing space (so prefixes and
suffixes are captured), and n-grams slide over the entire padded string — they \*may
therefore span the spaces between words\*. This is deliberate (it helps fuzzy
name/SKU matching) and differs from scikit-learn’s per-word `char_wb` analyzer. A
text too short for any `n` in the range falls back to the padded string itself, so
tiny or empty inputs still yield a (single) feature rather than an all-zero vector.

```pycon
>>> char_ngrams('cat', (2, 3))
[' c', 'ca', 'at', 't ', ' ca', 'cat', 'at ']
>>> char_ngrams('', (3, 4))          # too short for any n: falls back to the padding
['  ']
```

### equate.featurize.featurizers *= <equate.registry.Registry object>*

the featurize-stage strategy registry (name -> lazy factory)

### equate.featurize.identity(objects)

The identity featurizer: use the objects themselves as their representation.

This is the degenerate case that recovers exact matching (φ = id).

### equate.featurize.mk_tfidf(\*learn_texts, ngram_range=(2, 4), lowercase=True)

Fit a char-n-gram TF-IDF on the given collection(s) and return the fitted,
callable batch featurizer — the pure numpy/scipy replacement for the legacy
grub-backed `equate.util.mk_text_to_vect`.

### equate.featurize.resolve_featurizer(spec=None, , corpus=None, \*\*config)

Resolve `spec` to a ready-to-use batch featurizer.

- `None` -> the default `'tfidf'`;
- a registered name (`str`) -> built via its factory (`**config` is forwarded, so
  an unknown option raises rather than being silently ignored);
- a callable -> returned as-is (assumed already a batch featurizer).

A *fittable* featurizer (one exposing `fit`, e.g. TF-IDF) is fit on `corpus`
here — and a corpus is then required: resolving a fittable featurizer without one
raises immediately (a clear error at the call site) instead of returning a
half-built object that only fails later inside `transform`.

### Modules

| [`text`](equate.featurize.text.html.md#module-equate.featurize.text)   | Dense text-embedding featurizers, behind optional extras.                                                                 |
|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [`tfidf`](equate.featurize.tfidf.html.md#module-equate.featurize.tfidf) | Pure-numpy/scipy character-n-gram TF-IDF — equate's zero-heavy-dependency default text featurizer (decision register D9). |
