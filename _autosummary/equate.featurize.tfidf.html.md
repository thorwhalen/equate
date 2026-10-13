# equate.featurize.tfidf

Pure-numpy/scipy character-n-gram TF-IDF — equate’s zero-heavy-dependency default
text featurizer (decision register D9). No scikit-learn or grub required.

Character n-grams make it robust to typos and morphology, which suits fuzzy name/SKU
matching; the output vectors are L2-normalized, so cosine similarity is a dot product.

### Functions

| [`mk_tfidf`](#equate.featurize.tfidf.mk_tfidf)(\*learn_texts[, ngram_range, lowercase])   | Fit a char-n-gram TF-IDF on the given collection(s) and return the fitted, callable batch featurizer — the pure numpy/scipy replacement for the legacy grub-backed `equate.util.mk_text_to_vect`.   |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`char_ngrams`](#equate.featurize.tfidf.char_ngrams)(text[, ngram_range, lowercase])         | Character n-grams of `text` over the inclusive `ngram_range`.                                                                                                                                       |

### Classes

| [`TfidfFeaturizer`](#equate.featurize.tfidf.TfidfFeaturizer)(\*[, ngram_range, lowercase])   | Fittable char-n-gram TF-IDF featurizer producing L2-normalized sparse vectors.   |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|

### *class* equate.featurize.tfidf.TfidfFeaturizer(, ngram_range=(2, 4), lowercase=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Fittable char-n-gram TF-IDF featurizer producing L2-normalized sparse vectors.

Fit on a corpus to learn the vocabulary and inverse document frequencies, then
transform texts to vectors. A fitted instance is itself a batch featurizer, so
`vecs = featurizer(texts)` works.

#### meta *= FeaturizerMeta(output_kinds=('vector',), normalize=True, dim=None, license='core', max_seq_len=None, query_prefix=None, passage_prefix=None, truncatable_to=())*

declared metadata (see equate.base.FeaturizerMeta)

### equate.featurize.tfidf.char_ngrams(text, ngram_range=(2, 4), , lowercase=True)

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

### equate.featurize.tfidf.mk_tfidf(\*learn_texts, ngram_range=(2, 4), lowercase=True)

Fit a char-n-gram TF-IDF on the given collection(s) and return the fitted,
callable batch featurizer — the pure numpy/scipy replacement for the legacy
grub-backed `equate.util.mk_text_to_vect`.
