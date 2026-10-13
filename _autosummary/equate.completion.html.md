# equate.completion

Tools to complete missing data from other sources of the same data.

Merging/joining tables is very common instance, yet only a small part of what is
possible, and often needed. Without moving away from tabular data:

- Find the columns to match (join keys) by comparing how well the values of the column
  match.
- Comparing the values of the columns with something more flexible than hard equality.

For example, correlation, similarity, etc.

- Find near duplicate columns
- Find rows to align, based on flexible comparison of fuzzily matched cells

#### NOTE
This module is a stale early duplicate of `equate.util` and is slated to be
rewritten as a thin facade over the redesigned stages (roadmap #13). Its `grub`
import is lazy (behind the optional `equate[grub]` extra) so importing the module
does not fail on a default numpy/scipy-only install.

### Functions

| `mk_text_to_vect`(\*learn_texts)                                                            |                                                              |
|---------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [`similarity_matrix`](#equate.completion.similarity_matrix)(keys, values, \*[, ...]) | Return a matrix of the similarity of the keys to the values. |
| `transform_text`(text, transformer)                                                         |                                                              |

### equate.completion.similarity_matrix(keys, values, \*, text_to_vect=None, similarity_func=<function cosine_similarity>)

Return a matrix of the similarity of the keys to the values.
By default, the `text_to_vect` will be learned from the keys and values,
using the tfidf vectorizer from `grub`.
You can pass in a different `text_to_vect` function to use a different vectorizer:
For example, pre-computed embeddings such as word2vec or fasttext, or openai
embeddings.

```pycon
>>> keys = ['apple pie', 'apple crumble', 'banana split']
>>> values = ['american pie', 'big apple', 'american girl', 'banana republic']
>>> m = similarity_matrix(keys, values)
>>> m.round(2).tolist()
[[0.54, 0.38, 0.0, 0.0], [0.0, 0.33, 0.0, 0.0], [0.0, 0.0, 0.0, 0.41]]
```
