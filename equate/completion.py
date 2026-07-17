"""Tools to complete missing data from other sources of the same data.

Merging/joining tables is very common instance, yet only a small part of what is
possible, and often needed. Without moving away from tabular data:

- Find the columns to match (join keys) by comparing how well the values of the column
match.
- Comparing the values of the columns with something more flexible than hard equality.
For example, correlation, similarity, etc.
- Find near duplicate columns
- Find rows to align, based on flexible comparison of fuzzily matched cells

.. note::
   This module is a stale early duplicate of ``equate.util`` and is slated to be
   rewritten as a thin facade over the redesigned stages (roadmap #13). Its ``grub``
   import is lazy (behind the optional ``equate[grub]`` extra) so importing the module
   does not fail on a default numpy/scipy-only install.
"""

from functools import partial
from itertools import chain

from equate._dependencies import require
from equate._vector import cosine_similarity


def transform_text(text, transformer):
    if isinstance(text, str):
        return transformer([text])[0]
    else:
        return transformer(text)


def mk_text_to_vect(*learn_texts):
    grub = require("grub", extra="grub", purpose="the legacy grub TF-IDF featurizer")

    docs = dict(enumerate(chain(*learn_texts)))
    s = grub.SearchStore(docs)
    return partial(transform_text, transformer=s.tfidf.transform)


def similarity_matrix(
    keys, values, *, text_to_vect=None, similarity_func=cosine_similarity
):
    """Return a matrix of the similarity of the keys to the values.
    By default, the `text_to_vect` will be learned from the keys and values,
    using the tfidf vectorizer from `grub`.
    You can pass in a different `text_to_vect` function to use a different vectorizer:
    For example, pre-computed embeddings such as word2vec or fasttext, or openai
    embeddings.

    >>> keys = ['apple pie', 'apple crumble', 'banana split']
    >>> values = ['american pie', 'big apple', 'american girl', 'banana republic']
    >>> m = similarity_matrix(keys, values)
    >>> m.round(2).tolist()
    [[0.54, 0.38, 0.0, 0.0], [0.0, 0.33, 0.0, 0.0], [0.0, 0.0, 0.0, 0.41]]
    """
    text_to_vect = text_to_vect or mk_text_to_vect(keys, values)
    key_vectors, value_vectors = map(text_to_vect, (keys, values))
    return similarity_func(key_vectors, value_vectors)


# The matchers below used to be verbatim copies of ``equate.util``'s, and they silently
# drifted: the copies kept the networkx unordered-edge bug (transposed pairs) and the
# densify-the-holes bug (an absent cell read as a real score of 0.0) long after ``util``
# was fixed. Re-exporting instead of re-copying makes that divergence impossible, and keeps
# ``from equate.completion import maximal_matching`` working for anyone already importing it.
from equate.util import (  # noqa: F401  (re-exported for back-compat)
    greedy_matching,
    hungarian_matching,
    maximal_matching,
    stable_marriage_matching,
    kuhn_munkres_matching,
)
