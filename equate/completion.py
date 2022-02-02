"""Tools to complete missing data from other sources of the same data.

Merging/joining tables is very common instance, yet only a small part of what is
possible, and often needed. Without moving away from tabular data:

- Find the columns to match (join keys) by comparing how well the values of the column
match.
- Comparing the values of the columns with something more flexible than hard equality.
For example, correlation, similarity, etc.
- Find near duplicate columns
- Find rows to align, based on flexible comparison of fuzzily matched cells
"""

from functools import lru_cache, partial
from grub import SearchStore
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
from scipy.optimize import linear_sum_assignment


def transform_text(text, transformer):
    if isinstance(text, str):
        return transformer([text])[0]
    else:
        return transformer(text)


# @lru_cache
def mk_text_to_vect(*learn_texts):
    docs = dict(enumerate(chain(*learn_texts)))
    s = SearchStore(docs)
    return partial(transform_text, transformer=s.tfidf.transform)


def similarity_matrix(u_strings, v_strings, text_to_vect=None):
    text_to_vect = text_to_vect or mk_text_to_vect(u_strings, v_strings)
    u_vectors, v_vectors = map(text_to_vect, (u_strings, v_strings))
    return cosine_similarity(u_vectors, v_vectors)

