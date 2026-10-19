"""Tabular completion: find join keys, near-duplicate columns, and fill a table from another.

Merging tables on a known key is a small part of what is needed. Without leaving tables:

- **find the columns to join on** (:func:`find_join_keys`) by how much one column's
  values are *contained* in another's — no header names needed;
- **find near-duplicate columns** (:func:`find_near_duplicate_columns`) in one table;
- **complete a table** (:func:`complete_table`): discover the join key, align rows by
  fuzzy matching of the key values, and fill the target's missing cells from the source.

These are facades over the same seams as everything else (research doc 07 §8, doc 10
§8.7) — not a parallel implementation: a column is an object featurized as its value set
(``featurize='value_set'``), compared with a set comparator (``'containment'``,
``'jaccard'``, ``'overlap'``), and selected with the top-k :class:`~equate.interactive.
CandidateStore`; rows are aligned with :func:`equate.match` / :func:`equate.score_matrix`.

A *table* is any of: a mapping of column name to values (``{'name': [...], ...}``), a list
of row mappings, or anything DataFrame-like (``.columns`` and ``table[column]``). Results
are structured dataclasses; :class:`Completion` gives back columns (``.table``) or rows
(``.rows()``).

>>> left = {'cust': ['c1', 'c2', 'c3'], 'city': [None, 'Lyon', None]}
>>> right = {'id': ['c3', 'c1', 'c2', 'c9'], 'city': ['Nice', 'Paris', 'Lyon', 'Lille']}
>>> [(m.left, m.right, m.score) for m in find_join_keys(left, right)][0]
('cust', 'id', 1.0)
>>> complete_table(left, right).table['city']
['Paris', 'Lyon', 'Nice']

The functions further down (``similarity_matrix``, ``mk_text_to_vect``, ...) are the
pre-redesign API, kept for back-compat; new code should use :func:`equate.score_matrix`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Any, Optional

from equate._dependencies import require
from equate._vector import cosine_similarity
from equate.base import Explanation
from equate.compare import resolve_comparator
from equate.facade import score_matrix
from equate.featurize import resolve_featurizer
from equate.featurize.columns import is_missing
from equate.interactive import CandidateStore
from equate.matching import resolve_matcher

__all__ = [
    "ColumnMatch",
    "Completion",
    "as_columns",
    "find_join_keys",
    "find_near_duplicate_columns",
    "complete_table",
    # the pre-redesign API, kept importable (also by `import *`)
    "similarity_matrix",
    "mk_text_to_vect",
    "transform_text",
    "greedy_matching",
    "hungarian_matching",
    "maximal_matching",
    "stable_marriage_matching",
    "kuhn_munkres_matching",
]

DFLT_KEY_COMPARE = (
    "containment"  # joinability: how much of the target key is in the source
)
DFLT_MIN_KEY_SCORE = 0.5
DFLT_DUPLICATE_COMPARE = "jaccard"
DFLT_DUPLICATE_THRESHOLD = 0.9
DFLT_MIN_ROW_SCORE = 0.9  # fuzzy key matches only; exact matches always align


# --- tables ---------------------------------------------------------------------------


def as_columns(table) -> dict:
    """``{column name: list of values}`` from a column mapping, a list of row mappings,
    or a DataFrame-like object (``.columns`` + ``table[column]``).

    Column names are kept as they are (an ``int`` stays an ``int``). Rows missing a
    column get ``None`` there; column order is first-seen order. A column must be a
    sequence of cells, not a string.

    >>> as_columns([{'a': 1}, {'a': 2, 'b': 'x'}])
    {'a': [1, 2], 'b': [None, 'x']}
    """
    if isinstance(table, Mapping):
        columns = dict(table.items())
    elif hasattr(table, "columns") and not isinstance(table, (list, tuple)):
        columns = {c: table[c] for c in table.columns}
    elif isinstance(table, Sequence) and all(isinstance(r, Mapping) for r in table):
        names = list(dict.fromkeys(c for row in table for c in row))
        return {c: [row.get(c) for row in table] for c in names}
    else:
        raise TypeError(
            "a table is a mapping of column -> values, a list of row mappings, or a "
            f"DataFrame-like object; got {type(table).__name__}"
        )
    for name, values in columns.items():
        if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
            raise TypeError(
                f"column {name!r} must be a sequence of cells, got "
                f"{type(values).__name__} (a string would be read character by character)"
            )
    return {c: list(v) for c, v in columns.items()}


def _n_rows(columns: dict) -> int:
    lengths = {len(v) for v in columns.values()}
    if len(lengths) > 1:
        raise ValueError(f"columns have different lengths: {sorted(lengths)}")
    return lengths.pop() if lengths else 0


# --- results --------------------------------------------------------------------------


@dataclass
class ColumnMatch:
    """A scored correspondence between a ``left`` and a ``right`` column."""

    left: Any
    right: Any
    score: float
    explanation: Optional[Explanation] = None


@dataclass
class Completion:
    """The completed table and how it was completed.

    ``table`` is the target as columns, with missing cells filled; ``key`` the join key
    used; ``rows`` the aligned ``(target_row, source_row, score)`` triples (``score`` is
    ``None`` for an exact key match, the fuzzy score otherwise); ``filled`` the number of
    cells filled per column; ``columns`` the ``{target: source}`` columns used.
    """

    table: dict
    key: ColumnMatch
    rows: list
    filled: dict
    columns: dict = field(default_factory=dict)

    def as_rows(self) -> list:
        """The completed table as a list of row dicts."""
        names = list(self.table)
        n = _n_rows(self.table)
        return [{c: self.table[c][i] for c in names} for i in range(n)]


# --- column matching ------------------------------------------------------------------


def _column_scores(left: dict, right: dict, compare):
    feat = resolve_featurizer("value_set")
    sets_l, sets_r = feat(left.values()), feat(right.values())
    comparator = resolve_comparator(compare)
    return sets_l, sets_r, score_matrix(sets_l, sets_r, compare=comparator)


def _explain(sl, sr, score):
    shared = sl & sr
    return Explanation(
        summary=f"{len(shared)} shared values ({score:.2f})",
        details={
            "shared": len(shared),
            "left_distinct": len(sl),
            "right_distinct": len(sr),
            "examples": sorted(map(str, shared))[:5],
        },
    )


def find_join_keys(
    left,
    right,
    *,
    compare=DFLT_KEY_COMPARE,
    k: int = 1,
    min_score: float = DFLT_MIN_KEY_SCORE,
) -> list:
    """Columns of ``right`` that ``left``'s columns could be joined on, best first.

    Each column is featurized as its value set and compared with ``compare`` (default
    ``'containment'``: the share of the left column's distinct values found in the right
    one — a foreign key scores 1.0 against its primary key whatever else that holds).
    Each left column keeps its top-``k`` right columns scoring at least ``min_score``.
    Returns :class:`ColumnMatch` es sorted by score; ties go to the pair sharing more
    distinct values (a real key over a low-cardinality column), then to column order.
    """
    left, right = as_columns(left), as_columns(right)
    sets_l, sets_r, sm = _column_scores(left, right, compare)
    store = CandidateStore.from_scores(sm.data, k=k, sense=sm.sense)
    names_l, names_r = list(left), list(right)
    found = [
        ColumnMatch(names_l[i], names_r[j], s, _explain(sets_l[i], sets_r[j], s))
        for i, cands in store.candidates.items()
        for j, s in cands
        if s >= min_score
    ]
    order = {
        (a, b): n for n, (a, b) in enumerate((a, b) for a in names_l for b in names_r)
    }
    # ties on the score are broken by evidence: more shared distinct values first, so a
    # many-valued key beats a 2-value column that happens to be fully contained too
    return sorted(
        found,
        key=lambda m: (
            -m.score,
            -m.explanation.details["shared"],
            order[(m.left, m.right)],
        ),
    )


def find_near_duplicate_columns(
    table,
    *,
    compare=DFLT_DUPLICATE_COMPARE,
    threshold: float = DFLT_DUPLICATE_THRESHOLD,
) -> list:
    """Pairs of columns of one table whose value sets are near-identical.

    Each column is featurized as its value set and every pair compared with ``compare``
    (default ``'jaccard'``); pairs scoring at least ``threshold`` are returned as
    :class:`ColumnMatch` es, best first — earlier column on the left for a symmetric
    comparator, both directions for a directional one (``'containment'``).
    """
    columns = as_columns(table)
    sets, _, sm = _column_scores(columns, columns, compare)
    names = list(columns)
    meta = getattr(resolve_comparator(compare), "meta", None)
    symmetric = getattr(meta, "is_symmetric", True)
    found = [
        ColumnMatch(names[i], names[j], s, _explain(sets[i], sets[j], s))
        for i, j, s in sm.stored_entries()
        if (i < j if symmetric else i != j) and s >= threshold
    ]
    return sorted(
        found, key=lambda m: (-m.score, names.index(m.left), names.index(m.right))
    )


# --- table completion -----------------------------------------------------------------


def _resolve_key(target, source, on, compare, min_key_score):
    if on is not None:
        t, s = on
        for name, cols in ((t, target), (s, source)):
            if name not in cols:
                raise KeyError(f"no column {name!r}; columns are {list(cols)}")
        return ColumnMatch(t, s, float("nan"), Explanation(summary="given"))
    keys = find_join_keys(target, source, compare=compare, min_score=min_key_score)
    if not keys:
        raise ValueError(
            f"no column of the source contains at least {min_key_score:.0%} of a target "
            "column's values; pass on=(target_column, source_column) or lower "
            "min_key_score"
        )
    return keys[0]


def _align_rows(keys_t, keys_s, *, row_compare, how, min_row_score):
    """``(target_row, source_rows, score)`` for each aligned target row.

    Keys are normalized (:func:`~equate.featurize.columns.normalize_value`) the same way
    join-key discovery normalizes them. Equal normalized keys align **exactly** (a hash
    join: linear time, score ``None``); only the target keys left over are matched
    **fuzzily** (``row_compare``) against the distinct source keys, and an ambiguous best
    (a tie between two source keys) aligns nothing. ``source_rows`` lists every source row
    carrying the aligned key, in order. With a one-to-one ``how``, each source row is
    used at most once.
    """
    from equate.featurize.columns import normalize_value

    by_key = {}
    for j, v in enumerate(keys_s):
        if not is_missing(v):
            by_key.setdefault(normalize_value(v), []).append(j)
    one_to_one = how != "best"
    used = set()
    rows, leftover = [], []
    for i, v in enumerate(keys_t):
        if is_missing(v):
            continue
        sources = [j for j in by_key.get(normalize_value(v), ()) if j not in used]
        if sources:
            if one_to_one:
                sources = sources[:1]
                used.add(sources[0])
            rows.append((i, sources, None))
        else:
            leftover.append(i)
    if leftover and row_compare is not None:
        rows.extend(
            _fuzzy_rows(
                keys_t,
                leftover,
                by_key,
                used,
                row_compare=row_compare,
                how=how,
                min_row_score=min_row_score,
            )
        )
    return sorted(rows, key=lambda r: r[0])


def _fuzzy_rows(keys_t, leftover, by_key, used, *, row_compare, how, min_row_score):
    from equate.featurize.columns import normalize_value

    free = {k: [j for j in js if j not in used] for k, js in by_key.items()}
    free_keys = [k for k, js in free.items() if js]
    if not free_keys:
        return []
    texts = [normalize_value(keys_t[i]) for i in leftover]
    sm = score_matrix(texts, free_keys, compare=row_compare)
    out = []
    if how == "best":
        store = CandidateStore.from_scores(sm.data, k=2, sense=sm.sense)
        for r, cands in store.candidates.items():
            if not cands or (len(cands) > 1 and cands[0][1] == cands[1][1]):
                continue  # nothing, or a tie: an ambiguous key fills nothing
            c, score = cands[0]
            out.append((leftover[r], free[free_keys[c]], float(score)))
    else:
        for r, c in resolve_matcher(how)(sm):
            out.append((leftover[r], free[free_keys[c]][:1], float(sm.score_at(r, c))))
    return [
        row
        for row in out
        if min_row_score is None or _passes(row[2], min_row_score, sm.sense)
    ]


def _passes(score, bound, sense):
    return score >= bound if sense == "maximize" else score <= bound


def complete_table(
    target,
    source,
    *,
    on: Optional[tuple] = None,
    columns: Optional[Mapping] = None,
    key_compare=DFLT_KEY_COMPARE,
    min_key_score: float = DFLT_MIN_KEY_SCORE,
    row_compare="ratio",
    how: str = "best",
    min_row_score: Optional[float] = DFLT_MIN_ROW_SCORE,
) -> Completion:
    """Fill ``target``'s missing cells from ``source``.

    1. **Key** — ``on=(target_column, source_column)``, or discovered with
       :func:`find_join_keys` (``key_compare``, ``min_key_score``).
    2. **Rows** — key values are normalized (case, spacing, ``1.0`` = ``1``) and joined
       **exactly** first. Only target keys with no exact partner are matched
       **fuzzily** with ``row_compare`` (default the ``'ratio'`` edit similarity, so a
       stray character still meets its key; ``None`` turns fuzzy matching off) and kept only
       when they score at least ``min_row_score`` (default ``0.9``; for a *distance*
       comparator it is a maximum) and are not tied between two source keys.
       ``how='best'`` (default) is a lookup: each target row takes its source key, so
       many target rows may share one. A matcher name (``'optimal'``, ``'greedy'``,
       ``'stable'``) aligns rows one-to-one instead.
    3. **Fill** — for each ``{target_column: source_column}`` in ``columns`` (default:
       the columns both tables share, except the key), a missing target cell (``None``,
       blank, NaN, pandas ``NA``) takes the value of the first aligned source row that
       has one. A target column that does not exist yet is created. Present values are
       never overwritten.
    """
    t_cols, s_cols = as_columns(target), as_columns(source)
    n = _n_rows(t_cols)
    _n_rows(s_cols)
    key = _resolve_key(t_cols, s_cols, on, key_compare, min_key_score)
    if columns is None:
        columns = {
            c: c for c in t_cols if c in s_cols and c not in (key.left, key.right)
        }
    missing_src = [s for s in columns.values() if s not in s_cols]
    if missing_src:
        raise KeyError(f"no source column(s) {missing_src}; columns are {list(s_cols)}")
    aligned = _align_rows(
        t_cols[key.left],
        s_cols[key.right],
        row_compare=row_compare,
        how=how,
        min_row_score=min_row_score,
    )
    table = {c: list(v) for c, v in t_cols.items()}
    filled = {}
    for t_col, s_col in columns.items():
        values = table.setdefault(t_col, [None] * n)
        count = 0
        for i, sources, _ in aligned:
            if not is_missing(values[i]):
                continue
            new = next(
                (s_cols[s_col][j] for j in sources if not is_missing(s_cols[s_col][j])),
                None,
            )
            if new is not None:
                values[i] = new
                count += 1
        filled[t_col] = count
    rows = [(i, sources[0], score) for i, sources, score in aligned]
    return Completion(
        table=table, key=key, rows=rows, filled=filled, columns=dict(columns)
    )


# --- legacy API (pre-redesign; kept for back-compat) ---------------------------------


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
