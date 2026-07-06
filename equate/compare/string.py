"""String comparators (stage ②): pairwise similarity between two strings.

Core comparators use only the standard library (``difflib`` and a pure-python edit
distance); faster/richer ones (rapidfuzz, jellyfish, py_stringmatching) are optional
extras imported lazily via :func:`equate._dependencies.require`. Each carries a
:class:`equate.base.ComparatorMeta` declaring its polarity, boundedness, whether it is a
metric, and whether it is symmetric — so the framework can adapt (never treat a
non-metric as a metric, keep directional comparators directional).
"""

from difflib import SequenceMatcher

from equate.base import ComparatorMeta
from equate._dependencies import require

__all__ = [
    'ratio',
    'levenshtein_distance',
    'levenshtein',
    'monge_elkan',
    'jaro_winkler',
    'phonetic_match',
]

_SIM = ComparatorMeta(polarity='similarity', bounded=True, is_metric=False, is_symmetric=True)


def ratio(a, b):
    """``difflib.SequenceMatcher`` ratio — a bounded [0, 1] similarity (stdlib, core)."""
    return SequenceMatcher(None, a, b).ratio()


ratio.meta = _SIM


def levenshtein_distance(a, b):
    """Levenshtein edit distance (pure python, core) — a true metric, unbounded."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


levenshtein_distance.meta = ComparatorMeta(
    polarity='distance', bounded=False, is_metric=True, is_symmetric=True
)


def levenshtein(a, b):
    """Length-normalized Levenshtein *similarity* in [0, 1] (core)."""
    m = max(len(a), len(b))
    return 1.0 - levenshtein_distance(a, b) / m if m else 1.0


levenshtein.meta = _SIM


def monge_elkan(a, b, *, sim=ratio, tokenize=str.split):
    """Monge-Elkan hybrid token similarity — **directional** (``ME(a,b) != ME(b,a)``).

    Each token of ``a`` is scored by its best match among the tokens of ``b`` (via the
    inner ``sim``), then averaged over ``a``'s tokens. Kept directional on purpose (see
    decision register D3); symmetrize only at the matcher boundary if a scalar is needed.
    """
    ta, tb = tokenize(a), tokenize(b)
    if not ta:
        return 0.0
    return sum(max((sim(x, y) for y in tb), default=0.0) for x in ta) / len(ta)


monge_elkan.meta = ComparatorMeta(
    polarity='similarity', bounded=True, is_metric=False, is_symmetric=False
)


def jaro_winkler(a, b):
    """Jaro-Winkler similarity via ``rapidfuzz`` — requires ``equate[fuzzy]``."""
    rf = require('rapidfuzz', extra='fuzzy', purpose='Jaro-Winkler similarity')
    return rf.distance.JaroWinkler.normalized_similarity(a, b)


jaro_winkler.meta = _SIM


def phonetic_match(a, b, *, algorithm='metaphone'):
    """Boolean phonetic-code agreement via ``jellyfish`` — requires ``equate[phonetic]``.

    Returns 1.0 when the two strings share a phonetic code (``metaphone`` by default,
    or ``soundex`` / ``nysiis``), else 0.0.
    """
    jf = require('jellyfish', extra='phonetic', purpose='phonetic matching')
    encode = getattr(jf, algorithm)
    return 1.0 if encode(a) == encode(b) else 0.0


phonetic_match.meta = ComparatorMeta(
    polarity='similarity', bounded=True, is_metric=False, is_symmetric=True
)
