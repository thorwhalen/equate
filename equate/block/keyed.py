"""Keyed blocking (stage ② scalability): emit candidate pairs that share a blocking key.

A *blocker* proposes which pairs are worth scoring, avoiding the O(n*m) all-pairs
blow-up — equivalently, it decides which cells of the score matrix to leave uncomputed
(decision register D1). Keyed blocking groups items by a cheap key (or several keys per
item, e.g. token / q-gram blocking) and emits the within-group pairs. Standard, token,
q-gram, and phonetic blocking are all *one* algorithm parameterized by the key function.
"""

from itertools import groupby, zip_longest

from equate._dependencies import require

__all__ = [
    'key_blocking',
    'sorted_neighborhood',
    'first_chars',
    'qgram_keys',
    'whole_value',
    'phonetic_key',
]


def key_blocking(key_fn):
    """Return a blocker emitting candidate ``(i, j)`` for items sharing a blocking key.

    ``key_fn`` maps an item to an *iterable of keys* (an item may land in several blocks
    — token/q-gram blocking). Cross-collection ``blocker(A, B)`` yields ``(i, j)`` with
    ``i`` indexing ``A`` and ``j`` indexing ``B``; a self-join ``blocker(A)`` (dedup)
    yields each ``i < j`` pair once. Lazy: candidate pairs are generated, never a matrix.
    """

    def blocker(A, B=None):
        self_join = B is None or B is A
        A = list(A)
        B = A if self_join else list(B)
        index: dict = {}  # key -> list of B indices
        for j, b in enumerate(B):
            for k in key_fn(b):
                index.setdefault(k, []).append(j)
        for i, a in enumerate(A):
            js = set()
            for k in key_fn(a):
                js.update(index.get(k, ()))
            for j in sorted(js):
                if self_join and i >= j:
                    continue
                yield (i, j)

    return blocker


def sorted_neighborhood(sort_key=str, *, window=3):
    """Sorted-neighborhood blocker: sort items by ``sort_key`` and emit pairs within a
    sliding window of ``window`` neighbours. Robust to the key-equality brittleness of
    standard blocking.
    """

    def blocker(A, B=None):
        self_join = B is None or B is A
        A = list(A)
        if self_join:
            order = sorted(range(len(A)), key=lambda i: sort_key(A[i]))
            for p in range(len(order)):
                for q in range(p + 1, min(p + window, len(order))):
                    i, j = order[p], order[q]
                    yield (i, j) if i < j else (j, i)
        else:
            B = list(B)
            tagged = [(sort_key(x), 0, i) for i, x in enumerate(A)]
            tagged += [(sort_key(x), 1, j) for j, x in enumerate(B)]
            tagged.sort(key=lambda t: t[0])
            # Interleave A and B items within each equal-key run so A-B pairs sit
            # adjacent and land inside the window (else a tie group of all-A-then-all-B
            # pushes A items a full block away from any B — dropping true pairs).
            ordered = []
            for _, run in groupby(tagged, key=lambda t: t[0]):
                grp = list(run)
                a_items = [t for t in grp if t[1] == 0]
                b_items = [t for t in grp if t[1] == 1]
                for xa, xb in zip_longest(a_items, b_items):
                    if xa is not None:
                        ordered.append(xa)
                    if xb is not None:
                        ordered.append(xb)
            for p in range(len(ordered)):
                for q in range(p + 1, min(p + window, len(ordered))):
                    _, sa, ia = ordered[p]
                    _, sb, jb = ordered[q]
                    if sa != sb:
                        yield (ia, jb) if sa == 0 else (jb, ia)

    return blocker


# --- key functions (compose with key_blocking) ------------------------------------


def first_chars(n=1, *, key=str, lowercase=True):
    """Key function: the first ``n`` characters (standard blocking)."""

    def kf(item):
        s = str(key(item))
        s = s.lower() if lowercase else s
        return [s[:n]]

    return kf


def qgram_keys(q=2, *, key=str, lowercase=True):
    """Key function: the set of character q-grams (q-gram / token blocking)."""

    def kf(item):
        s = str(key(item))
        s = s.lower() if lowercase else s
        return {s[i : i + q] for i in range(len(s) - q + 1)} or {s}

    return kf


def whole_value(*, key=str, lowercase=True):
    """Key function: the whole (optionally lowercased) value — exact-key blocking."""

    def kf(item):
        s = str(key(item))
        return [s.lower() if lowercase else s]

    return kf


def phonetic_key(*, algorithm='soundex', key=str):
    """Key function: a phonetic code (``soundex`` / ``metaphone`` / ``nysiis``) via
    ``jellyfish`` — requires ``equate[phonetic]``.
    """

    def kf(item):
        jf = require('jellyfish', extra='phonetic', purpose='phonetic blocking')
        return [getattr(jf, algorithm)(str(key(item)))]

    return kf
