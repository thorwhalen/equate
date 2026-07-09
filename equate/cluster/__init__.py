"""The resolve/cluster stage (③ grouping): turn pairwise matches into entity groups.

Grading the comparison breaks transitivity (``a≈b`` and ``b≈c`` need not imply ``a≈c``),
so recovering *groups* from pairwise links is an explicit, swappable choice — never
hard-wired into the matcher (decision register D4, and the equivalence-relation framing in
``00-taxonomy``). ``connected_components`` (transitive closure via union-find) is the cheap
default; ``correlation_clustering`` is the robust alternative that tolerates a few
disagreements rather than collapsing two entities on one spurious edge. The output is a
:class:`equate.base.Partition`; :func:`canonicalize` merges each group into a golden record.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping

from equate.registry import Registry
from equate.base import Partition

__all__ = [
    "clusterers",
    "resolve_clusterer",
    "connected_components",
    "correlation_clustering",
    "canonicalize",
]


def _edges(pairs, n, *, threshold, sense):
    """Yield validated edges ``(i, j)`` from an iterable of ``(i, j)`` or ``(i, j, score)``.

    2-tuples are unconditional edges; a 3-tuple is kept only when its score passes the
    sense-aware threshold (``s >= threshold`` for ``'maximize'``, ``s <= threshold`` for
    ``'minimize'``). Every index is validated ``0 <= idx < n`` with a clear error (a raw
    negative index would otherwise silently alias to another item).
    """
    keep = (
        (lambda s: s >= threshold)
        if sense == "maximize"
        else (lambda s: s <= threshold)
    )
    for p in pairs:
        if len(p) == 3:
            i, j, s = p
            if threshold is not None and not keep(s):
                continue
        else:
            i, j = p
        for idx in (i, j):
            if not (0 <= idx < n):
                raise ValueError(f"cluster index {idx} out of range [0, {n})")
        yield i, j


def connected_components(pairs, n, *, threshold=None, sense="maximize"):
    """Partition ``n`` items into connected components of the match graph (union-find).

    Accepts either 2-tuple edges ``(i, j)`` or 3-tuple scored edges ``(i, j, score)`` — in
    the latter case only edges passing ``threshold`` (sense-aware) are used — so it serves
    both as the low-level primitive and as a clusterer via :func:`resolve_clusterer`. This
    is the *transitive closure* of the links: fragile by design (one spurious edge merges
    two entities). Use :func:`correlation_clustering` when that matters.

    >>> connected_components([(0, 2), (1, 3)], 5).groups()
    {0: [0, 2], 1: [1, 3], 2: [4]}
    """
    parent = list(range(n))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for i, j in _edges(pairs, n, threshold=threshold, sense=sense):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    remap: dict = {}
    labels = []
    for x in range(n):
        r = find(x)
        if r not in remap:
            remap[r] = len(remap)
        labels.append(remap[r])
    return Partition(labels)


def correlation_clustering(scored_pairs, n, *, threshold=0.0, sense="maximize"):
    """Cluster ``n`` items to minimize disagreements (the greedy Pivot algorithm).

    ``scored_pairs`` is an iterable of ``(i, j, score)``; an edge is *positive* when its
    score passes ``threshold`` (sense-aware: ``>=`` for ``'maximize'``, ``<=`` for
    ``'minimize'``). Pivot processes items in **index order** (deterministic — note the
    randomized-Pivot 3-approximation bound holds only in expectation over a random pivot),
    clustering each still-unclustered item with its still-unclustered positive neighbours.
    Unlike connected-components it does not chain transitively, so a single stray edge
    cannot collapse two clusters.
    """
    positive: dict = defaultdict(set)
    for i, j in _edges(scored_pairs, n, threshold=threshold, sense=sense):
        positive[i].add(j)
        positive[j].add(i)
    labels = [-1] * n
    cid = 0
    for pivot in range(n):
        if labels[pivot] != -1:
            continue
        members = [pivot] + [x for x in positive[pivot] if labels[x] == -1]
        for x in members:
            labels[x] = cid
        cid += 1
    return Partition(labels)


#: the resolve/cluster strategy registry — a clusterer is
#: ``(scored_pairs, n, *, threshold) -> Partition``.
clusterers = Registry("clusterer")
# Both accept the uniform (scored_pairs, n, *, threshold, sense) contract, so either the
# string spec or the public callable works with resolve_clusterer.
clusterers.register("connected_components", lambda: connected_components)
clusterers.register("correlation", lambda: correlation_clustering)


def resolve_clusterer(spec="connected_components"):
    """Resolve ``spec`` to a clusterer ``(scored_pairs, n, *, threshold, sense) -> Partition``."""
    if callable(spec):
        return spec
    if isinstance(spec, str):
        return clusterers.create(spec)
    raise TypeError(
        f"clusterer spec must be a registered name or a callable, "
        f"got {type(spec).__name__}"
    )


# --- canonicalization (golden record) ---------------------------------------------


def _completeness(record):
    if isinstance(record, Mapping):
        return sum(1 for v in record.values() if v is not None)
    return 0 if record is None else 1


def _mode(values):
    """Most common value; robust to unhashable values (falls back to the first)."""
    try:
        return Counter(values).most_common(1)[0][0]
    except TypeError:  # unhashable values (list/dict/set) can't go in a Counter
        return values[0]


def _majority_merge(records):
    if records and not any(isinstance(r, Mapping) for r in records):
        # scalar / string records: majority-vote over the values themselves
        present = [r for r in records if r is not None]
        return _mode(present) if present else None
    fields = {f for r in records if isinstance(r, Mapping) for f in r}
    out = {}
    for f in fields:
        vals = [
            r[f] for r in records if isinstance(r, Mapping) and r.get(f) is not None
        ]
        if vals:
            out[f] = _mode(vals)
    return out


def _merge(records, policy):
    if not records:
        return None
    if policy == "first":
        return records[0]
    if policy == "most_complete":
        return max(
            records, key=_completeness
        )  # ties -> first-max (first in cluster order)
    return _majority_merge(records)


_CANONICALIZE_POLICIES = ("first", "most_complete", "majority")


def canonicalize(partition, records, *, policy="most_complete"):
    """Merge each cluster of ``partition`` into one golden record from ``records``.

    ``policy``: ``'first'`` (keep the first member), ``'most_complete'`` (fewest missing
    fields, the default; ties keep the first member), or ``'majority'`` (per-field majority
    vote for mapping records, or a value vote for scalar records; unhashable values fall
    back to the first). Returns ``{cluster_id: golden_record}``.
    """
    if policy not in _CANONICALIZE_POLICIES:
        raise ValueError(
            f"unknown canonicalize policy {policy!r}; choose from {_CANONICALIZE_POLICIES}"
        )
    return {
        cid: _merge([records[m] for m in members], policy)
        for cid, members in partition.groups().items()
    }
