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
    'clusterers',
    'resolve_clusterer',
    'connected_components',
    'correlation_clustering',
    'canonicalize',
]


def connected_components(pairs, n):
    """Partition ``n`` items into connected components of the match graph (union-find).

    This is the *transitive closure* of the pairwise links — fragile by design: one
    spurious edge merges two entities. Use :func:`correlation_clustering` when that risk
    matters.

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

    for i, j in pairs:
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


def correlation_clustering(scored_pairs, n, *, threshold=0.0):
    """Cluster ``n`` items to minimize disagreements (the greedy Pivot algorithm).

    ``scored_pairs`` is an iterable of ``(i, j, score)``; an edge is *positive* when
    ``score >= threshold``. Pivot processes items in index order, clustering each
    still-unclustered item with its still-unclustered positive neighbours. Unlike
    connected-components it does not chain transitively, so a single stray edge cannot
    collapse two clusters.
    """
    positive: dict = defaultdict(set)
    for i, j, s in scored_pairs:
        if s >= threshold:
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
clusterers = Registry('clusterer')


def _connected_components_clusterer(scored_pairs, n, *, threshold=0.0):
    pairs = [(i, j) for i, j, s in scored_pairs if s >= threshold]
    return connected_components(pairs, n)


clusterers.register('connected_components', lambda: _connected_components_clusterer)
clusterers.register('correlation', lambda: correlation_clustering)


def resolve_clusterer(spec='connected_components'):
    """Resolve ``spec`` to a clusterer ``(scored_pairs, n, *, threshold) -> Partition``."""
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


def _majority_merge(records):
    fields = {f for r in records if isinstance(r, Mapping) for f in r}
    out = {}
    for f in fields:
        vals = [r[f] for r in records if isinstance(r, Mapping) and r.get(f) is not None]
        if vals:
            out[f] = Counter(vals).most_common(1)[0][0]
    return out


def _merge(records, policy):
    if not records:
        return None
    if policy == 'first':
        return records[0]
    if policy == 'most_complete':
        return max(records, key=_completeness)
    if policy == 'majority':
        return _majority_merge(records)
    raise ValueError(f"unknown canonicalize policy {policy!r}")


def canonicalize(partition, records, *, policy='most_complete'):
    """Merge each cluster of ``partition`` into one golden record from ``records``.

    ``policy``: ``'first'`` (keep the first member), ``'most_complete'`` (fewest missing
    fields, the default), or ``'majority'`` (per-field majority vote over mapping records).
    Returns ``{cluster_id: golden_record}``.
    """
    return {
        cid: _merge([records[m] for m in members], policy)
        for cid, members in partition.groups().items()
    }
