"""Convert between the two granularities: pairs and cluster labels.

Pairwise metrics take pair sets; cluster metrics take a label per record. The
conversion is a deliberate act (research doc 14, §2.2): :func:`pairs_to_labels` takes
the **transitive closure** of a pair set (union-find), which can merge records a matcher
never linked directly, so a user who evaluates a 1:1 matcher with a cluster metric sees
exactly what they are asking for.
"""

from collections.abc import Iterable, Sequence


def pairs_to_labels(pairs: Iterable[tuple], *, n: int) -> list[int]:
    """Cluster label per record ``0..n-1`` from the transitive closure of ``pairs``.

    Labels are canonical: numbered in order of each cluster's smallest record, so two
    pair sets that induce the same partition give equal label lists. A record no pair
    mentions is a singleton.

    >>> pairs_to_labels([(0, 1), (1, 2), (4, 5)], n=6)
    [0, 0, 0, 1, 2, 2]
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"pair {(i, j)} is outside 0..{n - 1}")
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    canon: dict[int, int] = {}
    return [canon.setdefault(find(i), len(canon)) for i in range(n)]


def labels_to_pairs(labels: Sequence) -> set[tuple[int, int]]:
    """Every within-cluster pair ``(i, j)`` with ``i < j``.

    >>> sorted(labels_to_pairs([0, 0, 1, 0]))
    [(0, 1), (0, 3), (1, 3)]
    """
    groups: dict = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    return {
        (members[a], members[b])
        for members in groups.values()
        for a in range(len(members))
        for b in range(a + 1, len(members))
    }


def cross_pairs_to_labels(pairs: Iterable[tuple], *, n_a: int, n_b: int) -> list[int]:
    """Labels over ``A + B`` (B's records offset by ``n_a``) from two-table ``(i, j)`` pairs.

    The bridge from a matching between two collections to the entity view a cluster
    metric or an entity split needs.

    >>> cross_pairs_to_labels([(0, 1)], n_a=2, n_b=2)
    [0, 1, 2, 0]
    """
    return pairs_to_labels(((i, n_a + j) for i, j in pairs), n=n_a + n_b)
