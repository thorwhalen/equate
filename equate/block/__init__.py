"""The block stage (② scalability): candidate generation.

A *blocker* is ``Callable[(A[, B]), Iterable[(i, j)]]`` yielding candidate index pairs
**lazily** — never a materialized matrix. The default :func:`all_pairs` is the
correctness baseline (the dense path is a degenerate blocker). Blocking is precisely the
decision to leave most cells of the score matrix uncomputed (decision register D1);
scoring a candidate stream yields a sparse ``ScoreMatrix`` (:func:`score_candidates`).

Select a blocker by name via the :data:`blockers` registry or pass a callable. Blocking
quality is measured with :func:`blocking_metrics` (PC / RR / PQ).

A blocker requests a **self-join** (dedup within one collection, yielding ``i < j``) via
the single-argument form ``blocker(A)``. The two-argument ``blocker(A, B)`` is a
cross-join. Self-join is detected by object *identity*, so ``blocker(A, A)`` is also a
self-join, but ``blocker(A, list(A))`` — a value-equal *copy* — is a cross-join; use the
single-argument form to be unambiguous.
"""

from equate.registry import Registry
from equate.block.keyed import (
    key_blocking,
    sorted_neighborhood,
    first_chars,
    qgram_keys,
    whole_value,
    phonetic_key,
)
from equate.block.metrics import blocking_metrics
from equate.block.score import score_candidates
from equate.block import metablock
from equate.block.ann import brute_knn_blocking, ann_blocking
from equate.block.lsh import minhash_lsh_blocking

__all__ = [
    'blockers',
    'all_pairs',
    'resolve_blocker',
    'key_blocking',
    'sorted_neighborhood',
    'first_chars',
    'qgram_keys',
    'whole_value',
    'phonetic_key',
    'blocking_metrics',
    'score_candidates',
    'metablock',
    'brute_knn_blocking',
    'ann_blocking',
    'minhash_lsh_blocking',
]

#: the block-stage strategy registry (name -> lazy factory)
blockers = Registry('blocker')


def all_pairs(A, B=None):
    """The correctness-baseline blocker: every pair (dense). A self-join yields ``i < j``.

    >>> list(all_pairs(['a', 'b'], ['x', 'y']))
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> list(all_pairs(['a', 'b', 'c']))          # self-join: i < j
    [(0, 1), (0, 2), (1, 2)]
    """
    self_join = B is None or B is A
    A = list(A)
    B = A if self_join else list(B)
    for i in range(len(A)):
        js = range(i + 1, len(A)) if self_join else range(len(B))
        for j in js:
            yield (i, j)


def _const_blocker(fn, name):
    """Factory returning a fixed blocker; passing config is a clear error."""

    def factory(**config):
        if config:
            raise TypeError(
                f"blocker {name!r} takes no configuration; got {list(config)}"
            )
        return fn

    return factory


blockers.register('all_pairs', _const_blocker(all_pairs, 'all_pairs'))
blockers.register('key', key_blocking)
blockers.register('sorted_neighborhood', sorted_neighborhood)
blockers.register('brute_knn', brute_knn_blocking)
blockers.register('ann', ann_blocking)
blockers.register('minhash_lsh', minhash_lsh_blocking)


def resolve_blocker(spec=None, **config):
    """Resolve ``spec`` to a blocker ``(A[, B]) -> Iterable[(i, j)]``.

    - ``None`` -> :func:`all_pairs` (the dense baseline);
    - a registered name (``str``) -> built via its factory (``**config`` forwarded);
    - a callable -> returned as-is.
    """
    if spec is None:
        return all_pairs
    if callable(spec):
        return spec
    if isinstance(spec, str):
        return blockers.create(spec, **config)
    raise TypeError(
        f"blocker spec must be None, a registered name, or a callable, "
        f"got {type(spec).__name__}"
    )
