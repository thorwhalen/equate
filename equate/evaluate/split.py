"""Leakage-aware splitting: split over entities, so test entities are unseen.

The most common way a matching benchmark lies is a pair-level split: the same real-world
entity shows up in train and test, and the model memorises it (research doc 14, §7.2).
:func:`entity_split` makes the right split the easy one — it partitions **entity ids**,
so every record of an entity lands in exactly one part — and :func:`assert_no_leakage`
is the cheap post-hoc guard for a split made some other way.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    """Record indices of each part; the entity ids of a part never occur in another."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]


def entity_split(
    entity_ids: Sequence, *, ratios: tuple = (0.6, 0.2, 0.2), seed: int = 0
) -> Split:
    """Split records into train / val / test by entity, never by record or pair.

    ``entity_ids[i]`` is record ``i``'s true entity (e.g. from
    :func:`~equate.evaluate.cross_pairs_to_labels` over a two-table gold matching).
    Entities, not records, are shuffled (``seed``) and cut by ``ratios``, so the parts
    are disjoint in entities and roughly — not exactly — proportional in records.

    >>> s = entity_split([0, 0, 1, 1, 2, 3, 4, 4, 5, 6], seed=1)
    >>> assert_no_leakage([0, 0, 1, 1, 2, 3, 4, 4, 5, 6], s.train, s.test)
    >>> sorted(s.train + s.val + s.test) == list(range(10))
    True
    """
    if (
        len(ratios) != 3
        or not all(np.isfinite(r) and r >= 0 for r in ratios)
        or not sum(ratios)
    ):
        raise ValueError(f"ratios must be three non-negative numbers, got {ratios!r}")
    ids = np.asarray(entity_ids)
    entities = np.unique(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(entities)
    fractions = np.cumsum(ratios) / sum(ratios)
    cut1, cut2 = (np.round(fractions[:2] * len(entities))).astype(int)
    parts = (set(entities[:cut1]), set(entities[cut1:cut2]), set(entities[cut2:]))
    idx = [
        tuple(int(i) for i in np.flatnonzero([e in part for e in ids]))
        for part in parts
    ]
    return Split(*idx)


def assert_no_leakage(entity_ids: Sequence, *parts: Sequence[int]) -> None:
    """Raise ``AssertionError`` if any entity has records in more than one of ``parts``.

    >>> assert_no_leakage([0, 0, 1], [0], [1])
    Traceback (most recent call last):
        ...
    AssertionError: entity 0 appears in parts 0 and 1 — the split leaks it
    """
    seen: dict = {}
    for k, part in enumerate(parts):
        for i in part:
            e = entity_ids[i]
            first = seen.setdefault(e, k)
            if first != k:
                raise AssertionError(
                    f"entity {e!r} appears in parts {first} and {k} — the split leaks it"
                )
