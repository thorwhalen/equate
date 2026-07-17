"""The equate facade: the one-line ``match(A, B)`` entry point.

Wires the pipeline — featurize → compare (over the blocker's candidates) → match — with
sensible defaults, and returns a structured :class:`equate.base.Matching`. Every stage is
swappable by one keyword (decision register D4): ``featurize=``, ``compare=``, ``block=``,
``how=``. This is the canonical case (doc 10 §1): two collections -> optimally-matched
(and scored) pairs, zero config.
"""

from equate.base import Matching, ScoreMatrix
from equate.compare import resolve_comparator, direct, featurized
from equate.block import resolve_blocker, score_candidates, all_pairs
from equate.matching import resolve_matcher, soft_match, harden
from equate.cluster import resolve_clusterer

__all__ = ["match", "dedupe", "resolve"]


def match(
    A,
    B=None,
    *,
    featurize=None,
    compare=None,
    block=None,
    how="assign",
    threshold=None,
    cluster="connected_components",
    sense="maximize",
):
    """Match two collections into a :class:`~equate.base.Matching` of scored ``(i, j)`` pairs.

    Pipeline (each stage swappable by one keyword):

    - **featurize / compare** — with ``compare=None`` (default), the *vector route*:
      featurize both sides with ``featurize`` (TF-IDF by default) and score by cosine.
      With a ``compare`` comparator (a registered name or a callable), the *direct
      pairwise route* on the raw items.
    - **block** — ``None`` (default) scores all pairs (dense); a blocker (name/callable)
      scores only the candidate pairs into a sparse matrix, and the matcher stays sparse-aware.
    - **how** — the matcher objective: ``'assign'`` (optimal 1:1, the default),
      ``'greedy'``, ``'stable'``, or ``'soft'`` (an optimal-transport plan; ``.plan`` is set).

    Returns a ``Matching``: iterate it for ``(i, j)`` pairs, ``dict(...)`` it for a mapping,
    ``.labeled_pairs()`` for ``(A[i], B[j])``; ``.scores`` holds each matched pair's score.

    >>> keys = ['apple pie', 'banana split', 'cherry tart']
    >>> values = ['cherry tarte', 'aple pie', 'banana split']
    >>> m = match(keys, values, compare='ratio')
    >>> dict(m.labeled_pairs())
    {'apple pie': 'aple pie', 'banana split': 'banana split', 'cherry tart': 'cherry tarte'}
    """
    if B is None:
        raise ValueError(
            "match() matches two collections; to deduplicate a single collection use "
            "equate.dedupe(A)."
        )
    if compare is not None and featurize is not None:
        raise ValueError(
            "featurize (the vector route) and compare (the direct pairwise route) are "
            "mutually exclusive — pass one, not both."
        )
    A = list(A)
    B = list(B)

    if block is not None:
        candidates = list(resolve_blocker(block)(A, B))
        scores = score_candidates(
            A, B, candidates, compare or "ratio", sense=sense
        ).data
    elif compare is not None:
        scores = direct(resolve_comparator(compare))(A, B)
    else:
        scores = featurized(featurize or "tfidf")(A, B)

    # the ScoreMatrix SSOT carries sense + labels and owns every densify (holes
    # worst-cased); a matched pair's retained score is read via sm.score_at(i, j)
    sm = ScoreMatrix(scores, sense=sense, row_labels=A, col_labels=B)

    if how == "cluster":
        # cluster the pooled A+B index space (B offset by len(A)) using the A-B match
        # edges above `threshold` — cross-collection entity resolution -> a Partition
        thr = 0.5 if threshold is None else threshold
        n_a = len(A)
        edges = [(int(i), n_a + int(j), v) for i, j, v in sm.stored_entries()]
        return resolve_clusterer(cluster)(
            edges, n_a + len(B), threshold=thr, sense=sense
        )

    if how == "soft":
        plan = soft_match(sm)
        # harden() solves over the DENSE transport plan and knows nothing of blocking, so a
        # forced-partial plan can put mass on a hole; drop those (D11) as every other path does
        pairs = sm.drop_holes(harden(plan))
        return Matching(
            pairs=pairs,
            scores=[sm.score_at(i, j) for i, j in pairs],
            plan=plan,
            row_labels=A,
            col_labels=B,
        )

    pairs = list(resolve_matcher(how)(sm))
    return Matching(
        pairs=pairs,
        scores=[sm.score_at(i, j) for i, j in pairs],
        row_labels=A,
        col_labels=B,
    )


def dedupe(
    A,
    *,
    compare="ratio",
    block=None,
    threshold=0.7,
    cluster="connected_components",
    sense=None,
):
    """Deduplicate a single collection into entity groups.

    Self-compares ``A``'s candidate pairs, keeps those passing ``threshold``, and clusters
    them -> a :class:`~equate.base.Partition` over ``A`` (iterate it for the duplicate
    ``(i, j)`` pairs; ``.groups()`` for the clusters). ``block`` (a blocker) restricts the
    candidate pairs (self-join); the default compares all ``i < j`` pairs. ``cluster`` is
    ``'connected_components'`` (transitive closure) or ``'correlation'``. ``sense`` defaults
    to the comparator's declared polarity — a *distance* comparator (e.g.
    ``levenshtein_distance``) uses ``'minimize'`` so *lower* scores link, a similarity uses
    ``'maximize'``.

    >>> from equate import dedupe
    >>> dedupe(['Jon Smith', 'Jon Smyth', 'Kate Doe'], threshold=0.8).groups()
    {0: [0, 1], 1: [2]}
    """
    A = list(A)
    comp = resolve_comparator(compare)
    if sense is None:
        meta = getattr(comp, "meta", None)
        sense = (
            "minimize" if getattr(meta, "polarity", None) == "distance" else "maximize"
        )
    candidates = (
        list(resolve_blocker(block)(A)) if block is not None else list(all_pairs(A))
    )
    scored = [(i, j, float(comp(A[i], A[j]))) for i, j in candidates]
    return resolve_clusterer(cluster)(scored, len(A), threshold=threshold, sense=sense)


def resolve(*collections, **kwargs):
    """Resolve entities across multiple collections: pool them (in order) and
    :func:`dedupe` -> a Partition over the pooled items. Generalizes cross-collection
    matching to n-way linkage. Accepts the same keywords as :func:`dedupe`.
    """
    pooled = [x for collection in collections for x in collection]
    return dedupe(pooled, **kwargs)
