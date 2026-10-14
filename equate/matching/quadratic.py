"""Quadratic matchers: align the *nodes* of two graphs (network alignment).

The second matcher arity (decision register D8, doc ``12`` §4). A flat matcher picks
pairs from a fixed score matrix; here how well ``u`` matches ``v`` *depends on how their
neighbors are matched* — a quadratic assignment problem — so the scores and the
correspondence are produced together, from the two graphs. Each quadratic matcher builds a
node-pair score matrix and hands it to a **flat** matcher (``extract=``, default the
optimal LAP) for the final 1:1 correspondence.

- ``'isorank'`` (core) — IsoRank (Singh, Xu & Berger 2008): "two nodes match if their
  neighbors match", the stationary point of a random walk on the product graph, blended
  with an optional prior ``node_affinity``.
- ``'rrwm'`` (``equate[graph-matching]``) — pygmtools' reweighted random walk solver on
  the full quadratic affinity (edge-weight agreement + ``node_affinity``).

Use :func:`align` (also ``equate.align``), or pick a strategy from the
:data:`quadratic_matchers` registry. ``seeds`` are known node correspondences
(anchors): they are kept as given and steer the rest.
"""

import functools

import numpy as np

from equate.base import Matching, ScoreMatrix
from equate.registry import Registry
from equate._dependencies import require
from equate._graph import as_graph, dense_adjacency

__all__ = [
    "quadratic_matchers",
    "resolve_quadratic_matcher",
    "align",
    "isorank",
    "isorank_scores",
    "rrwm",
    "DFLT_ISORANK_ALPHA",
    "DFLT_MAX_ITER",
    "DFLT_TOL",
    "DFLT_EDGE_SIGMA",
]

#: weight of the topology term vs the ``node_affinity`` prior (1 = topology only).
DFLT_ISORANK_ALPHA = 0.8
DFLT_MAX_ITER = 100
DFLT_TOL = 1e-9
#: bandwidth of the Gaussian edge-weight agreement used by ``'rrwm'``.
DFLT_EDGE_SIGMA = 1.0

#: the quadratic-matcher strategy registry (name -> lazy factory)
quadratic_matchers = Registry("quadratic matcher")


# --- shared plumbing ----------------------------------------------------------------


def _prior(g1, g2, node_affinity):
    """The ``(n1, n2)`` non-negative node-affinity prior (uniform when ``None``)."""
    shape = (g1.n, g2.n)
    if node_affinity is None:
        return np.ones(shape)
    if callable(node_affinity):
        E = np.array(
            [[float(node_affinity(u, v)) for v in g2.nodes] for u in g1.nodes],
            dtype=float,
        ).reshape(shape)
    else:
        E = np.asarray(node_affinity, dtype=float)
        if E.shape != shape:
            raise ValueError(
                f"node_affinity must have shape {shape} (nodes of G1 x nodes of G2), "
                f"got {E.shape}"
            )
    if not np.all(np.isfinite(E)) or (E < 0).any():
        raise ValueError("node_affinity must be finite and non-negative")
    return E


def _seed_indices(g1, g2, seeds):
    """Validate ``seeds`` (node pairs) -> a list of ``(i, j)`` index pairs."""
    if not seeds:
        return []
    idx1, idx2 = g1.index(), g2.index()
    pairs = []
    for u, v in seeds:
        if u not in idx1:
            raise KeyError(f"seed node {u!r} is not a node of G1")
        if v not in idx2:
            raise KeyError(f"seed node {v!r} is not a node of G2")
        pairs.append((idx1[u], idx2[v]))
    if len({i for i, _ in pairs}) != len(pairs) or len({j for _, j in pairs}) != len(
        pairs
    ):
        raise ValueError("seeds must be a 1:1 correspondence (a node appears twice)")
    return pairs


def _extract(S, g1, g2, seed_pairs, extract):
    """Seeds as given + a flat matcher's 1:1 assignment of the remaining nodes."""
    from equate.matching import resolve_matcher

    seeded_rows = {i for i, _ in seed_pairs}
    seeded_cols = {j for _, j in seed_pairs}
    rows = [i for i in range(g1.n) if i not in seeded_rows]
    cols = [j for j in range(g2.n) if j not in seeded_cols]
    pairs = list(seed_pairs)
    if rows and cols:
        sub = ScoreMatrix(S[np.ix_(rows, cols)], sense="maximize")
        pairs += [(rows[a], cols[b]) for a, b in resolve_matcher(extract)(sub)]
    pairs.sort()
    return Matching(
        pairs=pairs,
        scores=[float(S[i, j]) for i, j in pairs],
        plan=S,
        row_labels=list(g1.nodes),
        col_labels=list(g2.nodes),
    )


# --- IsoRank (core) -----------------------------------------------------------------


def _walk_matrix(A):
    """Column-stochastic random-walk matrix ``A D^-1`` (an isolated node's column is 0)."""
    deg = A.sum(axis=0)
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return A * inv[None, :]


def isorank_scores(
    G1,
    G2,
    *,
    node_affinity=None,
    seeds=None,
    alpha=DFLT_ISORANK_ALPHA,
    max_iter=DFLT_MAX_ITER,
    tol=DFLT_TOL,
    label=None,
):
    """The IsoRank node-pair score matrix ``R`` (``n1 x n2``, sums to 1).

    Iterates ``R <- alpha * W1 R W2^T + (1 - alpha) * E`` to its fixed point, where
    ``W`` is each graph's column-stochastic walk matrix and ``E`` the normalized prior
    (``node_affinity``; uniform when ``None``). ``seeds`` pin their rows/columns of the
    prior to the given partner, so their anchoring propagates to the neighbors. Returned
    as a ``ScoreMatrix`` (``sense='maximize'``) labeled with the two node lists.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    if g1.n == 0 or g2.n == 0:
        return ScoreMatrix(
            np.zeros((g1.n, g2.n)), row_labels=list(g1.nodes), col_labels=list(g2.nodes)
        )
    E = _prior(g1, g2, node_affinity)
    for i, j in _seed_indices(g1, g2, seeds):
        E[i, :] = 0.0
        E[:, j] = 0.0
        E[i, j] = 1.0
    total = E.sum()
    if total == 0:
        raise ValueError("node_affinity is all zero: there is no prior to propagate")
    E = E / total
    W1, W2 = _walk_matrix(dense_adjacency(g1)), _walk_matrix(dense_adjacency(g2))
    R = E.copy()
    for _ in range(max_iter):
        nxt = alpha * (W1 @ R @ W2.T) + (1 - alpha) * E
        s = nxt.sum()
        nxt = nxt / s if s > 0 else E.copy()
        done = np.abs(nxt - R).sum() < tol
        R = nxt
        if done:
            break
    return ScoreMatrix(
        R, sense="maximize", row_labels=list(g1.nodes), col_labels=list(g2.nodes)
    )


def isorank(
    G1, G2, *, node_affinity=None, seeds=None, extract="optimal", label=None, **config
):
    """Align two graphs with IsoRank -> a :class:`~equate.base.Matching` of node pairs.

    ``matching.labeled_pairs()`` yields ``(G1 node, G2 node)``; ``.plan`` holds the full
    IsoRank score matrix. ``extract`` is the flat matcher that turns it into a 1:1
    correspondence (any name or callable :func:`equate.matching.resolve_matcher` takes);
    ``**config`` goes to :func:`isorank_scores` (``alpha``, ``max_iter``, ``tol``).

    >>> G1 = {'a': ['b'], 'b': ['a', 'c', 'd'], 'c': ['b'], 'd': ['b', 'e'], 'e': ['d']}
    >>> G2 = {1: [2, 4], 2: [1], 3: [4], 4: [1, 3, 5], 5: [4]}  # the same graph, renamed
    >>> dict(isorank(G1, G2).labeled_pairs())['b'], dict(isorank(G1, G2).labeled_pairs())['d']
    (4, 1)
    """
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    S = isorank_scores(g1, g2, node_affinity=node_affinity, seeds=seeds, **config).data
    return _extract(S, g1, g2, _seed_indices(g1, g2, seeds), extract)


# --- RRWM via pygmtools (optional) ---------------------------------------------------


def rrwm(
    G1,
    G2,
    *,
    node_affinity=None,
    seeds=None,
    extract="optimal",
    label=None,
    sigma=DFLT_EDGE_SIGMA,
    **rrwm_params,
):
    """Align two graphs with pygmtools' RRWM — requires ``equate[graph-matching]``.

    Builds the quadratic affinity ``K`` from Gaussian edge-weight agreement (bandwidth
    ``sigma``) with ``node_affinity`` on its diagonal, solves it with
    ``pygmtools.rrwm`` (``**rrwm_params`` forwarded), and extracts a 1:1 correspondence
    with the flat ``extract`` matcher. Seeds are kept and their rows/columns excluded from
    the extraction. Uses the numpy backend and never changes pygmtools' global backend.
    """
    pygm = require(
        "pygmtools", extra="graph-matching", purpose="quadratic graph matching"
    )
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    seed_pairs = _seed_indices(g1, g2, seeds)
    if g1.n == 0 or g2.n == 0:
        return _extract(np.zeros((g1.n, g2.n)), g1, g2, seed_pairs, extract)
    be = "numpy"
    A1, A2 = dense_adjacency(g1)[None], dense_adjacency(g2)[None]
    conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1, backend=be)
    conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2, backend=be)
    n1, n2 = np.array([g1.n]), np.array([g2.n])
    aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=sigma, backend=be)
    K = pygm.utils.build_aff_mat(
        None,
        edge1,
        conn1,
        None,
        edge2,
        conn2,
        n1,
        ne1,
        n2,
        ne2,
        edge_aff_fn=aff,
        backend=be,
    )
    if node_affinity is not None:
        E = _prior(g1, g2, node_affinity)
        # pygmtools vectorizes X column-major: entry (i, j) sits at j * n1 + i
        diag = np.arange(K.shape[-1])
        K[0, diag, diag] += E.ravel(order="F")
    X = np.asarray(pygm.rrwm(K, n1, n2, backend=be, **rrwm_params))[0]
    return _extract(X, g1, g2, seed_pairs, extract)


# --- registry + facade --------------------------------------------------------------


def _const(fn, name):
    def factory(**config):
        return functools.partial(fn, **config) if config else fn

    factory.__name__ = f"_{name}_factory"
    return factory


quadratic_matchers.register("isorank", _const(isorank, "isorank"))
quadratic_matchers.register("rrwm", _const(rrwm, "rrwm"))


def resolve_quadratic_matcher(spec="isorank", **config):
    """A registered name (built with ``**config``) or a callable ``(G1, G2, **kw) -> Matching``."""
    if isinstance(spec, str):
        return quadratic_matchers.create(spec, **config)
    if callable(spec):
        return functools.partial(spec, **config) if config else spec
    raise TypeError(
        f"quadratic matcher spec must be a registered name or a callable, "
        f"got {type(spec).__name__}"
    )


def align(G1, G2, *, how="isorank", node_affinity=None, seeds=None, **config):
    """Align the nodes of two graphs -> a :class:`~equate.base.Matching` of node pairs.

    ``how`` picks the quadratic matcher (``'isorank'`` core, ``'rrwm'`` via
    ``equate[graph-matching]``, or a callable); ``node_affinity`` is an optional prior
    (an ``(n1, n2)`` array or a ``(u, v) -> float`` callable, non-negative); ``seeds`` are
    known ``(G1 node, G2 node)`` anchors. Other keywords configure the matcher (e.g.
    ``extract='greedy'``, ``alpha=0.9``). Graphs may be networkx graphs, ``{node:
    neighbors}`` mappings or adjacency matrices.

    >>> star = {'hub': ['x', 'y', 'z'], 'x': ['hub'], 'y': ['hub'], 'z': ['hub']}
    >>> other = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]}
    >>> dict(align(star, other).labeled_pairs())['hub']
    3
    """
    matcher = resolve_quadratic_matcher(how)
    return matcher(G1, G2, node_affinity=node_affinity, seeds=seeds, **config)
