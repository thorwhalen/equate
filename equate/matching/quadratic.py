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
- ``'final'`` (core) — FINAL (Zhang & Tong 2016): IsoRank's propagation gated by node
  attribute agreement (``label=`` / ``attribute_similarity=``).
- ``'regal'``, ``'cone'`` (core, :mod:`equate.matching.embedding_align`) — embedding
  aligners: REGAL/xNetMF (joint structural embeddings) and CONE-Align (per-graph NetMF
  embeddings aligned by Procrustes + optimal transport).
- ``'gw'`` (``equate[ot]``) — Gromov-Wasserstein optimal transport on shortest-path
  distances (fused with ``node_affinity`` when given); ``.plan`` is the transport plan.
- ``'rrwm'`` (``equate[graph-matching]``) — pygmtools' reweighted random walk solver on
  the full quadratic affinity (edge-weight agreement + ``node_affinity``).

PLANETALIGN is not wrapped: it has no PyPI distribution, and the in-house aligners above
cover its core methods (a git-sourced backend can register here later as another entry).

Use :func:`align` (also ``equate.align``), or pick a strategy from the
:data:`quadratic_matchers` registry. ``seeds`` are known node correspondences
(anchors): they are kept as given and steer the rest.
"""

import functools
import warnings

import numpy as np
from scipy.sparse import issparse

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
    "final",
    "final_scores",
    "gw",
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
#: fused GW: weight of the structure term vs the node-feature cost (1 = structure only).
DFLT_FGW_ALPHA = 0.5

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
        ).reshape(shape)  # reshape: keeps an empty side's shape
    elif issparse(node_affinity):
        E = np.asarray(node_affinity.toarray(), dtype=float)
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


def _pinned_prior(g1, g2, node_affinity, seeds):
    """The prior, seeds pinned (a seed's row and column keep only the seed, at the
    prior's top value), normalized to sum 1."""
    E = _prior(g1, g2, node_affinity)
    top = float(E.max()) or 1.0  # a seed is as strong as the strongest prior, any scale
    for i, j in _seed_indices(g1, g2, seeds):
        E[i, :] = 0.0
        E[:, j] = 0.0
        E[i, j] = top
    total = E.sum()
    if total == 0:
        raise ValueError("node_affinity is all zero: there is no prior to propagate")
    return E / total


def _seed_indices(g1, g2, seeds):
    """Validate ``seeds`` (node pairs) -> a list of ``(i, j)`` index pairs."""
    if not seeds:
        return []
    idx1, idx2 = g1.index(), g2.index()
    pairs = []
    for u, v in dict.fromkeys(tuple(s) for s in seeds):  # a repeated pair is one seed
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
    prior to the given partner (at the prior's own top value), so their anchoring
    propagates to the neighbors. A directed graph walks *against* its edges (the column
    normalization uses in-degrees). Warns when ``max_iter`` runs out before ``tol``.
    Returned as a ``ScoreMatrix`` (``sense='maximize'``) labeled with the two node lists.

    The prior does real work: from topology alone (``node_affinity=None``) the scores
    tend to the product of node degrees, which cannot tell equal-degree nodes apart. Pass
    a node similarity (attributes, names, degrees) for anything but degree-distinct graphs.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    if g1.n == 0 or g2.n == 0:
        return ScoreMatrix(
            np.zeros((g1.n, g2.n)),
            sense="maximize",
            row_labels=list(g1.nodes),
            col_labels=list(g2.nodes),
        )
    E = _pinned_prior(g1, g2, node_affinity, seeds)
    W1, W2 = _walk_matrix(dense_adjacency(g1)), _walk_matrix(dense_adjacency(g2))
    R = E.copy()
    for _ in range(max_iter):
        nxt = alpha * (W1 @ R @ W2.T) + (1 - alpha) * E
        s = nxt.sum()
        nxt = nxt / s if s > 0 else E.copy()
        delta = np.abs(nxt - R).sum()
        R = nxt
        if delta < tol:
            break
    else:
        warnings.warn(
            f"IsoRank did not converge in max_iter={max_iter} iterations "
            f"(last change {delta:.2e} > tol={tol:.0e}); raise max_iter or lower alpha",
            RuntimeWarning,
            stacklevel=2,
        )
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


# --- FINAL (core): IsoRank with node-attribute agreement ------------------------------


def _attribute_agreement(g1, g2, attribute_similarity):
    """``N``: how far node ``u``'s attributes agree with node ``v``'s, in ``[0, 1]``.

    ``None`` compares the graphs' node labels (``label=``): 1 when equal, 0 otherwise —
    all ones when neither graph carries labels. A callable ``(u, v) -> [0, 1]`` or an
    ``(n1, n2)`` array is used as given.
    """
    shape = (g1.n, g2.n)
    if attribute_similarity is None:
        l1, l2 = (
            np.asarray(g1.labels, dtype=object),
            np.asarray(g2.labels, dtype=object),
        )
        return (l1[:, None] == l2[None, :]).astype(float).reshape(shape)
    if callable(attribute_similarity):
        N = np.array(
            [[float(attribute_similarity(u, v)) for v in g2.nodes] for u in g1.nodes],
            dtype=float,
        ).reshape(shape)
    else:
        N = np.asarray(attribute_similarity, dtype=float)
        if N.shape != shape:
            raise ValueError(
                f"attribute_similarity must have shape {shape}, got {N.shape}"
            )
    if not np.all(np.isfinite(N)) or (N < 0).any() or (N > 1).any():
        raise ValueError("attribute_similarity must be in [0, 1]")
    return N


def final_scores(
    G1,
    G2,
    *,
    node_affinity=None,
    seeds=None,
    attribute_similarity=None,
    alpha=DFLT_ISORANK_ALPHA,
    max_iter=DFLT_MAX_ITER,
    tol=DFLT_TOL,
    label=None,
):
    """The FINAL node-pair score matrix (Zhang & Tong, KDD 2016; the FINAL-N variant).

    IsoRank's "two nodes match if their neighbors match", with every step of the walk on
    the product graph filtered by **node-attribute agreement** ``N`` (from ``label=``, or
    ``attribute_similarity``) and symmetrically degree-normalized:
    ``X <- alpha * N * D^-1/2 (A1 (N * D^-1/2 * X) A2^T) + (1 - alpha) * H``, ``H`` the
    normalized prior (``node_affinity``, seeds pinned as in IsoRank). Without attributes
    (``N`` all ones) it reduces to IsoRank-style propagation. Returned as a
    ``ScoreMatrix`` (``sense='maximize'``).
    """
    if not 0 <= alpha < 1:
        raise ValueError(f"alpha must be in [0, 1), got {alpha!r}")
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    labels = dict(row_labels=list(g1.nodes), col_labels=list(g2.nodes))
    if g1.n == 0 or g2.n == 0:
        return ScoreMatrix(np.zeros((g1.n, g2.n)), sense="maximize", **labels)
    H = _pinned_prior(g1, g2, node_affinity, seeds)
    N = _attribute_agreement(g1, g2, attribute_similarity)
    A1, A2 = dense_adjacency(g1), dense_adjacency(g2)
    degree = N * (
        A1 @ N @ A2.T
    )  # each node pair's degree in the filtered product graph
    root = np.sqrt(degree, where=degree > 0, out=np.zeros_like(degree))
    inv_root = np.divide(1.0, root, out=np.zeros_like(root), where=root > 0)
    X = H.copy()
    for _ in range(max_iter):
        nxt = alpha * N * inv_root * (A1 @ (N * inv_root * X) @ A2.T) + (1 - alpha) * H
        delta = np.abs(nxt - X).sum()
        X = nxt
        if delta < tol:
            break
    else:
        warnings.warn(
            f"FINAL did not converge in max_iter={max_iter} iterations "
            f"(last change {delta:.2e} > tol={tol:.0e})",
            RuntimeWarning,
            stacklevel=2,
        )
    return ScoreMatrix(X, sense="maximize", **labels)


def final(
    G1,
    G2,
    *,
    node_affinity=None,
    seeds=None,
    extract="optimal",
    label=None,
    **config,
):
    """Align two graphs with FINAL -> a :class:`~equate.base.Matching` of node pairs.

    Like :func:`isorank`, with node attributes (``label=`` or ``attribute_similarity=``)
    gating the propagation; ``**config`` goes to :func:`final_scores`.

    >>> G1 = {'a': ['b'], 'b': ['a', 'c'], 'c': ['b']}
    >>> G2 = {1: [2], 2: [1, 3], 3: [2]}
    >>> kinds = {'a': 'x', 'b': 'y', 'c': 'z', 1: 'z', 2: 'y', 3: 'x'}
    >>> dict(final(G1, G2, label=kinds.get).labeled_pairs())
    {'a': 3, 'b': 2, 'c': 1}
    """
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    S = final_scores(g1, g2, node_affinity=node_affinity, seeds=seeds, **config).data
    return _extract(S, g1, g2, _seed_indices(g1, g2, seeds), extract)


# --- Gromov-Wasserstein via POT (optional) --------------------------------------------


def _hop_distances(g):
    """All-pairs shortest-path lengths (edge weights as lengths), unreachable = max + 1,
    scaled to ``[0, 1]`` — the intra-graph structure Gromov-Wasserstein compares."""
    from scipy.sparse.csgraph import shortest_path

    A = g.adjacency.maximum(g.adjacency.T)  # distances on the undirected skeleton
    D = shortest_path(A, directed=False)
    finite = D[np.isfinite(D)]
    top = float(finite.max()) if finite.size else 0.0
    D[~np.isfinite(D)] = top + 1.0
    return D / D.max() if D.max() > 0 else D


def gw(
    G1,
    G2,
    *,
    node_affinity=None,
    seeds=None,
    extract="optimal",
    label=None,
    alpha=DFLT_FGW_ALPHA,
    **gw_params,
):
    """Align two graphs by Gromov-Wasserstein optimal transport — ``equate[ot]`` (POT).

    GW finds the coupling of the two node sets that best preserves *distances*: nodes
    far apart in ``G1`` should map to nodes far apart in ``G2`` (shortest-path lengths,
    scaled to ``[0, 1]``). With a ``node_affinity`` (or ``seeds``) it is *fused* GW: a
    node-feature cost ``1 - affinity/max`` weighted ``1 - alpha`` against the structure
    (``alpha``). ``.plan`` holds the transport plan, the soft correspondence; ``extract``
    hardens it. ``**gw_params`` go to POT (``max_iter``, ``tol_rel``, ...).
    """
    ot = require("ot", extra="ot", purpose="Gromov-Wasserstein graph alignment")
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    seed_pairs = _seed_indices(g1, g2, seeds)
    if g1.n == 0 or g2.n == 0:
        return _extract(np.zeros((g1.n, g2.n)), g1, g2, seed_pairs, extract)
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    C1, C2 = _hop_distances(g1), _hop_distances(g2)
    p, q = np.full(g1.n, 1 / g1.n), np.full(g2.n, 1 / g2.n)
    if node_affinity is None and not seed_pairs:
        plan = ot.gromov.gromov_wasserstein(C1, C2, p, q, "square_loss", **gw_params)
    else:
        E = _prior(g1, g2, node_affinity)
        M = 1.0 - E / E.max() if E.max() > 0 else np.zeros_like(E)
        for i, j in seed_pairs:  # a seed: free there, maximal cost to pair elsewhere
            M[i, :] = 1.0
            M[:, j] = 1.0
            M[i, j] = 0.0
        plan = ot.gromov.fused_gromov_wasserstein(
            M, C1, C2, p, q, "square_loss", alpha=alpha, **gw_params
        )
    return _extract(np.asarray(plan, dtype=float), g1, g2, seed_pairs, extract)


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
    ``sigma``) with ``node_affinity`` on its diagonal, solves it with ``pygmtools.rrwm``
    (``**rrwm_params`` forwarded), and extracts a 1:1 correspondence with the flat
    ``extract`` matcher. Seeds *condition* the solve: every assignment contradicting one
    (its node paired elsewhere) is removed from ``K`` and the seed pair gets the top
    affinity, so the rest is aligned around it; seeds are then kept as given. Graphs with
    no edges and no prior fall back to a uniform node affinity. Uses the numpy backend
    and never changes pygmtools' global backend.
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
    K = np.array(K, dtype=float)  # writable copy
    diag = np.arange(K.shape[-1])
    # pygmtools vectorizes X column-major: entry (i, j) sits at j * n1 + i
    if node_affinity is not None:
        K[0, diag, diag] += _prior(g1, g2, node_affinity).ravel(order="F")
    if not K.any():  # no edges, no prior: nothing to solve but a uniform guess
        K[0, diag, diag] = 1.0
    top = float(K.max())
    for i, j in seed_pairs:
        clash = [jj * g1.n + i for jj in range(g2.n) if jj != j]
        clash += [j * g1.n + ii for ii in range(g1.n) if ii != i]
        K[0, clash, :] = 0.0
        K[0, :, clash] = 0.0
        K[0, j * g1.n + i, j * g1.n + i] = top
    X = np.asarray(pygm.rrwm(K, n1, n2, backend=be, **rrwm_params))[0]
    return _extract(X, g1, g2, seed_pairs, extract)


# --- registry + facade --------------------------------------------------------------


def _const(fn, name):
    def factory(**config):
        return functools.partial(fn, **config) if config else fn

    factory.__name__ = f"_{name}_factory"
    return factory


quadratic_matchers.register("isorank", _const(isorank, "isorank"))
quadratic_matchers.register("final", _const(final, "final"))
quadratic_matchers.register("rrwm", _const(rrwm, "rrwm"))
quadratic_matchers.register("gw", _const(gw, "gw"))


def _embedding_factory(name):
    def factory(**config):
        from equate.matching import embedding_align

        fn = getattr(embedding_align, name)
        return functools.partial(fn, **config) if config else fn

    factory.__name__ = f"_{name}_factory"
    return factory


quadratic_matchers.register("regal", _embedding_factory("regal"))
quadratic_matchers.register("cone", _embedding_factory("cone"))


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
    (an ``(n1, n2)`` array or a ``(u, v) -> float`` callable, non-negative) and usually
    what makes topology-only alignment work (see :func:`isorank_scores`); ``seeds`` are
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
