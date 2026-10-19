"""Embedding-based network alignment: embed both graphs' nodes, then match the embeddings.

Two quadratic matchers (doc ``12`` §4) that scale past IsoRank-style propagation by
reasoning over node *embeddings* instead of the ``n1 x n2`` product graph, both in-house
numpy (no extra):

- ``'regal'`` — REGAL / xNetMF (Heimann et al., CIKM 2018): each node is described by the
  degree distribution of its k-hop neighbourhood (plus its label), the two graphs are
  embedded *jointly* through a Nyström factorization of that similarity, and nodes are
  matched by embedding similarity. Needs no correspondence to start from.
- ``'cone'`` — CONE-Align (Chen et al., CIKM 2020): each graph gets its own proximity
  embedding (NetMF), and the two embedding spaces are aligned by alternating an
  orthogonal Procrustes rotation with an entropic optimal-transport correspondence,
  starting from REGAL's structural similarity (or ``node_affinity``).

Both return a :class:`~equate.base.Matching` through the same flat ``extract`` step as
IsoRank, with the node-pair scores in ``.plan``. Use them as ``equate.align(G1, G2,
how='regal')`` or ``how='cone'``.
"""

import math
import warnings

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.special import logsumexp

from equate._graph import as_graph, dense_adjacency
from equate.matching.quadratic import _extract, _prior, _seed_indices

__all__ = ["regal", "regal_scores", "cone", "cone_scores", "sinkhorn_plan"]

DFLT_MAX_HOPS = 2  # xNetMF: neighbourhood depth
DFLT_HOP_DISCOUNT = 0.1  # xNetMF: weight of hop k is discount ** (k - 1)
DFLT_GAMMA_STRUCT = 1.0
DFLT_GAMMA_ATTR = 1.0
DFLT_LANDMARK_FACTOR = 10  # xNetMF: p = factor * log2(n) landmarks
DFLT_SEED = 0
DFLT_CONE_DIM = 64
DFLT_CONE_WINDOW = 10  # NetMF context window
DFLT_CONE_ITER = 30  # Procrustes / OT alternations
DFLT_SINKHORN_REG = 0.05
DFLT_SINKHORN_TOL = 1e-6  # relative violation of the row marginals
_CHECK_EVERY = 10  # Sinkhorn rounds between convergence checks
DFLT_SINKHORN_ITER = 1000
DFLT_INNER_TOL = 1e-4  # CONE's solves: its plan is a heuristic's, marginals to ~1e-4


# --- REGAL / xNetMF ---------------------------------------------------------------------


def _undirected(g):
    A = dense_adjacency(g)
    return ((A + A.T) > 0).astype(float)


def _degree_bin(degrees, n_bins):
    bins = np.zeros(len(degrees), dtype=int)
    positive = degrees > 0
    bins[positive] = np.floor(np.log2(degrees[positive])).astype(int) + 1
    return np.minimum(bins, n_bins - 1)


def _structural_features(A, *, max_hops, discount, n_bins):
    """xNetMF identity: per node, the log-binned degrees of its k-hop neighbours,
    hop ``k`` weighted ``discount ** (k - 1)``."""
    n = A.shape[0]
    features = np.zeros((n, n_bins))
    if n == 0:
        return features
    degree_bins = _degree_bin(A.sum(axis=1), n_bins)
    hops = dijkstra(A, directed=False, unweighted=True, limit=max_hops)
    for k in range(1, max_hops + 1):
        at_k = hops == k
        for b in range(n_bins):
            features[:, b] += discount ** (k - 1) * at_k[:, degree_bins == b].sum(
                axis=1
            )
    return features


def _xnetmf_embed(
    As, labels, *, max_hops, discount, gamma_struct, gamma_attr, dim, seed
):
    """Joint xNetMF embedding of several graphs' nodes (rows stacked in order)."""
    max_degree = max((A.sum(axis=1).max() if A.size else 0) for A in As)
    n_bins = int(math.floor(math.log2(max_degree))) + 2 if max_degree > 0 else 1
    F = np.vstack(
        [
            _structural_features(A, max_hops=max_hops, discount=discount, n_bins=n_bins)
            for A in As
        ]
    )
    codes = {}  # labels -> integer codes: a tuple label stays one label
    labels = np.array([codes.setdefault(v, len(codes)) for v in labels], dtype=int)
    n = F.shape[0]
    p = min(n, max(1, int(DFLT_LANDMARK_FACTOR * math.log2(max(n, 2)))))
    if dim is not None:
        p = min(n, max(int(dim), 1))
    landmarks = np.random.default_rng(seed).choice(n, size=p, replace=False)
    sq = ((F[:, None, :] - F[None, landmarks, :]) ** 2).sum(axis=2)
    mismatch = (labels[:, None] != labels[None, landmarks]).astype(float)
    C = np.exp(-gamma_struct * sq - gamma_attr * mismatch)  # n x p
    W = C[landmarks]  # p x p
    U, s, _ = np.linalg.svd(np.linalg.pinv(W, hermitian=True))
    Y = C @ U @ np.diag(np.sqrt(s))
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    return np.divide(Y, norms, out=np.zeros_like(Y), where=norms > 0)


def regal_scores(
    G1,
    G2,
    *,
    node_affinity=None,
    max_hops=DFLT_MAX_HOPS,
    discount=DFLT_HOP_DISCOUNT,
    gamma_struct=DFLT_GAMMA_STRUCT,
    gamma_attr=DFLT_GAMMA_ATTR,
    dim=None,
    seed=DFLT_SEED,
    label=None,
):
    """The REGAL node-pair score matrix ``exp(-||y_u - y_v||)`` (``n1 x n2``).

    Embeddings come from xNetMF over both graphs at once (so they share a space with no
    correspondence given): structural identity = degree histograms of the ``max_hops``
    neighbourhood, plus a label mismatch term (``label=``). ``dim`` sets the number of
    landmarks (default ``10 log2 n``); ``seed`` picks them reproducibly. A
    ``node_affinity`` prior multiplies the scores.
    """
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    if g1.n == 0 or g2.n == 0:
        return np.zeros((g1.n, g2.n))
    Y = _xnetmf_embed(
        [_undirected(g1), _undirected(g2)],
        list(g1.labels) + list(g2.labels),
        max_hops=max_hops,
        discount=discount,
        gamma_struct=gamma_struct,
        gamma_attr=gamma_attr,
        dim=dim,
        seed=seed,
    )
    Y1, Y2 = Y[: g1.n], Y[g1.n :]
    S = np.exp(-np.sqrt(_pairwise_sq(Y1, Y2)))  # n1 x n2, never n1 x n2 x dim
    if node_affinity is not None:
        S = S * _prior(g1, g2, node_affinity)
    return S


def regal(
    G1, G2, *, node_affinity=None, seeds=None, extract="optimal", label=None, **config
):
    """Align two graphs with REGAL (xNetMF embeddings) -> a :class:`~equate.base.Matching`.

    ``**config`` goes to :func:`regal_scores`; ``seeds`` are kept as given.

    >>> G1 = {'a': ['b'], 'b': ['a', 'c', 'd'], 'c': ['b'], 'd': ['b', 'e'], 'e': ['d']}
    >>> G2 = {1: [2, 4], 2: [1], 3: [4], 4: [1, 3, 5], 5: [4]}  # the same graph, renamed
    >>> dict(regal(G1, G2).labeled_pairs())['b']
    4
    """
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    S = regal_scores(g1, g2, node_affinity=node_affinity, **config)
    return _extract(S, g1, g2, _seed_indices(g1, g2, seeds), extract)


# --- CONE-Align -------------------------------------------------------------------------


def sinkhorn_plan(
    cost, *, reg=DFLT_SINKHORN_REG, max_iter=DFLT_SINKHORN_ITER, tol=DFLT_SINKHORN_TOL
):
    """Entropic optimal-transport plan between uniform marginals (log-domain Sinkhorn).

    Pure numpy (no POT): the plan ``P`` minimizes ``<P, cost> - reg * H(P)`` with rows
    summing to ``1/n`` and columns to ``1/m``. ``reg`` is relative to the cost's spread,
    so the same value works at any cost scale. Warns if ``max_iter`` runs out first.

    >>> P = sinkhorn_plan(np.array([[0.0, 1.0], [1.0, 0.0]]), reg=0.01)
    >>> P.round(3).tolist()
    [[0.5, 0.0], [0.0, 0.5]]
    """
    return _sinkhorn(cost, reg=reg, max_iter=max_iter, tol=tol)[0]


def _sinkhorn(
    cost,
    *,
    reg,
    max_iter=DFLT_SINKHORN_ITER,
    tol=DFLT_SINKHORN_TOL,
    warm=None,
    warn=True,
):
    """``(plan, potentials)``; ``warm`` = potentials of a previous, similar problem."""
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.zeros((n, m)), None
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg!r}")
    scale = cost.max() - cost.min() or 1.0
    K = -(cost - cost.min()) / (reg * scale)
    log_a, log_b = np.full(n, -math.log(n)), np.full(m, -math.log(m))
    u, v = warm if warm is not None else (np.zeros(n), np.zeros(m))
    for it in range(1, max_iter + 1):
        u = log_a - logsumexp(K + v[None, :], axis=1)
        v = log_b - logsumexp(K + u[:, None], axis=0)  # columns now match exactly
        # converged when the rows match too (relative marginal violation), checked
        # every few rounds: the standard criterion, robust on near-permutation plans
        if it % _CHECK_EVERY == 0 or it == max_iter:
            rows = logsumexp(K + u[:, None] + v[None, :], axis=1)
            if np.abs(np.expm1(rows - log_a)).max() < tol:
                break
    else:
        if not warn:
            return np.exp(K + u[:, None] + v[None, :]), (u, v)
        warnings.warn(
            f"Sinkhorn did not converge in max_iter={max_iter} iterations; raise reg "
            f"or max_iter",
            RuntimeWarning,
            stacklevel=3,
        )
    return np.exp(K + u[:, None] + v[None, :]), (u, v)


def _netmf(A, dim, window):
    """NetMF proximity embedding (Qiu et al., WSDM 2018), small-window closed form."""
    n = A.shape[0]
    degree = A.sum(axis=1)
    volume = degree.sum()
    if volume == 0:
        return np.zeros((n, dim))
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    P = inv_degree[:, None] * A
    S, power = np.zeros_like(A), np.eye(n)
    for _ in range(window):
        power = power @ P
        S += power
    M = volume / window * S * inv_degree[None, :]
    M = np.log(np.maximum(M, 1.0))
    U, s, _ = np.linalg.svd(M)
    Y = U[:, :dim] * np.sqrt(s[:dim])
    if (
        Y.shape[1] < dim
    ):  # a graph smaller than dim: pad, so both spaces have dim columns
        Y = np.hstack([Y, np.zeros((n, dim - Y.shape[1]))])
    return Y


def _pairwise_sq(Y1, Y2):
    return np.maximum(
        (Y1**2).sum(1)[:, None] + (Y2**2).sum(1)[None, :] - 2 * Y1 @ Y2.T, 0.0
    )


def cone_scores(
    G1,
    G2,
    *,
    node_affinity=None,
    seeds=None,
    dim=DFLT_CONE_DIM,
    window=DFLT_CONE_WINDOW,
    n_iter=DFLT_CONE_ITER,
    reg=DFLT_SINKHORN_REG,
    label=None,
):
    """The CONE-Align correspondence plan (``n1 x n2``, sums to 1).

    Embeds each graph on its own (NetMF, ``dim`` dimensions, ``window`` hops), then
    alternates ``n_iter`` times: a rotation ``Q`` aligning the spaces (orthogonal
    Procrustes on the current correspondence) and a correspondence ``P`` (entropic OT,
    ``reg``, on the distances after rotation). It starts from ``node_affinity`` when given,
    else from REGAL's structural similarity. Seed rows/columns are pinned in every step.
    The plan's marginals are accurate to about ``1e-4`` (relative): enough to rank and
    extract; :func:`sinkhorn_plan` is the place for a precise plan.
    """
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    if g1.n == 0 or g2.n == 0:
        return np.zeros((g1.n, g2.n))
    seed_pairs = _seed_indices(g1, g2, seeds)
    d = max(1, min(int(dim), g1.n, g2.n))
    Y1 = _netmf(_undirected(g1), d, window)
    Y2 = _netmf(_undirected(g2), d, window)

    def pin(cost):
        top = cost.max() + 1.0
        for i, j in seed_pairs:
            cost[i, :] = top
            cost[:, j] = top
            cost[i, j] = 0.0
        return cost

    if node_affinity is not None:
        start = _prior(g1, g2, node_affinity)
    else:
        start = regal_scores(g1, g2)  # g1, g2 already carry their labels
    rough = dict(reg=reg, tol=DFLT_INNER_TOL, warn=False)
    P, _ = _sinkhorn(pin(1.0 - start / (start.max() or 1.0)), **rough)
    warm = None
    for round_ in range(n_iter):
        U, _, Vt = np.linalg.svd(Y1.T @ P @ Y2)
        Q = U @ Vt  # rotation of G1's space onto G2's
        cost = pin(_pairwise_sq(Y1 @ Q, Y2))
        # consecutive problems differ little: warm-starting the potentials saves most of
        # Sinkhorn's iterations; only the last solve warns if it misses the tolerance
        last = round_ == n_iter - 1
        P, warm = _sinkhorn(cost, warm=warm, **{**rough, "warn": last})
    return P


def cone(
    G1, G2, *, node_affinity=None, seeds=None, extract="optimal", label=None, **config
):
    """Align two graphs with CONE-Align -> a :class:`~equate.base.Matching` of node pairs.

    ``**config`` goes to :func:`cone_scores`; ``.plan`` holds the final transport plan.
    """
    g1, g2 = as_graph(G1, label=label), as_graph(G2, label=label)
    P = cone_scores(g1, g2, node_affinity=node_affinity, seeds=seeds, **config)
    return _extract(P, g1, g2, _seed_indices(g1, g2, seeds), extract)
