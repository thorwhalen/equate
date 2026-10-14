"""Graph inputs, normalized once: a networkx graph, an adjacency mapping, or a matrix.

The graph featurizers (:mod:`equate.featurize.graph`) and the quadratic matchers
(:mod:`equate.matching.quadratic`) accept graphs in whatever form the caller has, without
importing networkx (it is duck-typed): :func:`as_graph` turns any of them into one small
:class:`GraphView` — an ordered node list, a sparse adjacency matrix, and node labels.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse

__all__ = ["GraphView", "as_graph", "dense_adjacency"]


@dataclass(frozen=True, eq=False)
class GraphView:
    """A graph as equate's structured strategies see it.

    ``nodes`` fixes the order (row ``i`` of ``adjacency`` is ``nodes[i]``);
    ``adjacency`` is a ``csr`` matrix of edge weights (``adjacency[i, j]`` is the edge
    ``nodes[i] -> nodes[j]``; an undirected graph is symmetric); ``labels`` holds one
    hashable label per node (all equal when the graph carries none).
    """

    nodes: tuple
    adjacency: Any
    labels: tuple

    @property
    def n(self) -> int:
        return len(self.nodes)

    def index(self) -> dict:
        """``{node: row index}``."""
        return {v: i for i, v in enumerate(self.nodes)}

    def neighbors(self, i) -> np.ndarray:
        """Row indices of the out-neighbors of node index ``i``."""
        a = self.adjacency
        return a.indices[a.indptr[i] : a.indptr[i + 1]]


def _weight(w) -> float:
    """An edge's weight: a number, ``None`` (unweighted: ``1``), or an attribute dict."""
    if isinstance(w, Mapping):
        w = w.get("weight", 1.0)
    return 1.0 if w is None else float(w)


def _label_of(node, attrs, label):
    if label is None:
        return 0
    if callable(label):
        return label(node)
    return attrs.get(label, 0) if attrs is not None else 0


def as_graph(G, *, label=None) -> GraphView:
    """Normalize ``G`` to a :class:`GraphView`.

    ``G`` may be:

    - a **networkx** graph (anything with ``.nodes`` and ``.adj``; edge ``weight``
      attributes are used, default ``1``);
    - a **mapping** ``{node: neighbors}`` where ``neighbors`` is an iterable of nodes or a
      ``{neighbor: weight}`` mapping (taken as given — list both directions for an
      undirected graph);
    - a square **adjacency matrix** (dense array-like or ``scipy.sparse``), nodes ``0..n-1``;
    - an existing :class:`GraphView` (returned unchanged; ``label`` must then be ``None``).

    A stored weight of ``0`` is no edge, whichever form it came in. A networkx
    ``MultiGraph`` contributes the *sum* of its parallel edges' weights.

    ``label`` names the node label: a node-attribute name (networkx), or a callable
    ``node -> label``; ``None`` gives every node the same label.

    >>> g = as_graph({'a': ['b'], 'b': ['a', 'c'], 'c': ['b']})
    >>> g.nodes, g.adjacency.toarray().tolist()
    (('a', 'b', 'c'), [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    >>> as_graph([[0, 2], [2, 0]], label=lambda v: v % 2).labels
    (0, 1)
    """
    if isinstance(G, GraphView):
        if label is not None:
            raise TypeError(
                "a GraphView already carries its labels; label= must be None"
            )
        return G
    if hasattr(G, "adj") and hasattr(G, "nodes"):  # networkx-like
        nodes = tuple(G.nodes)
        idx = {v: i for i, v in enumerate(nodes)}
        rows, cols, data = [], [], []
        multi = bool(getattr(G, "is_multigraph", lambda: False)())
        for u, nbrs in G.adj.items():
            for v, attrs in nbrs.items():
                rows.append(idx[u])
                cols.append(idx[v])
                if multi:  # {key: attrs} per parallel edge
                    data.append(sum(_weight(a or {}) for a in attrs.values()))
                else:
                    data.append(_weight(attrs or {}))
        node_attrs = dict(G.nodes(data=True))
        labels = tuple(_label_of(v, node_attrs.get(v), label) for v in nodes)
    elif isinstance(G, Mapping):
        nodes = list(G)
        seen = set(nodes)
        for nbrs in G.values():  # neighbors that are not keys are still nodes
            for v in nbrs:
                if v not in seen:
                    seen.add(v)
                    nodes.append(v)
        nodes = tuple(nodes)
        idx = {v: i for i, v in enumerate(nodes)}
        rows, cols, data = [], [], []
        for u, nbrs in G.items():
            weights = nbrs if isinstance(nbrs, Mapping) else dict.fromkeys(nbrs, 1.0)
            for v, w in weights.items():
                rows.append(idx[u])
                cols.append(idx[v])
                data.append(_weight(w))
        if label is not None and not callable(label):
            raise TypeError(
                "an adjacency-mapping graph has no node attributes; pass label= as a "
                "callable node -> label"
            )
        labels = tuple(_label_of(v, None, label) for v in nodes)
    else:
        A = G.tocsr() if issparse(G) else np.asarray(G, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"a graph must be networkx-like, a {{node: neighbors}} mapping, or a "
                f"square adjacency matrix; got shape {getattr(A, 'shape', None)}"
            )
        nodes = tuple(range(A.shape[0]))
        if label is not None and not callable(label):
            raise TypeError("an adjacency-matrix graph needs label= as a callable")
        labels = tuple(_label_of(v, None, label) for v in nodes)
        adjacency = csr_matrix(A, dtype=float)
        adjacency.eliminate_zeros()
        return GraphView(nodes, adjacency, labels)
    n = len(nodes)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    return GraphView(nodes, adjacency, labels)


def dense_adjacency(g: GraphView) -> np.ndarray:
    """The adjacency of a :class:`GraphView` as a dense float array."""
    return np.asarray(g.adjacency.toarray(), dtype=float)
