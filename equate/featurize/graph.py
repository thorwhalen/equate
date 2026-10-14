"""Graph featurizers (stage ①): turn whole graphs into vectors (graph kernels).

A graph kernel maps each graph to a feature vector, so matching a collection of graphs
goes back to the *indexable* featurize-then-compare route: cosine scoring, ANN/LSH
blocking, the flat matchers — with no special-casing in the match layer (doc ``12`` §3.4,
decision D8).

- :class:`WLFeaturizer` (``'wl'``, core) — the **Weisfeiler-Lehman subtree** feature map
  (Shervashidze et al. 2011): relabel each node by its own label plus the multiset of
  its neighbors' labels, ``iterations`` times, and count the labels seen. Linear in the
  number of edges per iteration; pure numpy/scipy.
- :func:`grakel_featurizer` (``'grakel'``, ``equate[graph-kernels]``) — any GraKeL
  kernel (WL, shortest-path, graphlet, ...) as an *empirical kernel map*: a graph's
  vector is its kernel value against each corpus graph.

Graphs may be networkx graphs, ``{node: neighbors}`` mappings or adjacency matrices
(see :func:`equate._graph.as_graph`).
"""

import hashlib
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from equate.base import FeaturizerMeta
from equate._dependencies import require
from equate._graph import as_graph
from equate._vector import l2_normalize

__all__ = ["WLFeaturizer", "grakel_featurizer", "DFLT_WL_ITERATIONS", "GRAKEL_META"]

#: WL refinement rounds (the subtree depth). 3-5 is the usual range.
DFLT_WL_ITERATIONS = 3

GRAKEL_META = FeaturizerMeta(
    output_kinds=("vector",), normalize=False, license="BSD-3-Clause"
)


def _digest(text):
    """A short, process-independent label id (Python's ``hash`` is salted per process)."""
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def _wl_label_sequences(g, iterations):
    """The WL labels of every node of ``g``, per round (round 0 = the node's own label).

    Each round compresses ``(own label, sorted multiset of neighbor labels)`` into a fixed
    digest, so labels stay short at any depth and are comparable *across* graphs: two
    nodes in two graphs share a round-``h`` label iff their depth-``h`` unfolding trees
    (with labels) are equal (up to digest collisions).
    """
    current = [_digest(repr(("L", lab))) for lab in g.labels]
    rounds = [current]
    for _ in range(iterations):
        current = [
            _digest(
                current[i] + "|" + ",".join(sorted(current[j] for j in g.neighbors(i)))
            )
            for i in range(g.n)
        ]
        rounds.append(current)
    return rounds


class WLFeaturizer:
    """Weisfeiler-Lehman subtree featurizer: ``graphs -> L2-normalized sparse counts``.

    Fit on a corpus of graphs to fix the vocabulary of WL labels (one column per label
    seen in any round), then transform graphs into count vectors over it; labels never
    seen in the corpus are ignored. A fitted instance is itself a batch featurizer.

    Two isomorphic graphs (same labels) always get the same vector; the converse holds
    up to the WL test's power (it cannot tell some regular graphs apart).

    >>> path3 = {0: [1], 1: [0, 2], 2: [1]}
    >>> triangle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    >>> wl = WLFeaturizer(iterations=2).fit([path3, triangle])
    >>> X = wl([path3, {'a': ['b'], 'b': ['a', 'c'], 'c': ['b']}, triangle])
    >>> cos = (X @ X.T).toarray()  # rows are L2-normalized
    >>> float(round(cos[0, 1], 6)), bool(cos[0, 2] < 1)  # isomorphic -> 1; path vs triangle < 1
    (1.0, True)
    """

    meta = FeaturizerMeta(output_kinds=("vector",), normalize=True, license="core")

    def __init__(self, *, iterations=DFLT_WL_ITERATIONS, label=None):
        if iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {iterations!r}")
        self.iterations = iterations
        self.label = label
        self.vocabulary_ = None

    def _counts(self, graph):
        g = as_graph(graph, label=self.label)
        rounds = _wl_label_sequences(g, self.iterations)
        return Counter(f"{h}:{sig}" for h, sigs in enumerate(rounds) for sig in sigs)

    def fit(self, graphs):
        """Learn the WL-label vocabulary from ``graphs``; returns ``self``."""
        vocab = {}
        for graph in graphs:
            for key in self._counts(graph):
                vocab.setdefault(key, len(vocab))
        self.vocabulary_ = vocab
        return self

    def transform(self, graphs):
        """Count vectors (``csr``, one row per graph), L2-normalized."""
        if self.vocabulary_ is None:
            raise ValueError("WLFeaturizer must be fit on a corpus before transform")
        rows, cols, data = [], [], []
        graphs = list(graphs)
        for r, graph in enumerate(graphs):
            for key, count in self._counts(graph).items():
                col = self.vocabulary_.get(key)
                if col is not None:
                    rows.append(r)
                    cols.append(col)
                    data.append(float(count))
        X = csr_matrix(
            (data, (rows, cols)),
            shape=(len(graphs), len(self.vocabulary_)),
            dtype=float,
        )
        return l2_normalize(X)

    def __call__(self, graphs):
        return self.transform(graphs)


def grakel_featurizer(*, kernel="WeisfeilerLehman", label=None, **kernel_params):
    """A GraKeL graph kernel as a fittable featurizer — requires ``equate[graph-kernels]``.

    ``kernel`` names a class in ``grakel.kernels`` (``'WeisfeilerLehman'``,
    ``'ShortestPath'``, ``'GraphletSampling'``, ...), built with ``normalize=True`` and
    ``**kernel_params``. After ``fit(corpus)``, a graph's vector is its (normalized)
    kernel value against each corpus graph — the empirical kernel map, so cosine/ANN work
    on it like on any embedding.
    """
    grakel = require("grakel", extra="graph-kernels", purpose="graph kernels")
    kernels = require("grakel.kernels", extra="graph-kernels", purpose="graph kernels")
    try:
        kernel_cls = getattr(kernels, kernel)
    except AttributeError:
        raise ValueError(f"grakel.kernels has no kernel named {kernel!r}") from None

    def to_grakel(graph):
        g = as_graph(graph, label=label)
        edges = {
            i: {int(j): float(g.adjacency[i, j]) for j in g.neighbors(i)}
            for i in range(g.n)
        }
        return grakel.Graph(edges, node_labels=dict(enumerate(g.labels)))

    class _GrakelFeaturizer:
        meta = GRAKEL_META

        def __init__(self):
            self.kernel_ = None

        def fit(self, graphs):
            self.kernel_ = kernel_cls(normalize=True, **kernel_params)
            self.kernel_.fit([to_grakel(g) for g in graphs])
            return self

        def __call__(self, graphs):
            if self.kernel_ is None:
                raise ValueError("the GraKeL featurizer must be fit on a corpus first")
            return np.asarray(
                self.kernel_.transform([to_grakel(g) for g in graphs]), dtype=float
            )

    return _GrakelFeaturizer()
