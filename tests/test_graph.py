"""Tests for the graph families: graph inputs, the Weisfeiler-Lehman featurizer, and the
quadratic matchers (IsoRank in-house; GraKeL / pygmtools wiring via fake modules)."""

import sys
import types

import numpy as np
import pytest

from equate import match, align, quadratic_matchers, MissingDependencyError
from equate._graph import as_graph
from equate.featurize import featurizers, resolve_featurizer, WLFeaturizer
from equate.matching.quadratic import isorank, isorank_scores, resolve_quadratic_matcher


def _relabel(G, mapping):
    return {mapping[u]: [mapping[v] for v in nbrs] for u, nbrs in G.items()}


def _undirected(edges):
    G = {}
    for u, v in edges:
        G.setdefault(u, []).append(v)
        G.setdefault(v, []).append(u)
    return G


# an asymmetric tree (no non-trivial automorphism), so the alignment is unique
TREE = _undirected([(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6), (6, 7), (2, 8)])


# --- graph inputs -----------------------------------------------------------------


def test_as_graph_accepts_mapping_matrix_and_networkx_like():
    g = as_graph({"a": {"b": 2.0}, "b": ["c"]})  # 'c' only appears as a neighbor
    assert g.nodes == ("a", "b", "c")
    assert g.adjacency.toarray().tolist() == [[0, 2, 0], [0, 0, 1], [0, 0, 0]]
    assert as_graph(np.eye(2)).nodes == (0, 1)
    with pytest.raises(ValueError, match="square"):
        as_graph(np.ones((2, 3)))
    with pytest.raises(TypeError, match="callable"):
        as_graph({"a": []}, label="color")


def test_as_graph_networkx():
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    G.add_node("a", color="red")
    G.add_edge("a", "b", weight=3.0)
    g = as_graph(G, label="color")
    assert g.nodes == ("a", "b")
    assert g.adjacency.toarray().tolist() == [[0, 3], [3, 0]]
    assert g.labels == ("red", 0)


# --- Weisfeiler-Lehman featurizer --------------------------------------------------


def test_wl_is_isomorphism_invariant_and_separates_structure():
    perm = {v: (v * 5 + 2) % 9 for v in TREE}  # a bijection on 0..8
    path = _undirected([(i, i + 1) for i in range(8)])
    wl = WLFeaturizer().fit([TREE, path])
    X = wl([TREE, _relabel(TREE, perm), path]).toarray()
    assert np.allclose(X[0], X[1])
    assert X[0] @ X[2] < 0.99
    assert np.allclose(np.linalg.norm(X, axis=1), 1)


def test_wl_labels_matter_and_unseen_labels_are_ignored():
    G = {0: [1], 1: [0]}
    red_blue = WLFeaturizer(label=lambda v: "rb"[v]).fit([G])
    plain = WLFeaturizer().fit([G])
    assert red_blue.vocabulary_.keys() != plain.vocabulary_.keys()
    # a graph made of labels the corpus never saw has an all-zero vector
    other = WLFeaturizer(label=lambda v: "xy"[v]).fit([G])
    other.vocabulary_ = red_blue.vocabulary_
    assert other([G]).nnz == 0


def test_wl_iterations_zero_is_a_label_histogram():
    wl = WLFeaturizer(iterations=0).fit([TREE])
    assert len(wl.vocabulary_) == 1  # every node has the same (empty) label
    with pytest.raises(ValueError):
        WLFeaturizer(iterations=-1)
    with pytest.raises(ValueError, match="fit"):
        WLFeaturizer().transform([TREE])


def test_wl_is_registered_and_drives_match():
    assert "wl" in featurizers
    triangle = _undirected([(0, 1), (1, 2), (2, 0)])
    star = _undirected([(0, 1), (0, 2), (0, 3)])
    path = _undirected([(0, 1), (1, 2), (2, 3)])
    A = [triangle, star, path]
    B = [
        _relabel(path, {0: "d", 1: "c", 2: "b", 3: "a"}),
        _relabel(triangle, {0: 5, 1: 6, 2: 7}),
        _relabel(star, {0: "h", 1: "x", 2: "y", 3: "z"}),
    ]
    m = match(A, B, featurize="wl")
    assert dict(m) == {0: 1, 1: 2, 2: 0}
    assert np.allclose(m.scores, 1)
    feat = resolve_featurizer("wl", corpus=A, iterations=1)
    assert feat.iterations == 1


# --- IsoRank / align --------------------------------------------------------------


def _degree_prior(G1, G2):
    d1 = {u: len(n) for u, n in G1.items()}
    d2 = {u: len(n) for u, n in G2.items()}
    return lambda u, v: 1.0 / (1 + abs(d1[u] - d2[v]))


def test_isorank_recovers_a_relabeled_asymmetric_graph():
    # IsoRank blends topology with a node prior (the paper uses sequence similarity);
    # a cheap degree-similarity prior is enough to pin down this tree
    perm = {v: f"n{(v * 5 + 2) % 9}" for v in TREE}
    G2 = _relabel(TREE, perm)
    m = align(TREE, G2, node_affinity=_degree_prior(TREE, G2), alpha=0.5)
    assert dict(m.labeled_pairs()) == perm
    assert len(m) == len(TREE)
    assert m.plan.shape == (9, 9)


def test_isorank_scores_are_a_distribution():
    sm = isorank_scores(TREE, TREE)
    assert sm.sense == "maximize"
    assert sm.data.sum() == pytest.approx(1.0)
    assert (sm.data >= 0).all()
    assert sm.row_labels == list(TREE)


def test_isorank_rectangular_and_partial():
    small = _undirected([(0, 1), (1, 2)])
    m = align(small, TREE)
    assert len(m) == 3  # every node of the smaller graph is matched, injectively
    assert len({j for _, j in m}) == 3


def test_node_affinity_and_seeds_break_symmetry():
    # a 4-cycle is symmetric: topology alone cannot tell which node is which
    cycle = _undirected([(0, 1), (1, 2), (2, 3), (3, 0)])
    target = _relabel(cycle, {0: "a", 1: "b", 2: "c", 3: "d"})
    want = {0: "a", 1: "b", 2: "c", 3: "d"}
    prior = lambda u, v: 1.0 if "abcd"[u] == v else 0.1  # noqa: E731
    assert dict(align(cycle, target, node_affinity=prior).labeled_pairs()) == want
    # one seed plus a mirror-breaking prior: the seed is kept, the rest follows
    m = align(cycle, target, seeds=[(0, "a")], node_affinity=prior)
    assert dict(m.labeled_pairs()) == want
    # a seed is kept even when it disagrees with the topology
    forced = align(TREE, TREE, seeds=[(0, 7)])
    assert dict(forced.labeled_pairs())[0] == 7


def test_align_validates_inputs():
    with pytest.raises(KeyError, match="G1"):
        align(TREE, TREE, seeds=[("nope", 0)])
    with pytest.raises(ValueError, match="1:1"):
        align(TREE, TREE, seeds=[(0, 0), (0, 1)])
    with pytest.raises(ValueError, match="shape"):
        align(TREE, TREE, node_affinity=np.ones((2, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        align(TREE, TREE, node_affinity=-np.ones((9, 9)))
    with pytest.raises(ValueError, match="alpha"):
        align(TREE, TREE, alpha=2)
    assert len(align({}, TREE)) == 0


def test_extract_and_callable_strategies():
    assert set(quadratic_matchers.names()) >= {"isorank", "rrwm"}
    greedy = align(TREE, TREE, extract="greedy")
    assert dict(greedy) == {i: i for i in range(9)}
    custom = lambda G1, G2, **kw: "custom"  # noqa: E731
    assert align(TREE, TREE, how=custom) == "custom"
    assert (
        resolve_quadratic_matcher("isorank", alpha=0.5)(TREE, TREE).pairs
        == isorank(TREE, TREE, alpha=0.5).pairs
    )


# --- optional backends (fakes) ----------------------------------------------------


def test_missing_backends_name_their_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygmtools", None)
    monkeypatch.setitem(sys.modules, "grakel", None)
    with pytest.raises(MissingDependencyError, match=r"equate\[graph-matching\]"):
        align(TREE, TREE, how="rrwm")
    with pytest.raises(MissingDependencyError, match=r"equate\[graph-kernels\]"):
        resolve_featurizer("grakel", corpus=[TREE])


def test_grakel_featurizer_wiring(monkeypatch):
    seen = {}

    class FakeGraph:
        def __init__(self, edges, node_labels):
            self.edges, self.node_labels = edges, node_labels

    class FakeWL:
        def __init__(self, **params):
            seen["params"] = params

        def fit(self, graphs):
            seen["fit"] = graphs
            return self

        def transform(self, graphs):
            return [[len(g.edges), 1.0] for g in graphs]

    grakel = types.ModuleType("grakel")
    kernels = types.ModuleType("grakel.kernels")
    kernels.WeisfeilerLehman = FakeWL
    grakel.Graph, grakel.kernels = FakeGraph, kernels
    monkeypatch.setitem(sys.modules, "grakel", grakel)
    monkeypatch.setitem(sys.modules, "grakel.kernels", kernels)

    feat = resolve_featurizer("grakel", corpus=[TREE, {0: [1], 1: [0]}], n_iter=2)
    assert seen["params"] == {"normalize": True, "n_iter": 2}
    g = seen["fit"][1]
    assert g.edges == {0: {1: 1.0}, 1: {0: 1.0}} and g.node_labels == {0: 0, 1: 0}
    assert feat([TREE]).tolist() == [[9.0, 1.0]]
    with pytest.raises(ValueError, match="no kernel"):
        resolve_featurizer("grakel", corpus=[TREE], kernel="Nope")


def test_rrwm_wiring(monkeypatch):
    calls = {}

    def dense_to_sparse(A, backend):
        return ("conn", A.shape), ("edge", A.shape), ("ne", A.shape)

    def build_aff_mat(nf1, e1, c1, nf2, e2, c2, n1, ne1, n2, ne2, edge_aff_fn, backend):
        calls["n"] = (int(n1[0]), int(n2[0]))
        k = int(n1[0]) * int(n2[0])
        return np.zeros((1, k, k))

    def rrwm_solver(K, n1, n2, backend, **kw):
        calls["K_diag"] = np.diag(K[0]).copy()
        calls["kw"] = kw
        n1, n2 = int(n1[0]), int(n2[0])
        X = np.zeros((1, n1, n2))
        X[0, np.arange(n1), (np.arange(n1) + 1) % n2] = 1.0  # a known shifted solution
        return X

    pygm = types.ModuleType("pygmtools")
    pygm.utils = types.SimpleNamespace(
        dense_to_sparse=dense_to_sparse,
        build_aff_mat=build_aff_mat,
        gaussian_aff_fn=lambda *a, **k: None,
    )
    pygm.rrwm = rrwm_solver
    monkeypatch.setitem(sys.modules, "pygmtools", pygm)

    G1, G2 = {0: [1], 1: [0], 2: []}, {"a": ["b"], "b": ["a"]}
    E = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    m = align(G1, G2, how="rrwm", node_affinity=E, max_iter=7)
    assert calls["n"] == (3, 2) and calls["kw"] == {"max_iter": 7}
    # node affinities land on K's diagonal in pygmtools' column-major order
    assert calls["K_diag"].tolist() == [1, 3, 5, 2, 4, 6]
    assert dict(m.labeled_pairs()) == {0: "b", 1: "a"}
    # seeds are kept and excluded from the extraction
    seeded = align(G1, G2, how="rrwm", seeds=[(2, "a")])
    assert dict(seeded.labeled_pairs())[2] == "a" and len(seeded) == 2


def test_rrwm_real_pygmtools_recovers_an_isomorphism():
    pytest.importorskip("pygmtools")
    rng = np.random.default_rng(1)
    A = rng.random((6, 6))
    A = np.triu(A, 1) * (np.triu(A, 1) > 0.4)
    A = A + A.T  # a weighted undirected graph
    perm = rng.permutation(6)
    P = np.eye(6)[perm]
    m = align(A, P @ A @ P.T, how="rrwm")
    # node a of G2 is node perm[a] of G1
    assert {i: j for i, j in m} == {int(perm[a]): a for a in range(6)}
