# equate.featurize.graph

Graph featurizers (stage ①): turn whole graphs into vectors (graph kernels).

A graph kernel maps each graph to a feature vector, so matching a collection of graphs
goes back to the *indexable* featurize-then-compare route: cosine scoring, ANN/LSH
blocking, the flat matchers — with no special-casing in the match layer (doc `12` §3.4,
decision D8).

- [`WLFeaturizer`](#equate.featurize.graph.WLFeaturizer) (`'wl'`, core) — the **Weisfeiler-Lehman subtree** feature map
  (Shervashidze et al. 2011): relabel each node by its own label plus the multiset of
  its neighbors’ labels, `iterations` times, and count the labels seen. Linear in the
  number of edges per iteration; pure numpy/scipy.
- [`grakel_featurizer()`](#equate.featurize.graph.grakel_featurizer) (`'grakel'`, `equate[graph-kernels]`) — any GraKeL
  kernel (WL, shortest-path, graphlet, …) as an *empirical kernel map*: a graph’s
  vector is its kernel value against each corpus graph.

Graphs may be networkx graphs, `{node: neighbors}` mappings or adjacency matrices
(see `equate._graph.as_graph()`).

### Module Attributes

| [`DFLT_WL_ITERATIONS`](#equate.featurize.graph.DFLT_WL_ITERATIONS)   | WL refinement rounds (the subtree depth).   |
|-----------------------------------------------------------------------|---------------------------------------------|

### Functions

| [`grakel_featurizer`](#equate.featurize.graph.grakel_featurizer)(\*[, kernel, label])   | A GraKeL graph kernel as a fittable featurizer — requires `equate[graph-kernels]`.   |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|

### Classes

| [`WLFeaturizer`](#equate.featurize.graph.WLFeaturizer)(\*[, iterations, label])   | Weisfeiler-Lehman subtree featurizer: `graphs -> L2-normalized sparse counts`.   |
|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|

### equate.featurize.graph.DFLT_WL_ITERATIONS *= 3*

WL refinement rounds (the subtree depth). 3-5 is the usual range.

### *class* equate.featurize.graph.WLFeaturizer(, iterations=3, label=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Weisfeiler-Lehman subtree featurizer: `graphs -> L2-normalized sparse counts`.

Fit on a corpus of graphs to fix the vocabulary of WL labels (one column per label
seen in any round), then transform graphs into count vectors over it; labels never
seen in the corpus are ignored. A fitted instance is itself a batch featurizer.

Two isomorphic graphs (same labels) always get the same vector; the converse holds
up to the WL test’s power (it cannot tell some regular graphs apart). Neighbors are
*out*-neighbors (a directed graph is refined along its edges), and edge weights are
ignored – the subtree kernel counts structure, not strength.

```pycon
>>> path3 = {0: [1], 1: [0, 2], 2: [1]}
>>> triangle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
>>> wl = WLFeaturizer(iterations=2).fit([path3, triangle])
>>> X = wl([path3, {'a': ['b'], 'b': ['a', 'c'], 'c': ['b']}, triangle])
>>> cos = (X @ X.T).toarray()  # rows are L2-normalized
>>> float(round(cos[0, 1], 6)), bool(cos[0, 2] < 1)  # isomorphic -> 1; path vs triangle < 1
(1.0, True)
```

#### fit(graphs)

Learn the WL-label vocabulary from `graphs`; returns `self`.

#### transform(graphs)

Count vectors (`csr`, one row per graph), L2-normalized.

### equate.featurize.graph.grakel_featurizer(, kernel='WeisfeilerLehman', label=None, \*\*kernel_params)

A GraKeL graph kernel as a fittable featurizer — requires `equate[graph-kernels]`.

`kernel` names a class in `grakel.kernels` (`'WeisfeilerLehman'`,
`'ShortestPath'`, `'GraphletSampling'`, …), built with `normalize=True` and
`**kernel_params`. After `fit(corpus)`, a graph’s vector is its (normalized)
kernel value against each corpus graph — the empirical kernel map, so cosine/ANN work
on it like on any embedding.
