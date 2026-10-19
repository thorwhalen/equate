# equate.matching.embedding_align

Embedding-based network alignment: embed both graphs’ nodes, then match the embeddings.

Two quadratic matchers (doc `12` §4) that scale past IsoRank-style propagation by
reasoning over node *embeddings* instead of the `n1 x n2` product graph, both in-house
numpy (no extra):

- `'regal'` — REGAL / xNetMF (Heimann et al., CIKM 2018): each node is described by the
  degree distribution of its k-hop neighbourhood (plus its label), the two graphs are
  embedded *jointly* through a Nyström factorization of that similarity, and nodes are
  matched by embedding similarity. Needs no correspondence to start from.
- `'cone'` — CONE-Align (Chen et al., CIKM 2020): each graph gets its own proximity
  embedding (NetMF), and the two embedding spaces are aligned by alternating an
  orthogonal Procrustes rotation with an entropic optimal-transport correspondence,
  starting from REGAL’s structural similarity (or `node_affinity`).

Both return a [`Matching`](equate.base.html.md#equate.base.Matching) through the same flat `extract` step as
IsoRank, with the node-pair scores in `.plan`. Use them as `equate.align(G1, G2,
how='regal')` or `how='cone'`.

### Functions

| [`regal`](#equate.matching.embedding_align.regal)(G1, G2, \*[, node_affinity, seeds, ...])   | Align two graphs with REGAL (xNetMF embeddings) -> a [`Matching`](equate.base.html.md#equate.base.Matching).   |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| [`regal_scores`](#equate.matching.embedding_align.regal_scores)(G1, G2, \*[, node_affinity, ...])   | The REGAL node-pair score matrix `exp(-||y_u - y_v||)` (`n1 x n2`).                                                                    |
| [`cone`](#equate.matching.embedding_align.cone)(G1, G2, \*[, node_affinity, seeds, ...])    | Align two graphs with CONE-Align -> a [`Matching`](equate.base.html.md#equate.base.Matching) of node pairs.    |
| [`cone_scores`](#equate.matching.embedding_align.cone_scores)(G1, G2, \*[, node_affinity, ...])    | The CONE-Align correspondence plan (`n1 x n2`, sums to 1).                                                                             |
| [`sinkhorn_plan`](#equate.matching.embedding_align.sinkhorn_plan)(cost, \*[, reg, max_iter, tol])    | Entropic optimal-transport plan between uniform marginals (log-domain Sinkhorn).                                                       |

### equate.matching.embedding_align.cone(G1, G2, , node_affinity=None, seeds=None, extract='optimal', label=None, \*\*config)

Align two graphs with CONE-Align -> a [`Matching`](equate.base.html.md#equate.base.Matching) of node pairs.

`**config` goes to [`cone_scores()`](#equate.matching.embedding_align.cone_scores); `.plan` holds the final transport plan.

### equate.matching.embedding_align.cone_scores(G1, G2, , node_affinity=None, seeds=None, dim=64, window=10, n_iter=30, reg=0.05, label=None)

The CONE-Align correspondence plan (`n1 x n2`, sums to 1).

Embeds each graph on its own (NetMF, `dim` dimensions, `window` hops), then
alternates `n_iter` times: a rotation `Q` aligning the spaces (orthogonal
Procrustes on the current correspondence) and a correspondence `P` (entropic OT,
`reg`, on the distances after rotation). It starts from `node_affinity` when given,
else from REGAL’s structural similarity. Seed rows/columns are pinned in every step.
The plan’s marginals are accurate to about `1e-4` (relative): enough to rank and
extract; [`sinkhorn_plan()`](#equate.matching.embedding_align.sinkhorn_plan) is the place for a precise plan.

### equate.matching.embedding_align.regal(G1, G2, , node_affinity=None, seeds=None, extract='optimal', label=None, \*\*config)

Align two graphs with REGAL (xNetMF embeddings) -> a [`Matching`](equate.base.html.md#equate.base.Matching).

`**config` goes to [`regal_scores()`](#equate.matching.embedding_align.regal_scores); `seeds` are kept as given.

```pycon
>>> G1 = {'a': ['b'], 'b': ['a', 'c', 'd'], 'c': ['b'], 'd': ['b', 'e'], 'e': ['d']}
>>> G2 = {1: [2, 4], 2: [1], 3: [4], 4: [1, 3, 5], 5: [4]}  # the same graph, renamed
>>> dict(regal(G1, G2).labeled_pairs())['b']
4
```

### equate.matching.embedding_align.regal_scores(G1, G2, , node_affinity=None, max_hops=2, discount=0.1, gamma_struct=1.0, gamma_attr=1.0, dim=None, seed=0, label=None)

The REGAL node-pair score matrix `exp(-||y_u - y_v||)` (`n1 x n2`).

Embeddings come from xNetMF over both graphs at once (so they share a space with no
correspondence given): structural identity = degree histograms of the `max_hops`
neighbourhood, plus a label mismatch term (`label=`). `dim` sets the number of
landmarks (default `10 log2 n`); `seed` picks them reproducibly. A
`node_affinity` prior multiplies the scores.

### equate.matching.embedding_align.sinkhorn_plan(cost, , reg=0.05, max_iter=1000, tol=1e-06)

Entropic optimal-transport plan between uniform marginals (log-domain Sinkhorn).

Pure numpy (no POT): the plan `P` minimizes `<P, cost> - reg * H(P)` with rows
summing to `1/n` and columns to `1/m`. `reg` is relative to the cost’s spread,
so the same value works at any cost scale. Warns if `max_iter` runs out first.

```pycon
>>> P = sinkhorn_plan(np.array([[0.0, 1.0], [1.0, 0.0]]), reg=0.01)
>>> P.round(3).tolist()
[[0.5, 0.0], [0.0, 0.5]]
```
