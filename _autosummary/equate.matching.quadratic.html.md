# equate.matching.quadratic

Quadratic matchers: align the *nodes* of two graphs (network alignment).

The second matcher arity (decision register D8, doc `12` §4). A flat matcher picks
pairs from a fixed score matrix; here how well `u` matches `v` \*depends on how their
neighbors are matched\* — a quadratic assignment problem — so the scores and the
correspondence are produced together, from the two graphs. Each quadratic matcher builds a
node-pair score matrix and hands it to a **flat** matcher (`extract=`, default the
optimal LAP) for the final 1:1 correspondence.

- `'isorank'` (core) — IsoRank (Singh, Xu & Berger 2008): “two nodes match if their
  neighbors match”, the stationary point of a random walk on the product graph, blended
  with an optional prior `node_affinity`.
- `'final'` (core) — FINAL (Zhang & Tong 2016): IsoRank’s propagation gated by node
  attribute agreement (`label=` / `attribute_similarity=`).
- `'regal'`, `'cone'` (core, [`equate.matching.embedding_align`](equate.matching.embedding_align.html.md#module-equate.matching.embedding_align)) — embedding
  aligners: REGAL/xNetMF (joint structural embeddings) and CONE-Align (per-graph NetMF
  embeddings aligned by Procrustes + optimal transport).
- `'gw'` (`equate[ot]`) — Gromov-Wasserstein optimal transport on shortest-path
  distances (fused with `node_affinity` when given); `.plan` is the transport plan.
- `'rrwm'` (`equate[graph-matching]`) — pygmtools’ reweighted random walk solver on
  the full quadratic affinity (edge-weight agreement + `node_affinity`).

PLANETALIGN is not wrapped: it has no PyPI distribution, and the in-house aligners above
cover its core methods (a git-sourced backend can register here later as another entry).

Use [`align()`](#equate.matching.quadratic.align) (also `equate.align`), or pick a strategy from the
[`quadratic_matchers`](#equate.matching.quadratic.quadratic_matchers) registry. `seeds` are known node correspondences
(anchors): they are kept as given and steer the rest.

### Module Attributes

| [`DFLT_ISORANK_ALPHA`](#equate.matching.quadratic.DFLT_ISORANK_ALPHA)   | weight of the topology term vs the `node_affinity` prior (1 = topology only).   |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`DFLT_EDGE_SIGMA`](#equate.matching.quadratic.DFLT_EDGE_SIGMA)      | bandwidth of the Gaussian edge-weight agreement used by `'rrwm'`.               |
| [`quadratic_matchers`](#equate.matching.quadratic.quadratic_matchers)   | the quadratic-matcher strategy registry (name -> lazy factory)                  |

### Functions

| [`resolve_quadratic_matcher`](#equate.matching.quadratic.resolve_quadratic_matcher)([spec])                | A registered name (built with `**config`) or a callable `(G1, G2, **kw) -> Matching`.                                            |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [`align`](#equate.matching.quadratic.align)(G1, G2, \*[, how, node_affinity, seeds])   | Align the nodes of two graphs -> a [`Matching`](equate.base.html.md#equate.base.Matching) of node pairs. |
| [`isorank`](#equate.matching.quadratic.isorank)(G1, G2, \*[, node_affinity, seeds, ...]) | Align two graphs with IsoRank -> a [`Matching`](equate.base.html.md#equate.base.Matching) of node pairs. |
| [`isorank_scores`](#equate.matching.quadratic.isorank_scores)(G1, G2, \*[, node_affinity, ...]) | The IsoRank node-pair score matrix `R` (`n1 x n2`, sums to 1).                                                                   |
| [`rrwm`](#equate.matching.quadratic.rrwm)(G1, G2, \*[, node_affinity, seeds, ...])    | Align two graphs with pygmtools' RRWM — requires `equate[graph-matching]`.                                                       |
| [`final`](#equate.matching.quadratic.final)(G1, G2, \*[, node_affinity, seeds, ...])   | Align two graphs with FINAL -> a [`Matching`](equate.base.html.md#equate.base.Matching) of node pairs.   |
| [`final_scores`](#equate.matching.quadratic.final_scores)(G1, G2, \*[, node_affinity, ...])   | The FINAL node-pair score matrix (Zhang & Tong, KDD 2016; the FINAL-N variant).                                                  |
| [`gw`](#equate.matching.quadratic.gw)(G1, G2, \*[, node_affinity, seeds, ...])      | Align two graphs by Gromov-Wasserstein optimal transport — `equate[ot]` (POT).                                                   |

### equate.matching.quadratic.DFLT_EDGE_SIGMA *= 1.0*

bandwidth of the Gaussian edge-weight agreement used by `'rrwm'`.

### equate.matching.quadratic.DFLT_ISORANK_ALPHA *= 0.8*

weight of the topology term vs the `node_affinity` prior (1 = topology only).

### equate.matching.quadratic.align(G1, G2, , how='isorank', node_affinity=None, seeds=None, \*\*config)

Align the nodes of two graphs -> a [`Matching`](equate.base.html.md#equate.base.Matching) of node pairs.

`how` picks the quadratic matcher (`'isorank'` core, `'rrwm'` via
`equate[graph-matching]`, or a callable); `node_affinity` is an optional prior
(an `(n1, n2)` array or a `(u, v) -> float` callable, non-negative) and usually
what makes topology-only alignment work (see [`isorank_scores()`](#equate.matching.quadratic.isorank_scores)); `seeds` are
known `(G1 node, G2 node)` anchors. Other keywords configure the matcher (e.g.
`extract='greedy'`, `alpha=0.9`). Graphs may be networkx graphs, `{node:
neighbors}` mappings or adjacency matrices.

```pycon
>>> star = {'hub': ['x', 'y', 'z'], 'x': ['hub'], 'y': ['hub'], 'z': ['hub']}
>>> other = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]}
>>> dict(align(star, other).labeled_pairs())['hub']
3
```

### equate.matching.quadratic.final(G1, G2, , node_affinity=None, seeds=None, extract='optimal', label=None, \*\*config)

Align two graphs with FINAL -> a [`Matching`](equate.base.html.md#equate.base.Matching) of node pairs.

Like [`isorank()`](#equate.matching.quadratic.isorank), with node attributes (`label=` or `attribute_similarity=`)
gating the propagation; `**config` goes to [`final_scores()`](#equate.matching.quadratic.final_scores).

```pycon
>>> G1 = {'a': ['b'], 'b': ['a', 'c'], 'c': ['b']}
>>> G2 = {1: [2], 2: [1, 3], 3: [2]}
>>> kinds = {'a': 'x', 'b': 'y', 'c': 'z', 1: 'z', 2: 'y', 3: 'x'}
>>> dict(final(G1, G2, label=kinds.get).labeled_pairs())
{'a': 3, 'b': 2, 'c': 1}
```

### equate.matching.quadratic.final_scores(G1, G2, , node_affinity=None, seeds=None, attribute_similarity=None, alpha=0.8, max_iter=100, tol=1e-09, label=None)

The FINAL node-pair score matrix (Zhang & Tong, KDD 2016; the FINAL-N variant).

IsoRank’s “two nodes match if their neighbors match”, with every step of the walk on
the product graph filtered by **node-attribute agreement** `N` (from `label=`, or
`attribute_similarity`) and symmetrically degree-normalized:
`X <- alpha * N * D^-1/2 (A1 (N * D^-1/2 * X) A2^T) + (1 - alpha) * H`, `H` the
normalized prior (`node_affinity`, seeds pinned as in IsoRank). Without attributes
(`N` all ones) it reduces to IsoRank-style propagation. Returned as a
`ScoreMatrix` (`sense='maximize'`).

### equate.matching.quadratic.gw(G1, G2, , node_affinity=None, seeds=None, extract='optimal', label=None, alpha=0.5, \*\*gw_params)

Align two graphs by Gromov-Wasserstein optimal transport — `equate[ot]` (POT).

GW finds the coupling of the two node sets that best preserves *distances*: nodes
far apart in `G1` should map to nodes far apart in `G2` (shortest-path lengths,
scaled to `[0, 1]`). With a `node_affinity` (or `seeds`) it is *fused* GW: a
node-feature cost `1 - affinity/max` weighted `1 - alpha` against the structure
(`alpha`). `.plan` holds the transport plan, the soft correspondence; `extract`
hardens it. `**gw_params` go to POT (`max_iter`, `tol_rel`, …); options that
change POT’s return type (`log`) are refused.

### equate.matching.quadratic.isorank(G1, G2, , node_affinity=None, seeds=None, extract='optimal', label=None, \*\*config)

Align two graphs with IsoRank -> a [`Matching`](equate.base.html.md#equate.base.Matching) of node pairs.

`matching.labeled_pairs()` yields `(G1 node, G2 node)`; `.plan` holds the full
IsoRank score matrix. `extract` is the flat matcher that turns it into a 1:1
correspondence (any name or callable [`equate.matching.resolve_matcher()`](equate.matching.html.md#equate.matching.resolve_matcher) takes);
`**config` goes to [`isorank_scores()`](#equate.matching.quadratic.isorank_scores) (`alpha`, `max_iter`, `tol`).

```pycon
>>> G1 = {'a': ['b'], 'b': ['a', 'c', 'd'], 'c': ['b'], 'd': ['b', 'e'], 'e': ['d']}
>>> G2 = {1: [2, 4], 2: [1], 3: [4], 4: [1, 3, 5], 5: [4]}  # the same graph, renamed
>>> dict(isorank(G1, G2).labeled_pairs())['b'], dict(isorank(G1, G2).labeled_pairs())['d']
(4, 1)
```

### equate.matching.quadratic.isorank_scores(G1, G2, , node_affinity=None, seeds=None, alpha=0.8, max_iter=100, tol=1e-09, label=None)

The IsoRank node-pair score matrix `R` (`n1 x n2`, sums to 1).

Iterates `R <- alpha * W1 R W2^T + (1 - alpha) * E` to its fixed point, where
`W` is each graph’s column-stochastic walk matrix and `E` the normalized prior
(`node_affinity`; uniform when `None`). `seeds` pin their rows/columns of the
prior to the given partner (at the prior’s own top value), so their anchoring
propagates to the neighbors. A directed graph walks *against* its edges (the column
normalization uses in-degrees). Warns when `max_iter` runs out before `tol`.
Returned as a `ScoreMatrix` (`sense='maximize'`) labeled with the two node lists.

The prior does real work: from topology alone (`node_affinity=None`) the scores
tend to the product of node degrees, which cannot tell equal-degree nodes apart. Pass
a node similarity (attributes, names, degrees) for anything but degree-distinct graphs.

### equate.matching.quadratic.quadratic_matchers *= <equate.registry.Registry object>*

the quadratic-matcher strategy registry (name -> lazy factory)

### equate.matching.quadratic.resolve_quadratic_matcher(spec='isorank', \*\*config)

A registered name (built with `**config`) or a callable `(G1, G2, **kw) -> Matching`.

### equate.matching.quadratic.rrwm(G1, G2, , node_affinity=None, seeds=None, extract='optimal', label=None, sigma=1.0, \*\*rrwm_params)

Align two graphs with pygmtools’ RRWM — requires `equate[graph-matching]`.

Builds the quadratic affinity `K` from Gaussian edge-weight agreement (bandwidth
`sigma`) with `node_affinity` on its diagonal, solves it with `pygmtools.rrwm`
(`**rrwm_params` forwarded), and extracts a 1:1 correspondence with the flat
`extract` matcher. Seeds *condition* the solve: every assignment contradicting one
(its node paired elsewhere) is removed from `K` and the seed pair gets the top
affinity, so the rest is aligned around it; seeds are then kept as given. Graphs with
no edges and no prior fall back to a uniform node affinity. Uses the numpy backend
and never changes pygmtools’ global backend.
