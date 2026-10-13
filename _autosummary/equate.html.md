# equate

Tools for matching things: featurize, compare, and match collections of objects.

Matching is fuzzy correspondence, not exact equality (`==` is the degenerate case).
The public surface is being redesigned around the `featurize -> compare -> match`
decomposition; see `docs/research/` and the `equate-dev-architecture` skill. The
foundation contracts live in [`equate.base`](equate.base.html.md#module-equate.base).

### Modules

| [`base`](equate.base.html.md#module-equate.base)               | Foundational contracts and data structures for equate.                              |
|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`block`](equate.block.html.md#module-equate.block)             | The block stage (② scalability): candidate generation.                              |
| [`cluster`](equate.cluster.html.md#module-equate.cluster)         | The resolve/cluster stage (③ grouping): turn pairwise matches into entity groups.   |
| [`compare`](equate.compare.html.md#module-equate.compare)         | The compare stage (②): score how alike two (featurized) items are.                  |
| [`completion`](equate.completion.html.md#module-equate.completion)   | Tools to complete missing data from other sources of the same data.                 |
| [`evaluate`](equate.evaluate.html.md#module-equate.evaluate)       | Evaluate matchers honestly: metrics, leakage-aware splits, and a benchmark harness. |
| [`facade`](equate.facade.html.md#module-equate.facade)           | The equate facade: the one-line `match(A, B)` entry point.                          |
| [`featurize`](equate.featurize.html.md#module-equate.featurize)     | The featurize stage (①): map objects to comparable representations.                 |
| [`interactive`](equate.interactive.html.md#module-equate.interactive) | The interactive stage: human-in-the-loop review and re-optimization.                |
| [`matching`](equate.matching.html.md#module-equate.matching)       | The match stage (③): turn a score matrix into a correspondence.                     |
| [`registry`](equate.registry.html.md#module-equate.registry)       | A tiny name-keyed strategy registry with lazy factories (open-closed dispatch).     |
| [`util`](equate.util.html.md#module-equate.util)               | Utils for equate.                                                                   |
