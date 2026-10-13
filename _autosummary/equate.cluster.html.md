# equate.cluster

The resolve/cluster stage (③ grouping): turn pairwise matches into entity groups.

Grading the comparison breaks transitivity (`a≈b` and `b≈c` need not imply `a≈c`),
so recovering *groups* from pairwise links is an explicit, swappable choice — never
hard-wired into the matcher (decision register D4, and the equivalence-relation framing in
`00-taxonomy`). `connected_components` (transitive closure via union-find) is the cheap
default; `correlation_clustering` is the robust alternative that tolerates a few
disagreements rather than collapsing two entities on one spurious edge. The output is a
[`equate.base.Partition`](equate.base.html.md#equate.base.Partition); [`canonicalize()`](#equate.cluster.canonicalize) merges each group into a golden record.

### Module Attributes

| [`clusterers`](#equate.cluster.clusterers)   | the resolve/cluster strategy registry — a clusterer is `(scored_pairs, n, *, threshold) -> Partition`.   |
|---------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|

### Functions

| [`resolve_clusterer`](#equate.cluster.resolve_clusterer)([spec])                          | Resolve `spec` to a clusterer `(scored_pairs, n, *, threshold, sense) -> Partition`.   |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`connected_components`](#equate.cluster.connected_components)(pairs, n, \*[, ...])          | Partition `n` items into connected components of the match graph (union-find).         |
| [`correlation_clustering`](#equate.cluster.correlation_clustering)(scored_pairs, n, \*[, ...]) | Cluster `n` items to minimize disagreements (the greedy Pivot algorithm).              |
| [`canonicalize`](#equate.cluster.canonicalize)(partition, records, \*[, policy])     | Merge each cluster of `partition` into one golden record from `records`.               |

### equate.cluster.canonicalize(partition, records, , policy='most_complete')

Merge each cluster of `partition` into one golden record from `records`.

`policy`: `'first'` (keep the first member), `'most_complete'` (fewest missing
fields, the default; ties keep the first member), or `'majority'` (per-field majority
vote for mapping records, or a value vote for scalar records; unhashable values fall
back to the first). Returns `{cluster_id: golden_record}`.

### equate.cluster.clusterers *= <equate.registry.Registry object>*

the resolve/cluster strategy registry — a clusterer is
`(scored_pairs, n, *, threshold) -> Partition`.

### equate.cluster.connected_components(pairs, n, , threshold=None, sense='maximize')

Partition `n` items into connected components of the match graph (union-find).

Accepts either 2-tuple edges `(i, j)` or 3-tuple scored edges `(i, j, score)` — in
the latter case only edges passing `threshold` (sense-aware) are used — so it serves
both as the low-level primitive and as a clusterer via [`resolve_clusterer()`](#equate.cluster.resolve_clusterer). This
is the *transitive closure* of the links: fragile by design (one spurious edge merges
two entities). Use [`correlation_clustering()`](#equate.cluster.correlation_clustering) when that matters.

```pycon
>>> connected_components([(0, 2), (1, 3)], 5).groups()
{0: [0, 2], 1: [1, 3], 2: [4]}
```

### equate.cluster.correlation_clustering(scored_pairs, n, , threshold=0.0, sense='maximize')

Cluster `n` items to minimize disagreements (the greedy Pivot algorithm).

`scored_pairs` is an iterable of `(i, j, score)`; an edge is *positive* when its
score passes `threshold` (sense-aware: `>=` for `'maximize'`, `<=` for
`'minimize'`). Pivot processes items in **index order** (deterministic — note the
randomized-Pivot 3-approximation bound holds only in expectation over a random pivot),
clustering each still-unclustered item with its still-unclustered positive neighbours.
Unlike connected-components it does not chain transitively, so a single stray edge
cannot collapse two clusters.

### equate.cluster.resolve_clusterer(spec='connected_components')

Resolve `spec` to a clusterer `(scored_pairs, n, *, threshold, sense) -> Partition`.

### Modules

| [`classify`](equate.cluster.classify.html.md#module-equate.cluster.classify)   | Classification (the optional decide stage): map a comparison score to a decision.   |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
