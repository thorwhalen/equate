# equate.block.metablock

Meta-blocking (stage ②): refine a candidate set to raise precision without losing
much recall — deduplicate pairs and prune those touching over-connected (uninformative)
items. A modular `refine(candidates) -> candidates` post-processor keeps recall-first
blocking cheap and precision recovery optional (doc 02-blocking).

### Functions

| [`dedupe_pairs`](#equate.block.metablock.dedupe_pairs)(candidate_pairs)                   | Yield unique candidate pairs, order-preserving.                                                                                                            |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`prune_frequent`](#equate.block.metablock.prune_frequent)(candidate_pairs, \*, max_degree) | Drop candidate pairs touching an item that appears in more than `max_degree` candidates (block-filtering: over-connected items are usually uninformative). |

### equate.block.metablock.dedupe_pairs(candidate_pairs)

Yield unique candidate pairs, order-preserving.

### equate.block.metablock.prune_frequent(candidate_pairs, , max_degree, self_join=False)

Drop candidate pairs touching an item that appears in more than `max_degree`
candidates (block-filtering: over-connected items are usually uninformative).

Materializes the candidate set (a degree count needs a full pass). For a
`self_join` (`i` and `j` index the same collection), degree is counted per item
so left- and right-slot appearances of the same item are combined.
