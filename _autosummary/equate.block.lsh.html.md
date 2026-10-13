# equate.block.lsh

MinHash-LSH blocking for set/token (Jaccard) similarity — requires `equate[lsh]`
(datasketch). Emits pairs whose estimated Jaccard similarity clears a threshold, without
comparing all pairs.

### Functions

| [`minhash_lsh_blocking`](#equate.block.lsh.minhash_lsh_blocking)([tokenize, threshold, ...])   | Blocker: MinHash-LSH over token sets; emit candidate pairs whose estimated Jaccard similarity is at least `threshold`.   |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|

### equate.block.lsh.minhash_lsh_blocking(tokenize=<function \_default_tokenize>, \*, threshold=0.5, num_perm=128)

Blocker: MinHash-LSH over token sets; emit candidate pairs whose estimated Jaccard
similarity is at least `threshold`. Requires `equate[lsh]`.
