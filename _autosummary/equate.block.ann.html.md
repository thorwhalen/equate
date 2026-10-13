# equate.block.ann

ANN / dense-vector blocking: candidate generation by nearest-neighbour search.

A brute-force k-NN (pure numpy, core) is the exact correctness baseline; hnswlib gives a
fast approximate index behind `equate[ann]`. Both featurize the objects and return each
A-item’s nearest B-items. Reusing the *same* embedder for blocking and scoring is the
cheap default; measure PC/RR/PQ to check it isn’t capping recall (decision register D6).

### Functions

| [`brute_knn_blocking`](#equate.block.ann.brute_knn_blocking)([featurizer, k])               | Blocker: featurize both sides and emit each A-item's top-`k` nearest B-items by cosine (pure numpy; exact; fine for small/medium collections).   |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ann_blocking`](#equate.block.ann.ann_blocking)([featurizer, k, ef_construction, m]) | Blocker using an hnswlib approximate-NN cosine index — requires `equate[ann]`.                                                                   |

### equate.block.ann.ann_blocking(featurizer='tfidf', , k=5, ef_construction=200, m=16)

Blocker using an hnswlib approximate-NN cosine index — requires `equate[ann]`.

### equate.block.ann.brute_knn_blocking(featurizer='tfidf', , k=5)

Blocker: featurize both sides and emit each A-item’s top-`k` nearest B-items by
cosine (pure numpy; exact; fine for small/medium collections). A self-join excludes
the trivial self-match *before* ranking (so ties among duplicates can’t over-produce)
and emits each pair once as `(i, j)` with `i < j`.
