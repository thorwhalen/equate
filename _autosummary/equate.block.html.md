# equate.block

The block stage (② scalability): candidate generation.

A *blocker* is `Callable[(A[, B]), Iterable[(i, j)]]` yielding candidate index pairs
**lazily** — never a materialized matrix. The default [`all_pairs()`](#equate.block.all_pairs) is the
correctness baseline (the dense path is a degenerate blocker). Blocking is precisely the
decision to leave most cells of the score matrix uncomputed (decision register D1);
scoring a candidate stream yields a sparse `ScoreMatrix` ([`score_candidates()`](#equate.block.score_candidates)).

Select a blocker by name via the [`blockers`](#equate.block.blockers) registry or pass a callable. Blocking
quality is measured with [`blocking_metrics()`](#equate.block.blocking_metrics) (PC / RR / PQ).

A blocker requests a **self-join** (dedup within one collection, yielding `i < j`) via
the single-argument form `blocker(A)`. The two-argument `blocker(A, B)` is a
cross-join. Self-join is detected by object *identity*, so `blocker(A, A)` is also a
self-join, but `blocker(A, list(A))` — a value-equal *copy* — is a cross-join; use the
single-argument form to be unambiguous.

### Module Attributes

| [`blockers`](#equate.block.blockers)   | the block-stage strategy registry (name -> lazy factory)   |
|-------------------------------------------------------------|------------------------------------------------------------|

### Functions

| [`all_pairs`](#equate.block.all_pairs)(A[, B])                                 | The correctness-baseline blocker: every pair (dense).                                                                                          |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| [`resolve_blocker`](#equate.block.resolve_blocker)([spec])                           | Resolve `spec` to a blocker `(A[, B]) -> Iterable[(i, j)]`.                                                                                    |
| [`key_blocking`](#equate.block.key_blocking)(key_fn)                              | Return a blocker emitting candidate `(i, j)` for items sharing a blocking key.                                                                 |
| [`sorted_neighborhood`](#equate.block.sorted_neighborhood)([sort_key, window])           | Sorted-neighborhood blocker: sort items by `sort_key` and emit pairs within a sliding window of `window` neighbours.                           |
| [`first_chars`](#equate.block.first_chars)([n, key, lowercase])                  | Key function: the first `n` characters (standard blocking).                                                                                    |
| [`qgram_keys`](#equate.block.qgram_keys)([q, key, lowercase])                   | Key function: the set of character q-grams (q-gram / token blocking).                                                                          |
| [`whole_value`](#equate.block.whole_value)(\*[, key, lowercase])                 | Key function: the whole (optionally lowercased) value — exact-key blocking.                                                                    |
| [`phonetic_key`](#equate.block.phonetic_key)(\*[, algorithm, key])                | Key function: a phonetic code (`soundex` / `metaphone` / `nysiis`) via `jellyfish` — requires `equate[phonetic]`.                              |
| [`blocking_metrics`](#equate.block.blocking_metrics)(candidate_pairs, ...[, n_b])     | Return `{pair_completeness, reduction_ratio, pairs_quality, n_candidates}`.                                                                    |
| [`score_candidates`](#equate.block.score_candidates)(A, B, candidate_pairs[, ...])    | Score each candidate `(i, j)` pair with `comparator` -> a sparse ScoreMatrix.                                                                  |
| [`brute_knn_blocking`](#equate.block.brute_knn_blocking)([featurizer, k])               | Blocker: featurize both sides and emit each A-item's top-`k` nearest B-items by cosine (pure numpy; exact; fine for small/medium collections). |
| [`ann_blocking`](#equate.block.ann_blocking)([featurizer, k, ef_construction, m]) | Blocker using an hnswlib approximate-NN cosine index — requires `equate[ann]`.                                                                 |
| [`minhash_lsh_blocking`](#equate.block.minhash_lsh_blocking)([tokenize, threshold, ...])  | Blocker: MinHash-LSH over token sets; emit candidate pairs whose estimated Jaccard similarity is at least `threshold`.                         |

### equate.block.all_pairs(A, B=None)

The correctness-baseline blocker: every pair (dense). A self-join yields `i < j`.

```pycon
>>> list(all_pairs(['a', 'b'], ['x', 'y']))
[(0, 0), (0, 1), (1, 0), (1, 1)]
>>> list(all_pairs(['a', 'b', 'c']))          # self-join: i < j
[(0, 1), (0, 2), (1, 2)]
```

### equate.block.ann_blocking(featurizer='tfidf', , k=5, ef_construction=200, m=16)

Blocker using an hnswlib approximate-NN cosine index — requires `equate[ann]`.

### equate.block.blockers *= <equate.registry.Registry object>*

the block-stage strategy registry (name -> lazy factory)

### equate.block.blocking_metrics(candidate_pairs, true_pairs, , n_a, n_b=None)

Return `{pair_completeness, reduction_ratio, pairs_quality, n_candidates}`.

- **pair_completeness (PC)** = `|candidates ∩ true| / |true|` — blocking *recall*
  (upper-bounds end-to-end recall);
- **reduction_ratio (RR)** = `1 - |candidates| / |all pairs|` — how much blocking pruned;
- **pairs_quality (PQ)** = `|candidates ∩ true| / |candidates|` — blocking *precision*.

`n_b=None` means a self-join (dedup) over `n_a` items, so
`|all pairs| = n_a*(n_a-1)/2`; otherwise `|all pairs| = n_a*n_b`. For a self-join,
pairs are canonicalized to unordered `(min, max)` so orientation doesn’t matter and
`(i, j)` / `(j, i)` count once. Empty `true_pairs` yields `pair_completeness=1.0`
(vacuously perfect recall); `reduction_ratio` is clamped to `[0, 1]`.

```pycon
>>> m = blocking_metrics([(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)], n_a=3, n_b=3)
>>> round(m['pair_completeness'], 2), round(m['pairs_quality'], 2)
(0.67, 1.0)
```

### equate.block.brute_knn_blocking(featurizer='tfidf', , k=5)

Blocker: featurize both sides and emit each A-item’s top-`k` nearest B-items by
cosine (pure numpy; exact; fine for small/medium collections). A self-join excludes
the trivial self-match *before* ranking (so ties among duplicates can’t over-produce)
and emits each pair once as `(i, j)` with `i < j`.

### equate.block.first_chars(n=1, \*, key=<class 'str'>, lowercase=True)

Key function: the first `n` characters (standard blocking).

### equate.block.key_blocking(key_fn)

Return a blocker emitting candidate `(i, j)` for items sharing a blocking key.

`key_fn` maps an item to an *iterable of keys* (an item may land in several blocks
— token/q-gram blocking). Cross-collection `blocker(A, B)` yields `(i, j)` with
`i` indexing `A` and `j` indexing `B`; a self-join `blocker(A)` (dedup)
yields each `i < j` pair once. Lazy: candidate pairs are generated, never a matrix.

### equate.block.minhash_lsh_blocking(tokenize=<function \_default_tokenize>, \*, threshold=0.5, num_perm=128)

Blocker: MinHash-LSH over token sets; emit candidate pairs whose estimated Jaccard
similarity is at least `threshold`. Requires `equate[lsh]`.

### equate.block.phonetic_key(\*, algorithm='soundex', key=<class 'str'>)

Key function: a phonetic code (`soundex` / `metaphone` / `nysiis`) via
`jellyfish` — requires `equate[phonetic]`.

### equate.block.qgram_keys(q=2, \*, key=<class 'str'>, lowercase=True)

Key function: the set of character q-grams (q-gram / token blocking).

### equate.block.resolve_blocker(spec=None, \*\*config)

Resolve `spec` to a blocker `(A[, B]) -> Iterable[(i, j)]`.

- `None` -> [`all_pairs()`](#equate.block.all_pairs) (the dense baseline);
- a registered name (`str`) -> built via its factory (`**config` forwarded);
- a callable -> returned as-is.

### equate.block.score_candidates(A, B, candidate_pairs, comparator='ratio', , sense='maximize')

Score each candidate `(i, j)` pair with `comparator` -> a sparse ScoreMatrix.

`comparator` is a pairwise comparator (a registered name or a callable). Duplicate
candidate pairs are collapsed (scored once) so a sparse matrix never sums them.

### equate.block.sorted_neighborhood(sort_key=<class 'str'>, \*, window=3)

Sorted-neighborhood blocker: sort items by `sort_key` and emit pairs within a
sliding window of `window` neighbours. Robust to the key-equality brittleness of
standard blocking.

### equate.block.whole_value(\*, key=<class 'str'>, lowercase=True)

Key function: the whole (optionally lowercased) value — exact-key blocking.

### Modules

| [`metablock`](equate.block.metablock.html.md#module-equate.block.metablock)   | Meta-blocking (stage ②): refine a candidate set to raise precision without losing much recall — deduplicate pairs and prune those touching over-connected (uninformative) items.   |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ann`](equate.block.ann.html.md#module-equate.block.ann)               | ANN / dense-vector blocking: candidate generation by nearest-neighbour search.                                                                                                     |
| [`keyed`](equate.block.keyed.html.md#module-equate.block.keyed)           | Keyed blocking (stage ② scalability): emit candidate pairs that share a blocking key.                                                                                              |
| [`lsh`](equate.block.lsh.html.md#module-equate.block.lsh)               | MinHash-LSH blocking for set/token (Jaccard) similarity — requires `equate[lsh]` (datasketch).                                                                                     |
| [`metrics`](equate.block.metrics.html.md#module-equate.block.metrics)       | Blocking evaluation metrics (stage ②): quantify the recall/efficiency trade-off of a candidate set.                                                                                |
| [`score`](equate.block.score.html.md#module-equate.block.score)           | Score only the blocker's candidate pairs into a sparse ScoreMatrix (decision D1).                                                                                                  |
