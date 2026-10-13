# equate.block.score

Score only the blocker’s candidate pairs into a sparse ScoreMatrix (decision D1).

This is the seam where block → compare → match meet: the blocker’s lazy candidate stream
is scored pairwise into a sparse matrix (the compare→match SSOT), with the cells the
blocker skipped left structurally absent (a non-candidate, which the matcher treats as
worst-case — see [`equate.base.to_cost()`](equate.base.html.md#equate.base.to_cost)).

### Functions

| [`score_candidates`](#equate.block.score.score_candidates)(A, B, candidate_pairs[, ...])   | Score each candidate `(i, j)` pair with `comparator` -> a sparse ScoreMatrix.   |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|

### equate.block.score.score_candidates(A, B, candidate_pairs, comparator='ratio', , sense='maximize')

Score each candidate `(i, j)` pair with `comparator` -> a sparse ScoreMatrix.

`comparator` is a pairwise comparator (a registered name or a callable). Duplicate
candidate pairs are collapsed (scored once) so a sparse matrix never sums them.
