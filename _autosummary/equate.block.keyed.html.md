# equate.block.keyed

Keyed blocking (stage ② scalability): emit candidate pairs that share a blocking key.

A *blocker* proposes which pairs are worth scoring, avoiding the O(n\*m) all-pairs
blow-up — equivalently, it decides which cells of the score matrix to leave uncomputed
(decision register D1). Keyed blocking groups items by a cheap key (or several keys per
item, e.g. token / q-gram blocking) and emits the within-group pairs. Standard, token,
q-gram, and phonetic blocking are all *one* algorithm parameterized by the key function.

### Functions

| [`key_blocking`](#equate.block.keyed.key_blocking)(key_fn)                    | Return a blocker emitting candidate `(i, j)` for items sharing a blocking key.                                       |
|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| [`sorted_neighborhood`](#equate.block.keyed.sorted_neighborhood)([sort_key, window]) | Sorted-neighborhood blocker: sort items by `sort_key` and emit pairs within a sliding window of `window` neighbours. |
| [`first_chars`](#equate.block.keyed.first_chars)([n, key, lowercase])        | Key function: the first `n` characters (standard blocking).                                                          |
| [`qgram_keys`](#equate.block.keyed.qgram_keys)([q, key, lowercase])         | Key function: the set of character q-grams (q-gram / token blocking).                                                |
| [`whole_value`](#equate.block.keyed.whole_value)(\*[, key, lowercase])       | Key function: the whole (optionally lowercased) value — exact-key blocking.                                          |
| [`phonetic_key`](#equate.block.keyed.phonetic_key)(\*[, algorithm, key])      | Key function: a phonetic code (`soundex` / `metaphone` / `nysiis`) via `jellyfish` — requires `equate[phonetic]`.    |

### equate.block.keyed.first_chars(n=1, \*, key=<class 'str'>, lowercase=True)

Key function: the first `n` characters (standard blocking).

### equate.block.keyed.key_blocking(key_fn)

Return a blocker emitting candidate `(i, j)` for items sharing a blocking key.

`key_fn` maps an item to an *iterable of keys* (an item may land in several blocks
— token/q-gram blocking). Cross-collection `blocker(A, B)` yields `(i, j)` with
`i` indexing `A` and `j` indexing `B`; a self-join `blocker(A)` (dedup)
yields each `i < j` pair once. Lazy: candidate pairs are generated, never a matrix.

### equate.block.keyed.phonetic_key(\*, algorithm='soundex', key=<class 'str'>)

Key function: a phonetic code (`soundex` / `metaphone` / `nysiis`) via
`jellyfish` — requires `equate[phonetic]`.

### equate.block.keyed.qgram_keys(q=2, \*, key=<class 'str'>, lowercase=True)

Key function: the set of character q-grams (q-gram / token blocking).

### equate.block.keyed.sorted_neighborhood(sort_key=<class 'str'>, \*, window=3)

Sorted-neighborhood blocker: sort items by `sort_key` and emit pairs within a
sliding window of `window` neighbours. Robust to the key-equality brittleness of
standard blocking.

### equate.block.keyed.whole_value(\*, key=<class 'str'>, lowercase=True)

Key function: the whole (optionally lowercased) value — exact-key blocking.
