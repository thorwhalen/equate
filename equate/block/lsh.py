"""MinHash-LSH blocking for set/token (Jaccard) similarity — requires ``equate[lsh]``
(datasketch). Emits pairs whose estimated Jaccard similarity clears a threshold, without
comparing all pairs.
"""

from equate._dependencies import require

__all__ = ["minhash_lsh_blocking"]


def _default_tokenize(item):
    return set(str(item).split())


def minhash_lsh_blocking(tokenize=_default_tokenize, *, threshold=0.5, num_perm=128):
    """Blocker: MinHash-LSH over token sets; emit candidate pairs whose estimated Jaccard
    similarity is at least ``threshold``. Requires ``equate[lsh]``.
    """
    datasketch = require("datasketch", extra="lsh", purpose="MinHash-LSH blocking")
    MinHash = datasketch.MinHash
    MinHashLSH = datasketch.MinHashLSH

    def _minhash(item):
        m = MinHash(num_perm=num_perm)
        for token in tokenize(item):
            m.update(str(token).encode("utf8"))
        return m

    def blocker(A, B=None):
        self_join = B is None or B is A
        A = list(A)
        B = A if self_join else list(B)
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        b_hashes = [_minhash(b) for b in B]
        for j, m in enumerate(b_hashes):
            lsh.insert(j, m)
        for i, a in enumerate(A):
            a_hash = b_hashes[i] if self_join else _minhash(a)
            for j in lsh.query(a_hash):
                j = int(j)
                if self_join and i >= j:
                    continue
                yield (i, j)

    return blocker
