"""Pairwise and cluster-level match metrics, dependency-light.

Two granularities, kept explicit (research doc 14, §2): :func:`pairwise_prf` scores pair
sets; :func:`bcubed`, :func:`ceaf` and :func:`cluster_metrics` score a label per record.
**B³ is the default cluster metric** — it alone satisfies all of Amigó's formal
constraints (§2.4) — with CEAF, ARI and V-measure as complements. ARI and V-measure
delegate to ``sklearn.metrics`` when it is installed and fall back to the numpy
implementations here otherwise; CEAF's cluster alignment goes through equate's own
matcher seam rather than a second Hungarian.
"""

from collections.abc import Iterable, Sequence
from math import comb

import numpy as np

from equate._dependencies import have
from equate.evaluate.scores import Scores


def _canon_pairs(pairs: Iterable[tuple], *, self_join: bool) -> set[tuple]:
    if self_join:
        return {(min(i, j), max(i, j)) for i, j in pairs if i != j}
    return {(i, j) for i, j in pairs}


def pairwise_prf(pred_pairs, true_pairs, *, self_join: bool = False) -> Scores:
    """Pairwise precision / recall / F1 of predicted pairs against gold pairs.

    ``self_join=True`` (deduplication within one collection) treats ``(i, j)`` and
    ``(j, i)`` as one pair and drops ``(i, i)``; otherwise pairs are ``(row of A, row
    of B)`` and orientation matters.

    >>> s = pairwise_prf([(0, 0), (1, 2)], [(0, 0), (1, 1)])
    >>> s.precision, s.recall, s.support
    (0.5, 0.5, 2)
    """
    pred = _canon_pairs(pred_pairs, self_join=self_join)
    true = _canon_pairs(true_pairs, self_join=self_join)
    return Scores.from_counts(tp=len(pred & true), n_pred=len(pred), n_true=len(true))


def _check_labels(pred_labels, true_labels) -> tuple[np.ndarray, np.ndarray]:
    pred, true = np.asarray(pred_labels), np.asarray(true_labels)
    if pred.shape != true.shape or pred.ndim != 1:
        raise ValueError(
            "pred_labels and true_labels must be 1-D and the same length (one label "
            f"per record); got shapes {pred.shape} and {true.shape}. Pair sets? Use "
            "pairs_to_labels first, or pairwise_prf."
        )
    return pred, true


def _contingency(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """``table[t, p]`` = number of records in true cluster ``t`` and predicted cluster ``p``."""
    _, t_idx = np.unique(true, return_inverse=True)
    _, p_idx = np.unique(pred, return_inverse=True)
    table = np.zeros((t_idx.max() + 1, p_idx.max() + 1), dtype=np.int64)
    np.add.at(table, (t_idx, p_idx), 1)
    return table


def bcubed(pred_labels: Sequence, true_labels: Sequence) -> Scores:
    """B³ precision / recall / F1 — the default cluster metric.

    Per record: precision is the share of its predicted cluster that truly belongs with
    it, recall the share of its true cluster that the prediction kept with it; both are
    averaged over records.

    >>> s = bcubed([0, 0, 1, 1], [0, 0, 0, 1])
    >>> round(s.precision, 3), round(s.recall, 3)
    (0.75, 0.667)
    """
    pred, true = _check_labels(pred_labels, true_labels)
    n = len(pred)
    if n == 0:
        return Scores.from_pr(1.0, 1.0, support=0)
    table = _contingency(pred, true).astype(float)
    pred_sizes, true_sizes = table.sum(axis=0), table.sum(axis=1)
    # Each record in cell (t, p) has precision table[t,p]/|p| and recall table[t,p]/|t|.
    precision = (table**2 / pred_sizes[None, :]).sum() / n
    recall = (table**2 / true_sizes[:, None]).sum() / n
    return Scores.from_pr(precision, recall, support=n)


def ceaf(
    pred_labels: Sequence,
    true_labels: Sequence,
    *,
    variant: str = "entity",
    matcher="optimal",
) -> Scores:
    """CEAF: score the best one-to-one alignment of predicted to true clusters.

    ``variant='entity'`` (CEAF-e, φ4) weighs each aligned pair by
    ``2|K∩R| / (|K|+|R|)`` and normalises by the number of clusters;
    ``variant='mention'`` (CEAF-m, φ3) weighs by ``|K∩R|`` and normalises by the
    number of records. The alignment is a maximum-weight assignment, resolved through
    equate's :data:`~equate.matching.matchers` registry (``matcher``), never a
    private Hungarian.

    >>> round(ceaf([0, 0, 1, 1], [0, 0, 0, 1]).f1, 3)
    0.733
    """
    from equate.matching import resolve_matcher

    pred, true = _check_labels(pred_labels, true_labels)
    if len(pred) == 0:
        return Scores.from_pr(1.0, 1.0, support=0)
    table = _contingency(pred, true).astype(float)
    if variant == "entity":
        sizes = table.sum(axis=1)[:, None] + table.sum(axis=0)[None, :]
        sim = 2 * table / sizes
        n_true, n_pred = table.shape
        denom_p, denom_r = n_pred, n_true
    elif variant == "mention":
        sim = table
        denom_p = denom_r = len(pred)
    else:
        raise ValueError(f"variant must be 'entity' or 'mention', got {variant!r}")
    match = resolve_matcher(matcher)
    total = float(sum(sim[i, j] for i, j in match(sim, sense="maximize")))
    return Scores.from_pr(total / denom_p, total / denom_r, support=len(true))


def _ari_numpy(pred: np.ndarray, true: np.ndarray) -> float:
    table = _contingency(pred, true)
    n = int(table.sum())
    sum_cells = sum(comb(int(x), 2) for x in table.ravel())
    sum_true = sum(comb(int(x), 2) for x in table.sum(axis=1))
    sum_pred = sum(comb(int(x), 2) for x in table.sum(axis=0))
    total = comb(n, 2)
    expected = sum_true * sum_pred / total if total else 0.0
    max_index = (sum_true + sum_pred) / 2
    if max_index == expected:  # both partitions trivial (all-one or all-singletons)
        return 1.0
    return (sum_cells - expected) / (max_index - expected)


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def _homogeneity_completeness_numpy(pred: np.ndarray, true: np.ndarray) -> tuple:
    table = _contingency(pred, true).astype(float)
    n = table.sum()
    h_true, h_pred = _entropy(table.sum(axis=1)), _entropy(table.sum(axis=0))
    nz = table > 0
    # conditional entropies H(true | pred) and H(pred | true)
    h_true_given_pred = -(
        table[nz]
        / n
        * np.log(table[nz] / np.broadcast_to(table.sum(axis=0), table.shape)[nz])
    ).sum()
    h_pred_given_true = -(
        table[nz]
        / n
        * np.log(
            table[nz] / np.broadcast_to(table.sum(axis=1)[:, None], table.shape)[nz]
        )
    ).sum()
    homogeneity = 1.0 if h_true == 0 else 1 - h_true_given_pred / h_true
    completeness = 1.0 if h_pred == 0 else 1 - h_pred_given_true / h_pred
    return float(homogeneity), float(completeness)


def cluster_metrics(
    pred_labels: Sequence, true_labels: Sequence, *, backend: str = "auto"
) -> dict:
    """B³ (the headline), CEAF-e, ARI, V-measure, homogeneity and completeness.

    ``backend='auto'`` uses ``sklearn.metrics`` for ARI / V-measure when sklearn is
    installed and the numpy implementations otherwise; ``'numpy'`` forces the latter,
    ``'sklearn'`` requires the former. The two agree; the choice only exists so the
    core install needs neither.

    >>> m = cluster_metrics([0, 0, 1, 1], [0, 0, 0, 1], backend='numpy')
    >>> sorted(m)  # doctest: +NORMALIZE_WHITESPACE
    ['ari', 'b3_f1', 'b3_precision', 'b3_recall', 'ceaf_e_f1', 'completeness',
     'homogeneity', 'v_measure']
    """
    pred, true = _check_labels(pred_labels, true_labels)
    if backend not in ("auto", "numpy", "sklearn"):
        raise ValueError(
            f"backend must be 'auto', 'numpy' or 'sklearn', got {backend!r}"
        )
    if len(pred) == 0:  # nothing to cluster: every metric is vacuously perfect
        names = (
            "ari",
            "b3_f1",
            "b3_precision",
            "b3_recall",
            "ceaf_e_f1",
            "completeness",
            "homogeneity",
            "v_measure",
        )
        return dict.fromkeys(names, 1.0)
    b3 = bcubed(pred, true)
    out = {
        "b3_precision": b3.precision,
        "b3_recall": b3.recall,
        "b3_f1": b3.f1,
        "ceaf_e_f1": ceaf(pred, true).f1,
    }
    use_sklearn = backend == "sklearn" or (backend == "auto" and have("sklearn"))
    if use_sklearn:
        from equate._dependencies import require

        skm = require("sklearn.metrics", extra="eval", purpose="ARI and V-measure")
        h, c, v = skm.homogeneity_completeness_v_measure(true, pred)
        out["ari"] = float(skm.adjusted_rand_score(true, pred))
    else:
        h, c = _homogeneity_completeness_numpy(pred, true)
        v = 2 * h * c / (h + c) if h + c else 0.0
        out["ari"] = float(_ari_numpy(pred, true))
    out.update(homogeneity=float(h), completeness=float(c), v_measure=float(v))
    return dict(sorted(out.items()))
