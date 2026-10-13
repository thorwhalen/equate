"""Threshold-free and ranking metrics, and threshold selection as its own step.

:func:`pr_auc` (average precision) summarises a scorer without choosing a threshold and
stays honest under the heavy class imbalance of matching, where ROC-AUC flatters
(research doc 14, §4). :func:`select_threshold` is deliberately separate from scoring:
tuning on the test set should be something a user does on purpose, never by accident
(§3, §7.2).
"""

from collections.abc import Mapping, Sequence

import numpy as np

from equate.evaluate.scores import Scores


def _sorted_by_score(scores, y_true) -> tuple[np.ndarray, np.ndarray]:
    s, raw = np.asarray(scores, dtype=float), np.asarray(y_true)
    if s.shape != raw.shape or s.ndim != 1:
        raise ValueError(
            f"scores and y_true must be 1-D and the same length; got {s.shape}, {raw.shape}"
        )
    if raw.size and not set(np.unique(raw).tolist()) <= {0, 1}:
        raise ValueError(
            "y_true must hold 0/1 (or False/True); got values "
            f"{sorted(set(np.unique(raw).tolist()))[:5]} — map a -1/+1 encoding first"
        )
    y = raw.astype(bool)
    order = np.argsort(-s, kind="mergesort")
    return s[order], y[order]


def pr_auc(scores: Sequence[float], y_true: Sequence) -> float:
    """Average precision: the area under the precision-recall step curve.

    Records with tied scores enter the ranking together (one threshold step), so the
    figure does not depend on the input order of ties. No positives → ``nan``.

    >>> round(pr_auc([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0]), 3)
    0.833
    """
    s, y = _sorted_by_score(scores, y_true)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    # threshold steps = the last index of each run of equal scores
    last_of_run = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[last_of_run]
    precision = tp / (last_of_run + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def select_threshold(
    scores: Sequence[float], y_true: Sequence, *, optimize: str = "f1"
) -> tuple[float, Scores]:
    """The score threshold (``score >= t`` predicts a match) that maximises ``optimize``.

    Pass **validation** scores and labels. ``optimize`` is ``'f1'``, ``'precision'`` or
    ``'recall'``; ties go to the higher threshold (fewer predicted matches). Returns the
    threshold and the :class:`Scores` it achieves on the data it was chosen on — which
    are therefore optimistic, and are not a test result.

    >>> t, s = select_threshold([0.9, 0.8, 0.4, 0.3], [1, 1, 0, 1])
    >>> t, round(s.f1, 3)
    (0.3, 0.857)
    """
    if optimize not in ("f1", "precision", "recall"):
        raise ValueError(f"optimize must be 'f1', 'precision' or 'recall', got {optimize!r}")
    s, y = _sorted_by_score(scores, y_true)
    if len(s) == 0:
        raise ValueError("cannot select a threshold from no scores")
    n_pos = int(y.sum())
    last_of_run = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[last_of_run]
    best = None
    for k, idx in enumerate(last_of_run):
        cand = Scores.from_counts(tp=int(tp[k]), n_pred=int(idx + 1), n_true=n_pos)
        value = getattr(cand, optimize)
        if best is None or value > best[0]:
            best = (value, float(s[idx]), cand)
    return best[1], best[2]


def ranking_metrics(
    ranked: Mapping[object, Sequence], gold: Mapping[object, object], *, ks=(1, 5, 10)
) -> dict:
    """Recall@k, MRR and MAP for ranked candidate lists.

    ``ranked[q]`` is query ``q``'s candidates, best first; ``gold[q]`` is its true match
    or a collection of them. Queries with no gold are skipped; a query whose gold never
    appears contributes 0.

    >>> m = ranking_metrics({'a': ['x', 'y'], 'b': ['z', 'w']}, {'a': 'y', 'b': 'z'}, ks=(1,))
    >>> m['recall_at_1'], m['mrr']
    (0.5, 0.75)
    """
    recalls = {k: [] for k in ks}
    rr, ap = [], []
    for q, truth in gold.items():
        truths = (
            set(truth)
            if isinstance(truth, (set, frozenset, list, tuple))
            else {truth}
        )
        if not truths:
            continue
        cands = list(dict.fromkeys(ranked.get(q, ())))  # a repeated candidate counts once
        hits = [i for i, c in enumerate(cands) if c in truths]
        for k in ks:
            recalls[k].append(len([i for i in hits if i < k]) / len(truths))
        rr.append(1 / (hits[0] + 1) if hits else 0.0)
        ap.append(
            sum((n + 1) / (i + 1) for n, i in enumerate(hits)) / len(truths)
            if hits
            else 0.0
        )
    if not rr:
        raise ValueError("no query has a gold answer")
    out = {f"recall_at_{k}": float(np.mean(v)) for k, v in recalls.items()}
    out.update(mrr=float(np.mean(rr)), map=float(np.mean(ap)), n_queries=len(rr))
    return out
