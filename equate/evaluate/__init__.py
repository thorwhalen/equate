"""Evaluate matchers honestly: metrics, leakage-aware splits, and a benchmark harness.

Dependency-light (numpy/scipy only; sklearn is used for ARI / V-measure when present).
Precision and recall always travel together (:class:`Scores`), the pairwise and cluster
granularities are explicit (:func:`pairs_to_labels` converts on purpose), B³ is the
default cluster metric, thresholds are selected on validation data by a separate call,
and :func:`benchmark_matcher` splits by entity so test entities are unseen.
See ``docs/research/14-evaluation-benchmarks-and-methodology.md``.

>>> from equate.evaluate import pairwise_prf, bcubed
>>> pairwise_prf([(0, 0), (1, 1)], [(0, 0), (1, 2)]).f1
0.5
"""

from equate.block.metrics import blocking_metrics
from equate.evaluate.scores import Scores
from equate.evaluate.labels import pairs_to_labels, labels_to_pairs, cross_pairs_to_labels
from equate.evaluate.metrics import pairwise_prf, bcubed, ceaf, cluster_metrics
from equate.evaluate.ranking import pr_auc, select_threshold, ranking_metrics
from equate.evaluate.split import Split, entity_split, assert_no_leakage
from equate.evaluate.benchmark import (
    Benchmark,
    BenchmarkReport,
    BenchmarkRow,
    benchmarks,
    list_benchmarks,
    load_benchmark,
    benchmark_matcher,
)

__all__ = [
    "Scores",
    "pairwise_prf",
    "bcubed",
    "ceaf",
    "cluster_metrics",
    "blocking_metrics",
    "pr_auc",
    "select_threshold",
    "ranking_metrics",
    "pairs_to_labels",
    "labels_to_pairs",
    "cross_pairs_to_labels",
    "Split",
    "entity_split",
    "assert_no_leakage",
    "Benchmark",
    "BenchmarkReport",
    "BenchmarkRow",
    "benchmarks",
    "list_benchmarks",
    "load_benchmark",
    "benchmark_matcher",
]
