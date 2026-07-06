"""Tools for matching things: featurize, compare, and match collections of objects.

Matching is fuzzy correspondence, not exact equality (``==`` is the degenerate case).
The public surface is being redesigned around the ``featurize -> compare -> match``
decomposition; see ``docs/research/`` and the ``equate-dev-architecture`` skill. The
foundation contracts live in :mod:`equate.base`.
"""

from equate.base import (
    to_cost,
    ScoreMatrix,
    Candidate,
    Matching,
    Explanation,
    Partition,
    FeaturizerMeta,
    ComparatorMeta,
)

from equate._dependencies import require, have, MissingDependencyError
from equate.registry import Registry
from equate.featurize import featurizers, resolve_featurizer
from equate.compare import comparators, resolve_comparator, direct, featurized
from equate.block import (
    blockers,
    resolve_blocker,
    all_pairs,
    blocking_metrics,
    score_candidates,
)

from equate.util import (
    match_keys_to_values,
    match_greedily,
    similarity_matrix,
    hungarian_matching,
)

# from equate.examples.site_names import (
#     DFLT_SITE_PKG_DIR,
#     site_packages_info_df,
#     print_n_null_elements_in_each_column_containing_at_least_one,
#     Lidx,
# )
