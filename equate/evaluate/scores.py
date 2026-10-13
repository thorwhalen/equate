"""The one score type every metric returns: precision and recall, never F1 alone.

:class:`Scores` is the SSOT for how a match quality figure is represented (research
doc 14, §2.5): an F1 without the precision/recall it averages hides whether a matcher
is over- or under-linking, so the type carries all three plus the support they were
computed over.
"""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class Scores:
    """Precision, recall and their harmonic mean, plus what they were computed over.

    >>> s = Scores.from_counts(tp=3, n_pred=4, n_true=6)
    >>> s.precision, s.recall, round(s.f1, 3)
    (0.75, 0.5, 0.6)
    >>> s.support
    6
    """

    precision: float
    recall: float
    f1: float
    #: the size of the gold side (true pairs, or records for a cluster metric)
    support: Optional[int] = None

    @classmethod
    def from_pr(cls, precision: float, recall: float, *, support=None) -> "Scores":
        """Build from precision and recall; F1 is derived, so it can never disagree."""
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else 0.0
        return cls(float(precision), float(recall), float(f1), support)

    @classmethod
    def from_counts(cls, *, tp: int, n_pred: int, n_true: int) -> "Scores":
        """Build from true-positive, predicted and gold counts.

        An empty prediction has precision 1.0 (it asserted nothing false); an empty gold
        set has recall 1.0 (there was nothing to find). Both empty is a perfect score.
        """
        precision = tp / n_pred if n_pred else 1.0
        recall = tp / n_true if n_true else 1.0
        return cls.from_pr(precision, recall, support=n_true)

    def as_dict(self) -> dict:
        """The fields as a plain dict (for tables, JSON, a dataframe row)."""
        return asdict(self)
