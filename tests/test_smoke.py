"""Smoke tests for the equate package.

Minimal, deterministic checks that keep CI meaningful during the redesign:
the public API is importable, and the standard-library-based greedy matcher
(no heavy numeric deps involved in the assertion) produces its known result.
"""

import equate


def test_public_api_is_importable():
    """The documented root-level names should be importable from the package."""
    for name in (
        "match_greedily",
        "match_keys_to_values",
        "similarity_matrix",
        "hungarian_matching",
    ):
        assert hasattr(equate, name), f"equate.{name} should be importable from the root"


def test_match_greedily_basic():
    """Each key is paired with its best still-available value (stdlib difflib)."""
    result = dict(
        equate.match_greedily(
            ["apple", "banana", "carrot"],
            ["car", "app", "carob", "cabana"],
        )
    )
    assert result == {"apple": "app", "banana": "cabana", "carrot": "carob"}
