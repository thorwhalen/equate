"""Tests for equate.completion: join keys, near-duplicate columns, table completion."""

import math

import pytest

from equate.completion import (
    ColumnMatch,
    as_columns,
    complete_table,
    find_join_keys,
    find_near_duplicate_columns,
)
from equate.compare.sets import containment, jaccard, overlap

ORDERS = {
    "order": ["o1", "o2", "o3", "o4"],
    "customer": ["C-001", "C-002", "C-001", "C-003"],
    "city": [None, "Lyon", "", None],
    "qty": [1, 2, 3, 4],
}
CUSTOMERS = {
    "cid": ["C-003", "C-001", "C-002", "C-004"],
    "name": ["Kate", "Jon", "Ann", "Bob"],
    "city": ["Nice", "Paris", "Lille", "Lyon"],
}


def test_set_comparators():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert overlap({1, 2}, {1, 2, 3, 4}) == 1.0
    assert (containment({1, 2}, {1, 2, 3, 4}), containment({1, 2, 3, 4}, {1, 2})) == (1.0, 0.5)
    assert jaccard(set(), set()) == overlap(set(), set()) == containment(set(), {1}) == 0.0


def test_as_columns_accepts_column_maps_rows_and_dataframes():
    assert as_columns({"a": (1, 2)}) == {"a": [1, 2]}
    assert as_columns([{"a": 1}, {"b": 2}]) == {"a": [1, None], "b": [None, 2]}

    class Frame:  # duck-typed DataFrame
        columns = ["x", "y"]

        def __getitem__(self, c):
            return {"x": [1, 2], "y": [3, 4]}[c]

    assert as_columns(Frame()) == {"x": [1, 2], "y": [3, 4]}
    with pytest.raises(TypeError):
        as_columns(42)


def test_find_join_keys_by_containment():
    keys = find_join_keys(ORDERS, CUSTOMERS)
    best = keys[0]
    assert (best.left, best.right, best.score) == ("customer", "cid", 1.0)
    assert best.explanation.details["shared"] == 3
    # directional: orders.customer is fully inside customers.cid, not the reverse
    reverse = find_join_keys(CUSTOMERS, ORDERS)
    assert (reverse[0].left, reverse[0].right, reverse[0].score) == ("cid", "customer", 0.75)


def test_find_join_keys_normalizes_values():
    left = {"k": ["Paris ", "LYON", 3.0]}
    right = {"k": ["paris", "lyon", 3]}
    assert find_join_keys(left, right)[0].score == 1.0


def test_find_join_keys_min_score_and_k():
    assert find_join_keys({"a": ["x"]}, {"b": ["y"]}) == []
    many = find_join_keys({"a": ["x", "y"]}, {"b": ["x", "y"], "c": ["x"], "d": ["y", "z"]}, k=2)
    assert [(m.right, m.score) for m in many] == [("b", 1.0), ("c", 0.5)]


def test_near_duplicate_columns():
    table = {
        "id": [1, 2, 3, 4],
        "id_copy": ["1", "2", "3", "4"],
        "label": ["a", "b", "c", "d"],
        "label_mostly": ["a", "b", "c", "e"],
    }
    dups = find_near_duplicate_columns(table, threshold=0.5)
    assert [(m.left, m.right) for m in dups] == [("id", "id_copy"), ("label", "label_mostly")]
    assert dups[0].score == 1.0 and math.isclose(dups[1].score, 3 / 5)
    assert find_near_duplicate_columns(table) == [dups[0]]  # default threshold 0.9


def test_complete_table_discovers_key_and_fills_only_missing_cells():
    done = complete_table(ORDERS, CUSTOMERS)
    assert (done.key.left, done.key.right) == ("customer", "cid")
    assert done.table["city"] == ["Paris", "Lyon", "Paris", "Nice"]  # 'Lyon' kept as is
    assert done.filled == {"city": 3}
    assert done.columns == {"city": "city"}
    assert done.table["qty"] == ORDERS["qty"] and ORDERS["city"][0] is None  # input untouched
    assert done.as_rows()[3] == {"order": "o4", "customer": "C-003", "city": "Nice", "qty": 4}


def test_complete_table_adds_named_columns_and_matches_fuzzy_keys():
    target = [{"cust": "c001"}, {"cust": "C-002 "}, {"cust": None}]
    done = complete_table(
        target, CUSTOMERS, on=("cust", "cid"), columns={"customer_name": "name"}
    )
    assert done.table["customer_name"] == ["Jon", "Ann", None]  # a missing key fills nothing
    assert done.filled == {"customer_name": 2}


def test_complete_table_row_floor_and_one_to_one():
    target = {"k": ["C-001", "zzz"], "name": [None, None]}
    done = complete_table(target, CUSTOMERS, on=("k", "cid"))
    assert done.table["name"] == ["Jon", None]  # 'zzz' resembles nothing: not filled
    assert len(done.rows) == 1
    loose = complete_table(target, CUSTOMERS, on=("k", "cid"), min_row_score=None)
    assert loose.table["name"][1] is not None
    # one-to-one: two target rows cannot both take source row C-001
    twins = {"k": ["C-001", "C-001"], "name": [None, None]}
    one = complete_table(twins, CUSTOMERS, on=("k", "cid"), how="optimal", min_row_score=None)
    assert len({j for _, j, _ in one.rows}) == 2


def test_complete_table_errors():
    with pytest.raises(ValueError, match="pass on="):
        complete_table({"a": ["x"]}, {"b": ["y"]})
    with pytest.raises(KeyError):
        complete_table(ORDERS, CUSTOMERS, on=("nope", "cid"))
    with pytest.raises(KeyError):
        complete_table(ORDERS, CUSTOMERS, columns={"city": "nope"})
    with pytest.raises(ValueError, match="different lengths"):
        complete_table({"a": [1, 2], "b": [1]}, CUSTOMERS)


def test_column_match_is_a_plain_record():
    m = ColumnMatch("a", "b", 0.5)
    assert (m.left, m.right, m.score, m.explanation) == ("a", "b", 0.5, None)
