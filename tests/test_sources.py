"""Tests for multi-source alignment (``align_sources``) and field-by-field fusion (``fuse``)."""

import json
import statistics
from collections import namedtuple
from dataclasses import asdict

import numpy as np
import pytest

from equate import ScoreMatrix, align_sources, fuse, match, record_similarity
from equate.sources import free_order, monotone_order

WORDS = [
    "bonjour",
    "merci beaucoup",
    "au revoir",
    "s'il vous plait",
    "excusez-moi",
    "bonne nuit",
    "a demain",
]


def _recs(texts, **extra):
    return [
        {"text": t, **{k: v[i] for k, v in extra.items()}} for i, t in enumerate(texts)
    ]


def _members(al):
    return [g.members for g in al]


# --- the four acceptance cases ----------------------------------------------------------


def test_gap_in_one_source_does_not_shift_the_rest():
    sheet = _recs(WORDS)
    audio = _recs([w for k, w in enumerate(WORDS) if k != 2])  # item 2 skipped
    ocr = _recs(WORDS)
    al = align_sources({"sheet": sheet, "audio": audio, "ocr": ocr})
    assert len(al) == len(WORDS)
    for k, g in enumerate(al):
        assert g.members["sheet"] == k and g.members["ocr"] == k
        if k == 2:
            assert "audio" not in g.members
            assert al.missing(g) == ["audio"]
        else:
            assert g.records["audio"]["text"] == WORDS[k]
    assert not any(g.reordered for g in al)


def test_gap_in_the_backbone_source():
    """An item only the later sources hold becomes its own group, in place."""
    sheet = _recs([w for k, w in enumerate(WORDS) if k != 3])
    audio = _recs(WORDS)
    al = align_sources({"sheet": sheet, "audio": audio})
    texts = [g.records.get("audio", g.records.get("sheet"))["text"] for g in al]
    assert texts == WORDS
    assert al[3].members == {"audio": 3}


def test_local_order_swap_is_recovered_not_cascaded():
    sheet = _recs(WORDS)
    swapped = WORDS[:3] + [WORDS[4], WORDS[3]] + WORDS[5:]
    audio = _recs(swapped)
    al = align_sources({"sheet": sheet, "audio": audio})
    assert len(al) == len(WORDS)  # nothing left unmatched
    for g in al:
        assert g.records["sheet"]["text"] == g.records["audio"]["text"]
    assert [g.reordered for g in al] == [k in (3, 4) for k in range(len(WORDS))]


def test_swap_beyond_the_window_is_left_as_gaps():
    sheet = _recs(WORDS)
    audio = _recs([WORDS[6]] + WORDS[:6])  # the last item read first: 6 positions away
    al = align_sources({"sheet": sheet, "audio": audio}, window=1)
    assert all(not g.reordered for g in al)
    assert {"audio": 0} in _members(al)  # stayed a gap: never linked across the list
    unbounded = align_sources({"sheet": sheet, "audio": audio}, window=None)
    assert len(unbounded) == len(WORDS)  # repaired at any distance
    assert all(g.records["sheet"] == g.records["audio"] for g in unbounded)


def test_interleaved_kinds_read_in_the_other_order():
    """Items split into per-kind records (zh then fr); one item read fr-first.

    Records of different kinds share no field, so they never align; the swap of an item's
    two records is a local swap, recovered and flagged.
    """
    sheet = [
        {"zh": "nihao", "item": 0},
        {"fr": "bonjour", "item": 0},
        {"zh": "xiexie", "item": 1},
        {"fr": "merci", "item": 1},
        {"zh": "zaijian", "item": 2},
        {"fr": "au revoir", "item": 2},
    ]
    audio = [
        {"zh": "nihau", "start": 0.0},
        {"fr": "bonjour", "start": 1.0},
        {"fr": "merci", "start": 2.0},  # fr first for item 1
        {"zh": "xiexi", "start": 3.0},
        {"zh": "zaijian", "start": 4.0},
        {"fr": "au revoir", "start": 5.0},
    ]
    al = align_sources({"sheet": sheet, "audio": audio})
    assert len(al) == 6 and all(len(g.members) == 2 for g in al)
    for g in al:
        assert set(g.records["sheet"]) - {"item"} == set(g.records["audio"]) - {"start"}
    assert [g.reordered for g in al] == [False, False, True, True, False, False]
    fused = fuse(al)
    assert [f.values["item"] for f in fused] == [0, 0, 1, 1, 2, 2]


def test_source_with_fields_the_others_lack():
    sheet = _recs(WORDS[:3], pinyin=["a", "b", "c"])
    audio = _recs(["bonjou", "merci bocou", "o revoir"], start=[0.0, 1.0, 2.0])
    al = align_sources({"sheet": sheet, "audio": audio})
    assert _members(al) == [{"sheet": k, "audio": k} for k in range(3)]
    fused = fuse(al, trust={"audio": {"text": 0.2}})
    for k, item in enumerate(fused):
        assert item.values == {
            "text": WORDS[k],
            "pinyin": "abc"[k],
            "start": float(k),
        }
        assert item.provenance == {"text": "sheet", "pinyin": "sheet", "start": "audio"}
        assert not item.conflicts  # a low-trust source disagreeing is not a conflict


@pytest.mark.parametrize("order", ["monotone", "free"])
def test_two_sources_reduce_to_match_on_shared_fields(order):
    A = _recs(WORDS, page=list(range(len(WORDS))))
    noisy = ["bonjou", "merci bocou", "o revoir", "sil vous plai", "excuse moi"]
    B = _recs(noisy + ["xyz", "qqq"], start=[float(k) for k in range(7)])
    sim = record_similarity()
    al = align_sources({"a": A, "b": B}, order=order, threshold=0.5)
    m = match(A, B, compare=sim)
    expected = {(i, j) for i, j in m if sim(A[i], B[j]) > 0.5}
    got = {(g.members["a"], g.members["b"]) for g in al if len(g.members) == 2}
    assert got == expected == {(k, k) for k in range(5)}
    # every record appears exactly once
    for s, data in (("a", A), ("b", B)):
        assert sorted(g.members[s] for g in al if s in g.members) == list(
            range(len(data))
        )


# --- ordering strategies ----------------------------------------------------------------


def test_free_order_aligns_a_shuffled_source():
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(WORDS))
    al = align_sources(
        {"a": _recs(WORDS), "b": _recs([WORDS[k] for k in perm])}, order="free"
    )
    for g in al:
        assert g.records["a"]["text"] == g.records["b"]["text"]


def test_monotone_never_crosses_without_the_repair():
    gain = np.array([[-1.0, 1.0], [1.0, -1.0]])
    assert len(monotone_order(gain, window=0)) == 1
    assert sorted(monotone_order(gain)) == [(0, 1), (1, 0)]
    assert free_order(gain) == [(0, 1), (1, 0)]
    assert monotone_order(np.empty((0, 3))) == [] and free_order(np.empty((2, 0))) == []


def test_order_callable_and_its_contract():
    sources = {"a": _recs(WORDS[:3]), "b": _recs(WORDS[:3])}
    seen = []

    def first_only(gain):
        seen.append(gain)
        return [(0, 0), (1, 2)]  # (1, 2) has no positive gain: dropped

    al = align_sources(sources, order=first_only)
    assert isinstance(seen[0], ScoreMatrix) and seen[0].sense == "maximize"
    assert _members(al)[0] == {"a": 0, "b": 0}
    assert len(al) == 5 and al.order == "first_only"
    twice = {"a": _recs(WORDS[:1]), "b": _recs([WORDS[0], WORDS[0]])}
    with pytest.raises(ValueError, match="one-to-one"):
        align_sources(twice, order=lambda gain: [(0, 0), (0, 1)])
    with pytest.raises(ValueError, match="unknown order"):
        align_sources(sources, order="sideways")


@pytest.mark.parametrize("order", ["greedy", "stable", "assign"])
def test_any_registered_matcher_is_an_order(order):
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(WORDS))
    al = align_sources(
        {"a": _recs(WORDS), "b": _recs([WORDS[k] for k in perm])}, order=order
    )
    assert al.order == order
    assert all(g.records["a"] == g.records["b"] for g in al)


def test_records_sharing_no_field_never_align_even_at_threshold_zero():
    with pytest.warns(UserWarning, match="shares a compared field"):
        al = align_sources(
            {"a": [{"zh": "nihao"}], "b": [{"fr": "bonjour"}]}, threshold=-1.0
        )
    assert _members(al) == [{"a": 0}, {"b": 0}]


# --- comparison, margins, inputs ---------------------------------------------------------


def test_margin_is_low_for_an_ambiguous_item():
    sheet = _recs(["ni hao", "ni hao ma", "xie xie"])
    audio = _recs(["ni hao", "ni hao ma", "xie xie"])
    al = align_sources({"sheet": sheet, "audio": audio})
    margins = [g.margin for g in al]
    assert margins[0] < margins[2] and margins[1] < margins[2]
    clear = align_sources({"sheet": sheet[2:], "audio": audio[2:]})
    assert clear[0].margin == pytest.approx(1.0)  # no rival at all


def test_per_field_comparators_and_injected_domain_knowledge():
    def toneless(a, b):  # a caller's domain comparator, injected
        strip = str.maketrans("āáǎàēéěèīíǐìōóǒòūúǔù", "aaaaeeeeiiiioooouuuu")
        return 1.0 if a.translate(strip) == b.translate(strip) else 0.0

    sheet = [{"pinyin": "nǐ hǎo", "zh": "你好"}, {"pinyin": "xiè xie", "zh": "谢谢"}]
    audio = [{"pinyin": "xie xie", "zh": "谢"}, {"pinyin": "ni hao", "zh": "泥好"}]
    al = align_sources(
        {"sheet": sheet, "audio": audio},
        compare={"pinyin": toneless},
        order="free",
    )
    assert {(g.members["sheet"], g.members["audio"]) for g in al} == {(0, 1), (1, 0)}
    with pytest.raises(ValueError, match="distance"):
        align_sources({"a": sheet, "b": audio}, compare={"zh": "levenshtein_distance"})
    with pytest.raises(TypeError):
        record_similarity(compare=["zh"])


def test_record_types_and_degenerate_inputs():
    Rec = namedtuple("Rec", "text")

    class Obj:
        def __init__(self, text):
            self.text = text

    al = align_sources({"a": [Rec("x y z"), Rec("abc")], "b": [Obj("abc")]})
    assert _members(al) == [{"a": 0}, {"a": 1, "b": 0}]
    assert len(align_sources({"a": [], "b": _recs(WORDS[:2])})) == 2
    single = align_sources({"only": _recs(WORDS[:2])})
    assert [g.score for g in single] == [None, None]
    assert fuse(single).unmatched == []
    with pytest.raises(TypeError):
        align_sources([_recs(WORDS)])


# --- fuse ----------------------------------------------------------------------------------


def test_fuse_votes_trust_and_conflicts():
    a = align_sources(
        {
            "sheet": [{"text": "au revior"}],  # a typo on the trusted list
            "ocr": [{"text": "au revoir"}],
            "audio": [{"text": "au revoir"}],
        }
    )
    fused = fuse(a, trust={"audio": {"text": 0.3}, "ocr": 0.8})
    (item,) = fused
    assert item.values["text"] == "au revoir"  # 0.8 + 0.3 outvote 1.0
    assert item.provenance["text"] == "ocr"  # the most trusted source in the vote
    (conflict,) = item.conflicts
    assert conflict.value == "au revoir"
    assert [(v.sources, round(v.weight, 2)) for v in conflict.votes] == [
        (["ocr", "audio"], 1.1),
        (["sheet"], 1.0),
    ]
    assert fused.conflicts == [(0, conflict)] and item.needs_review
    # raise the bar: the losing vote must weigh as much as the winner
    assert not fuse(a, trust={"audio": {"text": 0.3}, "ocr": 0.8}, conflict_ratio=1.0)[
        0
    ].conflicts


def test_fuse_agree_predicates_fields_and_unmatched():
    al = align_sources(
        {
            "a": _recs(["Bonjour ", "merci"], n=[1, 2]),
            "b": _recs(["bonjour", "xyzzy"], n=[1, 3]),
        }
    )
    fused = fuse(al)
    assert not fused[0].conflicts  # default agreement ignores case and spacing
    assert [len(item.members) for item in fused.unmatched] == [1, 1]
    assert sorted(item.missing[0] for item in fused.unmatched) == ["a", "b"]
    only_text = fuse(al, fields=["text"])
    assert all(set(item.values) == {"text"} for item in only_text)
    strict = fuse(al, agree={"text": lambda x, y: x == y})
    assert strict[0].conflicts and strict[0].values["text"] == "Bonjour "  # tie: first
    with pytest.raises(TypeError):
        fuse(al, agree=3)


def test_results_are_json_able():
    al = align_sources({"a": _recs(WORDS[:3]), "b": _recs(WORDS[:3])})
    fused = fuse(al)
    json.dumps(asdict(al))
    json.dumps([asdict(item) for item in fused])


# --- the README worked example: the fusion beats every single source ---------------------


def test_fusion_beats_every_single_source():
    truth = [
        {"text": t, "en": e}
        for t, e in zip(
            WORDS[:6],
            [
                "hello",
                "thank you very much",
                "goodbye",
                "please",
                "excuse me",
                "good night",
            ],
        )
    ]
    sheet = [dict(r) for r in truth]
    sheet[2]["text"] = "au revior"  # typo
    audio = _recs(
        ["bonjour", "merci bocou", "au revoir", "excusez moi", "sil vous plait"],
        start=[0.0, 1.2, 2.6, 3.9, 5.1],
    )  # noisy, items 3 and 4 swapped, item 5 skipped
    ocr = _recs(
        [
            "bonjour",
            "merci beaucoup",
            "au revoir",
            "s'il vous p1ait",
            "excusez-moi",
            "bonne nuit",
        ],
        page=[1, 1, 1, 1, 2, 2],
    )
    al = align_sources({"sheet": sheet, "audio": audio, "ocr": ocr})
    fused = fuse(al, trust={"audio": {"text": 0.3}, "ocr": {"text": 0.8}})

    def text_accuracy(texts):
        return sum(t == r["text"] for t, r in zip(texts, truth)) / len(truth)

    fused_acc = text_accuracy([item.values["text"] for item in fused])
    assert fused_acc == 1.0
    assert text_accuracy([r["text"] for r in sheet]) < 1.0
    assert text_accuracy([r["text"] for r in ocr]) < 1.0
    assert text_accuracy([g.records.get("audio", {}).get("text") for g in al]) < 1.0
    # timing reached the sheet's items, and every planted error is flagged
    assert [item.values.get("start") for item in fused] == [
        0.0,
        1.2,
        2.6,
        5.1,
        3.9,
        None,
    ]
    assert [k for k, _ in fused.conflicts] == [2, 3]  # the sheet typo, the OCR error
    assert [item.reordered for item in fused] == [False] * 3 + [True, True, False]
    assert fused[5].missing == ["audio"]


# --- regressions from the adversarial review ----------------------------------------------


def test_unmatched_record_never_lands_inside_a_swapped_pair():
    pies = ["apple pie", "banana split", "cherry tart"]
    audio = ["banana split", "apple pie", "xylophone", "xylophone", "cherry tart"]
    al = align_sources(
        {"sheet": _recs(pies), "audio": _recs(audio), "ocr": _recs(pies)}
    )
    full = [g for g in al if len(g.members) == 3]
    assert len(full) == 3  # no item split across two groups
    texts = [next(iter(g.records.values()))["text"] for g in al]
    assert texts[:2] == ["apple pie", "banana split"]  # the swapped pair stays together
    assert texts[-1] == "cherry tart"


def test_one_noisy_member_does_not_keep_a_clean_match_out():
    table = {("s", "a"): 0.6, ("s", "o"): 0.9, ("a", "o"): 0.05}

    def sim(x, y):
        return table.get((x["src"], y["src"]), table.get((y["src"], x["src"]), 0.0))

    def recs(src, n):
        return [{"src": src, "k": k} for k in range(n)]

    def keyed(x, y):  # the table above, only for records of the same item
        return sim(x, y) if x["k"] == y["k"] else 0.0

    al = align_sources(
        {"s": recs("s", 3), "a": recs("a", 3), "o": recs("o", 3)}, compare=keyed
    )
    assert [g.members for g in al] == [{"s": k, "a": k, "o": k} for k in range(3)]


@pytest.mark.parametrize("window", [1, 2, 3])
def test_an_item_moved_exactly_window_places_is_recovered(window):
    moved = WORDS[1 : window + 1] + [WORDS[0]] + WORDS[window + 1 :]
    al = align_sources({"sheet": _recs(WORDS), "audio": _recs(moved)}, window=window)
    assert len(al) == len(WORDS)
    assert all(g.records["sheet"] == g.records["audio"] for g in al)
    too_far = align_sources(
        {"sheet": _recs(WORDS), "audio": _recs(moved)}, window=window - 1
    )
    assert len(too_far) == len(WORDS) + 1  # one item left as two singletons


def test_free_order_margins_see_rivals_anywhere():
    a = _recs(["alpha beta", "c", "d", "e", "f", "g", "alpha betta"])
    b = _recs(["alpha beta"])
    al = align_sources({"a": a, "b": b}, order="free")
    (pair,) = [g for g in al if len(g.members) == 2]
    assert pair.margin < 0.1  # the near-duplicate six places away is a rival


def test_nan_similarity_is_no_score_not_a_crash():
    def sim(x, y):
        return float("nan") if x["text"] == WORDS[0] else float(x["text"] == y["text"])

    for order in ("monotone", "free"):
        al = align_sources(
            {"a": _recs(WORDS[:4]), "b": _recs(WORDS[:4])}, compare=sim, order=order
        )
        assert sum(len(g.members) == 2 for g in al) == 3
    g = np.eye(5) - 0.5
    g[0, 4] = np.nan
    assert len(monotone_order(g)) == 5


def test_string_and_slotted_records_and_mixed_source_names():
    from dataclasses import dataclass

    @dataclass(slots=True)
    class Line:
        text: str

    strings = align_sources({0: ["bonjour", "merci"], "b": ["bonjour", "merci"]})
    assert [g.members for g in strings] == [{0: 0, "b": 0}, {0: 1, "b": 1}]
    slotted = align_sources({"a": [Line("bonjour")], "b": [Line("bonjour")]})
    assert slotted[0].members == {"a": 0, "b": 0}


def test_order_pairs_out_of_bounds_raise():
    with pytest.raises(ValueError, match="outside"):
        align_sources(
            {"a": _recs(WORDS[:2]), "b": _recs(WORDS[:2])},
            order=lambda gain: [(-1, 0)],
        )


def test_fuse_zero_trust_is_never_a_conflict_and_nan_agrees():
    al = align_sources(
        {
            "a": [{"text": "x", "v": float("nan")}],
            "b": [{"text": "x", "v": float("nan")}],
        }
    )
    assert not fuse(al)[0].conflicts
    al = align_sources({"a": [{"text": "x", "n": 1}], "b": [{"text": "x", "n": 2}]})
    assert fuse(al)[0].conflicts
    zero = fuse(al, trust={"a": {"n": 0}, "b": {"n": 0}})
    assert not zero[0].conflicts and zero[0].values["n"] == 1


@pytest.mark.parametrize(
    "combine, expected",
    [(max, 1.0), (min, 0.0), (sum, 1.0), (statistics.median, 0.5)],
)
def test_record_similarity_takes_number_reducers(combine, expected):
    sim = record_similarity({"fr": "ratio", "zh": "ratio"}, combine=combine)
    assert sim({"fr": "merci", "zh": "ab"}, {"fr": "merci", "zh": "cd"}) == expected


def test_record_similarity_mapping_combiners_still_get_field_names():
    from equate.compare.vectorize import weighted_sum

    sim = record_similarity(
        {"fr": "ratio", "zh": "ratio"}, combine=weighted_sum({"fr": 2.0, "zh": 0.0})
    )
    assert sim({"fr": "merci", "zh": "ab"}, {"fr": "merci", "zh": "cd"}) == 2.0
