"""Tests for equate.service: the JSON-in/JSON-out service core and its qh HTTP surface."""

import json

import pytest

from equate.service import (
    BOUNDARY_TYPES,
    Edit,
    MatchService,
    SessionNotFound,
    boundary_fields,
    explain_pair,
    record_text,
    to_jsonable,
)

A = [
    {"name": "Jon Smith", "city": "Paris"},
    {"name": "Kate Doe", "city": "Lyon"},
    {"name": "Ann Lee", "city": "Nice"},
]
B = [
    {"name": "Kate Do", "city": "Lyon"},
    {"name": "John Smith", "city": "paris"},
    {"name": "Anne Li", "city": "Nice"},
]
TRUTH = {0: 1, 1: 0, 2: 2}


def _pairs(views):
    return {p.item: p.candidate for p in views}


# --- stateless match ---------------------------------------------------------------


def test_match_strings_and_records():
    svc = MatchService()
    r = svc.match(["apple pie", "banana split"], ["banana splitt", "aple pie"])
    assert [(p.left, p.right) for p in r.pairs] == [
        ("apple pie", "aple pie"),
        ("banana split", "banana splitt"),
    ]
    # records go through record_text on the default vector route
    assert _pairs(svc.match(A, B).pairs) == TRUTH


def test_match_is_cached_by_content_and_accounted():
    svc = MatchService()
    first = svc.match(A, B)
    assert not first.cached
    assert set(first.accounting.tiers) == {"score", "match"}
    assert first.accounting.tiers["score"].items == 9
    second = svc.match(A, B)
    assert second.cached and _pairs(second.pairs) == TRUTH
    assert not MatchService(cache=None).match(A, B).cached


def test_match_threshold_and_cluster():
    svc = MatchService()
    kept = svc.match(A, B, threshold=0.6).pairs
    assert all(p.score >= 0.6 for p in kept) and len(kept) < 3
    groups = svc.match(
        ["ab", "zz"], ["ab", "qq"], compare="ratio", how="cluster"
    ).groups
    assert [0, 2] in groups  # A[0] with B[0] (offset by len(A))


def test_on_cost_hook_sees_every_tier():
    seen = []
    MatchService(on_cost=lambda tier, s, n: seen.append(tier)).match(A, B)
    assert seen == ["score", "match"]


# --- sessions ----------------------------------------------------------------------


def _session(svc=None, **kw):
    svc = svc or MatchService()
    return svc, svc.open_session(A, B, **kw)


def test_open_session_assigns_and_queues_most_uncertain_first():
    svc, v = _session(k=2)
    assert _pairs(v.assignment) == TRUTH
    assert all(len(row.candidates) == 2 for row in v.review)
    margins = [row.margin for row in v.review]
    assert margins == sorted(margins)
    assert all(p.status == "proposed" for p in v.assignment)


def test_candidates_read_and_refresh_without_rescoring():
    svc, v = _session(k=1)
    sid = v.session_id
    assert len(svc.candidates(sid, 0)) == 1
    refreshed = svc.candidates(sid, 0, k=3)
    assert len(refreshed) == 3
    assert [c.score for c in refreshed] == sorted(
        (c.score for c in refreshed), reverse=True
    )
    assert len(svc.candidates(sid, 0)) == 3  # the refresh is stored
    assert svc.view(sid).accounting.tiers["score"].calls == 1  # never re-scored


def test_confirm_forces_and_reoptimizes_locally():
    svc, v = _session()
    r = svc.edit(v.session_id, [{"action": "confirm", "item": 0, "candidate": 0}])
    after = _pairs(r.assignment)
    assert after[0] == 0 and after[1] == 1  # the displaced item moves
    assert r.changed == [0, 1]
    statuses = {p.item: p.status for p in r.assignment}
    assert statuses[0] == "confirmed" and statuses[1] == "proposed"
    review = svc.view(v.session_id).review
    assert review[-1].item == 0 and review[-1].status == "confirmed"  # decided -> last


def test_reassign_releases_the_previous_holder():
    svc, v = _session()
    sid = v.session_id
    svc.edit(sid, [Edit("confirm", 0, 1)])
    r = svc.edit(sid, [Edit("reassign", 2, 1)])  # item 2 takes candidate 1
    after = _pairs(r.assignment)
    assert after[2] == 1 and after.get(0) != 1
    assert {p.item for p in r.assignment if p.status == "confirmed"} == {2}


def test_reject_all_candidates_leaves_item_unmatched():
    svc = MatchService()
    v = svc.open_session(["ab", "cd"], ["ab", "cd"], compare="ratio")
    r = svc.edit(
        v.session_id,
        [Edit("reject", 0, 0), Edit("reject", 0, 1)],
    )
    assert _pairs(r.assignment) == {1: 1}
    statuses = {row.item: row.status for row in svc.review(v.session_id)}
    assert statuses[0] == "unmatched"


def test_clear_undoes_a_reject():
    svc, v = _session()
    sid = v.session_id
    svc.edit(sid, [Edit("reject", 0, 1)])
    assert _pairs(svc.view(sid).assignment)[0] != 1
    svc.edit(sid, [Edit("clear", 0, 1)])
    assert _pairs(svc.view(sid).assignment) == TRUTH


def test_a_zero_score_is_still_a_candidate_after_a_reject():
    svc = MatchService()
    exact = lambda a, b: float(a == b)  # noqa: E731
    v = svc.open_session(["a", "b"], ["x", "b"], compare=exact)
    r = svc.edit(v.session_id, [Edit("reject", 1, 1)])
    # b's only real candidate is rejected; a's 0.0 cells are real scores, not holes
    assert 1 not in _pairs(r.assignment) or _pairs(r.assignment)[1] == 0


def test_invalid_edits_are_rejected_atomically():
    svc, v = _session()
    sid = v.session_id
    for bad in (
        [{"action": "bogus", "item": 0, "candidate": 0}],
        [{"action": "confirm", "item": 9, "candidate": 0}],
        [{"action": "confirm", "item": 0}],
        ["confirm"],
    ):
        with pytest.raises(ValueError):
            svc.edit(sid, bad)
    assert svc.view(sid).edits == []


def test_label_applies_and_returns_next_query():
    svc, v = _session()
    sid = v.session_id
    q = svc.next_query(sid)
    assert q.item == v.review[0].item  # the most uncertain item
    res = svc.label(sid, q.item, q.candidate.candidate, False)
    assert res.label.is_match is False
    nxt = res.next
    # the rejected pair is never asked again
    assert (nxt.item, nxt.candidate.candidate) != (q.item, q.candidate.candidate)
    for _ in range(10):  # confirm everything -> nothing left to ask
        if res.next is None:
            break
        res = svc.label(sid, res.next.item, res.next.candidate.candidate, True)
    assert res.next is None
    assert len(svc.view(sid).labels) >= 2


def test_unknown_session_and_close():
    svc, v = _session()
    with pytest.raises(SessionNotFound):
        svc.view("nope")
    assert svc.close_session(v.session_id) is True
    assert svc.close_session(v.session_id) is False


def test_sessions_live_in_any_mutable_mapping():
    class Recording(dict):
        writes = 0

        def __setitem__(self, k, v):
            type(self).writes += 1
            super().__setitem__(k, v)

    store = Recording()
    svc, v = _session(MatchService(sessions=store))
    svc.edit(v.session_id, [Edit("confirm", 0, 1)])
    assert v.session_id in store and Recording.writes >= 2  # written through on edit


def test_blocked_session_never_assigns_a_hole():
    # unbounded scores (CLAUDE.md: never test blocking with [0, 1]-only fixtures)
    svc = MatchService()
    blocker = lambda A, B: [(0, 0), (1, 0), (1, 1)]  # noqa: E731
    big = lambda a, b: 100.0 * (len(a) + len(b))  # noqa: E731
    v = svc.open_session(["aa", "b"], ["cc", "d", "e"], compare=big, block=blocker)
    assert _pairs(v.assignment) == {0: 0, 1: 1}
    assert {c.candidate for c in svc.candidates(v.session_id, 0)} == {0}  # stored only
    r = svc.edit(v.session_id, [Edit("reject", 1, 1)])
    # item 1's only other candidate is taken by item 0: one real pair, never a hole
    assert len(r.assignment) == 1
    assert all((p.item, p.candidate) in {(0, 0), (1, 0)} for p in r.assignment)


# --- JSON boundary -----------------------------------------------------------------


def test_views_are_json_serializable_with_no_infinities():
    svc = MatchService()
    v = svc.open_session(["solo"], ["a", "b"], k=1)  # k=1 -> margin is +inf
    payload = to_jsonable(v)
    text = json.dumps(payload, allow_nan=False)
    assert payload["review"][0]["margin"] is None
    assert "on_record" not in text


def test_explain_pair_field_diff():
    e = explain_pair({"name": "Jon"}, {"name": "Jon", "city": "Paris"}, 0.5)
    fields = e.details["fields"]
    assert [f["field"] for f in fields] == ["name", "city"]
    assert fields[0]["similarity"] == 1.0 and fields[1]["similarity"] is None
    assert explain_pair("abc", "abd", 0.6).details["fields"][0]["field"] == "value"


def test_record_text():
    assert record_text({"a": "x", "b": None, "c": 3}) == "x 3"
    assert record_text("plain") == "plain"


def test_ui_field_snapshot_is_current():
    """The UI checks its Zod schemas against ui/src/schema/fields.json; keep it current
    with ``python -m equate.service fields > ui/src/schema/fields.json``."""
    from pathlib import Path

    snapshot = Path(__file__).parent.parent / "ui" / "src" / "schema" / "fields.json"
    if not snapshot.exists():
        pytest.skip("the ui/ tree is not present (e.g. testing from an sdist)")
    assert json.loads(snapshot.read_text()) == boundary_fields()


def test_json_schemas():
    pytest.importorskip("pydantic")
    from equate.service import json_schemas

    schemas = json_schemas()
    assert set(schemas) == set(BOUNDARY_TYPES)
    assert schemas["Edit"]["properties"]["action"]["enum"] == [
        "confirm",
        "reject",
        "reassign",
        "clear",
    ]


# --- HTTP surface (qh) -------------------------------------------------------------


@pytest.fixture
def client():
    pytest.importorskip("qh")
    from qh.testing import test_app

    from equate.service import mk_app

    with test_app(mk_app()) as c:
        yield c


def test_http_match(client):
    r = client.post("/match", json={"A": A, "B": B})
    assert r.status_code == 200
    assert {p["item"]: p["candidate"] for p in r.json()["pairs"]} == TRUTH


def test_http_session_loop(client):
    v = client.post("/sessions", json={"A": A, "B": B, "k": 2}).json()
    sid = v["session_id"]
    assert client.get(f"/sessions/{sid}").json()["session_id"] == sid
    assert len(client.get(f"/sessions/{sid}/candidates/1").json()) == 2
    assert len(client.post(f"/sessions/{sid}/candidates/1", json={"k": 3}).json()) == 3
    r = client.post(
        f"/sessions/{sid}/edits",
        json={"edits": [{"action": "confirm", "item": 0, "candidate": 0}]},
    ).json()
    assert r["changed"] == [0, 1]
    nxt = client.get(f"/sessions/{sid}/next").json()
    lab = client.post(
        f"/sessions/{sid}/label",
        json={
            "item": nxt["item"],
            "candidate": nxt["candidate"]["candidate"],
            "is_match": True,
        },
    ).json()
    assert lab["label"]["is_match"] is True
    assert client.delete(f"/sessions/{sid}").json() == {"closed": True}


def test_http_error_contract(client):
    sid = client.post("/sessions", json={"A": ["a"], "B": ["a"]}).json()["session_id"]
    assert client.get("/sessions/nope").status_code == 404
    assert client.get(f"/sessions/{sid}/candidates/x").status_code == 422
    assert client.get(f"/sessions/{sid}/candidates/5").status_code == 422
    bad = {"edits": [{"action": "bogus", "item": 0, "candidate": 0}]}
    assert client.post(f"/sessions/{sid}/edits", json=bad).status_code == 422
    unknown = {"A": ["x"], "B": ["y"], "how": "nonesuch"}
    assert client.post("/match", json=unknown).status_code == 422


def test_a_session_pickles_so_any_dol_store_can_persist_it():
    import pickle

    seen = []
    svc = MatchService(
        on_cost=lambda *a: seen.append(a)
    )  # the hook is not session state
    v = svc.open_session(A, B)
    svc.edit(v.session_id, [Edit("confirm", 0, 1)])
    clone = pickle.loads(pickle.dumps(svc.sessions[v.session_id]))
    assert clone.assignment == svc.sessions[v.session_id].assignment
    assert seen  # ...yet it was called


# --- regressions from the adversarial review of #12 --------------------------------


def test_cache_distinguishes_key_order_of_records():
    svc = MatchService()
    B_ = [
        {"name": "Paris", "city": "Jon Smith"},
        {"name": "Jon Smith", "city": "Paris"},
    ]
    a1 = [{"name": "Jon Smith", "city": "Paris"}]
    a2 = [{"city": "Paris", "name": "Jon Smith"}]
    uncached = MatchService(cache=None)
    svc.match(a1, B_)
    assert _pairs(svc.match(a2, B_).pairs) == _pairs(uncached.match(a2, B_).pairs)
    assert svc.match(a2, B_).pairs[0].left == a2[0]


def test_cache_never_hands_out_its_own_objects():
    svc = MatchService()
    svc.match(A, B).pairs.clear()  # the miss's result
    svc.match(A, B).pairs.clear()  # a hit's result
    assert _pairs(svc.match(A, B).pairs) == TRUTH


def test_a_forced_non_candidate_pair_has_no_score():
    blocker = lambda A, B: [(0, 0), (1, 1)]  # noqa: E731
    dist = lambda a, b: 10.0 + abs(len(a) - len(b))  # noqa: E731
    svc = MatchService()
    v = svc.open_session(
        ["aa", "b"], ["cc", "d"], compare=dist, block=blocker, sense="minimize"
    )
    r = svc.edit(v.session_id, [Edit("confirm", 0, 1)])  # (0, 1) was blocked out
    scores = {(p.item, p.candidate): p.score for p in r.assignment}
    assert scores[(0, 1)] is None


def test_plain_number_items_on_the_default_route():
    assert len(MatchService().match([1, 22], [22, 1]).pairs) == 2


def test_sequences_still_reach_a_named_comparator_raw():
    r = MatchService().match(
        [[0, 1, 2], [5, 6, 7]], [[5, 6, 6, 7], [0, 1, 2]], compare="dtw"
    )
    assert _pairs(r.pairs) == {0: 1, 1: 0}


def test_k_and_is_match_are_validated():
    svc, v = _session()
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            svc.candidates(v.session_id, 0, k=bad)
    with pytest.raises(ValueError):
        svc.open_session(A, B, k=0)
    with pytest.raises(ValueError):
        svc.label(v.session_id, 0, 1, "false")
    assert svc.view(v.session_id).labels == []


def test_label_records_its_source_on_the_edit_atomically():
    svc, v = _session()
    svc.label(v.session_id, 0, 1, True, source="llm")
    view = svc.view(v.session_id)
    assert [(e.action, e.source) for e in view.edits] == [("confirm", "llm")]
    assert [lab.source for lab in view.labels] == ["llm"]


def test_a_failed_edit_leaves_the_session_untouched():
    svc, v = _session()
    before = svc.view(v.session_id)
    with pytest.raises(ValueError):
        svc.edit(v.session_id, [{"action": "bogus", "item": 0, "candidate": 0}])
    after = svc.view(v.session_id)
    assert after.edits == before.edits and after.accounting == before.accounting


def test_concurrent_edits_on_one_session_are_all_kept():
    from concurrent.futures import ThreadPoolExecutor

    svc = MatchService()
    n = 30
    v = svc.open_session([f"x{i}" for i in range(n)], [f"x{i}" for i in range(n)])
    with ThreadPoolExecutor(8) as pool:
        list(
            pool.map(
                lambda i: svc.edit(v.session_id, [Edit("reject", i, (i + 1) % n)]),
                range(n),
            )
        )
    assert len(svc.view(v.session_id).edits) == n


# --- session score floor (equate#21) -----------------------------------------------


def test_floor_keeps_a_reject_from_forcing_a_bad_pair():
    names_a = ["Ann Lee", "Bob Martin"]
    names_b = ["Anne Li", "Bob Martin", "Zoe Bernard"]
    svc = MatchService()
    loose = svc.open_session(names_a, names_b, compare="ratio")
    r = svc.edit(loose.session_id, [Edit("reject", 0, 0)])
    assert _pairs(r.assignment)[0] == 2  # without a floor: pushed onto "Zoe Bernard"

    v = svc.open_session(names_a, names_b, compare="ratio", floor=0.5)
    assert v.floor == 0.5
    assert {c.candidate for c in svc.candidates(v.session_id, 0)} == {0}  # only Anne Li
    r = svc.edit(v.session_id, [Edit("reject", 0, 0)])
    assert 0 not in _pairs(r.assignment)  # left unmatched instead
    assert {row.item: row.status for row in svc.review(v.session_id)}[0] == "unmatched"


def test_floor_is_a_ceiling_for_a_distance():
    svc = MatchService()
    v = svc.open_session(["abc", "xyz"], ["abd", "qqqq"], compare="levenshtein_distance", floor=1)
    assert v.sense == "minimize"
    assert _pairs(v.assignment) == {0: 0}  # xyz->qqqq (distance 4) is past the ceiling


def test_floor_with_blocking_and_unbounded_scores():
    blocker = lambda A, B: [(0, 0), (0, 1), (1, 1)]  # noqa: E731
    big = lambda a, b: 100.0 * len(set(a) & set(b))  # noqa: E731
    svc = MatchService()
    v = svc.open_session(["ab", "cd"], ["ab", "cx"], compare=big, block=blocker, floor=150)
    assert _pairs(v.assignment) == {0: 0}  # (1, 1) scores 100: below the floor
    assert svc.candidates(v.session_id, 1) == []


def test_floor_is_validated():
    for bad in ("high", float("nan"), True, 10**400, [1]):
        with pytest.raises(ValueError):
            MatchService().open_session(A, B, floor=bad)


def test_http_session_floor(client):
    v = client.post("/sessions", json={"A": A, "B": B, "floor": 0.99}).json()
    assert v["floor"] == 0.99 and v["assignment"] == []


def test_floor_accepts_numpy_scalars():
    np = pytest.importorskip("numpy")
    assert MatchService().open_session(A, B, floor=np.float32(0.25)).floor == 0.25


def test_floor_makes_a_borderline_item_uncertain_not_certain():
    # a -> x 0.51, a -> y 0.49; b -> y 0.95, b -> x 0.60 (exact scores, via a callable)
    S = {("a", "x"): 0.51, ("a", "y"): 0.49, ("b", "y"): 0.95, ("b", "x"): 0.60}
    svc = MatchService()
    v = svc.open_session(["a", "b"], ["x", "y"], compare=lambda p, q: S[(p, q)], floor=0.5)
    first = v.review[0]
    assert first.item == 0 and first.margin == pytest.approx(0.01)  # 0.51 vs the floor
    assert svc.next_query(v.session_id).item == 0


def test_floor_assignment_is_lexicographic_unlike_match_threshold():
    """Pinned (equate#21): a floor makes the cells past it non-candidates *before*
    solving, so the solve maximizes the number of acceptable pairs first; match's
    threshold solves first and then drops weak pairs."""
    # best total: a-y + b-x = 1.25 > a-x + b-y = 1.06; a-y is below the floor
    S = {("a", "x"): 0.51, ("a", "y"): 0.30, ("b", "x"): 0.95, ("b", "y"): 0.55}
    cmp = lambda p, q: S[(p, q)]  # noqa: E731
    svc = MatchService(cache=None)
    filtered = svc.match(["a", "b"], ["x", "y"], compare=cmp, threshold=0.5)
    assert _pairs(filtered.pairs) == {1: 0}  # best total first (a-y, b-x), then a-y dropped
    v = svc.open_session(["a", "b"], ["x", "y"], compare=cmp, floor=0.5)
    assert _pairs(v.assignment) == {0: 0, 1: 1}  # both acceptable pairs kept
