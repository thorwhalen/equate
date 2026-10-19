"""Tests for the benchmark store, the standard-suite loaders and the difficulty profile.

Everything runs offline: each loader is fed tiny fixtures in its publisher's real format
(written into a dict store under the keys the loader reads). One opt-in test
(``EQUATE_NETWORK_TESTS=1``) downloads the real files and checks the pinned hashes.
"""

import gzip
import hashlib
import io
import json
import os
import zipfile

import pytest

from equate import match
from equate.evaluate import (
    Benchmark,
    FileStore,
    Source,
    benchmark_matcher,
    data_dir,
    fetch,
    linearity,
    list_benchmarks,
    load_benchmark,
)
from equate.featurize import record_text

# --- store + fetch -----------------------------------------------------------------


def test_file_store_roundtrip_and_containment(tmp_path):
    s = FileStore(tmp_path)
    s["suite/a.csv"] = b"x"
    assert list(s) == ["suite/a.csv"] and s["suite/a.csv"] == b"x" and len(s) == 1
    assert "suite/a.csv" in s and "nope" not in s
    assert not list(tmp_path.rglob("*.part"))  # atomic write leaves no temp file
    for bad in ("../escape", "/abs/path", "", ".", "a//b", "..\\..\\evil", "c:/x", "f.part"):
        with pytest.raises(KeyError):
            s[bad] = b"x"
    del s["suite/a.csv"]
    assert list(s) == []


def test_fetch_caches_and_verifies():
    data = b"payload"
    good = Source("https://example.invalid/f", hashlib.sha256(data).hexdigest())
    calls = []
    get = lambda url: calls.append(url) or data  # noqa: E731
    store = {}
    assert fetch("k", good, store=store, get=get) == data
    assert fetch("k", good, store=store, get=get) == data
    assert calls == [good.url] and store["k"] == data  # downloaded once, then cached

    bad = Source("https://example.invalid/g", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        fetch("g", bad, store=store, get=lambda url: data)
    assert "g" not in store  # nothing cached on a mismatch
    with pytest.raises(FileNotFoundError):
        fetch("h", good, store=store, download=False, get=get)


def test_data_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EQUATE_DATA_DIR", str(tmp_path / "d"))
    assert data_dir() == tmp_path / "d"
    monkeypatch.delenv("EQUATE_DATA_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "x"))
    assert data_dir() == tmp_path / "x" / "equate" / "benchmarks"


# --- fixtures in the publishers' formats ---------------------------------------------


def _csv(rows, encoding="utf-8"):
    header = list(rows[0])
    lines = [",".join(f'"{h}"' for h in header)]
    lines += [",".join(f'"{r[h]}"' if r[h] != "" else "" for h in header) for r in rows]
    return ("\n".join(lines) + "\n").encode(encoding)


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


ABT = [
    {"id": "552", "name": "Sony Turntable PSLX350H", "description": "belt drive", "price": ""},
    {"id": "580", "name": "Bose Acoustimass 5 ® speaker", "description": "", "price": "399"},
    {"id": "604", "name": "Linksys EtherFast switch", "description": "8 port", "price": "29"},
]
BUY = [
    {"id": "10011646", "name": "Linksys EtherFast 8-port switch", "description": "", "manufacturer": "LINKSYS", "price": ""},
    {"id": "10140760", "name": "Sony PSLX350H turntable", "description": "", "manufacturer": "Sony", "price": ""},
    {"id": "10221960", "name": "Bose Acoustimass 5 speakers", "description": "", "manufacturer": "Bose", "price": ""},
]
MAP = [
    {"idAbt": "552", "idBuy": "10140760"},
    {"idAbt": "580", "idBuy": "10221960"},
    {"idAbt": "604", "idBuy": "10011646"},
]


def leipzig_store():
    archive = _zip(
        {
            "Abt.csv": _csv(ABT, "cp1252"),  # the real Abt.csv is Windows-1252
            "Buy.csv": _csv(BUY),
            "abt_buy_perfectMapping.csv": _csv(MAP),
        }
    )
    return {"abt_buy/Abt-Buy.zip": archive}


def deepmatcher_store():
    table = lambda rows: _csv([{"id": str(i), **{k: v for k, v in r.items() if k != "id"}} for i, r in enumerate(rows)])  # noqa: E731
    labelled = lambda rows: _csv([{"ltable_id": str(i), "rtable_id": str(j), "label": str(y)} for i, j, y in rows])  # noqa: E731
    return {
        "abt_buy/deepmatcher/tableA.csv": table(ABT),
        "abt_buy/deepmatcher/tableB.csv": table(BUY),
        "abt_buy/deepmatcher/train.csv": labelled([(0, 1, 1), (0, 0, 0)]),
        "abt_buy/deepmatcher/valid.csv": labelled([(1, 2, 1)]),
        "abt_buy/deepmatcher/test.csv": labelled([(2, 1, 0)]),  # (2, 0) is a match nobody labelled
    }


def _offer(id_, cluster, title):
    return {"id": id_, "brand": "None", "title": title, "description": "None", "price": "9", "priceCurrency": "USD", "cluster_id": cluster, "label": cluster, "unseen": "True"}


def wdc_store(*, leak=False, n_test_clusters=2):
    train = [_offer("1", "c1", "red shoe"), _offer("2", "c1", "red shoes"), _offer("3", "c2", "blue hat"), _offer("4", "c2", "hat blue")]
    valid = [_offer("5", "c3", "green cup"), _offer("6", "c3", "cup green")]
    test = [_offer("7", "c4", "black phone x1"), _offer("8", "c4", "phone x1 black"), _offer("9", "c5", "white phone x2"), _offer("10", "c5", "phone x2 white")]
    for k in range(2, n_test_clusters):
        test += [_offer(f"t{k}a", f"z{k}", f"model {k} gadget"), _offer(f"t{k}b", f"z{k}", f"gadget model {k}")]
    if leak:
        test[0] = dict(train[0], cluster_id="c4")  # a test offer that is also a train offer
        test[0]["id"] = "1"
    gz = lambda offers: gzip.compress("\n".join(json.dumps(o) for o in offers).encode())  # noqa: E731
    stem = "wdcproductsmulti80cc20rnd"
    files = {f"{stem}000un_train_large.json.gz": gz(train), f"{stem}000un_valid_large.json.gz": gz(valid)}
    for un in ("000", "050", "100"):
        files[f"{stem}{un}un_gs.json.gz"] = gz(test)
    return {"wdc_products/80multi.zip": _zip(files)}


# --- loaders ---------------------------------------------------------------------------


def test_registry_lists_the_standard_suites():
    assert {"abt_buy", "amazon_google", "dblp_acm", "wdc_products", "toy_names"} <= set(list_benchmarks())


def test_leipzig_loader_complete_gold_records_and_meta():
    store = leipzig_store()
    ds = load_benchmark("abt_buy", store=store, download=False, verify=False)
    assert (len(ds.a), len(ds.b)) == (3, 3)
    assert ds.gold == frozenset({(0, 1), (1, 2), (2, 0)})
    assert "id" not in ds.a[0] and ds.a[0]["price"] is None  # ids out of the record; '' -> None
    assert "®" in ds.a[1]["name"]  # decoded from Windows-1252
    assert ds.ids["a"] == ["552", "580", "604"] and "record_ids" not in ds.meta
    assert "CC BY 4.0" in ds.meta["license"] and "complete" in ds.meta["gold"]
    prof = ds.meta["difficulty_profile"]
    assert prof["n_gold"] == 3 and 0 <= prof["linearity"] <= 1 and prof["level"] in ("easy", "moderate", "hard")
    assert "linearity" in ds.meta["difficulty"]
    assert any(k.startswith("abt_buy/profile-v") for k in store)  # measured once, cached
    assert ds.pairs is None and ds.splits is None


def test_deepmatcher_source_keeps_canonical_pairs_and_flags_incomplete_gold():
    ds = load_benchmark("abt_buy", source="deepmatcher", store=deepmatcher_store(), download=False, verify=False)
    assert ds.name == "abt_buy_deepmatcher"  # never merged with the Leipzig rows
    assert ds.pairs["train"] == ((0, 1, 1), (0, 0, 0)) and ds.pairs["test"] == ((2, 1, 0),)
    assert ds.gold == frozenset({(0, 1), (1, 2)})  # labelled positives only
    assert ds.meta["gold"].startswith("incomplete")


def test_unknown_source_or_variant_is_an_error():
    with pytest.raises(ValueError):
        load_benchmark("abt_buy", source="nope", store={}, download=False)
    with pytest.raises(ValueError):
        load_benchmark("wdc_products", unseen=30, store={}, download=False)
    with pytest.raises(FileNotFoundError):  # not cached, and downloads are off
        load_benchmark("dblp_acm", store={}, download=False)


def test_wdc_loader_canonical_split_and_gold():
    ds = load_benchmark("wdc_products", store=wdc_store(), download=False, verify=False)
    assert ds.name == "wdc_products_80cc_100un_large"
    assert {k: (len(a), len(b)) for k, (a, b) in ds.splits.items()} == {"train": (2, 2), "val": (1, 1), "test": (2, 2)}
    test_a, test_b = ds.splits["test"]
    test_gold = {(i, j) for i, j in ds.gold if i in test_a and j in test_b}
    assert len(test_gold) == 2  # each two-offer test cluster is one a/b match
    assert all(ds.a[i]["title"].split()[1] in ds.b[j]["title"] for i, j in test_gold)
    assert all(set(r) == {"brand", "title", "description", "price", "priceCurrency"} for r in ds.a)
    assert ds.a[0]["brand"] is None  # the string 'None' is missing
    assert ds.meta["leakage"] == {"train_test": 0, "val_test": 0}
    assert ds.meta["variant"] == {"corner_cases": 80, "unseen": 100, "dev_size": "large"}


def test_wdc_loader_reports_publisher_leakage():
    with pytest.warns(UserWarning, match="also appear in train"):
        ds = load_benchmark("wdc_products", unseen=50, store=wdc_store(leak=True), download=False, verify=False)
    assert ds.meta["leakage"]["train_test"] == 1


# --- harness: canonical vs entity split ------------------------------------------------


def test_harness_uses_the_canonical_split_by_default():
    ds = load_benchmark("wdc_products", store=wdc_store(), download=False, verify=False)
    seen = []

    def matcher(a, b):
        seen.append((len(a), len(b)))
        return match(a, b)

    report = benchmark_matcher(matcher, on=[ds], n_seeds=5)
    assert seen == [(2, 2)]  # one run, on the published test set
    assert len(report.rows) == 1 and report.meta[ds.name]["split"] == "canonical"
    assert report.rows[0].f1 == 1.0


def test_harness_split_modes():
    wdc = load_benchmark("wdc_products", store=wdc_store(), download=False, verify=False)
    toy = load_benchmark("toy_names", n_entities=40)
    with pytest.raises(ValueError, match="no canonical split"):
        benchmark_matcher(on=[toy], split="canonical")
    with pytest.raises(ValueError, match="split must be"):
        benchmark_matcher(on=[toy], split="random")
    assert benchmark_matcher(on=[toy], n_seeds=2).meta["toy_names"]["split"] == "entity"
    report = benchmark_matcher(on=[wdc], n_seeds=1, split="entity", ratios=(0, 0, 1))
    assert report.meta[wdc.name]["split"] == "entity"


def test_canonical_fit_gets_train_and_val_parts():
    ds = load_benchmark("wdc_products", store=wdc_store(), download=False, verify=False)
    got = {}

    def fit(tr_a, tr_b, tr_pairs, va_a, va_b, va_pairs):
        got.update(train=(len(tr_a), len(tr_b), len(tr_pairs)), val=(len(va_a), len(va_b), len(va_pairs)))
        return match

    benchmark_matcher(on=[ds], fit=fit)
    assert got == {"train": (2, 2, 2), "val": (1, 1, 1)}


# --- difficulty + records --------------------------------------------------------------


def test_linearity_bounds_and_ties():
    assert linearity(["apple pie", "kiwi"], ["kiwi", "apple pie"], {(0, 1), (1, 0)}) == 1.0
    assert linearity(["a"], ["b"], set()) == 0.0
    # identical texts tie at one score: the only cut keeps all 4 pairs -> F1 = 2*2/(4+2)
    assert linearity(["x", "x"], ["x", "x"], {(0, 0), (1, 1)}) == pytest.approx(2 / 3)


def test_match_on_records_uses_their_text():
    a = [{"name": "Jon Smith", "city": "Paris"}, {"name": "Kate Doe", "city": None}]
    b = [{"name": "Kate Do"}, {"name": "John Smith", "city": "paris"}]
    assert dict(match(a, b)) == {0: 1, 1: 0}
    assert record_text({"a": "x", "b": None, "c": 3}) == "x 3"


def test_benchmark_container_stays_positional():
    ds = Benchmark("x", ["a"], ["a"], frozenset({(0, 0)}), {"source": "s"})
    assert ds.splits is None and ds.pairs is None and ds.base_rate == 1.0


# --- the real files (opt-in) -----------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("EQUATE_NETWORK_TESTS"),
    reason="downloads the real benchmark files; set EQUATE_NETWORK_TESTS=1",
)
def test_real_downloads_match_pinned_hashes(tmp_path):
    store = FileStore(tmp_path)
    expected = {"abt_buy": (1081, 1092, 1097), "amazon_google": (1363, 3226, 1300), "dblp_acm": (2616, 2294, 2224)}
    for name, sizes in expected.items():
        ds = load_benchmark(name, store=store, profile=False)
        assert (len(ds.a), len(ds.b), len(ds.gold)) == sizes
        dm = load_benchmark(name, source="deepmatcher", store=store, profile=False)
        assert len(dm.gold) < len(ds.gold)  # DeepMatcher's labelled positives are a subset
    wdc = load_benchmark("wdc_products", store=store, profile=False)
    assert {k: len(v[0]) for k, v in wdc.splits.items()}["test"] == 500


# --- regressions from the adversarial review of #15 ------------------------------------


def test_wdc_test_gold_is_not_the_diagonal():
    ds = load_benchmark("wdc_products", store=wdc_store(n_test_clusters=30), download=False, verify=False)
    identity = lambda a, b: [(i, i) for i in range(min(len(a), len(b)))]  # noqa: E731
    assert benchmark_matcher(identity, on=[ds]).rows[0].f1 < 0.3


def test_two_benchmarks_with_one_name_are_refused():
    ds = load_benchmark("toy_names", n_entities=20)
    with pytest.raises(ValueError, match="named 'toy_names'"):
        benchmark_matcher(on=[ds, ds], n_seeds=1)


def test_tfidf_refuses_non_text_instead_of_matching_missing_values():
    from equate import score_matrix

    with pytest.raises(TypeError, match="text or records"):
        score_matrix([None, "apple"], [None, "banana"])
    with pytest.raises(TypeError):
        score_matrix([["jon", "smith"]], [["jon", "smith"]])


def test_record_text_tolerates_array_values():
    np = pytest.importorskip("numpy")
    assert record_text({"v": np.array([1, 2]), "w": ""}) == "[1 2]"


def test_fetch_reverifies_a_cached_copy():
    data = b"payload"
    src = Source("https://example.invalid/f", hashlib.sha256(data).hexdigest())
    store = {"k": b"corrupt"}
    assert fetch("k", src, store=store, get=lambda url: data) == data  # re-fetched
    assert store["k"] == data
    store["k"] = b"corrupt"
    with pytest.raises(FileNotFoundError):  # stale and offline: refuse, never serve it
        fetch("k", src, store=store, download=False)


def test_file_store_concurrent_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    s = FileStore(tmp_path)
    blobs = [bytes([k]) * 200_000 for k in range(8)]

    def write(k):
        s["suite/big.bin"] = blobs[k]

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(write, range(8)))
    assert s["suite/big.bin"] in blobs  # one complete write, never a torn mix
    assert list(s) == ["suite/big.bin"] and not list(tmp_path.rglob("*.part"))


def test_linearity_refuses_huge_inputs():
    with pytest.raises(ValueError, match="max_pairs"):
        linearity(["a", "b"], ["a", "b"], {(0, 0)}, max_pairs=3)


def test_single_run_report_prints_no_fake_spread():
    ds = load_benchmark("wdc_products", store=wdc_store(), download=False, verify=False)
    table = benchmark_matcher(match, on=[ds]).to_markdown()
    assert "±" not in table.splitlines()[-1]
