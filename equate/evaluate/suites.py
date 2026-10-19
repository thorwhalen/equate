"""Loaders for the standard matching benchmarks, fetched on first use into a store.

Registered in :data:`~equate.evaluate.benchmarks` (research doc 14, §6 and §8.3):

- ``abt_buy``, ``amazon_google``, ``dblp_acm`` — the product and bibliographic tables the
  Magellan / DeepMatcher suite made standard. ``source='leipzig'`` (default) loads the
  original tables of the Leipzig database group with their **complete** gold mapping
  (CC BY 4.0). ``source='deepmatcher'`` loads DeepMatcher's preprocessed tables with its
  canonical labelled train/valid/test pairs in ``Benchmark.pairs``; its gold is the
  labelled positives only, which misses true matches its blocking dropped (``meta``
  says how many), so it is the source for comparing pair classifiers with published
  numbers, not for scoring whole-table matching.
- ``wdc_products`` — WDC Products (Peeters, Der & Bizer 2024), the difficulty-controlled
  product benchmark: ``corner_cases`` (80/50/20 %), ``unseen`` (0/50/100 % of test
  entities absent from training) and ``dev_size`` (small/medium/large). Built from its
  multi-class files, with its train/valid/test sets as ``Benchmark.splits``.

Every loader takes ``store=`` (any ``MutableMapping``; default
:func:`~equate.evaluate.store.default_store`), ``download=`` and ``get=`` (the
downloader), checks each download against a pinned SHA-256, and fills ``meta`` with
``source``, ``license``, a ``difficulty`` sentence and a measured ``difficulty_profile``
(:func:`~equate.evaluate.difficulty.difficulty_profile`, cached in the store).
"""

import csv
import gzip
import io
import json
import warnings
import zipfile
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Callable, Optional

from equate.evaluate.benchmark import Benchmark, benchmarks, subset_pairs
from equate.evaluate.difficulty import difficulty_profile
from equate.evaluate.store import Source, default_store, fetch

__all__ = ["abt_buy", "amazon_google", "dblp_acm", "wdc_products", "MAGELLAN_SOURCES"]

_LEIPZIG = "https://dbs.uni-leipzig.de/files/datasets"
_DEEPMATCHER = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data"
_WDC = "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products"

LEIPZIG_LICENSE = (
    "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) — credit the Leipzig "
    "database group (https://dbs.uni-leipzig.de/research/projects/"
    "benchmark-datasets-for-entity-resolution) and Köpcke, Thor & Rahm, VLDB 2010"
)
DEEPMATCHER_LICENSE = (
    "no separate license stated by DeepMatcher; the tables are preprocessed from the "
    "Leipzig originals (" + LEIPZIG_LICENSE + "); cite Mudgal et al., SIGMOD 2018"
)
WDC_LICENSE = (
    "not stated by the publisher (WDC Products page, checked 2026-10-19); the offers "
    "were extracted from Common Crawl — check the terms before any commercial use; "
    "cite Peeters, Der & Bizer, EDBT 2024"
)

# What the literature reports, to read next to a score (research doc 14, §6 and §7.6).
# The numbers are pair-classification F1 on DeepMatcher's canonical test pairs.
_MAGELLAN = {
    "abt_buy": dict(
        title="Abt-Buy",
        leipzig=(
            "Abt-Buy.zip",
            "b9ff2937d97371b7a00b0a97033213bfbdee24e8f1a0718a41ea8364472a8ff8",
            ("Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv"),
        ),
        deepmatcher="Textual/Abt-Buy",
        sha_dm={
            "tableA.csv": "20b79995258c4ef80bb96b731762a6fe869650cc42dcb2c58994bda91b0d81aa",
            "tableB.csv": "e6ebe54c98fd69c5ad4b8364dde53c323ae872c49cf3c2ad5f7b9acd246e1eb6",
            "train.csv": "efa7ee36e35915aa2e5c8218bb90a94ff11a6cc7a9c4a45cefff1bfc16d64061",
            "valid.csv": "25ac7b6035327c82a9e45c93e02e24dd2fffe7afff779ae9381e5c32f727c3c0",
            "test.csv": "fd815442d37368cecc4cd3b5a0af33930ce5b7b353b222b810d8079211e81d9a",
        },
        difficulty=(
            "hard for attribute matchers: long free-text product descriptions; "
            "DeepMatcher reported F1 ~63 and Ditto ~89 on its test pairs"
        ),
    ),
    "amazon_google": dict(
        title="Amazon-Google",
        leipzig=(
            "Amazon-GoogleProducts.zip",
            "fcd72e223d51ce5b0d6b1d68034a438ba95cf2940431e209a57910b067cc4811",
            (
                "Amazon.csv",
                "GoogleProducts.csv",
                "Amzon_GoogleProducts_perfectMapping.csv",
            ),
        ),
        deepmatcher="Structured/Amazon-Google",
        sha_dm={
            "tableA.csv": "04a797511d83af9e27469778081c50397857b199d49e2f83c119390f14301947",
            "tableB.csv": "00c0d19c0dcd014b00d6d341e15e2b3b3caa616b4f6d710d0c9cd2dfc3f4717f",
            "train.csv": "cdc5e88d714e77febc26a81a61c3295a0000baa51d6debf93043f0cbd5193b9f",
            "valid.csv": "4999067ea47fa2ae03e0c4a68fee60cf68728943208fcde46415cb895844dd22",
            "test.csv": "09e376c0c5391bd4a6c6d8efde148c994441b6043ef6b7f9eab2a8772a6fb362",
        },
        difficulty=(
            "hard: software products with near-duplicate editions and sparse "
            "manufacturer fields; DeepMatcher reported F1 ~69 and Ditto ~76 on its test pairs"
        ),
    ),
    "dblp_acm": dict(
        title="DBLP-ACM",
        leipzig=(
            "DBLP-ACM.zip",
            "e37e7ed8e06722499e6d9e94583fdec279207b13ce6270a37595b6a5d594bc40",
            ("DBLP2.csv", "ACM.csv", "DBLP-ACM_perfectMapping.csv"),
        ),
        deepmatcher="Structured/DBLP-ACM",
        sha_dm={
            "tableA.csv": "a83dfac196a4e263f3adac7aaf095c7198254a98fcaed0ec68d59130c74c43a7",
            "tableB.csv": "bd103ffdccdff4d8b9d04c18d90d04110b70b6fc87dec83c4e52e9616c58431a",
            "train.csv": "ad94b36b178bbf76023d3cee689565fbda1fe01b19d9a3926a51db382f45f0a5",
            "valid.csv": "862f848ed3f3f005ae6c8997ecf571984bd575f6bbd319fc1a2170830a91132b",
            "test.csv": "e49adc4590d24c18b1a9bbd96011d9c745e10432e10e93e050d856a206fac394",
        },
        difficulty=(
            "easy, effectively solved: clean bibliographic records; published matchers "
            "reach F1 ~98-99 — a sanity check, not evidence a matcher generalizes"
        ),
    ),
}

#: the selectable sources of the three Magellan/DeepMatcher tables
MAGELLAN_SOURCES = ("leipzig", "deepmatcher")

_WDC_SHA = {
    80: "5385e2491e82c7f6c208d8eac7ec777a4a44180769f3a339c99803ff2b068981",
    50: "fed08b5543e746e19df545903b4dd573755ad580f8363ca9ba7954aedc69539f",
    20: "e30a9d3086ea4a16decec74e29746c98fec203986096748c7e1a055ab3324450",
}
_WDC_FIELDS = ("brand", "title", "description", "price", "priceCurrency")
WDC_CORNER_CASES = (80, 50, 20)
WDC_UNSEEN = (0, 50, 100)
WDC_DEV_SIZES = ("small", "medium", "large")


# --- parsing helpers -----------------------------------------------------------------


def _decode(data: bytes) -> str:
    # the Leipzig files mix UTF-8 and Windows-1252 (e.g. a raw 0xAE for "®")
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _rows(data: bytes) -> list:
    return list(csv.DictReader(io.StringIO(_decode(data))))


def _clean(value):
    """Missing values (``''``, the string ``'None'``) become ``None``."""
    return None if value in ("", "None", None) else value


def _record(row: dict, drop=("id",)) -> dict:
    return {k: _clean(v) for k, v in row.items() if k not in drop}


def _check_choice(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _profile(store, key, a, b, gold, *, compute):
    """The difficulty profile, read from the store when already measured."""
    if not compute:
        return None
    if key in store:
        return json.loads(store[key])
    profile = difficulty_profile(a, b, gold)
    store[key] = json.dumps(profile).encode()
    return profile


def _difficulty_sentence(base: str, profile: Optional[dict]) -> str:
    if not profile:
        return base
    return (
        f"{base}. Measured: linearity {profile['linearity']:.2f} "
        f"({profile['level']}; best F1 of one TF-IDF cosine threshold)"
    )


# --- Magellan / DeepMatcher tables ----------------------------------------------------


def _magellan(
    name,
    *,
    source="leipzig",
    store: Optional[MutableMapping] = None,
    download: bool = True,
    get: Optional[Callable] = None,
    profile: bool = True,
) -> Benchmark:
    _check_choice("source", source, MAGELLAN_SOURCES)
    spec = _MAGELLAN[name]
    store = default_store() if store is None else store
    kw = dict(store=store, download=download, get=get)
    if source == "leipzig":
        zip_name, sha, (file_a, file_b, file_map) = spec["leipzig"]
        archive = zipfile.ZipFile(
            io.BytesIO(
                fetch(f"{name}/{zip_name}", Source(f"{_LEIPZIG}/{zip_name}", sha), **kw)
            )
        )
        rows_a, rows_b = _rows(archive.read(file_a)), _rows(archive.read(file_b))
        mapping = _rows(archive.read(file_map))
        index_a = {r["id"]: i for i, r in enumerate(rows_a)}
        index_b = {r["id"]: j for j, r in enumerate(rows_b)}
        key_a, key_b = list(mapping[0])
        gold = frozenset((index_a[m[key_a]], index_b[m[key_b]]) for m in mapping)
        pairs = None
        meta = {
            "source": f"{_LEIPZIG}/{zip_name} (Köpcke, Thor & Rahm, VLDB 2010)",
            "license": LEIPZIG_LICENSE,
            "gold": "complete: the publisher's perfect mapping",
        }
    else:
        base = f"{_DEEPMATCHER}/{spec['deepmatcher']}/exp_data"
        files = {
            f: fetch(f"{name}/deepmatcher/{f}", Source(f"{base}/{f}", sha), **kw)
            for f, sha in spec["sha_dm"].items()
        }
        rows_a, rows_b = _rows(files["tableA.csv"]), _rows(files["tableB.csv"])
        index_a = {r["id"]: i for i, r in enumerate(rows_a)}
        index_b = {r["id"]: j for j, r in enumerate(rows_b)}
        pairs = {
            part: tuple(
                (index_a[r["ltable_id"]], index_b[r["rtable_id"]], int(r["label"]))
                for r in _rows(files[f"{fname}.csv"])
            )
            for part, fname in (("train", "train"), ("val", "valid"), ("test", "test"))
        }
        gold = frozenset((i, j) for part in pairs.values() for i, j, y in part if y)
        meta = {
            "source": f"{base}/ (Mudgal et al., SIGMOD 2018)",
            "license": DEEPMATCHER_LICENSE,
            "gold": (
                f"incomplete: the {len(gold)} labelled positives only — true matches "
                "DeepMatcher's blocking dropped are absent, so whole-table precision is "
                "understated; use source='leipzig' for that, and Benchmark.pairs for "
                "the canonical pair-classification task"
            ),
        }
    a = [_record(r) for r in rows_a]
    b = [_record(r) for r in rows_b]
    prof = _profile(store, f"{name}/profile-{source}.json", a, b, gold, compute=profile)
    meta.update(
        {
            "title": spec["title"],
            "record_ids": {
                "a": [r["id"] for r in rows_a],
                "b": [r["id"] for r in rows_b],
            },
            "difficulty": _difficulty_sentence(spec["difficulty"], prof),
            "difficulty_profile": prof,
        }
    )
    return Benchmark(name, a, b, gold, meta=meta, pairs=pairs)


def abt_buy(**config) -> Benchmark:
    """Abt-Buy: 1081 × 1092 product offers, 1097 matches (Leipzig; textual, hard)."""
    return _magellan("abt_buy", **config)


def amazon_google(**config) -> Benchmark:
    """Amazon-Google: 1363 × 3226 software products, 1300 matches (Leipzig; hard)."""
    return _magellan("amazon_google", **config)


def dblp_acm(**config) -> Benchmark:
    """DBLP-ACM: 2616 × 2294 publications, 2224 matches (Leipzig; easy)."""
    return _magellan("dblp_acm", **config)


for _name, _fn in (
    ("abt_buy", abt_buy),
    ("amazon_google", amazon_google),
    ("dblp_acm", dblp_acm),
):
    benchmarks.register(_name, _fn)


# --- WDC Products ----------------------------------------------------------------------


def _wdc_part(offers):
    """Split one part's offers into two tables: within each entity (cluster), offers
    alternate between ``a`` and ``b`` in id order — a two-offer cluster (every test set)
    becomes one ``a``/``b`` match."""
    by_cluster = defaultdict(list)
    for o in offers:
        by_cluster[o["cluster_id"]].append(o)
    a, b = [], []
    for cluster in sorted(by_cluster):
        for k, o in enumerate(sorted(by_cluster[cluster], key=lambda o: o["id"])):
            (a if k % 2 == 0 else b).append(o)
    return a, b


def wdc_products(
    *,
    corner_cases: int = 80,
    unseen: int = 100,
    dev_size: str = "large",
    store: Optional[MutableMapping] = None,
    download: bool = True,
    get: Optional[Callable] = None,
    profile: bool = True,
) -> Benchmark:
    """WDC Products as a two-table benchmark with its canonical train/valid/test split.

    The defaults pick the hardest generalization variant — 80 % corner cases, every test
    entity unseen in training (research doc 14, §7.6) — with the large development set.
    Gold pairs every ``a``/``b`` offer of the same product. ``meta['leakage']`` counts test
    offers that also appear (identically) in train or valid: WDC's own files have such
    overlap in the 50 %-unseen variants, and a warning says so.
    """
    _check_choice("corner_cases", corner_cases, WDC_CORNER_CASES)
    _check_choice("unseen", unseen, WDC_UNSEEN)
    _check_choice("dev_size", dev_size, WDC_DEV_SIZES)
    store = default_store() if store is None else store
    zip_name = f"{corner_cases}multi.zip"
    archive = zipfile.ZipFile(
        io.BytesIO(
            fetch(
                f"wdc_products/{zip_name}",
                Source(f"{_WDC}/{zip_name}", _WDC_SHA[corner_cases]),
                store=store,
                download=download,
                get=get,
            )
        )
    )
    stem = f"wdcproductsmulti{corner_cases}cc{100 - corner_cases}rnd"

    def offers(suffix):
        raw = gzip.decompress(archive.read(f"{stem}{suffix}.json.gz")).decode("utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    parts = {
        "train": offers(f"000un_train_{dev_size}"),
        "val": offers(f"000un_valid_{dev_size}"),
        "test": offers(f"{unseen:03d}un_gs"),
    }
    a, b, splits, cluster_a, cluster_b, ids = [], [], {}, [], [], {"a": [], "b": []}
    for part, part_offers in parts.items():
        pa, pb = _wdc_part(part_offers)
        splits[part] = (
            tuple(range(len(a), len(a) + len(pa))),
            tuple(range(len(b), len(b) + len(pb))),
        )
        for side, rows, clusters, side_ids in (
            (a, pa, cluster_a, ids["a"]),
            (b, pb, cluster_b, ids["b"]),
        ):
            side.extend({f: _clean(o.get(f)) for f in _WDC_FIELDS} for o in rows)
            clusters.extend(o["cluster_id"] for o in rows)
            side_ids.extend(o["id"] for o in rows)
    rows_of = defaultdict(list)
    for j, c in enumerate(cluster_b):
        rows_of[c].append(j)
    gold = frozenset((i, j) for i, c in enumerate(cluster_a) for j in rows_of[c])

    test_ids = {o["id"] for o in parts["test"]}
    leakage = {
        "train_test": len(test_ids & {o["id"] for o in parts["train"]}),
        "val_test": len(test_ids & {o["id"] for o in parts["val"]}),
    }
    if any(leakage.values()):
        warnings.warn(
            f"WDC Products {corner_cases}% corner cases / {unseen}% unseen: "
            f"{leakage['train_test']} test offers also appear in train and "
            f"{leakage['val_test']} in valid (identical records, in the publisher's own "
            "files) — a trained matcher can memorise them"
        )
    test_a, test_b = splits["test"]
    sub = subset_pairs(gold, test_a, test_b)
    prof = _profile(
        store,
        f"wdc_products/profile-{corner_cases}cc-{unseen}un.json",
        [a[i] for i in test_a],
        [b[j] for j in test_b],
        sub,
        compute=profile,
    )
    if prof:
        prof.update(corner_cases=corner_cases, unseen=unseen, dev_size=dev_size)
    base = (
        f"controlled: {corner_cases}% corner-case pairs, {unseen}% of test products unseen "
        f"in training, {dev_size} development set (profile measured on the test split)"
    )
    meta = {
        "title": "WDC Products",
        "source": f"{_WDC}/{zip_name} (Peeters, Der & Bizer, EDBT 2024)",
        "license": WDC_LICENSE,
        "gold": "complete within the offers: every a/b pair of the same product",
        "variant": {
            "corner_cases": corner_cases,
            "unseen": unseen,
            "dev_size": dev_size,
        },
        "record_ids": ids,
        "leakage": leakage,
        "difficulty": _difficulty_sentence(base, prof),
        "difficulty_profile": prof,
    }
    name = f"wdc_products_{corner_cases}cc_{unseen}un_{dev_size}"
    return Benchmark(name, a, b, gold, meta=meta, splits=splits)


benchmarks.register("wdc_products", wdc_products)
