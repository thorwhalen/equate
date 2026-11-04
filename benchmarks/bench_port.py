#!/usr/bin/env python3
"""Benchmark the ``equate.compare.sequence`` backends end to end (equate#35).

Times the public functions (validation, cost matrix, kernels and the boundary
included) on deterministic inputs at the sizes the port assessment named, for every
requested backend, side by side in one process: ratios from one run are valid on a
noisy VM, absolute times from two machines are not comparable. Stdlib + numpy only::

    python benchmarks/bench_port.py                    # numpy, and rust if installed
    python benchmarks/bench_port.py --quick            # smallest size per case, 3 repeats
    python benchmarks/bench_port.py --backends numpy dtaidistance --json out.json

Each cell is the median of ``--repeat`` (default 7) timing runs, each auto-ranged to
at least 0.2 s, with the min-max range as the spread. It refuses to time a debug build
of ``equate_rust`` (``equate_rust.DEBUG_BUILD``): build it with ``--release`` first.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import timeit
from statistics import median

import numpy as np

from equate._dependencies import have
from equate.compare import sequence as Q

SEED = 0
_PACKAGE = {"rust": "equate_rust", "dtaidistance": "dtaidistance", "biopython": "Bio"}


def _series(n, rng):
    return np.cumsum(rng.normal(size=n)).tolist()  # random walks: realistic DTW inputs


def _dna(n, rng):
    return "".join(rng.choice(list("ACGT"), size=n))


# (label, function, factory(size, rng) -> (args, kwargs), sizes, backends that apply)
CASES = [
    (
        "dtw unconstrained",
        Q.dtw_alignment,
        lambda n, rng: ((_series(n, rng), _series(n, rng)), {"window": None}),
        [100, 1_000],
        ("numpy", "rust", "dtaidistance"),
    ),
    (
        "dtw 10% band",
        Q.dtw_alignment,
        lambda n, rng: ((_series(n, rng), _series(n, rng)), {"window": 0.1}),
        [100, 1_000],
        ("numpy", "rust", "dtaidistance"),
    ),
    (
        "subsequence_dtw",
        Q.subsequence_dtw_alignment,
        lambda n, rng: ((_series(n // 10, rng), _series(n, rng)), {}),
        [100, 1_000],
        ("numpy", "rust", "dtaidistance"),
    ),
    (
        "needleman_wunsch",
        Q.needleman_wunsch_alignment,
        lambda n, rng: ((_dna(n, rng), _dna(n, rng)), {}),
        [100, 1_000],
        ("numpy", "rust", "biopython"),
    ),
    (
        "smith_waterman",
        Q.smith_waterman_alignment,
        lambda n, rng: ((_dna(n, rng), _dna(n, rng)), {}),
        [100, 1_000],
        ("numpy", "rust", "biopython"),
    ),
    (
        "nw custom substitution",
        Q.needleman_wunsch_alignment,
        lambda n, rng: (
            (_dna(n, rng), _dna(n, rng)),
            {"substitution": lambda x, y: 2.0 if x == y else -1.0},
        ),
        [100],
        ("numpy", "rust"),
    ),
]


def conditions():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next(
                line.split(":", 1)[1].strip() for line in f if "model name" in line
            )
    except (OSError, StopIteration):
        pass
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "python": platform.python_version(),
        "build": "GIL" if gil else "free-threaded",
        "numpy": np.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
        "cpu": cpu,
        "cpus": os.cpu_count(),
    }


def time_call(fn, args, kwargs, *, repeat):
    """(median, min, max) seconds per call over ``repeat`` auto-ranged timing runs."""
    timer = timeit.Timer(lambda: fn(*args, **kwargs))
    number, _ = timer.autorange()
    runs = [t / number for t in timer.repeat(repeat=repeat, number=number)]
    return median(runs), min(runs), max(runs)


def _fmt(seconds):
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smallest size, 3 repeats")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--backends", nargs="+", default=["numpy", "rust"])
    parser.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    opts = parser.parse_args(argv)

    backends = [b for b in opts.backends if b == "numpy" or have(_PACKAGE.get(b, b))]
    if "rust" in backends:
        import equate_rust

        if equate_rust.DEBUG_BUILD:
            print("refusing to benchmark a debug build of equate_rust: build --release")
            return 2
    repeat = 3 if opts.quick else opts.repeat
    cond = conditions()
    print("conditions: " + ", ".join(f"{k}={v}" for k, v in cond.items()))
    print(f"backends: {', '.join(backends)}; median of {repeat}, spread = min-max\n")
    print("| case | size | backend | median | spread | vs numpy |")
    print("|---|---|---|---|---|---|")
    rows = []
    for label, fn, factory, sizes, applies in CASES:
        for size in sizes[:1] if opts.quick else sizes:
            args, kwargs = factory(size, np.random.default_rng(SEED))
            base = None
            for backend in (b for b in backends if b in applies):
                med, lo, hi = time_call(
                    fn, args, {**kwargs, "backend": backend}, repeat=repeat
                )
                base = med if backend == "numpy" else base
                ratio = f"{base / med:,.1f}x" if base else ""
                print(
                    f"| {label} | {len(args[0])} x {len(args[1])} | {backend} | {_fmt(med)} "
                    f"| {_fmt(lo)} - {_fmt(hi)} | {ratio} |",
                    flush=True,
                )
                rows.append(
                    {
                        "case": label,
                        "size": f"{len(args[0])} x {len(args[1])}",
                        "backend": backend,
                        "median_s": med,
                        "min_s": lo,
                        "max_s": hi,
                        "speedup_vs_numpy": base / med if base else None,
                    }
                )
    if opts.json:
        with open(opts.json, "w", encoding="utf-8") as f:
            json.dump({"conditions": cond, "results": rows}, f, indent=2)
        print(f"\nwrote {opts.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
