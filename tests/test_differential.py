"""Differential tests: every alternative backend of ``equate.compare.sequence`` must
behave like ``backend="numpy"`` (equate#35).

The numpy backend is the **reference**: its kernels (``_dtw_accumulate``,
``_traceback_dtw``, ``_align_dp``, ...) are the specification a faster backend must
reproduce. For each call the test compares the *outcome* of the reference and of the
backend under test: the returned :class:`~equate.base.Alignment` (or float), or the
exception class.

- **Ported backends** (``backend="rust"``, from the separate ``equate-rust`` wheel): the
  whole domain, NaN/inf/-0.0 included. Scores agree within ``1e-12`` relative (NaN
  equals NaN, signed zero and inf must match); **paths are compared exactly**, so a
  different tie-breaking fails.
- **Third-party engines** (``dtaidistance``, ``biopython``, ``tslearn``) are cross-checks
  over the domain they implement (finite numbers, the default cost, non-empty
  sequences): scores only, since they report no path.

Profiles: ``fast`` (default, derandomized, for every PR) and ``thorough``::

    HYPOTHESIS_PROFILE=thorough python -m pytest tests/test_differential.py

``EQUATE_RUST_SHIM=1`` installs a pure-Python stand-in for ``equate_rust`` (the numpy
kernels behind the native module's interface) when the wheel is absent: the
reference-against-reference run that proves the harness and the facade routing, and
the way to check the test fails against a stub (point the shim at a stub).
"""

from __future__ import annotations

import copy
import math
import os
import sys
from fractions import Fraction
from functools import partial

import numpy as np
import pytest
from hypothesis import example, given, note, settings
from hypothesis import strategies as st

from equate.base import Alignment
from equate._dependencies import have
from equate.compare import sequence as Q

settings.register_profile("fast", max_examples=300, derandomize=True, deadline=None)
settings.register_profile("thorough", max_examples=5000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))

if os.environ.get("EQUATE_RUST_SHIM") and not have("equate_rust"):
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "equate_rust", os.path.join(os.path.dirname(__file__), "_rust_shim.py")
    )
    sys.modules["equate_rust"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["equate_rust"])


def _available(backend, family, package):
    return backend in family and have(package)


HAVE_RUST_DTW = _available("rust", Q._DTW_BACKENDS, "equate_rust")
HAVE_RUST_ALIGN = _available("rust", Q._ALIGN_BACKENDS, "equate_rust")
needs_rust_dtw = pytest.mark.skipif(
    not HAVE_RUST_DTW, reason="backend='rust' (equate-rust) is not installed"
)
needs_rust_align = pytest.mark.skipif(
    not HAVE_RUST_ALIGN, reason="backend='rust' (equate-rust) is not installed"
)


# --- comparing outcomes ---------------------------------------------------------------

REL = 1e-12  # scores: relative tolerance (equate#35); paths: exact


def same(a, b, *, rel=0.0):
    """Structural equality that also compares types. Floats: NaN equals NaN, the sign of
    zero matters, others within ``rel`` (infinities must be equal)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if a == b:
            return math.copysign(1.0, a) == math.copysign(1.0, b)
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=rel)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y, rel=rel) for x, y in zip(a, b))
    if isinstance(a, Alignment):
        return (
            same(a.score, b.score, rel=rel)
            and same(a.path, b.path)  # exact: tie-breaking is part of the contract
            and a.sense == b.sense
        )
    return a == b


def score_only(a, b, *, rel=REL):
    """For engines that report no path: compare the scores alone."""
    return same(float(a), float(b), rel=rel)


def outcome(fn, args, kwargs):
    args, kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - exception classes are part of the API
        return ("err", type(exc))


def check(ref_fn, port_fn, args, kwargs, *, compare=partial(same, rel=REL)):
    expected, actual = outcome(ref_fn, args, kwargs), outcome(port_fn, args, kwargs)
    note(f"reference: {expected!r}")
    note(f"port:      {actual!r}")
    if expected[0] == actual[0] == "ok":
        assert compare(actual[1], expected[1]), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def differential(fn, backend, *, args, kwargs, compare=None, examples=()):
    """A Hypothesis test: ``fn(..., backend=backend)`` behaves like ``backend='numpy'``.

    ``examples`` is the seed corpus, ``(args, kwargs)`` pairs always run first: every
    divergence found during the port goes there before it is fixed.
    """
    ref_fn, port_fn = partial(fn, backend="numpy"), partial(fn, backend=backend)
    kw = {} if compare is None else {"compare": compare}

    def test(args, kwargs):
        check(ref_fn, port_fn, args, kwargs, **kw)

    for ex_args, ex_kwargs in examples:
        test = example(args=ex_args, kwargs=ex_kwargs)(test)
    test = given(args=args, kwargs=kwargs)(test)
    test.__name__ = f"test_{fn.__name__}_{backend}_matches_numpy"
    return test


def trap_grid(fn, backend, *, cases, compare=None):
    """Every ``(args, kwargs)`` in ``cases``, deterministically: the conjunctions random
    generation rarely hits (one trap value *and* one exact parameter)."""
    ref_fn, port_fn = partial(fn, backend="numpy"), partial(fn, backend=backend)
    kw = {} if compare is None else {"compare": compare}

    @pytest.mark.parametrize(("case_args", "case_kwargs"), cases)
    def test(case_args, case_kwargs):
        check(ref_fn, port_fn, case_args, case_kwargs, **kw)

    test.__name__ = f"test_{fn.__name__}_{backend}_trap_grid"
    return test


# --- strategies: ties, non-finite values, band edges ----------------------------------
# Small repeated values make many warping paths / alignments tie, so the tie-breaking
# is exercised on most examples rather than by luck.

TIE_VALUES = (0.0, 1.0, 2.0, -1.0, 0.5)
TRAP_FLOATS = (math.nan, math.inf, -math.inf, -0.0, 5e-324, 1e308, -1e308)
finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
scalar = st.one_of(
    st.sampled_from(TIE_VALUES),
    st.sampled_from(TIE_VALUES),
    finite,
    st.sampled_from(TRAP_FLOATS),
)
finite_scalar = st.one_of(st.sampled_from(TIE_VALUES), finite)


def series(element=scalar, *, min_size=0, max_size=12):
    univariate = st.lists(element, min_size=min_size, max_size=max_size)
    multivariate = st.integers(1, 3).flatmap(
        lambda d: st.lists(
            st.lists(element, min_size=d, max_size=d),
            min_size=min_size,
            max_size=max_size,
        )
    )
    long = st.lists(element, min_size=max(min_size, 13), max_size=40)
    return st.one_of(univariate, univariate, multivariate, long)


def _abs_cost(x, y):
    return abs(float(np.sum(x)) - float(np.sum(y)))


def _zero_cost(x, y):
    return 0.0


def _negative_cost(x, y):
    return -1.0


def _int_cost(x, y):
    return int(np.sum(x) == np.sum(y))


WINDOWS = st.one_of(
    st.none(),
    st.integers(0, 6),
    st.floats(0.01, 1.0),
    st.sampled_from([10**30, np.int64(2), np.float32(0.25), 0.0, -1, True, "wide"]),
)
LOCAL_COSTS = st.sampled_from([None, None, None, _abs_cost, _zero_cost, _int_cost])

SCORES = st.one_of(
    st.integers(-3, 3),
    st.sampled_from([-0.5, -1 / 3, 0.0, -0.0, 2.5]),
    st.floats(-5, 5, allow_nan=False),
    st.sampled_from(
        [
            math.nan,
            math.inf,
            -math.inf,
            10**400,
            True,
            np.float64(-0.5),
            np.float32(-0.25),
        ]
    ),
    st.sampled_from([Fraction(-1, 3), np.int64(-2), 2**63]),
)
SYMBOL_SEQS = st.one_of(
    st.text("AB", max_size=12),
    st.text("ACGT", max_size=12),
    st.text(st.characters(), max_size=8),  # surrogates, astral, combining marks
    st.text("aİiİ\U0001f600\ud800\x00", max_size=6),
    st.lists(st.sampled_from(["the", "a", "A", "", "fox"]), max_size=10),
    st.lists(st.integers(-2, 2), max_size=10),
    st.lists(st.sampled_from([0, 1, 1.0, True, None, math.nan, "1", b"1"]), max_size=8),
    st.tuples(st.sampled_from(["x", "y"]), st.sampled_from(["x", "y"])),
    st.binary(max_size=8),
)


def _sub_tokens(x, y):
    return 2.0 if x == y else (1.0 if str(x)[:1] == str(y)[:1] else -2.0)


def _sub_nan(x, y):
    return math.nan if x == y else -1.0


def _sub_int(x, y):
    return int(x == y)


def _sub_raises(x, y):
    if x != y:
        raise KeyError((x, y))
    return 1.0


SUBSTITUTIONS = st.sampled_from(
    [None, None, None, _sub_tokens, _sub_nan, _sub_int, _sub_raises]
)

dtw_kwargs = st.fixed_dictionaries(
    {}, optional={"window": WINDOWS, "local_cost": LOCAL_COSTS}
)
align_kwargs = st.fixed_dictionaries(
    {},
    optional={
        "substitution": SUBSTITUTIONS,
        "match": SCORES,
        "mismatch": SCORES,
        "gap": SCORES,
    },
)
subseq_kwargs = st.fixed_dictionaries(
    {},
    optional={"local_cost": st.sampled_from([None, None, _abs_cost, _negative_cost])},
)


# --- the ported backend: rust ---------------------------------------------------------

DTW_SEEDS = [
    (([0, 0, 0], [0, 0, 0, 0]), {"window": None}),  # every path ties
    (([1, 0, 0], [0, 0, 1]), {"window": None}),
    (([math.nan, 0], [0]), {}),
    (([0, math.inf, 1], [1, 0]), {"window": 0}),
    (([0.0], [-0.0, 0.0]), {"window": 10**30}),
]
ALIGN_SEEDS = [
    (("AA", "A"), {}),
    (("ab", "ba"), {"gap": 0, "mismatch": 0}),
    (("aXa", "a"), {}),
    (("ab", "b"), {"gap": math.nan}),
    (("ab", "b"), {"substitution": _sub_nan}),
    (("\ud800a", "a\ud800"), {}),
    (([1, 1.0, True], [True, 1]), {}),
    (([math.nan], [math.nan]), {}),
    (("abc", "abd"), {"match": 10**400}),
    (("", ""), {}),
]

test_dtw_alignment_rust = needs_rust_dtw(
    differential(
        Q.dtw_alignment,
        "rust",
        args=st.tuples(series(), series()),
        kwargs=dtw_kwargs,
        examples=DTW_SEEDS,
    )
)
test_subsequence_dtw_alignment_rust = needs_rust_dtw(
    differential(
        Q.subsequence_dtw_alignment,
        "rust",
        args=st.tuples(series(max_size=6), series()),
        kwargs=subseq_kwargs,
        examples=[(([0, 0], [0, 0, 0]), {})],
    )
)
test_needleman_wunsch_alignment_rust = needs_rust_align(
    differential(
        Q.needleman_wunsch_alignment,
        "rust",
        args=st.tuples(SYMBOL_SEQS, SYMBOL_SEQS),
        kwargs=align_kwargs,
        examples=ALIGN_SEEDS,
    )
)
test_smith_waterman_alignment_rust = needs_rust_align(
    differential(
        Q.smith_waterman_alignment,
        "rust",
        args=st.tuples(SYMBOL_SEQS, SYMBOL_SEQS),
        kwargs=align_kwargs,
        examples=ALIGN_SEEDS,
    )
)

_GRID_SERIES = [
    [],
    [0],
    [math.nan],
    [math.inf, 0],
    [-0.0, 0.0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 1, 0],
    [1e308, -1e308],
    [[0, 1], [1, 0]],
]
_GRID_WINDOWS = [None, 0, 1, 2, 0.5, 10**30]
test_dtw_alignment_rust_grid = needs_rust_dtw(
    trap_grid(
        Q.dtw_alignment,
        "rust",
        cases=[
            ((a, b), {"window": w, "local_cost": lc})
            for a in _GRID_SERIES
            for b in _GRID_SERIES
            for w in _GRID_WINDOWS
            for lc in (None, _abs_cost)
        ],
    )
)
_GRID_SYMBOLS = [
    "",
    "A",
    "AA",
    "AB",
    "ab",
    "aXa",
    "\ud800",
    "İi",
    "\U0001f600a",
    ["a", "A"],
    b"ab",
]
_GRID_SCORES = [
    {},
    {"gap": 0, "mismatch": 0},
    {"match": 0, "mismatch": 0, "gap": 0},
    {"gap": -0.5, "mismatch": -1 / 3},
    {"gap": math.nan},
    {"match": math.inf},
    {"mismatch": -math.inf},
    {"match": 10**400},
    {"gap": 10**400},
    {"substitution": _sub_nan},
]
for _fn in (Q.needleman_wunsch_alignment, Q.smith_waterman_alignment):
    _test = needs_rust_align(
        trap_grid(
            _fn,
            "rust",
            cases=[
                ((a, b), kw)
                for a in _GRID_SYMBOLS
                for b in _GRID_SYMBOLS
                for kw in _GRID_SCORES
            ],
        )
    )
    globals()[_test.__name__] = _test
del _fn, _test


# --- third-party engines: cross-checks over their domains ------------------------------

needs = lambda package: pytest.mark.skipif(  # noqa: E731
    not have(package), reason=f"{package} is not installed"
)


def _same_dimension_pair(d):
    if d == 0:  # univariate
        one = st.lists(finite_scalar, min_size=1, max_size=20)
    else:
        one = st.lists(
            st.lists(finite_scalar, min_size=d, max_size=d), min_size=1, max_size=12
        )
    return st.tuples(one, one)


# dtaidistance's domain: steps of one dimension on both sides (it returns 0.0 for a
# univariate-vs-2-D pair, where numpy raises ValueError: see docs/port-log.md)
test_dtw_dtaidistance = needs("dtaidistance")(
    differential(
        Q.dtw_alignment,
        "dtaidistance",
        args=st.integers(0, 3).flatmap(_same_dimension_pair),
        kwargs=st.fixed_dictionaries(
            {},
            optional={
                "window": st.one_of(st.none(), st.integers(0, 6), st.floats(0.01, 1))
            },
        ),
        compare=score_only,
    )
)
_DNA = st.text("ACGT", min_size=1, max_size=12)
_SCALAR_SCORES = st.sampled_from([-2, -1, -0.5, 0, 1, 2, 0.25])
# Biopython's domain: non-empty sequences, gap scores <= 0 (with a positive gap its local
# mode disagrees with numpy's DP: see docs/port-log.md)
_bio_kwargs = st.fixed_dictionaries(
    {},
    optional={
        "match": _SCALAR_SCORES,
        "mismatch": _SCALAR_SCORES,
        "gap": st.sampled_from([-2, -1, -0.5, -1 / 3, 0]),
    },
)
test_needleman_wunsch_biopython = needs("Bio")(
    differential(
        Q.needleman_wunsch_alignment,
        "biopython",
        args=st.tuples(_DNA, _DNA),
        kwargs=_bio_kwargs,
        compare=score_only,
    )
)
test_smith_waterman_biopython = needs("Bio")(
    differential(
        Q.smith_waterman_alignment,
        "biopython",
        args=st.tuples(_DNA, _DNA),
        kwargs=_bio_kwargs,
        compare=score_only,
    )
)
# tslearn evaluates the soft-minimum in another order: equal to ~1e-9, not to the last bit
test_soft_dtw_tslearn = needs("tslearn")(
    differential(
        Q.soft_dtw,
        "tslearn",
        args=st.tuples(
            st.lists(
                finite_scalar.filter(lambda x: abs(x) < 50), min_size=1, max_size=10
            ),
            st.lists(
                finite_scalar.filter(lambda x: abs(x) < 50), min_size=1, max_size=10
            ),
        ),
        kwargs=st.fixed_dictionaries(
            {}, optional={"gamma": st.sampled_from([0.1, 0.5, 1.0, 2.0])}
        ),
        compare=lambda a, b: math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9),
    )
)


# --- the native module must be the one under test -------------------------------------


@needs_rust_dtw
def test_native_extension_is_loaded():
    """Fails if a stale build, the shim, or a source tree is what got imported."""
    if (
        os.environ.get("EQUATE_RUST_SHIM")
        and sys.modules.get("equate_rust") is not None
    ):
        if getattr(sys.modules["equate_rust"], "IS_SHIM", False):
            pytest.skip("EQUATE_RUST_SHIM: the pure-Python stand-in is under test")
    import equate_rust

    assert equate_rust._core.__file__.endswith((".so", ".pyd")), (
        equate_rust._core.__file__
    )
    assert not equate_rust.DEBUG_BUILD or os.environ.get("EQUATE_RUST_ALLOW_DEBUG")
