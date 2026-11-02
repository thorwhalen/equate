# Port log: Rust backend for `equate.compare.sequence`

The state of the port, for the next session. Method: the `pyo3-port` skill (oracle
first, one leaf function per step, each gated on a differential test and a review);
whether to port: the `rust-port-assess` skill.

Status: **in progress** (cloud session, 2026-11-02). Landing owed: see the end of this file.

## 1. Assessment

Fleet survey verdict (2026-11-01): score 81, **port (pilot)**. Re-checked in this session:
see "Assessment check" below.

## Baseline (before any change)

`python -m pytest tests/ -q` on the branch point (`434a35f`), CPython 3.13.18, numpy 2.5.4,
Linux x86_64: **442 passed, 0 failed, 55 skipped**.

## Assessment check (this session, `rust-port-assess`)

The brief is the consent for running code in the VM. Measured on the branch point
(CPython 3.13.18, numpy 2.5.4, Intel Xeon @ 2.10 GHz, 4 vCPU):

| kernel | 100 x 100 | 300 x 300 | per cell |
|---|---|---|---|
| `_dtw_accumulate` (unbanded) | 4.4 ms | 40 ms | ~445 ns (the survey's 716 ns was an estimate on another machine) |
| `needleman_wunsch_alignment` (whole call) | 26 ms | 209 ms | ~2.3 µs |
| `_traceback_dtw` | 0.1 ms | 0.3 ms | O(n + m), negligible |

- `hotspots.py` on `equate/compare` ranks `_align_dp` (260), `_soft_dtw_backward` (92),
  `_dtw_accumulate`, `subsequence_dtw_alignment`, `_soft_dtw_forward` (84 each),
  `levenshtein_distance` (16), `_traceback_dtw` (13): the survey's list, in its order of weight.
- **Gate 1 (fix in Python)** does not fire. There is no asymptotic fix (the DP is
  inherently O(nm)); the constant-factor fix, an anti-diagonal numpy sweep (prototype,
  bit-identical `D`), measured **1.3x at 100 x 100 and 4.2x at 1,000 x 1,000**. Matching
  scores many short pairs, where 1.3x is no answer.
- **Gate 2 (depend on a library)** does not fire for what moves: `dtaidistance` is already
  the score-only DTW backend, but no installed library gives equate's banded path with its
  tie-breaking, nor NW/SW with an arbitrary substitution matrix and a path (Biopython's
  backend is score-only, scalar scores only).
- Shape: data in (a float cost matrix, or two code arrays), data out (a number and a path):
  a coarse boundary, no callbacks inside the loop. The verdict **port (pilot)** stands.

## Port surface coverage (`port_coverage.py`)

Before: `_dtw_accumulate`, `_traceback_dtw`, `_align_dp`, `_band_columns`,
`_soft_dtw_backward`, and the `dtw` / NW / SW alignment facades at 100% lines and branches;
`subsequence_dtw_alignment` 89% / 80%, `_soft_dtw_forward` 92% / 83%,
`levenshtein_distance` 86% / 80%.

After `tests/test_port_surface.py` (28 characterization tests, written by a Sonnet
subagent, checked here): **every target at 100% lines and branches.** Behaviour it pins
that looks odd but is today's specification (a port must reproduce it):

- NaN or inf in a DTW series gives a NaN or inf score **and a path** (the traceback's `min`
  keeps the first candidate when comparisons are false).
- NaN substitution or gap in NW/SW gives a NaN score and a path; SW's end cell is the
  first NaN of `H` (numpy's `argmax` returns the first NaN), and `NaN <= 0` is false, so a
  NaN cell is not floored to zero.
- `soft_dtw` with an inf step returns `inf`, no exception.

## 2. The oracle (issue #35)

The numpy backend **is the frozen reference**. This is the separate-package shape (the
Rust kernels ship as their own wheel, `equate-rust`, selected with `backend="rust"`), so
nothing is replaced: `backend="numpy"` keeps calling the same in-module kernels
(`_dtw_accumulate`, `_traceback_dtw`, `_align_dp`), and those are the specification.
Decision: no separate `_reference.py` copy (two copies of one kernel would drift);
changing a numpy kernel is a behaviour change that needs a test and a changelog entry.

- `tests/test_sequence.py` is parametrised over every backend of each family
  (`Q._DTW_BACKENDS`, `Q._ALIGN_BACKENDS`, `Q._SOFT_DTW_BACKENDS`), skipped when the
  backend's package is absent, with capability flags (`path`, `local_cost`,
  `multivariate`, `substitution`, `empty`, `messages`) so score-only engines run the
  assertions they can answer. A backend joins automatically once the facade accepts it.
- **Tie-breaking pinned** (`test_dtw_traceback_keeps_the_first_minimum`,
  `test_alignment_fill_keeps_the_first_maximum`). Checked against hand mutants of the numpy
  kernels: a last-wins traceback and a reordered (left, up, diagonal) traceback each fail 3
  of the 5 DTW cases; a last-wins `argmax` fails all 9 NW/SW cases.
- `tests/test_differential.py` (Hypothesis, profiles `fast`/`thorough`): the ported
  backend against numpy on the whole domain (NaN, ±inf, -0.0, 1e308, subnormals, ties from
  small repeated values, multivariate steps, windows incl. `10**30`, numpy scalars, bools,
  bad types; symbols as `str` with surrogates and astral characters, bytes, token lists,
  mixed objects, `nan` tokens; scores incl. NaN, inf, `10**400`, `Fraction`, numpy scalars;
  substitution callables returning NaN, ints, or raising), scores within 1e-12 relative,
  **paths exact**, exception classes equal; seed corpus and a deterministic trap grid
  (DTW 1,200 cases, NW and SW 1,210 each). Third-party engines are cross-checked on their
  domains (scores only).

Cross-check findings about the **existing** third-party backends (not changed here, open
questions for the maintainers):

1. `backend="dtaidistance"` returns `0.0` for a univariate series against a 2-D one
   (`dtw([0.0], [[0.0, 0.0]])`), where numpy raises `ValueError` ("different dimensions").
2. `backend="biopython"` raises `ValueError` on an empty sequence, where numpy returns the
   all-gap score; with a *positive* gap score its local mode disagrees with numpy's DP
   (`smith_waterman('A', 'A', match=-2, gap=1)`: numpy 1.0, Biopython 0.0).
3. `backend="dtaidistance"` is **slower than numpy** (0.6-0.7x, table below): equate calls
   `dtaidistance.dtw.distance` without `use_c=True`, so dtaidistance runs its pure-Python
   implementation. "A faster engine is one keyword away" does not hold for it today.

On their domains all three agree with numpy: dtaidistance and Biopython to 1e-12
relative, tslearn's soft-DTW to 1e-9 (it evaluates the soft-minimum differently).

### Before (numpy backend), `benchmarks/bench_port.py`

Median of 7, auto-ranged runs, spread = min-max; end to end through the public function.
Machine: Intel Xeon @ 2.10 GHz (model 207), 4 vCPU, Linux x86_64 cloud VM; CPython 3.13.18
(GIL), numpy 2.5.4.

| case | size | numpy | spread | dtaidistance / biopython |
|---|---|---|---|---|
| dtw unconstrained | 100 x 100 | 4.39 ms | 3.58 - 4.73 ms | 6.12 ms (0.7x) |
| dtw unconstrained | 1000 x 1000 | 417 ms | 376 - 486 ms | 666 ms (0.6x) |
| dtw 10% band | 100 x 100 | 992 µs | 964 µs - 1.1 ms | 1.47 ms (0.7x) |
| dtw 10% band | 1000 x 1000 | 92.3 ms | 91.1 - 113 ms | 130 ms (0.7x) |
| subsequence_dtw | 10 x 100 | 439 µs | 418 - 561 µs | 1.11 ms (0.4x) |
| subsequence_dtw | 100 x 1000 | 41.8 ms | 40.3 - 46 ms | 90.3 ms (0.5x) |
| needleman_wunsch | 100 x 100 | 22.9 ms | 20.5 - 23.5 ms | 54.6 µs (420x, score only) |
| needleman_wunsch | 1000 x 1000 | 2.56 s | 2.3 - 2.88 s | 2.1 ms (1,217x, score only) |
| smith_waterman | 100 x 100 | 27.5 ms | 24 - 29.5 ms | 53.5 µs (514x, score only) |
| smith_waterman | 1000 x 1000 | 2.64 s | 2.41 - 2.74 s | 2.55 ms (1,036x, score only) |
| nw custom substitution | 100 x 100 | 26.3 ms | 22.2 - 41.6 ms | — |

The brief's "about 0.7 s" for 1,000 x 1,000 DTW was an estimate: measured, 0.42 s
unconstrained, 0.09 s in the default 10% band.

## 3-4. Layout and skeleton

**Layout: a separate package inside this repository** (`pyo3-port`'s "separate kernels
package" shape), because a merge of equate publishes it and wads builds only pure-Python
wheels until i2mint/wads#166 lands:

```text
rust/                       the equate-rust package (its own maturin build; not in equate's sdist)
├── Cargo.toml, Cargo.lock  workspace: core + py; crates at 0.0.0, publish = false
├── pyproject.toml          maturin; name equate-rust; module-name equate_rust._core; abi3-py310
├── core/                   equate-sequence-core: pure Rust, no PyO3
├── py/                     equate-rust-py: cdylib `_core`, PyO3 0.30.1 + numpy 0.30.0
└── python/equate_rust/     __init__.py (re-exports), _core.pyi, py.typed
```

equate's own wheel stays pure Python. `equate/compare/sequence.py` imports `equate_rust`
lazily (`require`, on the first `backend="rust"` call) and keeps every public signature,
default, validation message and docstring; `"rust"` joins `_DTW_BACKENDS` and
`_ALIGN_BACKENDS`, and the module docstring gains one paragraph about it. Soft-DTW keeps
numpy/tslearn only (not ported: see the end).

Decisions:

- **No `equate[rust]` extra yet.** `equate-rust` is not on PyPI, and an extra naming it
  would break `uv lock` and `pip install equate[rust]`; the missing-package message says
  `pip install equate_rust`. The landing session adds the extra once the wheel is
  published (and regenerates `uv.lock`).
- **The kernel seam.** DTW: `_dtw_kernels(backend)` returns the three kernels
  (accumulate, subsequence accumulate, traceback) as a `NamedTuple`; the numpy tuple is the
  reference, the rust one wraps `_core`. Alignment: `_align(..., backend)` routes to
  `_align_dp` (reference) or `_align_rust`. Two pure refactors of the reference, both
  covered by the characterization tests: `_subsequence_accumulate` (the loop formerly inline
  in `subsequence_dtw_alignment`) and `_substitution_scores` (the S matrix, now shared).
- **Domain routing (`_align_rust`).** Rust receives only what it reproduces exactly:
  - a gap is passed as a double only when its type's arithmetic with `np.float64` is exactly
    `float(gap)` (`int`, `float`, `bool`, numpy float16/32/64 and fixed-size ints) and
    `float(gap)` does not overflow; any other gap runs the numpy reference unchanged;
  - with the default scorer and plain match/mismatch scores, elements that are exact `str`
    or `int` are interned to int64 codes (dict equality is `==` for those types) and Rust
    scores the cells (`align_codes`);
  - otherwise the substitution matrix is built in Python by the reference's own code (so
    substitution callables run in the same order, raise the same exceptions) and Rust does
    the fill and traceback (`align_scores`).
- **DTW window**: `_core` takes `Option<usize>`; the facade clamps a half-width to
  `max(n, m)` before the call (exact: such a band already spans every row), so
  `window=10**30` cannot raise `OverflowError`.

Skeleton run (`_core` with every kernel raising `NotImplementedError`, the facade pointing
at it): **every `backend="rust"` differential test fails** (2,990 of 3,624 cases; the 634
passing are grid cases rejected by validation before any kernel runs), and every
`backend="rust"` case of `tests/test_sequence.py` that reaches a kernel fails.
Reference-against-reference (`EQUATE_RUST_SHIM=1`, the numpy kernels behind `_core`'s
interface): **3,624 passed**. That run found a harness bug (Hypothesis `note()` called
from the deterministic grid), fixed before the stub run above.

From the skeleton on, the rust tests fail locally until each kernel lands; hosted CI does
not install `equate-rust`, so they skip there and the branch stays green.

## 5. Leaves

### 5.1 `_dtw_accumulate` -> `equate_sequence_core::dtw::accumulate` (done)

- Core: `dtw::accumulate(MatrixView, Band) -> Matrix` with `Band::{Unconstrained,
  HalfWidth(usize)}`; the band's columns are the reference's `range(...)` in saturating
  arithmetic (`Band::columns`); each cell is `C + py_min3(diag, up, left)`, where
  `pyorder::py_min3` is Python's `min` (first value kept, replaced only by a strictly
  smaller one), so NaN propagates exactly as in the reference. Same additions in the same
  order: the matrix is bit-identical. `dtw::subsequence_accumulate` shares the fill; its
  binding lands in its own step.
- Facade: the rust DTW kernels use `_core.dtw_accumulate` (window clamped to `max(n, m)`),
  the numpy traceback and subsequence accumulation until those are ported; alignment stays
  on the reference (`_RUST_ALIGN_PORTED = False`).
- Differential: fast profile + grid green; **thorough (5,000 examples) + 1,200-case grid:
  1,201 passed**. Hand mutants: `f64::min` instead of `py_min3` (ignores NaN) -> 13
  failures; band upper edge off by one -> 95 failures.
- Kernel timing (release, same VM): unconstrained 100² 6.42 ms -> 0.023 ms (280x),
  1000² 415 ms -> 2.77 ms (150x); 10% band 100² 191x, 1000² 87x.
- Accepted difference: the numpy reference emits numpy's `RuntimeWarning` ("overflow
  encountered in scalar add") when a sum overflows to inf (e.g. steps of 1e308); the Rust
  DP computes the same inf silently. Values are equal; only the warning differs.

#### Review of 5.1 (`rust-reviewer`, fresh Opus subagent): changes requested, 0 blockers, 3 majors

| finding | resolution |
|---|---|
| major B1: `vec!` of an unallocatable padded matrix aborts the process (`_core.dtw_accumulate(np.empty((2**40, 0)), window=None)` exited 134) | fixed: `Matrix::filled` allocates with `try_reserve_exact`, `Error::Alloc` maps to `MemoryError` (the numpy reference raises `MemoryError` too); verified on the same probe |
| major P4: the O(nm) fill runs attached, the doc said it detached | **declined, doc fixed**: the kernels read caller-owned numpy buffers in place; detached (or free-threaded), another thread could write them mid-read, a data race. Copying first costs about half the kernel at 1000². Kernels on copied inputs (`align_codes`) detach |
| major P5: no `gil_used` decision | `#[pymodule(gil_used = true)]`, for the same reason; abi3-py310 wheels do not load on 3.14t anyway, so free-threaded support is untested and unclaimed |
| minor P3: facade calls the traceback without `stop_row0` | fixed in 5.2: the facade wrapper passes it |
| minor D2: `align_codes` has 6 arguments | `#[expect(clippy::too_many_arguments)]` with the reason, in 5.3 |
| minor A5: `Error::Shape` folded into `_` | explicit arm; `_` kept for `#[non_exhaustive]` |
| minor X: `filter_map` hid the "row exists" invariant | explicit loop with the invariant in a comment |

Also from the reviewer: stale-form search clean; `extension-module` and `panic = "abort"`
absent; `Cargo.lock` and `deny.toml` committed; the trap rows for `Band::columns` and the
fill (argument order of `min`, first-kept ties, NaN, -0.0, inf + -inf, overflow) all
handled and pinned.

Thorough cross-check run of the third-party engines (5,000 examples each): dtaidistance,
Biopython NW and SW, tslearn: **4 passed**.
