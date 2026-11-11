# Port log: Rust backend for `equate.compare.sequence`

The state of the port, for the next session. Method: the `pyo3-port` skill (oracle
first, one leaf function per step, each gated on a differential test and a review);
whether to port: the `rust-port-assess` skill.

Status: **ported (DTW, subsequence DTW, NW/SW), landing owed** (cloud session, 2026-11-02): see the end of this file.

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
- Landing session: under `np.seterr(all='raise')` (or any state other than ignore/warn) the
  rust backend now runs the numpy kernels, so overflow raises the same `FloatingPointError`
  (`_float_state_is_quiet`; test `test_rust_follows_numpys_float_error_state`). A warning
  filter that turns `RuntimeWarning` into an error is not detected: see the next item.
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

### 5.2 `_traceback_dtw` -> `equate_sequence_core::dtw::traceback` (done)

- Core: `dtw::traceback(MatrixView, end, Start::{Origin, FirstRow}) -> Result<Vec<(usize,
  usize)>, Error>`: candidates offered in the reference's order (diagonal, up, left; left
  withheld on the first row for `FirstRow`), the first kept unless a later one is strictly
  smaller (Python's `min(..., key=...)`), so ties and NaN behave as in the reference. An
  `end` outside the matrix is `Error::OutOfBounds` -> `IndexError` (the facade never
  passes one).
- Facade: the rust DTW kernels now use the native accumulation **and** traceback; the
  wrapper passes `stop_row0` explicitly (review minor P3).
- Differential: thorough (5,000 examples) + 1,200-case grid: **1,201 passed**;
  `tests/test_sequence.py` + `tests/test_port_surface.py`: 222 passed, 6 skipped
  (Biopython on empty inputs). Hand mutant: last-wins ties (`<=` for `<`) -> **415
  failures**.

### 5.3 `_align_dp` (fill + traceback) -> `equate_sequence_core::align::align` (done)

- Core: `align::align(&impl Substitution, Scoring { gap, mode: Mode::{Global, Local} })
  -> Alignment { score, path: Vec<(Option<usize>, Option<usize>)> }`. Two substitution
  sources behind one trait: a `MatrixView` (the score matrix the facade built) and
  `MatchMismatch` (interned codes, `matched`/`mismatched`), so the default scorer costs no
  Python call per cell. Each cell takes `pyorder::np_argmax` of (diagonal, up, left):
  numpy's first maximum, first NaN wins; local mode floors `<= 0` (NaN is not floored);
  the local end cell is `np_argmax` over the whole table (row-major); the traceback follows
  the recorded moves, never a recomputed float equality.
- Binding: `align_scores` (stays attached: reads the caller's buffer) and `align_codes`
  (copies the O(n + m) codes, then `py.detach` for the O(nm) work).
- Facade: `backend="rust"` alignment routes through `_align_rust` (domain rules in 3-4).
  Routing checked by a spy: `str`, token lists, `int` lists and `bytes` take
  `align_codes`; float tokens and substitution callables take `align_scores`; a
  `Fraction` gap runs the reference; `match=10**400` / `gap=10**400` raise
  `OverflowError` exactly where the reference does.
- Differential: fast + both 1,210-case grids: 2,422 passed; **thorough (5,000 examples
  each, NW and SW): passed**; `tests/test_sequence.py` + `tests/test_port_surface.py`
  green. Hand mutant: last-wins `argmax` (`<` for `<=`) -> **557 failures**.
- Timing (end to end, release): NW 100² 38 ms -> 0.061 ms (623x), 1000² 2.59 s -> 6.24 ms
  (415x).

### 5.4 subsequence DTW accumulation -> `equate_sequence_core::dtw::subsequence_accumulate` (done)

- Core: shares the DTW fill (unbanded, whole first row zero). The end point stays
  `np.argmin(D[n, 1:])` in the facade (numpy, O(m)), the traceback is 5.2's with
  `Start::FirstRow`.
- Binding `subsequence_dtw_accumulate`; the last `NotImplementedError` stub is gone, and
  `cargo clippy --workspace --all-targets` is clean (0 warnings).
- Differential: thorough (5,000 examples): passed; new deterministic grid (10 queries x 12
  series x 3 local costs incl. a negative one) + seeds: 362 passed.
- Mutants: skipping the last column -> 51 failures. Leaving `D[0][m]` at `+inf` survived
  and is **equivalent**: that cell is only ever the "up" candidate beside a diagonal `0`
  that wins the tie, and the traceback stops before reading row 0.

#### Review of 5.2-5.4 (`rust-reviewer`, fresh Opus subagent): changes requested, 1 blocker, 2 majors

The reviewer's own routing probe (NW and SW x 8 sequence pairs x 31 values for gap /
match / mismatch / substitution: ints at the float edge, `10**400`, `2**64 + 1`, numpy
float16/32, int8/64, uint32/64, longdouble, bool, NaN, ±inf, `Fraction`, `Decimal`, `"1"`,
`None`; 2,480 cases) found **no difference** in score, path or exception class.

| finding | resolution |
|---|---|
| blocker B1: `align` allocated its tables with `vec!`; `needleman_wunsch_alignment('a'*2**20, 'b'*2**20, backend='rust')` aborted the process (numpy raises `MemoryError`) | fixed: `align` returns `Result`, tables via a fallible `try_filled` (shared with `Matrix::filled`), `Error::Alloc` -> `MemoryError`; the probe now raises `MemoryError: cannot allocate a 1048577 x 1048577 matrix` |
| major B1: copying a non-contiguous array (`row_major`) could abort | fixed: `try_copy` with `try_reserve_exact` -> `MemoryError` (probe: a broadcast 2^20 x 2^20 view) |
| major B1: the code copies in `align_codes` could abort | fixed: same `try_copy` (probe: a broadcast 2^50 array) |
| minor B4: nothing pinned which kernel each input takes | fixed: `test_rust_alignment_routing` (10 inputs) and `test_rust_dtw_routing`, with a spy on `_rust()` |
| minor X: kernels do not check signals | documented in the binding's module docs (Ctrl-C acts when the call returns; ~6 ms for the largest benchmarked call); a signal check would need PyO3 in the core |
| minor O2: `copy_from_slice` could panic for a wrong-length buffer | fixed: element-wise `zip` |

## 6. After benchmark (same machine, same run, release build)

`python benchmarks/bench_port.py --backends numpy rust dtaidistance biopython`; median of
7 auto-ranged runs, spread = min-max of the rust column; end to end through the public
function (validation, cost matrix, kernels, boundary). Intel Xeon @ 2.10 GHz (model 207),
4 vCPU, Linux x86_64 cloud VM; CPython 3.13.18 (GIL), numpy 2.5.4; `equate_rust` 0.1.0,
release (`DEBUG_BUILD` false). Ratios are valid within this run; absolute times are not
comparable with another machine.

| case | size | before (numpy) | after (rust) | ratio | rust spread |
|---|---|---|---|---|---|
| dtw unconstrained | 100 x 100 | 3.64 ms | 52.4 µs | 69x | 44.4 - 60.4 µs |
| dtw unconstrained | 1000 x 1000 | 462 ms | 8.72 ms | **53x** | 7.42 - 9.52 ms |
| dtw 10% band (the default window) | 100 x 100 | 1.07 ms | 39.4 µs | 27x | 37.5 - 42 µs |
| dtw 10% band (the default window) | 1000 x 1000 | 97.7 ms | 4.40 ms | 22x | 4.19 - 5.73 ms |
| subsequence_dtw | 10 x 100 | 438 µs | 19.7 µs | 22x | 15.9 - 22.5 µs |
| subsequence_dtw | 100 x 1000 | 40.6 ms | 564 µs | 72x | 488 - 630 µs |
| needleman_wunsch | 100 x 100 | 27.3 ms | 44.1 µs | 619x | 39.1 - 68 µs |
| needleman_wunsch | 1000 x 1000 | 2.45 s | 5.81 ms | **422x** | 5.0 - 7.16 ms |
| smith_waterman | 100 x 100 | 28.7 ms | 55.5 µs | 517x | 44.7 - 77.8 µs |
| smith_waterman | 1000 x 1000 | 2.72 s | 7.68 ms | 354x | 5.95 - 9.34 ms |
| nw custom substitution | 100 x 100 | 27.7 ms | 927 µs | 30x | 657 µs - 1.2 ms |

The numpy column agrees with the before run of step 2 within its spread (e.g. DTW
1000² 417 ms then, 462 ms now; NW 1000² 2.56 s then, 2.45 s now). Same run, for
reference: dtaidistance 0.5-0.8x numpy (its pure-Python path, see the open questions),
Biopython 525-1,378x (score only, no path). Every ported function gains: none goes back
to the reference.

Where the remaining DTW time goes (1000², unconstrained, 8.7 ms): the kernel is 2.8 ms
(150x on its own); the rest is the Python-side squared-Euclidean cost matrix (`einsum`)
and returning the padded matrix to numpy for the traceback. A fused native kernel
(costs + accumulate + traceback in one call) is the next step if that matters; the
custom-substitution case is bound by the n*m Python callable calls, as expected.

## 7. Publish rule (`rust-port-assess`) for `equate-sequence-core`

```text
Publish: no: veto (API not ready to commit to; shaped for the facade)  [the rule itself passes: a=3, b=2]
  (a) 3/3  pure transformation: f64 matrices / i64 code slices in, a Matrix or (score, path) out;
           finite configuration (Band, Start, Mode, Scoring); no callbacks; deterministic
  (b) 2/3  DTW: augurs-dtw 0.10.3 (MIT OR Apache-2.0, released 2026-10-27, ~8k recent downloads)
           is distance-only and univariate: no warping path, no subsequence DTW; dtw_rs 0.10.0 has
           a non-standard licence. NW/SW: bio 4.2.2 (MIT, ~122k recent downloads) has pairwise
           alignment with paths, so the gap is DTW with path / band / subsequence, not alignment.
           Searched: crates.io q="dynamic time warping", q="needleman wunsch"; read augurs-dtw's
           public API (Dtw::{euclidean, manhattan, with_window, distance, distance_matrix}).
  name: equate-sequence-core  free (404 on crates.io, with '-' and '_')
  layout: (ii) stay unpublished (publish = false)
```

Why the veto: the core's API is the facade's seam, not a Rust user's API. `accumulate`
returns the reference's *padded* matrix and the traceback is a separate call over it, and
the NaN handling deliberately mirrors Python's `min` and numpy's `argmax` (right for
parity, surprising as a Rust contract). A crate worth publishing would expose
`dtw(a, b, &Options) -> Alignment` (plus subsequence, NW and SW twins) with the padded
kernels private. After that redesign the rule says "standard" (a >= 2, b = 2), and
`cargo publish --dry-run -p equate-sequence-core` becomes the gate. Nothing was
published from this session, and no `cargo publish --dry-run` was needed.

## 8-9. Stubs, packaging, CI

- `equate_rust/_core.pyi` (with `__all__`) and `py.typed`: `python -m mypy.stubtest
  equate_rust._core` -> **no issues**.
- Local release build: `maturin build --release --locked` -> `equate_rust-0.1.0-cp310-abi3-
  manylinux_2_34_x86_64.whl` (contents: `__init__.py`, `_core.abi3.so`, `_core.pyi`,
  `py.typed`, no bytecode); `maturin sdist` builds. The wheel installed into a fresh
  **CPython 3.10** venv with equate (non-editable) passed the differential, sequence and
  port-surface tests run from outside the checkout (`--import-mode=importlib`):
  **4,150 passed, 67 skipped** (the third-party engines were not installed there).
- `.github/workflows/rust-ci.yml` (PRs touching `rust/` or the sequence backend): fmt,
  clippy with warnings denied, core tests, a release wheel installed with equate, the
  differential + sequence + port-surface tests (fast; thorough on 3.13), the whole suite,
  cargo-deny. **Ran on this PR**: lint, deny and test (3.10, 3.13) green from
  `0563fb6` on (`lint` failed earlier on the skeleton's stub warnings, which the leaves
  removed). equate's own wads CI is green from the same commit (it had failed at
  collection while `tests/test_differential.py` imported hypothesis unconditionally).
- `.github/workflows/rust-release.yml`: the abi3 wheel matrix (manylinux x86_64 and
  aarch64, macOS arm64 and x86_64, Windows x64) + sdist, smoke-tested outside the
  checkout, published by PyPI trusted publishing on an `equate-rust-v*` tag or a manual
  dispatch with `publish`. **Not run**: `workflow_dispatch` works only once the workflow
  is on the default branch, and no tag was pushed. macOS x86_64 is cross-compiled on
  macos-14 and not smoke-tested. The trusted publisher for `equate-rust` must be
  configured on PyPI before the first tag.

## Not ported (and why)

- `_soft_dtw_forward` / `_soft_dtw_backward`: the soft-minimum goes through `exp`/`log`,
  whose libm results can differ between numpy and Rust in the last bits, so the parity
  of the other kernels (bit-identical) would become a tolerance; `soft_dtw_alignment` has
  no `backend=` seam; tslearn already exists as a soft-DTW backend. Candidate for a later
  step under a 1e-12 tolerance.
- `compare/string.py::levenshtein_distance`: no backend seam (adding a parameter changes
  a public signature); rapidfuzz (`equate[fuzzy]`) is the natural fast path there.

## Open questions

1. dtaidistance backend: `use_c=True` is not passed, so it runs dtaidistance's pure-Python
   code (0.5-0.8x numpy); and it returns 0.0 for a univariate-vs-2-D pair where numpy
   raises. Fix in a separate change (it would change that backend's behaviour).
2. Biopython backend raises on empty sequences and disagrees with numpy for positive gap
   scores (local mode).
3. The `equate[rust]` extra (deferred until `equate-rust` is on PyPI; regenerate
   `uv.lock` then).
4. A fused native DTW (costs + accumulate + traceback) would take DTW 1000² from 53x to
   roughly the kernel's 150x.

## Landing (owed)

This branch was made in a cloud session and is **not merged**. The package's own session
lands it (per `landing-a-branch`):

1. Rebuild locally (`cd rust && maturin develop --release --uv`), then run the whole
   suite and `HYPOTHESIS_PROFILE=thorough python -m pytest tests/test_differential.py -k
   rust`, and `python benchmarks/bench_port.py` on the Mac (ratios only; record them).
2. Run `rust-release.yml` by hand (after the merge, or from a temporary tag on a fork) to
   see every platform's wheel build and smoke test; check the macOS x86_64 runner choice.
3. Configure PyPI trusted publishing for `equate-rust`, publish it (TestPyPI first), then
   add the `equate[rust]` extra and regenerate `uv.lock`.
4. Squash-merge. Merging publishes equate (wads auto-bump): the CHANGELOG entry is under
   "Unreleased".

#### Final review of the fixes (`rust-reviewer`, fresh Opus subagent)

All six findings of the previous review confirmed resolved (the reviewer re-ran the
`MemoryError` probes and the routing tests). New: one major and three minors.

| finding | resolution |
|---|---|
| major B1: the alignment path grew with infallible `push` (32 B per step), so a long sequence against an empty one could abort after the tables allocated | fixed: `try_reserve_exact(n + m)` -> `Error::Alloc` |
| minor B1: the same in the DTW traceback | fixed: `try_reserve_exact(i + j + 1)` |
| minor X: PyO3 0.30.1's `PyTuple::new` panics (`PanicException`) if CPython cannot allocate the tuple, where numpy would raise `MemoryError` | **accepted and recorded**: PyO3-internal, reachable only after the O(nm) tables already fit in memory; returning the path as a fallibly allocated `(L, 2)` int64 array would avoid it, at the cost of converting it back to tuples in Python |
| minor X: `except OverflowError: pass` in the routing test hid which call raised | fixed: `pytest.raises` scoped to the overflowing input |

After these fixes: clippy 0 warnings, core tests 18 passed, whole suite 4,596 passed /
53 skipped, rust differential (fast + grids) green. No blocker or major remains open.

## Mac benchmark and workflow tightening (2026-11-11)

Re-run on the user's Mac (Apple M1 Max, arm64, CPython 3.12.1 GIL build, numpy 2.5.4), using the released `equate-rust` 0.1.0 macOS arm64 wheel from PyPI, `python benchmarks/bench_port.py --backends numpy rust`, median of 7. Ratios only; absolute times are not comparable with the cloud VM above.

| case | size | numpy | rust | ratio |
|---|---|---|---|---|
| dtw unconstrained | 100 x 100 | 5.72 ms | 48.8 µs | 117x |
| dtw unconstrained | 1000 x 1000 | 579 ms | 5.75 ms | 101x |
| dtw 10% band | 100 x 100 | 1.33 ms | 42.5 µs | 31x |
| dtw 10% band | 1000 x 1000 | 116 ms | 3.42 ms | 34x |
| subsequence_dtw | 10 x 100 | 581 µs | 17.7 µs | 33x |
| subsequence_dtw | 100 x 1000 | 57.3 ms | 492 µs | 116x |
| needleman_wunsch | 100 x 100 | 22.6 ms | 57.2 µs | 395x |
| needleman_wunsch | 1000 x 1000 | 2.3 s | 6.44 ms | 356x |
| smith_waterman | 100 x 100 | 22.9 ms | 53.8 µs | 425x |
| smith_waterman | 1000 x 1000 | 2.32 s | 7.28 ms | 319x |
| nw custom substitution | 100 x 100 | 22.4 ms | 864 µs | 26x |

The Mac ratios are the same order as the cloud VM's (21-619x); DTW is higher here because numpy is slower relative to the native kernel on this machine. `rust-release.yml` now publishes only from an `equate-rust-v*` tag (the `publish` input on manual dispatch is gone, so a manual run only builds and smoke-tests); the `version` job already fails a tag that differs from `rust/pyproject.toml`. `equate-rust` 0.1.0 was published to PyPI on 2026-11-10 (tag `equate-rust-v0.1.0`), so open question 3 (the `equate[rust]` extra) is addressed in the change that adds it.

