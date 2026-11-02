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
