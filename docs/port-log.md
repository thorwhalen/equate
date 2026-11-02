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
