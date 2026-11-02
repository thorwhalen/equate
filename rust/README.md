# equate-rust

Native (Rust) kernels for [`equate`](https://github.com/thorwhalen/equate)'s sequence
comparators. Install it next to equate and pass `backend="rust"`:

```python
from equate.compare.sequence import dtw_alignment, needleman_wunsch

dtw_alignment(a, b, backend="rust")          # banded exact DTW, with its warping path
needleman_wunsch("GATTACA", "GATACA", backend="rust")
```

It covers `dtw`, `subsequence_dtw`, `needleman_wunsch` and `smith_waterman` (and their
`*_alignment` twins). Results equal the numpy backend's: the same validation and errors,
the same paths (ties broken identically), scores equal in practice to the last bit (the
contract is 1e-12 relative). equate's `tests/test_differential.py` checks it.

The Python package `equate_rust` holds private kernels only; the public API is equate's.

## Layout

- `core/`: the pure-Rust crate `equate-sequence-core` (no Python types; `cargo test -p equate-sequence-core`).
- `py/`: the PyO3 binding crate, built by maturin as `equate_rust._core`.
- `python/equate_rust/`: the Python package (re-exports, the hand-written stub `_core.pyi`).

## Development

```bash
cd rust
maturin develop --release --uv     # into the active virtual environment
cargo test -p equate-sequence-core
cargo clippy --workspace --all-targets
cd .. && python -m pytest tests/test_differential.py tests/test_sequence.py
python benchmarks/bench_port.py
```
