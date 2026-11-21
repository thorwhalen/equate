# equate-sequence-core

Sequence-alignment kernels of [equate](https://github.com/thorwhalen/equate): banded and subsequence dynamic time warping, and Needleman-Wunsch / Smith-Waterman alignment with a linear gap, on row-major `f64` matrices.

Every comparison reproduces the Python (numpy) reference exactly, including its tie-breaking and NaN handling. Pure Rust, no Python types.
