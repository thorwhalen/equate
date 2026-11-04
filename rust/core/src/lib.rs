//! Sequence-alignment kernels of `equate`, reproducing its numpy backend exactly.
//!
//! Dynamic time warping (banded and subsequence) and Needleman-Wunsch / Smith-Waterman
//! alignment with a linear gap, on row-major `f64` matrices. Every comparison follows
//! the Python reference it replaces, including its tie-breaking and its handling of NaN
//! (see [`pyorder`]), so the paths are identical and the scores are bit-identical
//! (the same additions, in the same order).
//!
//! This crate has no Python types; `equate-rust-py` binds it as `equate_rust._core`.

pub mod align;
pub mod dtw;
mod error;
mod matrix;
pub mod pyorder;

pub use error::Error;
pub use matrix::{Matrix, MatrixView};
