//! The errors a caller of the kernels can cause.

/// What can go wrong when a kernel is called with inconsistent arguments.
///
/// The Python facade validates its inputs, so none of these reach an `equate` user;
/// they keep a direct caller of the binding from causing a panic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A matrix's buffer does not hold `rows * cols` values.
    #[error("a {rows} x {cols} matrix needs {expected} values, got {actual}")]
    Shape {
        /// Number of rows asked for.
        rows: usize,
        /// Number of columns asked for.
        cols: usize,
        /// `rows * cols`, or `usize::MAX` when that overflows.
        expected: usize,
        /// Length of the buffer given.
        actual: usize,
    },
    /// A cell index lies outside the matrix.
    #[error("cell ({row}, {col}) is outside a {rows} x {cols} matrix")]
    OutOfBounds {
        /// Row asked for.
        row: usize,
        /// Column asked for.
        col: usize,
        /// Number of rows of the matrix.
        rows: usize,
        /// Number of columns of the matrix.
        cols: usize,
    },
}
