//! Binding crate: the private `equate_rust._core` extension module.
//!
//! Thin shell over `equate-sequence-core`: converts numpy arrays to and from the core's
//! matrices and maps its errors onto Python exceptions. Defaults and validation belong
//! to equate's facade (`equate.compare.sequence`); every argument here is required.
//!
//! Interpreter and threads: the kernels read the caller's numpy buffers in place (no
//! copy). They stay attached while they do, and the module declares `gil_used = true`,
//! so no other Python thread can write those buffers mid-read; detaching (or running
//! free-threaded) would make that a data race, and copying first costs about half the
//! kernel time at 1000 x 1000. Kernels whose inputs are copied into Rust first
//! (`align_codes`) detach. Recorded in equate's `docs/port-log.md`.

use std::borrow::Cow;

use equate_sequence_core::{Error, Matrix, MatrixView};
use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyIndexError, PyMemoryError, PyValueError};
use pyo3::prelude::*;

/// The one place core errors become Python exceptions (used as `.map_err(to_pyerr)`).
///
/// The facade never triggers these (it validates first); they only reach a direct
/// caller of `_core` who passed inconsistent arguments.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the shape `.map_err` takes: FnOnce(Error) -> PyErr"
)]
fn to_pyerr(err: Error) -> PyErr {
    match err {
        Error::Alloc { .. } => PyMemoryError::new_err(err.to_string()),
        Error::OutOfBounds { .. } => PyIndexError::new_err(err.to_string()),
        Error::Shape { .. } => PyValueError::new_err(err.to_string()),
        // `Error` is non_exhaustive: a variant added later is an inconsistent argument
        _ => PyValueError::new_err(err.to_string()),
    }
}

/// The values of a 2-D array in row-major order: borrowed when the array is
/// C-contiguous (what the facade passes), copied otherwise.
fn row_major<'a>(array: &'a ArrayView2<'_, f64>) -> Cow<'a, [f64]> {
    match array.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(array.iter().copied().collect()),
    }
}

/// A core matrix as a new numpy array (the buffer moves, no copy).
fn to_numpy(py: Python<'_>, matrix: Matrix) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let shape = (matrix.rows(), matrix.cols());
    let array = Array2::from_shape_vec(shape, matrix.into_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

/// Run `kernel` on a numpy matrix viewed as a core [`MatrixView`].
fn with_view<T>(
    array: &ArrayView2<'_, f64>,
    kernel: impl FnOnce(MatrixView<'_>) -> T,
) -> PyResult<T> {
    let (rows, cols) = array.dim();
    let data = row_major(array);
    let view = MatrixView::new(&data, rows, cols).map_err(to_pyerr)?;
    Ok(kernel(view))
}

/// Native kernels of `equate.compare.sequence`; not part of any public API.
// gil_used = true: see the module docs (in-place reads of caller-owned numpy buffers)
#[pymodule(gil_used = true)]
mod _core {
    use equate_sequence_core::dtw::{self, Band};
    use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyNotImplementedError;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;

    use super::{to_numpy, to_pyerr, with_view};

    /// True in debug builds, so benchmarks can refuse them.
    #[pymodule_export]
    const DEBUG_BUILD: bool = cfg!(debug_assertions);

    fn not_yet(name: &str) -> PyErr {
        PyNotImplementedError::new_err(format!("equate_rust._core.{name} is not ported yet"))
    }

    /// Padded DTW accumulated-cost matrix of `costs` inside a Sakoe-Chiba band.
    #[pyfunction]
    #[pyo3(signature = (costs, /, *, window))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts array arguments by value"
    )]
    fn dtw_accumulate<'py>(
        py: Python<'py>,
        costs: PyReadonlyArray2<'py, f64>,
        window: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let band = window.map_or(Band::Unconstrained, Band::HalfWidth);
        let acc = with_view(&costs.as_array(), |c| dtw::accumulate(c, band))?;
        to_numpy(py, acc.map_err(to_pyerr)?)
    }

    /// Padded subsequence-DTW accumulated-cost matrix (free start on the first row).
    #[pyfunction]
    #[pyo3(signature = (costs, /))]
    fn subsequence_dtw_accumulate<'py>(
        costs: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _ = costs;
        Err(not_yet("subsequence_dtw_accumulate"))
    }

    /// Warping path from cell `(i, j)` back through the padded accumulated costs `acc`.
    #[pyfunction]
    #[pyo3(signature = (acc, i, j, /, *, stop_row0))]
    fn traceback_dtw<'py>(
        acc: PyReadonlyArray2<'py, f64>,
        i: usize,
        j: usize,
        stop_row0: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = (acc, i, j, stop_row0);
        Err(not_yet("traceback_dtw"))
    }

    /// Needleman-Wunsch (global) or Smith-Waterman (local) alignment of a score matrix.
    #[pyfunction]
    #[pyo3(signature = (scores, /, *, gap, local))]
    fn align_scores<'py>(
        scores: PyReadonlyArray2<'py, f64>,
        gap: f64,
        local: bool,
    ) -> PyResult<(f64, Bound<'py, PyTuple>)> {
        let _ = (scores, gap, local);
        Err(not_yet("align_scores"))
    }

    /// The same alignment for two code sequences scored by match / mismatch.
    #[pyfunction]
    #[pyo3(signature = (a, b, /, *, match_score, mismatch_score, gap, local))]
    fn align_codes<'py>(
        a: PyReadonlyArray1<'py, i64>,
        b: PyReadonlyArray1<'py, i64>,
        match_score: f64,
        mismatch_score: f64,
        gap: f64,
        local: bool,
    ) -> PyResult<(f64, Bound<'py, PyTuple>)> {
        let _ = (a, b, match_score, mismatch_score, gap, local);
        Err(not_yet("align_codes"))
    }
}
