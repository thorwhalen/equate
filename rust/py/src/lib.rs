//! Binding crate: the private `equate_rust._core` extension module.
//!
//! Thin shell over `equate-sequence-core`: converts numpy arrays to and from the core's
//! matrices and maps its errors onto Python exceptions. Defaults and validation belong
//! to equate's facade (`equate.compare.sequence`); every argument here is required.
//!
//! Interpreter and threads: the kernels read the caller's numpy buffers in place (no
//! copy). They stay attached while they do, and the module declares `gil_used = true`.
//! That narrows the window for a data race rather than closing it: a thread inside a
//! numpy loop that released the GIL could still write a buffer a caller shares. The
//! facade's inputs are temporaries no other thread can reach, so only a direct caller of
//! this private module is exposed. Detaching (or running free-threaded) would widen the
//! window, and copying first costs about half the kernel time at 1000 x 1000. Kernels
//! whose inputs are copied into Rust first (`align_codes`) detach. Recorded in equate's
//! `docs/port-log.md`.
//!
//! Allocation: every buffer of input size is allocated fallibly, so an impossible size
//! is a `MemoryError` (as in numpy), never a process abort. Interrupts: the kernels do
//! not check for signals; Ctrl-C takes effect when a call returns (milliseconds at
//! 1000 x 1000, about 6 ms for the largest benchmarked alignment).

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
#[expect(
    clippy::match_same_arms,
    reason = "Shape is named on purpose; the wildcard covers variants added later"
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

/// The elements of an array view in logical (row-major) order, copied into a buffer
/// allocated fallibly (`MemoryError`, never a process abort).
fn try_copy<'a, T: Copy + 'a>(len: usize, items: impl Iterator<Item = &'a T>) -> PyResult<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| PyMemoryError::new_err(format!("cannot copy an array of {len} elements")))?;
    out.extend(items.copied());
    Ok(out)
}

/// The values of a 2-D array in row-major order: borrowed when the array is
/// C-contiguous (what the facade passes), copied otherwise.
fn row_major<'a>(array: &'a ArrayView2<'_, f64>) -> PyResult<Cow<'a, [f64]>> {
    match array.as_slice() {
        Some(slice) => Ok(Cow::Borrowed(slice)),
        None => Ok(Cow::Owned(try_copy(array.len(), array.iter())?)),
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
    let data = row_major(array)?;
    let view = MatrixView::new(&data, rows, cols).map_err(to_pyerr)?;
    Ok(kernel(view))
}

/// Native kernels of `equate.compare.sequence`; not part of any public API.
// gil_used = true: see the module docs (in-place reads of caller-owned numpy buffers)
#[pymodule(gil_used = true)]
mod _core {
    use equate_sequence_core::align::{self, MatchMismatch, Mode, Scoring};
    use equate_sequence_core::dtw::{self, Band, Start};
    use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;

    use super::{to_numpy, to_pyerr, try_copy, with_view};

    /// True in debug builds, so benchmarks can refuse them.
    #[pymodule_export]
    const DEBUG_BUILD: bool = cfg!(debug_assertions);

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
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts array arguments by value"
    )]
    fn subsequence_dtw_accumulate<'py>(
        py: Python<'py>,
        costs: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let acc = with_view(&costs.as_array(), dtw::subsequence_accumulate)?;
        to_numpy(py, acc.map_err(to_pyerr)?)
    }

    /// Warping path from cell `(i, j)` back through the padded accumulated costs `acc`.
    #[pyfunction]
    #[pyo3(signature = (acc, i, j, /, *, stop_row0))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts array arguments by value"
    )]
    fn traceback_dtw<'py>(
        py: Python<'py>,
        acc: PyReadonlyArray2<'py, f64>,
        i: usize,
        j: usize,
        stop_row0: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let start = if stop_row0 {
            Start::FirstRow
        } else {
            Start::Origin
        };
        let path =
            with_view(&acc.as_array(), |d| dtw::traceback(d, (i, j), start))?.map_err(to_pyerr)?;
        PyTuple::new(py, path)
    }

    /// Needleman-Wunsch (global) or Smith-Waterman (local) alignment of a score matrix.
    #[pyfunction]
    #[pyo3(signature = (scores, /, *, gap, local))]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts array arguments by value"
    )]
    fn align_scores<'py>(
        py: Python<'py>,
        scores: PyReadonlyArray2<'py, f64>,
        gap: f64,
        local: bool,
    ) -> PyResult<(f64, Bound<'py, PyTuple>)> {
        let scoring = scoring(gap, local);
        let al = with_view(&scores.as_array(), |s| align::align(&s, scoring))?.map_err(to_pyerr)?;
        Ok((al.score, PyTuple::new(py, al.path)?))
    }

    /// The same alignment for two code sequences scored by match / mismatch.
    #[pyfunction]
    #[pyo3(signature = (a, b, /, *, match_score, mismatch_score, gap, local))]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the facade's keyword-only call; PyO3 has no options struct for kwargs"
    )]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts array arguments by value"
    )]
    fn align_codes<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<'py, i64>,
        b: PyReadonlyArray1<'py, i64>,
        match_score: f64,
        mismatch_score: f64,
        gap: f64,
        local: bool,
    ) -> PyResult<(f64, Bound<'py, PyTuple>)> {
        // owned copies (O(n + m)), so the O(n m) work can run detached
        let (a, b) = (a.as_array(), b.as_array());
        let a = try_copy(a.len(), a.iter())?;
        let b = try_copy(b.len(), b.iter())?;
        let scoring = scoring(gap, local);
        let al = py
            .detach(|| {
                let sub = MatchMismatch {
                    a: &a,
                    b: &b,
                    matched: match_score,
                    mismatched: mismatch_score,
                };
                align::align(&sub, scoring)
            })
            .map_err(to_pyerr)?;
        Ok((al.score, PyTuple::new(py, al.path)?))
    }

    fn scoring(gap: f64, local: bool) -> Scoring {
        let mode = if local { Mode::Local } else { Mode::Global };
        Scoring { gap, mode }
    }
}
