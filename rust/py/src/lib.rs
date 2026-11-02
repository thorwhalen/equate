//! Binding crate: the private `equate_rust._core` extension module.
//!
//! Thin shell over `equate-sequence-core`: converts numpy arrays to and from the core's
//! matrices, maps its errors onto Python exceptions, and releases the interpreter
//! (`py.detach`) around the O(n m) work. Defaults and validation belong to equate's
//! facade (`equate.compare.sequence`); every argument here is required.

use pyo3::prelude::*;

/// Native kernels of `equate.compare.sequence`; not part of any public API.
#[pymodule]
mod _core {
    use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyNotImplementedError;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;

    /// True in debug builds, so benchmarks can refuse them.
    #[pymodule_export]
    const DEBUG_BUILD: bool = cfg!(debug_assertions);

    fn not_yet(name: &str) -> PyErr {
        PyNotImplementedError::new_err(format!("equate_rust._core.{name} is not ported yet"))
    }

    /// Padded DTW accumulated-cost matrix of `costs` inside a Sakoe-Chiba band.
    #[pyfunction]
    #[pyo3(signature = (costs, /, *, window))]
    fn dtw_accumulate<'py>(
        costs: PyReadonlyArray2<'py, f64>,
        window: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _ = (costs, window);
        Err(not_yet("dtw_accumulate"))
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
