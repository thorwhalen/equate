//! Dynamic time warping: banded and subsequence accumulated costs.
//!
//! Matrices follow equate's numpy reference: the accumulated-cost matrix `D` is padded
//! by one row and one column, `D[i + 1][j + 1]` being the cheapest warping path ending
//! at cell `(i, j)` of the `n x m` step-cost matrix `C`. Every cell is
//! `C[i][j] + min(D[i][j], D[i][j + 1], D[i + 1][j])` with Python's `min`
//! ([`py_min3`]), so the floating-point result is the reference's, bit for bit.

use std::ops::Range;

use crate::pyorder::py_min3;
use crate::{Error, Matrix, MatrixView};

/// A Sakoe-Chiba band: which cells of each row may be on the warping path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    /// Every cell (unconstrained DTW).
    Unconstrained,
    /// Cells within this many columns of the diagonal (inclusive), the band widened by
    /// `|n - m|` so that series of different lengths can always be aligned.
    HalfWidth(usize),
}

impl Band {
    /// The columns of row `i` inside the band, for an `n x m` cost matrix.
    ///
    /// The reference's `range(max(0, i - max(0, n - m) - w), min(m - 1, i + max(0, m - n) + w) + 1)`,
    /// in saturating unsigned arithmetic (each saturation is the reference's `max(0, ..)`
    /// or `min(..)`).
    pub fn columns(self, i: usize, n: usize, m: usize) -> Range<usize> {
        match self {
            Band::Unconstrained => 0..m,
            Band::HalfWidth(w) => {
                let lo = i.saturating_sub(n.saturating_sub(m)).saturating_sub(w);
                let hi = i.saturating_add(m.saturating_sub(n)).saturating_add(w);
                lo..hi.saturating_add(1).min(m)
            }
        }
    }
}

/// The padded `(n + 1) x (m + 1)` accumulated-cost matrix of `costs` inside `band`.
///
/// `D[0][0] = 0`; every other cell outside the band (and the padding) is `+inf`.
///
/// # Errors
///
/// [`Error::Alloc`] when the padded matrix cannot be allocated.
pub fn accumulate(costs: MatrixView<'_>, band: Band) -> Result<Matrix, Error> {
    let (n, m) = (costs.rows(), costs.cols());
    let mut acc = padded(n, m)?;
    let data = acc.as_mut_slice();
    if let Some(origin) = data.first_mut() {
        *origin = 0.0;
    }
    fill(costs, data, |i| band.columns(i, n, m));
    Ok(acc)
}

/// The `(n + 1) x (m + 1)` matrix of `+inf` every accumulation starts from.
fn padded(n: usize, m: usize) -> Result<Matrix, Error> {
    let alloc = || Error::Alloc { rows: n, cols: m };
    let rows = n.checked_add(1).ok_or_else(alloc)?;
    let cols = m.checked_add(1).ok_or_else(alloc)?;
    Matrix::filled(rows, cols, f64::INFINITY)
}

/// The padded subsequence-DTW accumulated costs of `costs`: unbanded, and the whole
/// first row is `0` (the match may start anywhere in the series).
///
/// # Errors
///
/// [`Error::Alloc`] when the padded matrix cannot be allocated.
pub fn subsequence_accumulate(costs: MatrixView<'_>) -> Result<Matrix, Error> {
    let (n, m) = (costs.rows(), costs.cols());
    let mut acc = padded(n, m)?;
    let data = acc.as_mut_slice();
    if let Some(first_row) = data.get_mut(..=m) {
        first_row.fill(0.0);
    }
    fill(costs, data, |_| 0..m);
    Ok(acc)
}

/// Where a traceback stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// At cell `(0, 0)`: both series are consumed from their first step (DTW).
    Origin,
    /// On the first row: the series may start anywhere (subsequence DTW).
    FirstRow,
}

/// The warping path ending at cell `end = (i, j)`, walked back through the padded
/// accumulated costs `acc`, in order from its first cell.
///
/// At each cell the candidates are, in this order, the diagonal `(i - 1, j - 1)`, up
/// `(i - 1, j)` and left `(i, j - 1)` (left is not offered on the first row when
/// `start` is [`Start::FirstRow`]); the reference's `min(candidates, key=cost)` keeps
/// the **first** candidate unless a later one is strictly cheaper, so ties go to the
/// diagonal, then up, and a NaN cost never wins over an earlier candidate.
///
/// # Errors
///
/// [`Error::OutOfBounds`] when `end` is not a cell of the unpadded matrix, i.e. when
/// `(i + 1, j + 1)` is outside `acc`.
/// [`Error::Alloc`] when the path cannot be allocated.
pub fn traceback(
    acc: MatrixView<'_>,
    end: (usize, usize),
    start: Start,
) -> Result<Vec<(usize, usize)>, Error> {
    let (mut i, mut j) = end;
    let out_of_bounds = Error::OutOfBounds {
        row: i,
        col: j,
        rows: acc.rows().saturating_sub(1),
        cols: acc.cols().saturating_sub(1),
    };
    if i.checked_add(1).is_none_or(|r| r >= acc.rows())
        || j.checked_add(1).is_none_or(|c| c >= acc.cols())
    {
        return Err(out_of_bounds);
    }
    // every read below is at (row <= i + 1, col <= j + 1) with i, j only decreasing,
    // so inside acc; `cost` still returns NaN rather than panic if that were wrong
    let cost = |row: usize, col: usize| acc.get(row, col).unwrap_or(f64::NAN);
    // at most i + j + 1 cells; reserved fallibly
    let mut path = Vec::new();
    path.try_reserve_exact(i.saturating_add(j).saturating_add(1))
        .map_err(|_| Error::Alloc { rows: i, cols: j })?;
    path.push((i, j));
    while (i, j) != (0, 0) {
        if i == 0 && start == Start::FirstRow {
            break;
        }
        let mut best: Option<(f64, usize, usize)> = None;
        #[expect(
            clippy::neg_cmp_op_on_partial_ord,
            reason = "Python's min replaces only when `value < current`; NaN never does"
        )]
        let mut offer = |value: f64, ni: usize, nj: usize| match best {
            Some((current, _, _)) if !(value < current) => {}
            _ => best = Some((value, ni, nj)),
        };
        if i > 0 && j > 0 {
            offer(cost(i, j), i - 1, j - 1); // D is padded: D[i][j] is cell (i - 1, j - 1)
        }
        if i > 0 {
            offer(cost(i, j + 1), i - 1, j);
        }
        if j > 0 && !(start == Start::FirstRow && i == 0) {
            offer(cost(i + 1, j), i, j - 1);
        }
        // (i, j) != (0, 0) and not (i == 0 with a first-row start): one candidate exists
        let Some((_, ni, nj)) = best else { break };
        (i, j) = (ni, nj);
        path.push((i, j));
    }
    path.reverse();
    Ok(path)
}

/// The DTW recurrence over the rows of `costs`, writing into the padded `acc` (row-major,
/// `costs.cols() + 1` wide), visiting in row `i` the columns `columns(i)`, in order.
#[expect(
    clippy::indexing_slicing,
    reason = "acc has (n + 1) x (m + 1) cells and columns(i) lies in 0..m, so every index is in bounds"
)]
fn fill(costs: MatrixView<'_>, acc: &mut [f64], columns: impl Fn(usize) -> Range<usize>) {
    let width = costs.cols() + 1;
    for i in 0..costs.rows() {
        let Range { start, end } = columns(i);
        // `row(i)` is `Some` for every i < rows; the `else` arm only avoids an unwrap
        let Some(cost_row) = costs.row(i) else { break };
        if start >= end {
            continue;
        }
        // prev = padded row i (D[i][..]), cur = padded row i + 1 (D[i + 1][..])
        let (head, tail) = acc.split_at_mut((i + 1) * width);
        let prev = &head[i * width..];
        let cur = &mut tail[..width];
        let mut left = cur[start]; // D[i + 1][j] for j = start
        for j in start..end {
            let value = cost_row[j] + py_min3(prev[j], prev[j + 1], left);
            cur[j + 1] = value;
            left = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INF: f64 = f64::INFINITY;

    fn view(data: &[f64], rows: usize, cols: usize) -> MatrixView<'_> {
        MatrixView::new(data, rows, cols).unwrap()
    }

    #[test]
    fn band_columns_follow_the_reference() {
        // n = m = 5, w = 1: a three-wide diagonal band
        let b = Band::HalfWidth(1);
        assert_eq!(b.columns(0, 5, 5), 0..2);
        assert_eq!(b.columns(2, 5, 5), 1..4);
        assert_eq!(b.columns(4, 5, 5), 3..5);
        // widened by |n - m|: n = 2, m = 5, w = 0
        assert_eq!(Band::HalfWidth(0).columns(0, 2, 5), 0..4);
        assert_eq!(Band::HalfWidth(0).columns(1, 2, 5), 1..5);
        // n = 5, m = 1, w = 0: one column, every row
        assert_eq!(Band::HalfWidth(0).columns(4, 5, 1), 0..1);
        // a huge half-width is the whole row, without overflow
        assert_eq!(Band::HalfWidth(usize::MAX).columns(3, 4, 7), 0..7);
        assert_eq!(Band::Unconstrained.columns(3, 4, 7), 0..7);
    }

    #[test]
    fn accumulate_matches_a_hand_computed_matrix() {
        // C = [[0, 1], [1, 0]]: D = [[0, inf, inf], [inf, 0, 1], [inf, 1, 0]]
        let c = [0.0, 1.0, 1.0, 0.0];
        let d = accumulate(view(&c, 2, 2), Band::Unconstrained).unwrap();
        assert_eq!(
            d.into_vec(),
            vec![0.0, INF, INF, INF, 0.0, 1.0, INF, 1.0, 0.0]
        );
        // window 0 on a square matrix: the diagonal only
        let d = accumulate(view(&c, 2, 2), Band::HalfWidth(0)).unwrap();
        assert_eq!(
            d.into_vec(),
            vec![0.0, INF, INF, INF, 0.0, INF, INF, INF, 0.0]
        );
    }

    #[test]
    fn accumulate_propagates_nan_like_python_min() {
        // C = [[nan, 0]]: D[1][1] = nan + min(0, inf, inf) = nan;
        // D[1][2] = 0 + min(inf, inf, nan) = 0 + inf (nan never replaces in Python's min)
        let c = [f64::NAN, 0.0];
        let d = accumulate(view(&c, 1, 2), Band::Unconstrained)
            .unwrap()
            .into_vec();
        assert!(d[4].is_nan());
        assert_eq!(d[5], INF);
    }

    #[test]
    fn subsequence_starts_anywhere_on_the_first_row() {
        // query [1] in series [0, 1]: squared costs [[1, 0]]
        let c = [1.0, 0.0];
        let d = subsequence_accumulate(view(&c, 1, 2)).unwrap().into_vec();
        assert_eq!(d, vec![0.0, 0.0, 0.0, INF, 1.0, 0.0]);
    }

    #[test]
    fn traceback_keeps_the_first_minimum() {
        // all-zero costs, 3 x 4: every path ties; the diagonal wins, then row 0
        let c = [0.0; 12];
        let d = accumulate(view(&c, 3, 4), Band::Unconstrained).unwrap();
        let path = traceback(d.view(), (2, 3), Start::Origin).unwrap();
        assert_eq!(path, vec![(0, 0), (0, 1), (1, 2), (2, 3)]);
        // a NaN first candidate (the diagonal) is kept: nothing compares below it
        let d = [0.0, INF, INF, INF, f64::NAN, 5.0, INF, 1.0, 9.0];
        let path = traceback(view(&d, 3, 3), (1, 1), Start::Origin).unwrap();
        assert_eq!(path, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn subsequence_traceback_stops_on_the_first_row() {
        let c = [1.0, 0.0, 0.0, 1.0]; // a 2 x 2 cost matrix
        let d = subsequence_accumulate(view(&c, 2, 2)).unwrap();
        let path = traceback(d.view(), (1, 1), Start::FirstRow).unwrap();
        assert_eq!(path.first().map(|p| p.0), Some(0));
        assert_eq!(path.last(), Some(&(1, 1)));
    }

    #[test]
    fn traceback_rejects_a_cell_outside_the_matrix() {
        let d = accumulate(view(&[0.0], 1, 1), Band::Unconstrained).unwrap();
        assert!(matches!(
            traceback(d.view(), (1, 0), Start::Origin),
            Err(Error::OutOfBounds { .. })
        ));
        assert!(traceback(d.view(), (0, usize::MAX), Start::Origin).is_err());
        assert_eq!(
            traceback(d.view(), (0, 0), Start::Origin).unwrap(),
            vec![(0, 0)]
        );
    }

    #[test]
    fn empty_costs_give_the_padding_only() {
        let d = accumulate(view(&[], 0, 3), Band::HalfWidth(2)).unwrap();
        assert_eq!(d.into_vec(), vec![0.0, INF, INF, INF]);
        let d = accumulate(view(&[], 2, 0), Band::Unconstrained).unwrap();
        assert_eq!(d.into_vec(), vec![0.0, INF, INF]);
    }
}
