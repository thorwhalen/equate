//! Dynamic time warping: banded and subsequence accumulated costs.
//!
//! Matrices follow equate's numpy reference: the accumulated-cost matrix `D` is padded
//! by one row and one column, `D[i + 1][j + 1]` being the cheapest warping path ending
//! at cell `(i, j)` of the `n x m` step-cost matrix `C`. Every cell is
//! `C[i][j] + min(D[i][j], D[i][j + 1], D[i + 1][j])` with Python's `min`
//! ([`py_min3`]), so the floating-point result is the reference's, bit for bit.

use std::ops::Range;

use crate::pyorder::py_min3;
use crate::{Matrix, MatrixView};

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
pub fn accumulate(costs: MatrixView<'_>, band: Band) -> Matrix {
    let (n, m) = (costs.rows(), costs.cols());
    let mut acc = Matrix::filled(n + 1, m + 1, f64::INFINITY);
    let data = acc.as_mut_slice();
    if let Some(origin) = data.first_mut() {
        *origin = 0.0;
    }
    fill(costs, data, |i| band.columns(i, n, m));
    acc
}

/// The padded subsequence-DTW accumulated costs of `costs`: unbanded, and the whole
/// first row is `0` (the match may start anywhere in the series).
pub fn subsequence_accumulate(costs: MatrixView<'_>) -> Matrix {
    let (n, m) = (costs.rows(), costs.cols());
    let mut acc = Matrix::filled(n + 1, m + 1, f64::INFINITY);
    let data = acc.as_mut_slice();
    if let Some(first_row) = data.get_mut(..=m) {
        first_row.fill(0.0);
    }
    fill(costs, data, |_| 0..m);
    acc
}

/// The DTW recurrence over the rows of `costs`, writing into the padded `acc` (row-major,
/// `costs.cols() + 1` wide), visiting in row `i` the columns `columns(i)`, in order.
#[expect(
    clippy::indexing_slicing,
    reason = "acc has (n + 1) x (m + 1) cells and columns(i) lies in 0..m, so every index is in bounds"
)]
fn fill(costs: MatrixView<'_>, acc: &mut [f64], columns: impl Fn(usize) -> Range<usize>) {
    let width = costs.cols() + 1;
    for (i, cost_row) in (0..costs.rows()).filter_map(|i| costs.row(i).map(|r| (i, r))) {
        let Range { start, end } = columns(i);
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
        let d = accumulate(view(&c, 2, 2), Band::Unconstrained);
        assert_eq!(
            d.into_vec(),
            vec![0.0, INF, INF, INF, 0.0, 1.0, INF, 1.0, 0.0]
        );
        // window 0 on a square matrix: the diagonal only
        let d = accumulate(view(&c, 2, 2), Band::HalfWidth(0));
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
        let d = accumulate(view(&c, 1, 2), Band::Unconstrained).into_vec();
        assert!(d[4].is_nan());
        assert_eq!(d[5], INF);
    }

    #[test]
    fn subsequence_starts_anywhere_on_the_first_row() {
        // query [1] in series [0, 1]: squared costs [[1, 0]]
        let c = [1.0, 0.0];
        let d = subsequence_accumulate(view(&c, 1, 2)).into_vec();
        assert_eq!(d, vec![0.0, 0.0, 0.0, INF, 1.0, 0.0]);
    }

    #[test]
    fn empty_costs_give_the_padding_only() {
        let d = accumulate(view(&[], 0, 3), Band::HalfWidth(2));
        assert_eq!(d.into_vec(), vec![0.0, INF, INF, INF]);
        let d = accumulate(view(&[], 2, 0), Band::Unconstrained);
        assert_eq!(d.into_vec(), vec![0.0, INF, INF]);
    }
}
