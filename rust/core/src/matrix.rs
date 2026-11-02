//! Row-major `f64` matrices: an owned [`Matrix`] and a borrowed [`MatrixView`].

use crate::Error;

/// A borrowed, row-major, `rows x cols` matrix of `f64`; the shape is checked once, at
/// construction.
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a> {
    data: &'a [f64],
    rows: usize,
    cols: usize,
}

impl<'a> MatrixView<'a> {
    /// Views `data` as a `rows x cols` matrix.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when `data.len() != rows * cols`.
    pub fn new(data: &'a [f64], rows: usize, cols: usize) -> Result<Self, Error> {
        let expected = rows.saturating_mul(cols);
        if data.len() == expected {
            Ok(Self { data, rows, cols })
        } else {
            Err(Error::Shape {
                rows,
                cols,
                expected,
                actual: data.len(),
            })
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The value at `(row, col)`, or `None` outside the matrix.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.rows && col < self.cols {
            self.data.get(row * self.cols + col).copied()
        } else {
            None
        }
    }

    /// Row `row` as a slice, or `None` outside the matrix.
    pub fn row(&self, row: usize) -> Option<&'a [f64]> {
        if row < self.rows {
            self.data.get(row * self.cols..(row + 1) * self.cols)
        } else {
            None
        }
    }

    /// The values in row-major order.
    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }
}

/// An owned, row-major, `rows x cols` matrix of `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// A `rows x cols` matrix with every cell equal to `value`.
    ///
    /// # Errors
    ///
    /// [`Error::Alloc`] when `rows * cols` overflows `usize` or the allocation fails
    /// (an infallible `vec!` would abort the process instead).
    pub fn filled(rows: usize, cols: usize, value: f64) -> Result<Self, Error> {
        let alloc = Error::Alloc { rows, cols };
        let len = rows.checked_mul(cols).ok_or_else(|| alloc.clone())?;
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| alloc)?;
        data.resize(len, value);
        Ok(Self { data, rows, cols })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// A borrowed view of the whole matrix.
    pub fn view(&self) -> MatrixView<'_> {
        MatrixView {
            data: &self.data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// The values in row-major order, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// The values in row-major order, consuming the matrix (no copy).
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_checks_its_shape() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = MatrixView::new(&data, 2, 3).unwrap();
        assert_eq!(v.get(1, 2), Some(6.0));
        assert_eq!(v.get(2, 0), None);
        assert_eq!(v.get(0, 3), None);
        assert_eq!(v.row(1), Some(&data[3..]));
        assert!(matches!(
            MatrixView::new(&data, 4, 2),
            Err(Error::Shape { expected: 8, .. })
        ));
        assert!(MatrixView::new(&[], usize::MAX, 2).is_err());
        assert!(MatrixView::new(&[], 0, 7).is_ok());
    }

    #[test]
    fn an_impossible_matrix_is_an_error_not_an_abort() {
        assert!(matches!(
            Matrix::filled(usize::MAX, 2, 0.0),
            Err(Error::Alloc { .. })
        ));
        assert!(matches!(
            Matrix::filled(1 << 40, 1 << 10, 0.0),
            Err(Error::Alloc { .. })
        ));
        assert_eq!(
            Matrix::filled(2, 1, 1.5).unwrap().into_vec(),
            vec![1.5, 1.5]
        );
    }
}
