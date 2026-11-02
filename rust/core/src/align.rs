//! Needleman-Wunsch (global) and Smith-Waterman (local) alignment with a linear gap.
//!
//! The reference is equate's `_align_dp`: a fill that records back-pointers, then a
//! traceback that follows them (never a recomputed float equality). Each cell takes
//! the **first** maximum of `(diagonal, up, left)` as numpy's `argmax` does (the first
//! NaN, if any, wins); in local mode a value `<= 0` is floored to `0` and stops the
//! path (a NaN is not `<= 0`, so it is kept); the local end cell is the first maximum
//! of the whole matrix in row-major order. The arithmetic is the reference's, in the
//! same order, so scores are bit-identical.

use crate::Error;
use crate::matrix::try_filled;
use crate::pyorder::np_argmax;

/// Global or local alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Needleman-Wunsch: align both sequences entirely.
    Global,
    /// Smith-Waterman: the best-scoring pair of fragments.
    Local,
}

/// The scoring of an alignment besides the substitution scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scoring {
    /// Score added for each gap step (one skipped element); usually negative.
    pub gap: f64,
    /// Global or local.
    pub mode: Mode,
}

/// One step of an alignment path: `(Some(i), Some(j))` aligns `a[i]` with `b[j]`;
/// `(Some(i), None)` skips `a[i]` and `(None, Some(j))` skips `b[j]` (gaps).
pub type Step = (Option<usize>, Option<usize>);

/// An alignment's score and its path, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Alignment {
    /// The score of the reported path (`H` at its end cell).
    pub score: f64,
    /// The aligned steps, from the first to the last.
    pub path: Vec<Step>,
}

/// The substitution scores of an `rows x cols` alignment problem, one row at a time.
pub trait Substitution {
    /// Number of elements of the first sequence.
    fn rows(&self) -> usize;
    /// Number of elements of the second sequence.
    fn cols(&self) -> usize;
    /// Writes the scores of row `i` (`a[i]` against every `b[j]`) into `out`, which has
    /// `cols()` cells (cells beyond either length are left as they are).
    fn fill_row(&self, i: usize, out: &mut [f64]);
}

impl Substitution for crate::MatrixView<'_> {
    fn rows(&self) -> usize {
        crate::MatrixView::rows(self)
    }

    fn cols(&self) -> usize {
        crate::MatrixView::cols(self)
    }

    fn fill_row(&self, i: usize, out: &mut [f64]) {
        if let Some(row) = self.row(i) {
            for (cell, &value) in out.iter_mut().zip(row) {
                *cell = value;
            }
        }
    }
}

/// Two code sequences scored `matched` where the codes are equal and `mismatched`
/// elsewhere: the reference's default `match if x == y else mismatch` on elements
/// interned to integer codes.
#[derive(Debug, Clone, Copy)]
pub struct MatchMismatch<'a> {
    /// Codes of the first sequence.
    pub a: &'a [i64],
    /// Codes of the second sequence.
    pub b: &'a [i64],
    /// Score of two equal elements.
    pub matched: f64,
    /// Score of two different elements.
    pub mismatched: f64,
}

impl Substitution for MatchMismatch<'_> {
    fn rows(&self) -> usize {
        self.a.len()
    }

    fn cols(&self) -> usize {
        self.b.len()
    }

    fn fill_row(&self, i: usize, out: &mut [f64]) {
        if let Some(&x) = self.a.get(i) {
            for (cell, &y) in out.iter_mut().zip(self.b) {
                *cell = if x == y {
                    self.matched
                } else {
                    self.mismatched
                };
            }
        }
    }
}

/// A back-pointer: the move that produced a cell, in `argmax` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Move {
    Diagonal,
    Up,
    Left,
    Stop,
}

/// Aligns the two sequences described by `sub` under `scoring`.
///
/// # Errors
///
/// [`Error::Alloc`] when the `(n + 1) x (m + 1)` tables cannot be allocated.
#[expect(
    clippy::indexing_slicing,
    reason = "table and moves have (n + 1) x (m + 1) cells and every index below is i * width + j with i <= n, j <= m"
)]
pub fn align<S: Substitution + ?Sized>(sub: &S, scoring: Scoring) -> Result<Alignment, Error> {
    let (n, m) = (sub.rows(), sub.cols());
    let alloc = || Error::Alloc { rows: n, cols: m };
    let height = n.checked_add(1).ok_or_else(alloc)?;
    let width = m.checked_add(1).ok_or_else(alloc)?;
    let local = scoring.mode == Mode::Local;
    let gap = scoring.gap;
    let mut table = try_filled(height, width, 0.0_f64)?;
    let mut moves = try_filled(height, width, Move::Stop)?;
    if !local {
        for i in 1..=n {
            table[i * width] = table[(i - 1) * width] + gap;
            moves[i * width] = Move::Up;
        }
        for j in 1..=m {
            table[j] = table[j - 1] + gap;
            moves[j] = Move::Left;
        }
    }
    let mut scores = try_filled(1, m, 0.0_f64)?;
    for i in 1..=n {
        sub.fill_row(i - 1, &mut scores);
        let (head, tail) = table.split_at_mut(i * width);
        let prev = &head[(i - 1) * width..];
        let cur = &mut tail[..width];
        let row_moves = &mut moves[i * width..(i + 1) * width];
        let mut left = cur[0];
        for j in 1..=m {
            let candidates = [prev[j - 1] + scores[j - 1], prev[j] + gap, left + gap];
            let k = np_argmax(&candidates).unwrap_or(0);
            let (value, step) = if local && candidates[k] <= 0.0 {
                (0.0, Move::Stop)
            } else {
                let step = match k {
                    0 => Move::Diagonal,
                    1 => Move::Up,
                    _ => Move::Left,
                };
                (candidates[k], step)
            };
            cur[j] = value;
            row_moves[j] = step;
            left = value;
        }
    }
    let (mut i, mut j) = if local {
        let best = np_argmax(&table).unwrap_or(0);
        (best / width, best % width)
    } else {
        (n, m)
    };
    let score = table[i * width + j];
    let mut path = Vec::new();
    loop {
        match moves[i * width + j] {
            Move::Stop => break,
            Move::Diagonal => {
                path.push((Some(i - 1), Some(j - 1)));
                (i, j) = (i - 1, j - 1);
            }
            Move::Up => {
                path.push((Some(i - 1), None));
                i -= 1;
            }
            Move::Left => {
                path.push((None, Some(j - 1)));
                j -= 1;
            }
        }
    }
    path.reverse();
    Ok(Alignment { score, path })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MatrixView;

    fn codes(s: &str) -> Vec<i64> {
        s.chars().map(|c| i64::from(u32::from(c))).collect()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "a test helper taking the reference's parameters"
    )]
    fn nw(a: &str, b: &str, matched: f64, mismatched: f64, gap: f64, mode: Mode) -> Alignment {
        let (a, b) = (codes(a), codes(b));
        let sub = MatchMismatch {
            a: &a,
            b: &b,
            matched,
            mismatched,
        };
        align(&sub, Scoring { gap, mode }).unwrap()
    }

    #[test]
    fn global_alignment_matches_the_reference() {
        let al = nw("GATTACA", "GATACA", 1.0, -1.0, -1.0, Mode::Global);
        assert_eq!(al.score, 5.0);
        let al = nw("ACT", "AT", 1.0, -1.0, -1.0, Mode::Global);
        assert_eq!(
            al.path,
            vec![(Some(0), Some(0)), (Some(1), None), (Some(2), Some(1))]
        );
        // ties: diagonal beats up
        let al = nw("AA", "A", 1.0, -1.0, -1.0, Mode::Global);
        assert_eq!(al.path, vec![(Some(0), None), (Some(1), Some(0))]);
    }

    #[test]
    fn local_alignment_matches_the_reference() {
        let al = nw("xxABCyy", "zABCz", 1.0, -1.0, -1.0, Mode::Local);
        assert_eq!(al.score, 3.0);
        assert_eq!(
            al.path,
            vec![(Some(2), Some(1)), (Some(3), Some(2)), (Some(4), Some(3))]
        );
        let al = nw("abc", "xyz", 1.0, -1.0, -1.0, Mode::Local);
        assert_eq!((al.score, al.path.len()), (0.0, 0));
        // the first maximal cell, row-major
        let al = nw("aXa", "a", 1.0, -1.0, -1.0, Mode::Local);
        assert_eq!(al.path, vec![(Some(0), Some(0))]);
    }

    #[test]
    fn empty_sequences() {
        let al = nw("", "abc", 1.0, -1.0, -1.0, Mode::Global);
        assert_eq!(al.score, -3.0);
        assert_eq!(
            al.path,
            vec![(None, Some(0)), (None, Some(1)), (None, Some(2))]
        );
        let al = nw("", "", 1.0, -1.0, -1.0, Mode::Local);
        assert_eq!((al.score, al.path.len()), (0.0, 0));
    }

    #[test]
    fn a_score_matrix_and_codes_agree() {
        let (a, b) = (codes("kitten"), codes("sitting"));
        let s: Vec<f64> = a
            .iter()
            .flat_map(|x| b.iter().map(move |y| if x == y { 2.0 } else { -0.5 }))
            .collect();
        let view = MatrixView::new(&s, a.len(), b.len()).unwrap();
        for mode in [Mode::Global, Mode::Local] {
            let scoring = Scoring {
                gap: -1.0 / 3.0,
                mode,
            };
            let by_codes = MatchMismatch {
                a: &a,
                b: &b,
                matched: 2.0,
                mismatched: -0.5,
            };
            assert_eq!(align(&view, scoring), align(&by_codes, scoring));
        }
    }

    /// A problem too large to allocate: its size overflows `usize`.
    struct Huge(usize, usize);

    impl Substitution for Huge {
        fn rows(&self) -> usize {
            self.0
        }
        fn cols(&self) -> usize {
            self.1
        }
        fn fill_row(&self, _: usize, _: &mut [f64]) {}
    }

    #[test]
    fn an_unallocatable_table_is_an_error_not_an_abort() {
        let scoring = Scoring {
            gap: -1.0,
            mode: Mode::Global,
        };
        for huge in [
            Huge(usize::MAX, 1),
            Huge(1, usize::MAX),
            Huge(1 << 40, 1 << 30),
        ] {
            assert!(matches!(align(&huge, scoring), Err(Error::Alloc { .. })));
        }
    }

    #[test]
    fn nan_is_kept_not_floored() {
        // local, NaN gap: every gap move is NaN; argmax picks the first NaN
        let al = nw("ab", "b", 1.0, -1.0, f64::NAN, Mode::Local);
        assert!(al.score.is_nan());
    }
}
