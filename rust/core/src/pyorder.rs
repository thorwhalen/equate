//! Python's and numpy's orderings of `f64`, NaN included.
//!
//! The kernels must break ties and treat NaN exactly as the Python reference does, so
//! these helpers transcribe the reference's comparisons instead of using
//! [`f64::min`] (which ignores NaN) or [`Iterator::max_by`] (which keeps the *last*
//! maximum).

/// Python's `min(a, b, c)` on floats: keep the first value, replace it only by a later
/// one that is strictly smaller (`<`). A NaN first value is therefore kept, and a NaN
/// later value never replaces anything.
#[inline]
pub fn py_min3(a: f64, b: f64, c: f64) -> f64 {
    let mut best = a;
    if b < best {
        best = b;
    }
    if c < best {
        best = c;
    }
    best
}

/// numpy's `argmax` on a float array: the index of the first NaN if there is one, else
/// of the first maximum. `None` for an empty slice (numpy raises `ValueError`).
#[inline]
#[expect(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "numpy's `!(v <= best)` is true for a NaN `v`, which `v > best` is not"
)]
pub fn np_argmax(values: &[f64]) -> Option<usize> {
    let (&first, rest) = values.split_first()?;
    let (mut best, mut index) = (first, 0);
    if best.is_nan() {
        return Some(0);
    }
    for (k, &v) in rest.iter().enumerate() {
        // numpy: `if (!(*ip <= mp))`, i.e. a strictly greater value or a NaN
        if !(v <= best) {
            best = v;
            index = k + 1;
            if best.is_nan() {
                break;
            }
        }
    }
    Some(index)
}

/// numpy's `argmin` on a float array: the index of the first NaN if there is one, else
/// of the first minimum. `None` for an empty slice.
#[inline]
#[expect(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "numpy's `!(v >= best)` is true for a NaN `v`, which `v < best` is not"
)]
pub fn np_argmin(values: &[f64]) -> Option<usize> {
    let (&first, rest) = values.split_first()?;
    let (mut best, mut index) = (first, 0);
    if best.is_nan() {
        return Some(0);
    }
    for (k, &v) in rest.iter().enumerate() {
        if !(v >= best) {
            best = v;
            index = k + 1;
            if best.is_nan() {
                break;
            }
        }
    }
    Some(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAN: f64 = f64::NAN;

    #[test]
    fn py_min3_follows_python() {
        assert_eq!(py_min3(1.0, 0.0, 0.0).to_bits(), 0.0_f64.to_bits());
        assert!(py_min3(NAN, 0.0, -1.0).is_nan()); // min(nan, 0, -1) is nan in Python
        assert_eq!(py_min3(1.0, NAN, 0.0), 0.0); // min(1, nan, 0) == 0
        assert_eq!(py_min3(1.0, NAN, 2.0), 1.0);
        // ties keep the first: -0.0 first stays -0.0
        assert!(py_min3(-0.0, 0.0, 0.0).is_sign_negative());
        assert!(py_min3(0.0, -0.0, -0.0).is_sign_positive());
    }

    #[test]
    fn argmax_and_argmin_follow_numpy() {
        assert_eq!(np_argmax(&[]), None);
        assert_eq!(np_argmax(&[1.0, 3.0, 3.0]), Some(1)); // first maximum
        assert_eq!(np_argmax(&[1.0, NAN, 5.0, NAN]), Some(1)); // first NaN
        assert_eq!(np_argmax(&[NAN, 5.0]), Some(0));
        assert_eq!(np_argmax(&[0.0, -0.0]), Some(0));
        assert_eq!(np_argmax(&[f64::NEG_INFINITY, f64::NEG_INFINITY]), Some(0));
        assert_eq!(np_argmin(&[3.0, 1.0, 1.0]), Some(1));
        assert_eq!(np_argmin(&[3.0, NAN, 0.0]), Some(1));
        assert_eq!(np_argmin(&[f64::INFINITY, f64::INFINITY]), Some(0));
    }
}
