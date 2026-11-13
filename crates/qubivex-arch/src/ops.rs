//! Elementwise and attention operations, in f32 with fixed evaluation order.
//!
//! Every function computes each output from its inputs in one fixed order, so a
//! token's result does not depend on how many other tokens are processed with it.
//! Only IEEE-754 basic operations, fused multiply-add and the portable
//! [`qubivex_compute::math::exp`] are used, so the bits are the same on every
//! platform and on a GPU that evaluates them in the same order.

/// RMSNorm in place and its factor (defined in `qubivex-compute`, so a device backend
/// can give the same bits without depending on this crate).
pub use qubivex_compute::math::{rms_norm, rms_scale};

/// RoPE tables for `rotary_dim` dimensions: `cos` and `sin` of `pos · θ^(−2i/d)`.
#[derive(Clone, Debug)]
pub struct Rope {
    inv_freq: Vec<f32>,
}

impl Rope {
    /// Inverse frequencies as `transformers` computes them (f32).
    #[must_use]
    pub fn new(head_dim: usize, theta: f64) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let inv_freq = (0..head_dim / 2)
            .map(|i| {
                let e = (2 * i) as f32 / head_dim as f32;
                1.0 / (theta as f32).powf(e)
            })
            .collect();
        Self { inv_freq }
    }

    /// Pairs per head (`head_dim / 2`).
    #[must_use]
    pub fn half(&self) -> usize {
        self.inv_freq.len()
    }

    /// `cos` and `sin` of each pair's angle at `pos`, into `cs` as `[cos…, sin…]`
    /// (`2 · half` values).
    ///
    /// The angle is the f32 product `pos · inv_freq` (as `transformers` computes it);
    /// its sine and cosine are evaluated in f64 and rounded once, so the table is the
    /// correctly rounded value on every platform except in rare near-tie cases, rather
    /// than whatever the platform's `sinf` returns. Device kernels take this table
    /// instead of evaluating trigonometry themselves.
    pub fn table(&self, pos: usize, cs: &mut [f32]) {
        let half = self.inv_freq.len();
        debug_assert_eq!(cs.len(), 2 * half);
        #[allow(clippy::cast_precision_loss)]
        let p = pos as f32;
        let (c, s) = cs.split_at_mut(half);
        for i in 0..half {
            let a = f64::from(p * self.inv_freq[i]);
            #[allow(clippy::cast_possible_truncation)]
            {
                c[i] = a.cos() as f32;
                s[i] = a.sin() as f32;
            }
        }
    }

    /// Rotates one head in place with a [`Rope::table`] (NeoX layout: pairs `i` and
    /// `i + d/2`).
    pub fn rotate(head: &mut [f32], cs: &[f32]) {
        qubivex_compute::math::rope_rotate(head, cs);
    }

    /// Rotates one head in place at position `pos`.
    pub fn apply(&self, head: &mut [f32], pos: usize) {
        let mut cs = vec![0f32; 2 * self.half()];
        self.table(pos, &mut cs);
        Self::rotate(head, &cs);
    }
}

/// `silu(g) · u`, in place into `g` (defined in `qubivex-compute`, so a device backend
/// can give the same bits without depending on this crate).
pub use qubivex_compute::math::swiglu;

/// Softmax and top-k routing (defined in `qubivex-compute`, so a device backend can be
/// checked against them without depending on this crate).
pub use qubivex_compute::math::{route, softmax};

/// Causal attention of one query head (defined in `qubivex-compute`, so a device
/// backend can give the same bits without depending on this crate).
pub use qubivex_compute::math::attend;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit() {
        let mut x = vec![3.0f32, 4.0];
        rms_norm(&mut x, &[1.0, 2.0], 0.0);
        let r = (12.5f64).sqrt();
        assert!((f64::from(x[0]) - 3.0 / r).abs() < 1e-6);
        assert!((f64::from(x[1]) - 8.0 / r).abs() < 1e-6);
    }

    #[test]
    fn rope_at_zero_is_identity_and_preserves_norm() {
        let r = Rope::new(8, 10000.0);
        let mut h: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
        let orig = h.clone();
        r.apply(&mut h, 0);
        assert_eq!(h, orig);
        r.apply(&mut h, 17);
        let n0: f32 = orig.iter().map(|v| v * v).sum();
        let n1: f32 = h.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-4);
    }

    #[test]
    fn attention_of_one_position_is_its_value() {
        let (q, k, v) = (
            vec![1.0f32; 4],
            vec![0.5f32; 4],
            vec![2.0f32, -1.0, 0.0, 3.0],
        );
        let mut out = vec![0f32; 4];
        attend(&q, &k, &v, 0, &mut out, &mut Vec::new());
        assert_eq!(out, v);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0, 1000.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[3] > 0.999);
    }
}
