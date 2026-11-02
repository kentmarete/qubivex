//! Elementwise and attention operations, in f32 with fixed evaluation order.
//!
//! Every function computes each output from its inputs in one fixed order, so a
//! token's result does not depend on how many other tokens are processed with it.

/// RMSNorm in place: `x · w / sqrt(mean(x²) + eps)`. The mean is accumulated in f64.
pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f64) {
    debug_assert_eq!(x.len(), w.len());
    #[allow(clippy::cast_precision_loss)]
    let n = x.len() as f64;
    let ms: f64 = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / n;
    #[allow(clippy::cast_possible_truncation)]
    let r = (1.0 / (ms + eps).sqrt()) as f32;
    for (v, g) in x.iter_mut().zip(w) {
        *v = (*v * r) * g;
    }
}

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

    /// Rotates one head in place at position `pos` (NeoX layout: pairs `i` and `i + d/2`).
    pub fn apply(&self, head: &mut [f32], pos: usize) {
        let half = self.inv_freq.len();
        debug_assert_eq!(head.len(), 2 * half);
        #[allow(clippy::cast_precision_loss)]
        let p = pos as f32;
        for i in 0..half {
            let a = p * self.inv_freq[i];
            let (s, c) = a.sin_cos();
            let (x1, x2) = (head[i], head[i + half]);
            head[i] = x1 * c - x2 * s;
            head[i + half] = x2 * c + x1 * s;
        }
    }
}

/// `silu(g) · u`, in place into `g`.
pub fn swiglu(g: &mut [f32], u: &[f32]) {
    for (a, b) in g.iter_mut().zip(u) {
        let s = *a / (1.0 + (-*a).exp());
        *a = s * b;
    }
}

/// Softmax in place (max-subtracted, sequential sum).
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Sequential f32 dot product.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |s, (x, y)| x.mul_add(*y, s))
}

/// Causal attention of one query head over positions `0..=pos` of one KV head.
/// `k` and `v` hold `head_dim` values per position. `scores` is scratch of at
/// least `pos + 1`.
pub fn attend(q: &[f32], k: &[f32], v: &[f32], pos: usize, out: &mut [f32], scores: &mut Vec<f32>) {
    let d = q.len();
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0 / (d as f32).sqrt();
    scores.clear();
    scores.extend((0..=pos).map(|t| dot(q, &k[t * d..(t + 1) * d]) * scale));
    softmax(scores);
    out.fill(0.0);
    for (t, p) in scores.iter().enumerate() {
        for (o, x) in out.iter_mut().zip(&v[t * d..(t + 1) * d]) {
            *o = p.mul_add(*x, *o);
        }
    }
}

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
