//! Single-token attention decode: `H` heads × `D` dims over `T` cached positions.
//! K and V are stored head-major, `[H][T][D]` f32.

use crate::formats::dot_f32;
use candle_core::{Device, Result, Tensor};
use rayon::prelude::*;
use std::cell::RefCell;

pub struct AttnData {
    pub h: usize,
    pub d: usize,
    pub t: usize,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl AttnData {
    pub fn kv_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 4
    }
}

thread_local! {
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Own kernel: per head, fixed-order scores, max, exp-sum and V accumulation.
pub fn own(a: &AttnData, out: &mut [f32]) {
    let (d, t) = (a.d, a.t);
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0 / (d as f32).sqrt();
    out.par_chunks_mut(d).enumerate().for_each(|(h, o)| {
        let q = &a.q[h * d..(h + 1) * d];
        let kh = &a.k[h * t * d..(h + 1) * t * d];
        let vh = &a.v[h * t * d..(h + 1) * t * d];
        SCORES.with(|cell| {
            let mut s = cell.borrow_mut();
            s.resize(t, 0.0);
            let mut max = f32::NEG_INFINITY;
            for (sv, kr) in s.iter_mut().zip(kh.chunks_exact(d)) {
                *sv = dot_f32(q, kr) * scale;
                max = max.max(*sv);
            }
            let mut sum = 0f32;
            for sv in s.iter_mut() {
                *sv = (*sv - max).exp();
                sum += *sv;
            }
            o.fill(0.0);
            for (&p, vr) in s.iter().zip(vh.chunks_exact(d)) {
                for (ov, &vv) in o.iter_mut().zip(vr) {
                    *ov = p.mul_add(vv, *ov);
                }
            }
            let inv = 1.0 / sum;
            for ov in o.iter_mut() {
                *ov *= inv;
            }
        });
    });
}

pub struct CandleAttn {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: f64,
}

impl CandleAttn {
    pub fn new(a: &AttnData) -> Result<Self> {
        let dev = Device::Cpu;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (a.d as f64).sqrt();
        Ok(Self {
            q: Tensor::from_slice(&a.q, (a.h, 1, a.d), &dev)?,
            k: Tensor::from_slice(&a.k, (a.h, a.t, a.d), &dev)?,
            v: Tensor::from_slice(&a.v, (a.h, a.t, a.d), &dev)?,
            scale,
        })
    }

    /// `softmax(q·kᵀ·scale)·v`, as `candle-transformers` attention does it.
    pub fn forward(&self) -> Result<Tensor> {
        let s = (self.q.matmul(&self.k.t()?)? * self.scale)?;
        let p = candle_nn::ops::softmax_last_dim(&s)?;
        p.matmul(&self.v)
    }
}

/// f64 reference.
pub fn reference(a: &AttnData) -> Vec<f64> {
    let (d, t) = (a.d, a.t);
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0f64; a.h * d];
    for h in 0..a.h {
        let q = &a.q[h * d..(h + 1) * d];
        let s: Vec<f64> = (0..t)
            .map(|i| {
                let kr = &a.k[(h * t + i) * d..(h * t + i + 1) * d];
                q.iter().zip(kr).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum::<f64>() * scale
            })
            .collect();
        let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = s.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        for (i, p) in e.iter().enumerate() {
            let vr = &a.v[(h * t + i) * d..(h * t + i + 1) * d];
            for (o, &vv) in out[h * d..(h + 1) * d].iter_mut().zip(vr) {
                *o += p / sum * f64::from(vv);
            }
        }
    }
    out
}
