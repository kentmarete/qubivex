//! Explicit NEON twins of the portable row dots in `formats`.
//!
//! Lane mapping: accumulator vector `i`, lane `j` is portable lane `4i + j`. Each step
//! does the same `fma(w, x, acc)` per lane, and [`reduce`] is the same halving tree,
//! so results are bitwise equal to the portable kernels (tested in `lib.rs`).

#![allow(unsafe_code)]

use crate::formats::{LANES, Q4_K_BYTES, Q8_0_BYTES};
use core::arch::aarch64::*;
use half::f16;

const V: usize = LANES / 4;

#[inline]
#[target_feature(enable = "neon")]
fn reduce(mut a: [float32x4_t; V]) -> f32 {
    let mut w = V / 2;
    while w >= 1 {
        for i in 0..w {
            a[i] = vaddq_f32(a[i], a[i + w]);
        }
        w /= 2;
    }
    let v = a[0];
    (vgetq_lane_f32::<0>(v) + vgetq_lane_f32::<1>(v)) + (vgetq_lane_f32::<2>(v) + vgetq_lane_f32::<3>(v))
}

/// Loads 4 f32x4 of `x` at element offset `off` (16 floats).
///
/// # Safety
/// `off + 16 <= x.len()`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_x16(x: &[f32], off: usize) -> [float32x4_t; 4] {
    debug_assert!(off + 16 <= x.len());
    // SAFETY: the caller guarantees 16 floats are in bounds from `off`.
    unsafe {
        let p = x.as_ptr().add(off);
        [vld1q_f32(p), vld1q_f32(p.add(4)), vld1q_f32(p.add(8)), vld1q_f32(p.add(12))]
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn u8x16_to_f32(v: uint8x16_t) -> [float32x4_t; 4] {
    let lo = vmovl_u8(vget_low_u8(v));
    let hi = vmovl_high_u8(v);
    [
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
        vcvtq_f32_u32(vmovl_high_u16(lo)),
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
        vcvtq_f32_u32(vmovl_high_u16(hi)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn i8x16_to_f32(v: int8x16_t) -> [float32x4_t; 4] {
    let lo = vmovl_s8(vget_low_s8(v));
    let hi = vmovl_high_s8(v);
    [
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
        vcvtq_f32_s32(vmovl_high_s16(lo)),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
        vcvtq_f32_s32(vmovl_high_s16(hi)),
    ]
}

/// `acc[base..base+4] = fma(w, x, acc)` for 16 lanes.
#[inline]
#[target_feature(enable = "neon")]
fn fma4(acc: &mut [float32x4_t; V], base: usize, w: [float32x4_t; 4], x: [float32x4_t; 4]) {
    for i in 0..4 {
        acc[base + i] = vfmaq_f32(acc[base + i], w[i], x[i]);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

#[inline]
#[target_feature(enable = "neon")]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Q4_K row dot. `x.len() == row.len() / 144 * 256`.
#[target_feature(enable = "neon")]
fn q4k_dot_impl(row: &[u8], x: &[f32]) -> f32 {
    assert_eq!(row.len() / Q4_K_BYTES * 256, x.len());
    let mut acc = [vdupq_n_f32(0.0); V];
    let mut off = 0;
    for blk in row.chunks_exact(Q4_K_BYTES) {
        let d = f16_at(blk, 0);
        let dmin = f16_at(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for j in 0..4 {
            let (sc, m) = scale_min_k4(2 * j, scales);
            let (d1, m1) = (vdupq_n_f32(d * f32::from(sc)), vdupq_n_f32(dmin * f32::from(m)));
            let (sc, m) = scale_min_k4(2 * j + 1, scales);
            let (d2, m2) = (vdupq_n_f32(d * f32::from(sc)), vdupq_n_f32(dmin * f32::from(m)));
            // SAFETY: qs has 128 bytes; 32 * j + 32 <= 128.
            let (qa, qb) = unsafe { (vld1q_u8(qs.as_ptr().add(32 * j)), vld1q_u8(qs.as_ptr().add(32 * j + 16))) };
            let mask = vdupq_n_u8(0x0F);
            let parts = [
                (vandq_u8(qa, mask), d1, m1),
                (vandq_u8(qb, mask), d1, m1),
                (vshrq_n_u8::<4>(qa), d2, m2),
                (vshrq_n_u8::<4>(qb), d2, m2),
            ];
            for (p, (q, dd, mm)) in parts.into_iter().enumerate() {
                let qf = u8x16_to_f32(q);
                let mut w = qf;
                for v in &mut w {
                    *v = vsubq_f32(vmulq_f32(dd, *v), mm);
                }
                // SAFETY: off + 64 <= x.len() (asserted above, 256 per block).
                let xv = unsafe { load_x16(x, off + 16 * p) };
                fma4(&mut acc, 4 * p, w, xv);
            }
            off += 64;
        }
    }
    reduce(acc)
}

/// Q8_0 row dot (two 32-weight blocks per 64-lane chunk).
#[target_feature(enable = "neon")]
fn q8_0_dot_impl(row: &[u8], x: &[f32]) -> f32 {
    assert_eq!(row.len() / Q8_0_BYTES * 32, x.len());
    let mut acc = [vdupq_n_f32(0.0); V];
    for (c, pair) in row.chunks_exact(2 * Q8_0_BYTES).enumerate() {
        for (b, blk) in pair.chunks_exact(Q8_0_BYTES).enumerate() {
            let d = vdupq_n_f32(f16_at(blk, 0));
            for h in 0..2 {
                // SAFETY: blk has 34 bytes; 2 + 16 * h + 16 <= 34.
                let q = unsafe { vld1q_s8(blk.as_ptr().add(2 + 16 * h).cast()) };
                let mut w = i8x16_to_f32(q);
                for v in &mut w {
                    *v = vmulq_f32(*v, d);
                }
                let lane = 32 * b + 16 * h;
                // SAFETY: c * 64 + lane + 16 <= x.len() (asserted above).
                let xv = unsafe { load_x16(x, 64 * c + lane) };
                fma4(&mut acc, lane / 4, w, xv);
            }
        }
    }
    reduce(acc)
}

/// int8 row dot (scale applied by the caller).
#[target_feature(enable = "neon")]
fn i8_dot_impl(row: &[i8], x: &[f32]) -> f32 {
    assert_eq!(row.len(), x.len());
    assert_eq!(row.len() % LANES, 0);
    let mut acc = [vdupq_n_f32(0.0); V];
    for c in 0..row.len() / LANES {
        for p in 0..4 {
            let off = 64 * c + 16 * p;
            // SAFETY: off + 16 <= row.len() == x.len().
            let (q, xv) = unsafe { (vld1q_s8(row.as_ptr().add(off)), load_x16(x, off)) };
            fma4(&mut acc, 4 * p, i8x16_to_f32(q), xv);
        }
    }
    reduce(acc)
}

/// Safe entry point for `q4k_dot_impl`.
pub fn q4k_dot(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline, so the target feature is always present.
    unsafe { q4k_dot_impl(row, x) }
}

/// Safe entry point for `q8_0_dot_impl`.
pub fn q8_0_dot(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline, so the target feature is always present.
    unsafe { q8_0_dot_impl(row, x) }
}

/// Safe entry point for `i8_dot_impl`.
pub fn i8_dot(row: &[i8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline, so the target feature is always present.
    unsafe { i8_dot_impl(row, x) }
}
