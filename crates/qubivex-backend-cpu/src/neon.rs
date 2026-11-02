//! Fused NEON row dots for aarch64.
//!
//! Each kernel computes, bit for bit, what [`crate::formats::dequant_row`] followed by
//! [`crate::reduce::dot`] computes:
//! - the same dequantized value per weight (the same operations, the same rounding: a
//!   separate multiply and subtract, never fused);
//! - the same fused multiply-add into the same lane (element `i` of each 64-element
//!   chunk goes to lane `i`; accumulator vector `j` holds lanes `4j..4j+4`);
//! - the same halving-tree reduction.
//!
//! So switching between these and the portable path never changes results. The
//! `matmul` tests compare them directly.

#![allow(clippy::cast_possible_wrap)]

use core::arch::aarch64::{
    float32x4_t, int8x16_t, uint8x16_t, vaddq_f32, vandq_u8, vcvtq_f32_s32, vdupq_n_f32,
    vdupq_n_s8, vdupq_n_u8, vfmaq_f32, vget_low_s8, vget_low_s16, vget_low_u16, vgetq_lane_f32,
    vld1q_f32, vld1q_u8, vld1q_u16, vmovl_high_s8, vmovl_high_s16, vmovl_high_u16, vmovl_s8,
    vmovl_s16, vmovl_u16, vmulq_f32, vorrq_u8, vreinterpretq_f32_u32, vreinterpretq_s8_u8,
    vshlq_n_u8, vshlq_n_u32, vshrq_n_u8, vsubq_f32, vsubq_s8,
};

use crate::formats::{Q4_K_BYTES, Q6_K_BYTES, Q8_0_BYTES, QK_K, f16_to_f32};
use crate::reduce::LANES;

/// Accumulator vectors: 64 lanes / 4.
const V: usize = LANES / 4;

/// The portable halving tree on vectors: `a[j] += a[j + w]` for w = 8, 4, 2, 1
/// (lane `l` of vector `j` is lane `4j + l`, so this is `a[i] += a[i + 4w]`), then
/// `(l0 + l1) + (l2 + l3)`.
#[inline]
fn reduce(mut a: [float32x4_t; V]) -> f32 {
    // SAFETY: register-only NEON arithmetic; NEON is part of the aarch64 baseline.
    unsafe {
        let mut w = V / 2;
        while w >= 1 {
            for j in 0..w {
                a[j] = vaddq_f32(a[j], a[j + w]);
            }
            w /= 2;
        }
        let v = a[0];
        (vgetq_lane_f32::<0>(v) + vgetq_lane_f32::<1>(v))
            + (vgetq_lane_f32::<2>(v) + vgetq_lane_f32::<3>(v))
    }
}

/// 16 activations starting at `x[off]`.
///
/// # Safety
/// `off + 16 <= x.len()`.
#[inline]
unsafe fn load_x16(x: &[f32], off: usize) -> [float32x4_t; 4] {
    debug_assert!(off + 16 <= x.len());
    // SAFETY: the caller guarantees 16 floats are in bounds from `off`.
    unsafe {
        let p = x.as_ptr().add(off);
        [
            vld1q_f32(p),
            vld1q_f32(p.add(4)),
            vld1q_f32(p.add(8)),
            vld1q_f32(p.add(12)),
        ]
    }
}

/// `acc[base + i] = fma(w[i], x[i], acc[base + i])` for 16 lanes.
#[inline]
fn fma16(acc: &mut [float32x4_t; V], base: usize, w: [float32x4_t; 4], x: [float32x4_t; 4]) {
    // SAFETY: register-only NEON arithmetic; NEON is part of the aarch64 baseline.
    unsafe {
        for i in 0..4 {
            acc[base + i] = vfmaq_f32(acc[base + i], w[i], x[i]);
        }
    }
}

#[inline]
fn u8x16_to_f32(v: uint8x16_t) -> [float32x4_t; 4] {
    // SAFETY: register-only NEON conversions; NEON is part of the aarch64 baseline.
    unsafe {
        use core::arch::aarch64::{vcvtq_f32_u32, vget_low_u8, vmovl_high_u8, vmovl_u8};
        let lo = vmovl_u8(vget_low_u8(v));
        let hi = vmovl_high_u8(v);
        [
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
            vcvtq_f32_u32(vmovl_high_u16(lo)),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
            vcvtq_f32_u32(vmovl_high_u16(hi)),
        ]
    }
}

#[inline]
fn i8x16_to_f32(v: int8x16_t) -> [float32x4_t; 4] {
    // SAFETY: register-only NEON conversions; NEON is part of the aarch64 baseline.
    unsafe {
        let lo = vmovl_s8(vget_low_s8(v));
        let hi = vmovl_high_s8(v);
        [
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
            vcvtq_f32_s32(vmovl_high_s16(lo)),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
            vcvtq_f32_s32(vmovl_high_s16(hi)),
        ]
    }
}

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// GGML `get_scale_min_k4` (as in `formats`).
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Q4_K row · x.
///
/// # Safety
/// None beyond running on aarch64; lengths are asserted.
unsafe fn q4_k_impl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline. Every load is in bounds: the
    // asserts at the top fix the lengths, and each load notes its bound.
    unsafe {
        assert_eq!(row.len() / Q4_K_BYTES * QK_K, x.len());
        let mut acc = [vdupq_n_f32(0.0); V];
        let mask = vdupq_n_u8(0x0F);
        for (b, blk) in row.chunks_exact(Q4_K_BYTES).enumerate() {
            let d = f16_at(blk, 0);
            let dmin = f16_at(blk, 2);
            let scales = &blk[4..16];
            for j in 0..4 {
                let (sc, m) = scale_min_k4(2 * j, scales);
                let (d1, m1) = (
                    vdupq_n_f32(d * f32::from(sc)),
                    vdupq_n_f32(dmin * f32::from(m)),
                );
                let (sc, m) = scale_min_k4(2 * j + 1, scales);
                let (d2, m2) = (
                    vdupq_n_f32(d * f32::from(sc)),
                    vdupq_n_f32(dmin * f32::from(m)),
                );
                let q = &blk[16 + 32 * j..16 + 32 * j + 32];
                // SAFETY: `q` has 32 bytes.
                let (qa, qb) = { (vld1q_u8(q.as_ptr()), vld1q_u8(q.as_ptr().add(16))) };
                let parts = [
                    (vandq_u8(qa, mask), d1, m1),
                    (vandq_u8(qb, mask), d1, m1),
                    (vshrq_n_u8::<4>(qa), d2, m2),
                    (vshrq_n_u8::<4>(qb), d2, m2),
                ];
                let off = b * QK_K + 64 * j;
                for (p, (qv, dd, mm)) in parts.into_iter().enumerate() {
                    let mut w = u8x16_to_f32(qv);
                    for v in &mut w {
                        *v = vsubq_f32(vmulq_f32(dd, *v), mm);
                    }
                    // SAFETY: off + 16p + 16 <= (b + 1) * 256 <= x.len() (asserted above).
                    let xv = { load_x16(x, off + 16 * p) };
                    fma16(&mut acc, 4 * p, w, xv);
                }
            }
        }
        reduce(acc)
    }
}

/// Q6_K row · x.
///
/// # Safety
/// None beyond running on aarch64; lengths are asserted.
unsafe fn q6_k_impl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline. Every load is in bounds: the
    // asserts at the top fix the lengths, and each load notes its bound.
    unsafe {
        assert_eq!(row.len() / Q6_K_BYTES * QK_K, x.len());
        let mut acc = [vdupq_n_f32(0.0); V];
        let m4 = vdupq_n_u8(0x0F);
        let m2 = vdupq_n_u8(0x03);
        let k32 = vdupq_n_s8(32);
        for (b, blk) in row.chunks_exact(Q6_K_BYTES).enumerate() {
            let d = f16_at(blk, 208);
            for half in 0..2 {
                let ql = &blk[64 * half..64 * half + 64];
                let qh = &blk[128 + 32 * half..128 + 32 * half + 32];
                let sc = &blk[192 + 8 * half..192 + 8 * half + 8];
                let s = |i: usize| vdupq_n_f32(d * f32::from(sc[i] as i8));
                // Two 16-element vectors per 32-weight group: l in 0..16 and 16..32.
                for v16 in 0..2 {
                    let lo = 16 * v16;
                    // SAFETY: ql has 64 bytes and qh 32; lo + 16 <= 32 and lo + 32 + 16 <= 64.
                    let (a, c, h) = {
                        (
                            vld1q_u8(ql.as_ptr().add(lo)),
                            vld1q_u8(ql.as_ptr().add(lo + 32)),
                            vld1q_u8(qh.as_ptr().add(lo)),
                        )
                    };
                    let to_q = |low: uint8x16_t, high: uint8x16_t| -> int8x16_t {
                        vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(low, vshlq_n_u8::<4>(vandq_u8(high, m2)))),
                            k32,
                        )
                    };
                    // q1..q4 → elements lo.., lo+32.., lo+64.., lo+96.. of the 128-weight half.
                    let qs = [
                        (to_q(vandq_u8(a, m4), h), 0, s(v16)),
                        (to_q(vandq_u8(c, m4), vshrq_n_u8::<2>(h)), 32, s(v16 + 2)),
                        (to_q(vshrq_n_u8::<4>(a), vshrq_n_u8::<4>(h)), 64, s(v16 + 4)),
                        (to_q(vshrq_n_u8::<4>(c), vshrq_n_u8::<6>(h)), 96, s(v16 + 6)),
                    ];
                    for (q, base, scale) in qs {
                        let mut w = i8x16_to_f32(q);
                        for v in &mut w {
                            *v = vmulq_f32(scale, *v);
                        }
                        let e = 128 * half + base + lo; // element within the block
                        // SAFETY: b * 256 + e + 16 <= (b + 1) * 256 <= x.len().
                        let xv = { load_x16(x, b * QK_K + e) };
                        fma16(&mut acc, (e % LANES) / 4, w, xv);
                    }
                }
            }
        }
        reduce(acc)
    }
}

/// Q8_0 row · x; an odd block count ends in a half chunk (lanes 0..32), as the
/// portable tail does.
///
/// # Safety
/// None beyond running on aarch64; lengths are asserted.
unsafe fn q8_0_impl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline. Every load is in bounds: the
    // asserts at the top fix the lengths, and each load notes its bound.
    unsafe {
        assert_eq!(row.len() / Q8_0_BYTES * 32, x.len());
        let mut acc = [vdupq_n_f32(0.0); V];
        for (bi, blk) in row.chunks_exact(Q8_0_BYTES).enumerate() {
            let d = vdupq_n_f32(f16_at(blk, 0));
            for h in 0..2 {
                // SAFETY: the block has 34 bytes; 2 + 16h + 16 <= 34.
                let q = { core::arch::aarch64::vld1q_s8(blk.as_ptr().add(2 + 16 * h).cast()) };
                let mut w = i8x16_to_f32(q);
                for v in &mut w {
                    *v = vmulq_f32(*v, d);
                }
                let e = 32 * bi + 16 * h;
                // SAFETY: e + 16 <= 32 * (bi + 1) <= x.len().
                let xv = { load_x16(x, e) };
                fma16(&mut acc, (e % LANES) / 4, w, xv);
            }
        }
        reduce(acc)
    }
}

/// bf16 row · x; a tail shorter than 64 fills lanes `0..rem`.
///
/// # Safety
/// None beyond running on aarch64; lengths are asserted.
unsafe fn bf16_impl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline. Every load is in bounds: the
    // asserts at the top fix the lengths, and each load notes its bound.
    unsafe {
        let n = x.len();
        assert_eq!(row.len(), 2 * n);
        let mut acc = [vdupq_n_f32(0.0); V];
        let full16 = n / 16 * 16;
        let mut e = 0;
        while e < full16 {
            // SAFETY: 32 bytes from 2e are in bounds (e + 16 <= n); x likewise.
            let (lo, hi, xv) = {
                let p = row.as_ptr().add(2 * e).cast::<u16>();
                (vld1q_u16(p), vld1q_u16(p.add(8)), load_x16(x, e))
            };
            let w = [
                vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(lo)))),
                vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_high_u16(lo))),
                vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(hi)))),
                vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_high_u16(hi))),
            ];
            fma16(&mut acc, (e % LANES) / 4, w, xv);
            e += 16;
        }
        let mut a = [0f32; LANES];
        for (j, v) in acc.iter().enumerate() {
            // SAFETY: `a` has 64 floats; 4j + 4 <= 64.
            core::arch::aarch64::vst1q_f32(a.as_mut_ptr().add(4 * j), *v);
        }
        for i in full16..n {
            let w =
                f32::from_bits(u32::from(u16::from_le_bytes([row[2 * i], row[2 * i + 1]])) << 16);
            a[i % LANES] = w.mul_add(x[i], a[i % LANES]);
        }
        crate::reduce::reduce(&a)
    }
}

// Safe entry points.

/// Q4_K row dot.
pub(crate) fn q4_k(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is always available on aarch64.
    unsafe { q4_k_impl(row, x) }
}

/// Q6_K row dot.
pub(crate) fn q6_k(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is always available on aarch64.
    unsafe { q6_k_impl(row, x) }
}

/// Q8_0 row dot.
pub(crate) fn q8_0(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is always available on aarch64.
    unsafe { q8_0_impl(row, x) }
}

/// bf16 row dot.
pub(crate) fn bf16(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is always available on aarch64.
    unsafe { bf16_impl(row, x) }
}
