//! Register-blocked matmul for several activation rows (prefill).
//!
//! [`crate::reduce::dot`] gives each output 64 independent accumulator lanes: lane `l`
//! receives `w[64c + l] · x[64c + l]` for chunks `c = 0, 1, …` in order, by fused
//! multiply-add from zero, and the lanes are then summed by one fixed halving tree.
//! Because the lanes never interact before that tree, they can be computed in any
//! grouping. This kernel computes one 4-lane slice ("quad") at a time for a tile of
//! [`MR`] weight rows × up to [`NR`] activation rows, holding the `MR × NR`
//! accumulators in registers, and then applies the same tree. Each lane sees exactly
//! the same sequence of operations as in `dot`, so every output has the same bits as
//! the one-row-at-a-time path, whatever the batch size (ADR-002 amendment 1, ADR-010).
//!
//! A ragged final chunk (`k` not a multiple of 64) is padded with zeros in both
//! operands. That is exact: `fma(0, 0, acc) = acc + 0 = acc`, because an accumulator
//! that starts at `+0` can never become `-0` (a round-to-nearest sum is `-0` only when
//! both addends are).
//!
//! Operands are packed lane-major so the micro-kernel reads them contiguously:
//! element `(row, 64c + 4q + i)` of a block `width` rows wide is at
//! `((q · chunks + c) · width + row) · 4 + i`. Reading the row-major rows in place
//! instead was measured 2× slower: rows `k · 4` bytes apart map to the same L1 sets.

use qubivex_compute::WeightView;
use rayon::prelude::*;

use crate::formats;
use crate::reduce::{LANES, reduce};

/// f32 values per vector register.
const VL: usize = 4;
/// Quads per 64-lane chunk.
const QUADS: usize = LANES / VL;
/// Weight rows per micro-tile.
const MR: usize = 4;
/// Activation rows per micro-tile.
const NR: usize = 6;
/// The fewest activation rows for which the tiled kernel beats one dot product per
/// output. Below it, packing each dequantised weight row costs more than the tile
/// saves (measured on the Q4_K expert shape, ADR-002 amendment 1).
const MIN_ROWS: usize = 6;

/// Whether [`rows`] handles a product with `s` activation rows.
pub(crate) const fn applies(s: usize) -> bool {
    s >= MIN_ROWS
}

/// 64-lane chunks covering `k` values.
const fn chunks(k: usize) -> usize {
    k.div_ceil(LANES)
}

/// Writes one row into slot `j` of a `width`-wide packed block. Padding is never
/// written, so it keeps whatever the block held (zero).
#[inline(always)]
fn pack_row(dst: &mut [f32], width: usize, j: usize, k: usize, row: &[f32]) {
    let stride = chunks(k) * width * VL;
    let full = k / VL * VL;
    // Whole quads as fixed-size copies (a run-time length would call memcpy).
    for (e, quad) in row[..full].chunks_exact(VL).enumerate() {
        let (c, q) = (e / QUADS, e % QUADS);
        let o = q * stride + (c * width + j) * VL;
        let d: &mut [f32; VL] = (&mut dst[o..o + VL])
            .try_into()
            .unwrap_or_else(|_| unreachable!("one quad"));
        *d = quad.try_into().unwrap_or_else(|_| unreachable!("one quad"));
    }
    // A partial final quad.
    let e = full / VL;
    let o = (e % QUADS) * stride + ((e / QUADS) * width + j) * VL;
    for (i, v) in row[full..].iter().enumerate() {
        dst[o + i] = *v;
    }
}

/// `s` activation rows packed in blocks of [`NR`] rows (the last block may be
/// narrower). Block `b` starts at `b · NR · chunks · 64`.
pub(crate) struct PackedX {
    data: Vec<f32>,
    s: usize,
    ch: usize,
}

impl PackedX {
    /// Packs `xs` (`s` rows of `k`), one block per parallel task.
    pub(crate) fn new(xs: &[f32], s: usize, k: usize) -> Self {
        let ch = chunks(k);
        let block = NR * ch * LANES;
        let mut data = vec![0f32; s.div_ceil(NR) * block];
        data.par_chunks_mut(block).enumerate().for_each(|(b, dst)| {
            let width = NR.min(s - b * NR);
            for j in 0..width {
                let t = b * NR + j;
                pack_row(dst, width, j, k, &xs[t * k..(t + 1) * k]);
            }
        });
        Self { data, s, ch }
    }

    /// Block `b` and its width.
    fn block(&self, b: usize) -> (&[f32], usize) {
        let width = NR.min(self.s - b * NR);
        let start = b * NR * self.ch * LANES;
        (&self.data[start..start + width * self.ch * LANES], width)
    }
}

/// Per-thread scratch for [`rows`]: one dequantised row, the packed `MR`-row weight
/// panel, and the accumulator lanes of one tile.
pub(crate) struct Scratch {
    row: Vec<f32>,
    panel: Vec<f32>,
    lanes: Vec<f32>,
}

impl Scratch {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            row: vec![0f32; k],
            panel: vec![0f32; MR * chunks(k) * LANES],
            lanes: vec![0f32; MR * NR * LANES],
        }
    }
}

/// One micro-tile: `MR` packed weight rows × `N` packed activation rows, every quad.
/// Writes the 64 accumulator lanes of output `(r, t)` to `lanes[(r · N + t) · 64..]`.
#[inline(always)]
fn tile<const N: usize>(wp: &[f32], xp: &[f32], ch: usize, lanes: &mut [f32]) {
    let (wq_len, xq_len) = (ch * MR * VL, ch * N * VL);
    for q in 0..QUADS {
        let mut acc = [[[0f32; VL]; N]; MR];
        let wq = &wp[q * wq_len..(q + 1) * wq_len];
        let xq = &xp[q * xq_len..(q + 1) * xq_len];
        for (wc, xc) in wq.chunks_exact(MR * VL).zip(xq.chunks_exact(N * VL)) {
            let w: [[f32; VL]; MR] = std::array::from_fn(|r| {
                wc[r * VL..(r + 1) * VL]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("one quad"))
            });
            let x: [[f32; VL]; N] = std::array::from_fn(|t| {
                xc[t * VL..(t + 1) * VL]
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("one quad"))
            });
            for r in 0..MR {
                for t in 0..N {
                    for i in 0..VL {
                        acc[r][t][i] = w[r][i].mul_add(x[t][i], acc[r][t][i]);
                    }
                }
            }
        }
        for (r, ar) in acc.iter().enumerate() {
            for (t, a) in ar.iter().enumerate() {
                let o = (r * N + t) * LANES + q * VL;
                lanes[o..o + VL].copy_from_slice(a);
            }
        }
    }
}

/// Computes rows `r0 .. r0 + out.len() / s` of `w × xs` into `out` (row-major, `s`
/// values per row), with the same bits as one dot product per output.
#[inline(always)]
pub(crate) fn rows(w: &WeightView<'_>, xp: &PackedX, r0: usize, out: &mut [f32], sc: &mut Scratch) {
    let (k, s, ch) = (w.cols(), xp.s, xp.ch);
    let n = out.len() / s;
    let Scratch { row, panel, lanes } = sc;
    for g in (0..n).step_by(MR) {
        let live = MR.min(n - g);
        if live < MR {
            // Rows past the matrix: zero, so they stay finite; their outputs are dropped.
            panel.fill(0.0);
        }
        for j in 0..live {
            formats::dequant_row(w.dtype(), w.row(r0 + g + j), row);
            pack_row(panel, MR, j, k, row);
        }
        for b in 0..s.div_ceil(NR) {
            let (xb, nb) = xp.block(b);
            match nb {
                1 => tile::<1>(panel, xb, ch, lanes),
                2 => tile::<2>(panel, xb, ch, lanes),
                3 => tile::<3>(panel, xb, ch, lanes),
                4 => tile::<4>(panel, xb, ch, lanes),
                5 => tile::<5>(panel, xb, ch, lanes),
                _ => tile::<NR>(panel, xb, ch, lanes),
            }
            for r in 0..live {
                for t in 0..nb {
                    let o = (r * nb + t) * LANES;
                    let acc: &[f32; LANES] = lanes[o..o + LANES]
                        .try_into()
                        .unwrap_or_else(|_| unreachable!("64 lanes"));
                    out[(g + r) * s + b * NR + t] = reduce(acc);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_matches_the_documented_layout() {
        for (s, k) in [(1, 70), (6, 64), (8, 130), (13, 2048)] {
            let xs: Vec<f32> = (0..s * k).map(|i| i as f32 + 1.0).collect();
            let p = PackedX::new(&xs, s, k);
            let ch = chunks(k);
            for b in 0..s.div_ceil(NR) {
                let (blk, width) = p.block(b);
                for j in 0..width {
                    for e in 0..ch * LANES {
                        let (c, l) = (e / LANES, e % LANES);
                        let got = blk[(((l / VL) * ch + c) * width + j) * VL + l % VL];
                        let t = b * NR + j;
                        let want = if e < k { xs[t * k + e] } else { 0.0 };
                        assert_eq!(got.to_bits(), want.to_bits(), "s{s} k{k} t{t} e{e}");
                    }
                }
            }
        }
    }
}
