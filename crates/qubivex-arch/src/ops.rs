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
use qubivex_model::arch::RopeScaling;

/// RoPE tables for `rotary_dim` dimensions: `cos` and `sin` of `pos · θ^(−2i/d)`.
#[derive(Clone, Debug)]
pub struct Rope {
    inv_freq: Vec<f32>,
    /// The factor on every cos and sin (YaRN's `attention_factor`; 1 otherwise).
    scale: f32,
}

impl Rope {
    /// Inverse frequencies as `transformers` computes them (f32), with the power
    /// correctly rounded.
    #[must_use]
    pub fn new(head_dim: usize, theta: f64) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let inv_freq = (0..head_dim / 2)
            .map(|i| {
                // The power is computed in f64 and rounded once, so it is the correctly
                // rounded f32 on every platform. An f32 `powf` is not (θ = 500 000, d = 128,
                // i = 6 is one ulp off on macOS), and torch's is within one ulp of this.
                let e = (2 * i) as f32 / head_dim as f32;
                1.0 / theta.powf(f64::from(e)) as f32
            })
            .collect();
        Self {
            inv_freq,
            scale: 1.0,
        }
    }

    /// [`Rope::new`] with the inverse frequencies scaled as `transformers` scales them,
    /// operation for operation in f32, so the bits match its tables.
    #[must_use]
    pub fn with_scaling(head_dim: usize, theta: f64, scaling: &RopeScaling) -> Self {
        let mut r = Self::new(head_dim, theta);
        match *scaling {
            RopeScaling::None => {}
            RopeScaling::Banded {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            } => {
                for f in &mut r.inv_freq {
                    *f = banded(
                        *f,
                        factor,
                        low_freq_factor,
                        high_freq_factor,
                        original_max_position_embeddings,
                    );
                }
            }
            RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_max_position_embeddings,
                truncate,
                attention_factor,
            } => {
                r.inv_freq = yarn(
                    head_dim,
                    theta,
                    factor,
                    (beta_fast, beta_slow),
                    original_max_position_embeddings,
                    truncate,
                );
                #[allow(clippy::cast_possible_truncation)]
                {
                    r.scale = attention_factor as f32;
                }
            }
        }
        r
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
        // transformers multiplies the f32 cos and sin by the factor (YaRN); 1 is exact.
        if self.scale != 1.0 {
            for v in cs.iter_mut() {
                *v *= self.scale;
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

/// One inverse frequency under banded scaling, as `transformers` computes it. Python
/// scalars meet the f32 tensor rounded to f32, and torch evaluates `scalar / tensor` as
/// `reciprocal(tensor) · scalar`; both are mirrored, and each step rounds once (no
/// fused multiply-add), as torch's separate kernels do.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn banded(inv: f32, factor: f64, low: f64, high: f64, original: u64) -> f32 {
    let old = original as f64;
    let (factor_f, low_f) = (factor as f32, low as f32);
    let low_wavelen = (old / low) as f32;
    let high_wavelen = (old / high) as f32;
    let wavelen = (1.0 / inv) * (2.0 * std::f64::consts::PI) as f32;
    let scaled = if wavelen > low_wavelen {
        inv / factor_f
    } else {
        inv
    };
    if wavelen < high_wavelen || wavelen > low_wavelen {
        return scaled;
    }
    let smooth = ((1.0 / wavelen) * old as f32 - low_f) / (high - low) as f32;
    (1.0 - smooth) * scaled / factor_f + smooth * scaled
}

/// YaRN inverse frequencies as transformers' `_compute_yarn_parameters` computes them:
/// the correction range in f64 (Python floats), everything on the tensors in f32 with
/// Python scalars rounded to f32, one rounding per operation and no fused multiply-add.
/// The power `θ^(2i/d)` is the correctly rounded f32, as in [`Rope::new`]; torch's
/// f32 `pow` is within one ulp of it, so where torch's is not correctly rounded the
/// frequency can be one ulp from transformers'.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn yarn(
    head_dim: usize,
    theta: f64,
    factor: f64,
    (beta_fast, beta_slow): (f64, f64),
    original: u64,
    truncate: bool,
) -> Vec<f32> {
    let dim = head_dim as f64;
    let correction = |rotations: f64| {
        (dim * (original as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln())
            / (2.0 * theta.ln())
    };
    let (mut low, mut high) = (correction(beta_fast), correction(beta_slow));
    if truncate {
        low = low.floor();
        high = high.ceil();
    }
    let (low, mut high) = (low.max(0.0), high.min(dim - 1.0));
    if low == high {
        high += 0.001;
    }
    let factor_f = factor as f32;
    let (low_f, span_f) = (low as f32, (high - low) as f32);
    (0..head_dim / 2)
        .map(|i| {
            let e = (2 * i) as f32 / head_dim as f32;
            let pos = theta.powf(f64::from(e)) as f32;
            let extrapolation = 1.0 / pos;
            let interpolation = 1.0 / (factor_f * pos);
            let ramp = ((i as f32 - low_f) / span_f).clamp(0.0, 1.0);
            let extra = 1.0 - ramp;
            interpolation * (1.0 - extra) + extrapolation * extra
        })
        .collect()
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

    /// Plain RoPE's inverse frequencies against `transformers` 5.22.1's
    /// default rotary embedding (f32, CPU, arm64), at the common bases. Ours are the
    /// correctly rounded power; torch's vectorised f32 `pow` is within one ulp of it and
    /// misses in rare entries (θ = 10⁶, d = 128, i = 37), so this allows one ulp in at
    /// most one entry per base.
    #[test]
    fn plain_rope_matches_transformers_bits() {
        #[rustfmt::skip]
        let cases: [(usize, f64, &str); 4] = [
            (128, 500000.0, concat!(
                "3f8000003f508ac13f29e1c63f0a63843ee177bc3eb7ab7d3e959ee33e73c461",
                "3e4693b03e21c3a03e03c6a03dd6b19c3daee4ad3d8e78983d681e673d3d1684",
                "3d1a08c83cfaf53f3ccc6f493ca689393c87a9c33c5d06ec3c340d6d3c12ac7f",
                "3beef74f3bc2aa763b9e94023b812e353b5277203b2b72dc3b0baa413ae38c10",
                "3ab95d213a9700233a7603ea3a4868863a23418d3a04fdbf39d8ac8139b08199",
                "398fc8f8396a426f393ed4f4391b747538fd45c338ce51f538a8126b3888ea0f",
                "385f10c43835b687381406cb37f12b8037c4761037a00a6937825f3437546808",
                "372d07a7370cf40036e5a54d36bb12c7369864a7367848c2364a41b03624c2ff",
            )),
            (128, 1000000.0, concat!(
                "3f8000003f4e4bad3f263de03f05f6ef3ed7e89b3eadfcff3e8c35043e61f835",
                "3e3618873e12bd913dec7fd63dbe94c63d99940d3d7785133d47763f3d20bc1d",
                "3d0186e33cd0c1a83ca8398b3c8790083c5a7bf23c3010523c0de12d3be4aa46",
                "3bb8449c3b947dae3b6f520e3b40dac53b1b690d3afa78f03ac9d75c3aa2a6f7",
                "3a83126f3a533f283a2a3b443a092e0239dd172539b229fb398f927239676492",
                "393a77533916432438f22ce238c327b5389d43a4387d75d4384c3fbe382497ab",
                "3804a2b237d5c44137ac431d378ad0ed375fba4f37344a0e371148e336ea2732",
                "36bcb0c136980e023675107036457bab361f23e436003dec35ceaf7935a68e4c",
            )),
            (64, 10000.0, concat!(
                "3f8000003f3ff9113f0ff59a3ed7e89b3ea1e89b3e72d4233e3618873e088d77",
                "3dcccccd3d99940d3d6655c23d2cba153d0186e33cc2434f3c91ad393c5a7bf2",
                "3c23d70a3bf5b9b03bb8449c3b8a2e773b4f3e383b1b690d3ae915283aaec98e",
                "3a83126f3a44948c3a136a1639dd172539a5cb603978a815393a7753390bd472",
            )),
            (128, 10000.0, concat!(
                "3f8000003f5dafd73f3ff9113f263de03f0ff59a3ef953cf3ed7e89b3ebaf81b",
                "3ea1e89b3e8c35043e72d4233e5247ed3e3618873e1db0403e088d773dec7fd6",
                "3dcccccd3db159783d99940d3d84fe4d3d6655c23d47763f3d2cba153d159348",
                "3d0186e33ce054d23cc2434f3ca8398b3c91ad393c7c4d333c5a7bf23c3d3311",
                "3c23d70a3c0de12d3bf5b9b03bd4ca153bb8449c3b9f91cc3b8a2e773b6f520e",
                "3b4f3e383b33770f3b1b690d3b06946f3ae915283ac9d75c3aaec98e3a975c0e",
                "3a83126f3a6301e23a44948c3a2a3b443a136a1639ff4fac39dd172539bf74d7",
                "39a5cb60398f92723978a815395753e4393a775339217916390bd47238f22ce2",
            )),
        ];
        for (d, theta, hex) in cases {
            let want: Vec<u32> = (0..hex.len() / 8)
                .map(|i| u32::from_str_radix(&hex[8 * i..8 * i + 8], 16).unwrap())
                .collect();
            let got: Vec<u32> = Rope::new(d, theta)
                .inv_freq
                .iter()
                .map(|f| f.to_bits())
                .collect();
            assert_eq!(got.len(), want.len());
            let off: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
            assert!(off.len() <= 1, "d={d} theta={theta}: {off:?} differ");
            for i in off {
                assert_eq!(got[i].abs_diff(want[i]), 1, "d={d} theta={theta} i={i}");
            }
        }
    }

    /// Inverse frequencies under banded scaling, bit for bit against `transformers`
    /// 5.22.1 (f32, CPU): the factors of two released checkpoints (32 and 8 at θ = 500 000)
    /// and two made-up configs whose bands fall elsewhere. Each value is its f32 bits in hex.
    #[test]
    #[allow(clippy::float_cmp)]
    fn banded_scaling_matches_transformers_bits() {
        #[rustfmt::skip]
        let cases: [(usize, f64, f64, f64, f64, u64, &str); 4] = [
            (128, 500000.0, 32.0, 1.0, 4.0, 8192, concat!(
                "3f8000003f508ac13f29e1c63f0a63843ee177bc3eb7ab7d3e959ee33e73c461",
                "3e4693b03e21c3a03e03c6a03dd6b19c3daee4ad3d8e78983d681e673d3d1684",
                "3d1a08c83cfaf53f3ccc6f493ca689393c87a9c33c5d06ec3c340d6d3c12ac7f",
                "3beef74f3bc2aa763b9e94023b812e353b5277203b0ad4f83aa9279b3a47e547",
                "39e136203969959938cb98f737c8688637a3418d3784fdbf3758ac8137308199",
                "370fc8f836ea426f36bed4f4369b7475367d45c3364e51f53628126b3608ea0f",
                "35df10c435b5b687359406cb35712b803544761035200a6935025f3434d46808",
                "34ad07a7348cf4003465a54d343b12c7341864a733f848c233ca41b033a4c2ff",
            )),
            (128, 500000.0, 8.0, 1.0, 4.0, 8192, concat!(
                "3f8000003f508ac13f29e1c63f0a63843ee177bc3eb7ab7d3e959ee33e73c461",
                "3e4693b03e21c3a03e03c6a03dd6b19c3daee4ad3d8e78983d681e673d3d1684",
                "3d1a08c83cfaf53f3ccc6f493ca689393c87a9c33c5d06ec3c340d6d3c12ac7f",
                "3beef74f3bc2aa763b9e94023b812e353b5277203b0dfd063ab3d11d3a60979e",
                "3a0995d239a3f108393b2dd238c8688638a3418d3884fdbf3858ac8138308199",
                "380fc8f837ea426f37bed4f4379b7475377d45c3374e51f53728126b3708ea0f",
                "36df10c436b5b687369406cb36712b803644761036200a6936025f3435d46808",
                "35ad07a7358cf4003565a54d353b12c7351864a734f848c234ca41b034a4c2ff",
            )),
            (64, 10000.0, 4.0, 1.5, 3.0, 2048, concat!(
                "3f8000003f3ff9113f0ff59a3ed7e89b3ea1e89b3e72d4233e3618873e088d77",
                "3dcccccd3d99940d3d6655c23d2cba153d0186e33cc2434f3c91ad393c5a7bf2",
                "3c23d70a3bb172903b197c513a8a2e773a4f3e383a1b690d39e9152839aec98e",
                "3983126f3944948c39136a1638dd172538a5cb603878a815383a7753380bd472",
            )),
            (96, 1000000.0, 16.0, 0.5, 7.0, 4096, concat!(
                "3f8000003f3ff9113f0ff59a3ed7e89b3ea1e89b3e72d4253e3618873e088d78",
                "3dcccccc3d99940d3d6655c43d2cba143d0186e33cc2434e3c91ad3a3c5a7bf2",
                "3c1879263baae4a63b3f50733ad5d6a93a6e81ce3a04a09c3992ebe93921e0a1",
                "38b104743844948f38136a1437dd172537a5cb623778a80f373a7753370bd474",
                "36d1b713369d43a4366bdcf53630df4e3604a2b235c6ece035952c40355fba4f",
                "3527c5b034fb9f6734bcb0c1348d7f7634543781341f23e433eead3b33b2fb6f",
            )),
        ];
        for (d, theta, factor, low, high, old, hex) in cases {
            let r = Rope::with_scaling(
                d,
                theta,
                &RopeScaling::Banded {
                    factor,
                    low_freq_factor: low,
                    high_freq_factor: high,
                    original_max_position_embeddings: old,
                },
            );
            let want: Vec<u32> = (0..hex.len() / 8)
                .map(|i| u32::from_str_radix(&hex[8 * i..8 * i + 8], 16).unwrap())
                .collect();
            let got: Vec<u32> = r.inv_freq.iter().map(|f| f.to_bits()).collect();
            assert_eq!(got, want, "d={d} theta={theta} factor={factor}");
        }
        // No scaling is plain RoPE.
        assert_eq!(
            Rope::with_scaling(64, 1e4, &RopeScaling::None).inv_freq,
            Rope::new(64, 1e4).inv_freq
        );
    }

    /// YaRN inverse frequencies and attention factor against transformers 5.22
    /// (`ROPE_INIT_FUNCTIONS["yarn"]`) on three configs, the first a 64-wide head at
    /// θ = 150 000, factor 32, untruncated. Bit for bit, except one ulp at the listed
    /// frequencies: there torch's f32 `pow` for `θ^(2i/d)` is not correctly rounded
    /// (checked in Python: the correctly rounded powers give these bits exactly).
    #[test]
    fn yarn_scaling_matches_transformers_bits() {
        #[rustfmt::skip]
        let cases: [(usize, f64, f64, f64, f64, u64, bool, f64, &[usize], &str); 3] = [
            (64, 150000.0, 32.0, 32.0, 1.0, 4096, false, 1.346_573_590_279_972_7, &[6, 10], concat!(
                "3f8000003f3065353ef316a23ea77faa3e66d3fa3e1f0cfd3ddb2f9f3d970764",
                "3d5021943d01ddd53c9e646f3c3dec943bdea8403b7cfe143b0938033a89f778",
                "39ef543c390799b93820adc637dd6dfc37989328375242fa3710e12c36c7a82b",
                "3689928c363d96473602a24535b4066835781728352af20034eb93eb34a252d3",
            )),
            (128, 10000.0, 4.0, 32.0, 1.0, 2048, true, 1.138_629_436_111_989, &[], concat!(
                "3f8000003f5dafd73f3ff9113f263de03f0ff59a3ef953cf3ed7e89b3ebaf81b",
                "3ea1e89b3e8c35043e72d4233e5247ed3e3618873e1db0403e088d773dec7fd6",
                "3dcccccd3dac076d3d905d163d720c443d4ab1de3d298ae93d0da2d43cec5439",
                "3cc4e1823ca3c30a3c87fbeb3c616bd43c3a77533c19e76a3bfd710f3bd01e92",
                "3baa64c33b8b0ac13b62113a3b36ffb63b136a163aec29b43abbed463a9460f4",
                "3a681cbb3a33770f3a1b690d3a06946f39e9152839c9d75c39aec98e39975c0e",
                "3983126f396301e23944948c392a3b4439136a1638ff4fac38dd172538bf74d7",
                "38a5cb60388f92723878a815385753e4383a775338217916380bd47237f22ce2",
            )),
            (96, 1e6, 8.0, 16.0, 2.0, 8192, true, 1.207_944_154_167_983_6, &[], concat!(
                "3f8000003f3ff9113f0ff59a3ed7e89b3ea1e89b3e72d4253e3618873e088d78",
                "3dcccccc3d99940d3d6655c43d2cba143d0186e33cc2434e3c91ad3a3c5a7bf2",
                "3c11eb843bbff9133b779c323b1b74453abbd0643a55b07239da83d2392ec98f",
                "3903126f38c4948f38936a14385d17253825cb6237f8a80f37ba7753378bd474",
                "3751b713371d43a436ebdcf536b0df4e3684a2b23646ece036152c4035dfba4f",
                "35a7c5b0357b9f67353cb0c1350d7f7634d43781349f23e4346ead3b3432fb6f",
            )),
        ];
        for (d, theta, factor, bf, bs, old, truncate, af, off_by_one, hex) in cases {
            let r = Rope::with_scaling(
                d,
                theta,
                &RopeScaling::Yarn {
                    factor,
                    beta_fast: bf,
                    beta_slow: bs,
                    original_max_position_embeddings: old,
                    truncate,
                    // What the parser derives from the factor alone.
                    attention_factor: 0.1 * factor.ln() + 1.0,
                },
            );
            assert_eq!(0.1 * factor.ln() + 1.0, af, "d={d}: attention factor");
            let want: Vec<u32> = (0..hex.len() / 8)
                .map(|i| u32::from_str_radix(&hex[8 * i..8 * i + 8], 16).unwrap())
                .collect();
            let got: Vec<u32> = r.inv_freq.iter().map(|f| f.to_bits()).collect();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let ulps = g.abs_diff(*w);
                let allowed = u32::from(off_by_one.contains(&i));
                assert!(ulps <= allowed, "d={d} theta={theta} dim {i}: {ulps} ulp");
            }
            // The table carries the factor on cos and sin.
            let mut cs = vec![0f32; d];
            r.table(0, &mut cs);
            #[allow(clippy::cast_possible_truncation)]
            let f = af as f32;
            assert_eq!(cs[0], f, "cos 0 scaled");
            assert_eq!(cs[d / 2], 0.0, "sin 0");
        }
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
