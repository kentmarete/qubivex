//! An independent, deliberately naive f64 forward pass, checked against the engine on
//! every fixture (including the GGUF one that transformers cannot load). It shares only
//! config parsing and the dequantizers (verified against gguf-py) with the engine.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop
)]

use std::collections::BTreeMap;
use std::path::Path;

use qubivex_arch::{Logits, TraceEvent, TracePoint};
use qubivex_backend_cpu::formats::dequant_row;
use qubivex_core::Shape;
use qubivex_model::arch::{MoeConfig, QkNorm};
use qubivex_model::{ArchitectureRegistry, ExpertPart, ModelMetadata, TensorRole};
use qubivex_runtime::{Engine, EngineOptions};

/// Per position, per KV head: a key or value vector.
type HeadCache = Vec<Vec<Vec<f64>>>;
/// Logits per position, and the experts selected per (layer, token).
type Reference = (Vec<Vec<f64>>, BTreeMap<(u32, usize), Vec<u32>>);

/// A tensor as f64, row-major, with its dims.
fn tensor(meta: &ModelMetadata, name: &str) -> (Vec<u64>, Vec<f64>) {
    let t = meta.tensors.get(name).unwrap();
    let file = std::fs::read(&meta.tensors.files()[t.file].path).unwrap();
    let bytes = &file[t.tensor.offset as usize..(t.tensor.offset + t.tensor.bytes) as usize];
    let dims = t.tensor.shape.dims().to_vec();
    let cols = *dims.last().unwrap() as usize;
    let rows = (t.tensor.shape.numel().unwrap() as usize) / cols;
    let row_bytes = t
        .tensor
        .dtype
        .storage_bytes(&Shape::new(&[1, cols as u64]).unwrap())
        .unwrap() as usize;
    let mut out = Vec::with_capacity(rows * cols);
    let mut buf = vec![0f32; cols];
    for r in 0..rows {
        dequant_row(
            t.tensor.dtype,
            &bytes[r * row_bytes..(r + 1) * row_bytes],
            &mut buf,
        );
        out.extend(buf.iter().map(|v| f64::from(*v)));
    }
    (dims, out)
}

fn matvec(w: &[f64], cols: usize, x: &[f64]) -> Vec<f64> {
    w.chunks(cols)
        .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn rms(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    x.iter()
        .zip(w)
        .map(|(v, g)| v / (ms + eps).sqrt() * g)
        .collect()
}

fn rope(x: &mut [f64], pos: usize, theta: f64) {
    let d = x.len();
    for i in 0..d / 2 {
        let a = pos as f64 * theta.powf(-(2.0 * i as f64) / d as f64);
        let (s, c) = a.sin_cos();
        let (x1, x2) = (x[i], x[i + d / 2]);
        x[i] = x1 * c - x2 * s;
        x[i + d / 2] = x2 * c + x1 * s;
    }
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// Logits for every prompt position, and the selected experts per (layer, token).
fn reference(meta: &ModelMetadata, prompt: &[u32]) -> Reference {
    let c: &MoeConfig = meta.config.specific.downcast_ref().unwrap();
    let names = &c.names;
    let (h, d, nh, nkv) = (
        c.hidden as usize,
        c.head_dim as usize,
        c.heads as usize,
        c.kv_heads as usize,
    );
    let at = |p: &str, l: u32| p.replace("{L}", &l.to_string());
    let (_, emb) = tensor(meta, &names.embedding);
    let mut hs: Vec<Vec<f64>> = prompt
        .iter()
        .map(|&t| emb[t as usize * h..(t as usize + 1) * h].to_vec())
        .collect();
    // Routed expert matrices per (layer, expert): (gate, up, down), from the roles.
    let mut experts: BTreeMap<(u32, u32), BTreeMap<ExpertPart, Vec<f64>>> = BTreeMap::new();
    for (name, role) in &meta.roles {
        match *role {
            TensorRole::RoutedExpert {
                layer,
                expert,
                part,
            } => {
                experts
                    .entry((layer.0, expert.0))
                    .or_default()
                    .insert(part, tensor(meta, name).1);
            }
            TensorRole::RoutedExpertStack {
                layer,
                part,
                experts: n,
            } => {
                let w = tensor(meta, name).1;
                let per = w.len() / n as usize;
                for e in 0..n {
                    experts
                        .entry((layer.0, e))
                        .or_default()
                        .insert(part, w[e as usize * per..(e as usize + 1) * per].to_vec());
                }
            }
            _ => {}
        }
    }
    let ffn = c.expert_ffn as usize;
    let mut picks = BTreeMap::new();
    let (mut ks, mut vs): (HeadCache, HeadCache) = (vec![], vec![]);
    for l in 0..c.layers {
        let w = |p: &str| tensor(meta, &at(p, l)).1;
        let (wq, wk, wv, wo, wr) = (
            w(&names.q),
            w(&names.k),
            w(&names.v),
            w(&names.o),
            w(&names.router),
        );
        let (an, fnn) = (w(&names.attn_norm), w(&names.ffn_norm));
        let (qn, kn) = match c.qk_norm {
            QkNorm::None => (vec![], vec![]),
            _ => (w(&names.q_norm), w(&names.k_norm)),
        };
        ks.clear();
        vs.clear();
        let mut new_hs = Vec::new();
        for (t, ht) in hs.iter().enumerate() {
            let x = rms(ht, &an, c.rms_eps);
            let mut q = matvec(&wq, h, &x);
            let mut k = matvec(&wk, h, &x);
            let v = matvec(&wv, h, &x);
            match c.qk_norm {
                QkNorm::Full => {
                    q = rms(&q, &qn, c.rms_eps);
                    k = rms(&k, &kn, c.rms_eps);
                }
                QkNorm::PerHead => {
                    q = q.chunks(d).flat_map(|hd| rms(hd, &qn, c.rms_eps)).collect();
                    k = k.chunks(d).flat_map(|hd| rms(hd, &kn, c.rms_eps)).collect();
                }
                QkNorm::None => {}
            }
            let qh: Vec<Vec<f64>> = q
                .chunks(d)
                .map(|hd| {
                    let mut hd = hd.to_vec();
                    rope(&mut hd, t, c.rope_theta);
                    hd
                })
                .collect();
            let kh: Vec<Vec<f64>> = k
                .chunks(d)
                .map(|hd| {
                    let mut hd = hd.to_vec();
                    rope(&mut hd, t, c.rope_theta);
                    hd
                })
                .collect();
            ks.push(kh);
            vs.push(v.chunks(d).map(<[f64]>::to_vec).collect());
            let mut att = vec![0f64; nh * d];
            for hd in 0..nh {
                let g = hd / (nh / nkv);
                let sc: Vec<f64> = (0..=t)
                    .map(|p| {
                        qh[hd]
                            .iter()
                            .zip(&ks[p][g])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let p = softmax(&sc);
                for (pi, pw) in p.iter().enumerate() {
                    for i in 0..d {
                        att[hd * d + i] += pw * vs[pi][g][i];
                    }
                }
            }
            let o = matvec(&wo, nh * d, &att);
            let h1: Vec<f64> = ht.iter().zip(&o).map(|(a, b)| a + b).collect();
            let x = rms(&h1, &fnn, c.rms_eps);
            let probs = softmax(&matvec(&wr, h, &x));
            let mut idx: Vec<u32> = (0..c.experts).collect();
            idx.sort_by(|a, b| {
                probs[*b as usize]
                    .total_cmp(&probs[*a as usize])
                    .then(a.cmp(b))
            });
            idx.truncate(c.top_k as usize);
            let mut wts: Vec<f64> = idx.iter().map(|&e| probs[e as usize]).collect();
            if c.norm_topk_prob {
                let s: f64 = wts.iter().sum();
                wts.iter_mut().for_each(|v| *v /= s);
            }
            let mut moe = vec![0f64; h];
            for (&e, wt) in idx.iter().zip(&wts) {
                let ex = &experts[&(l, e)];
                let (gate, up): (Vec<f64>, Vec<f64>) = match ex.get(&ExpertPart::GateUp) {
                    Some(gu) => (gu[..ffn * h].to_vec(), gu[ffn * h..].to_vec()),
                    None => (ex[&ExpertPart::Gate].clone(), ex[&ExpertPart::Up].clone()),
                };
                let g = matvec(&gate, h, &x);
                let u = matvec(&up, h, &x);
                let a: Vec<f64> = g
                    .iter()
                    .zip(&u)
                    .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                let y = matvec(&ex[&ExpertPart::Down], ffn, &a);
                for i in 0..h {
                    moe[i] += wt * y[i];
                }
            }
            picks.insert((l, t), idx);
            new_hs.push(h1.iter().zip(&moe).map(|(a, b)| a + b).collect());
        }
        hs = new_hs;
    }
    let (_, fnw) = tensor(meta, &names.final_norm);
    let head = if c.tie_embeddings {
        emb
    } else {
        tensor(meta, &names.output).1
    };
    (
        hs.iter()
            .map(|ht| matvec(&head, h, &rms(ht, &fnw, c.rms_eps)))
            .collect(),
        picks,
    )
}

#[test]
fn engine_matches_an_independent_f64_reference() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models");
    let prompt = [
        72u32, 101, 108, 108, 111, 44, 32, 101, 120, 112, 101, 114, 116, 115,
    ];
    for id in ["olmoe-tiny", "olmoe-tiny-gguf", "qwen3-moe-tiny"] {
        let meta =
            ModelMetadata::load(&root.join(id), &ArchitectureRegistry::with_builtins()).unwrap();
        let (want, want_picks) = reference(&meta, &prompt);
        let e = Engine::load(meta, EngineOptions::default()).unwrap();
        let mut picks = BTreeMap::new();
        let mut cache = e.new_cache();
        let mut tr = |ev: TraceEvent<'_>| {
            if ev.point == TracePoint::RoutingWeights {
                picks.insert((ev.layer.unwrap(), ev.token), ev.experts.to_vec());
            }
        };
        let got = e
            .forward(&prompt, &mut cache, Logits::All, Some(&mut tr))
            .unwrap();
        assert_eq!(picks, want_picks, "{id}: selected experts");
        let vocab = got.len() / prompt.len();
        let mut worst = 0f64;
        for (t, w) in want.iter().enumerate() {
            let scale = w.iter().fold(0f64, |m, v| m.max(v.abs()));
            let err = got[t * vocab..(t + 1) * vocab]
                .iter()
                .zip(w)
                .fold(0f64, |m, (a, b)| m.max((f64::from(*a) - b).abs()));
            worst = worst.max(err / scale);
        }
        assert!(worst < 1e-5, "{id}: logits relative error {worst:.2e}");
        eprintln!("{id}: worst logits relative error vs f64 {worst:.2e}");
    }
}
