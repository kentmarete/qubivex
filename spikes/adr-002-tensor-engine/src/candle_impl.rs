//! Candle counterparts, used the way `candle-nn` / `candle-transformers` layers use them.

use crate::formats::{F32Mat, GgmlKind};
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use std::borrow::Cow;

pub enum CandleW {
    /// Dense weight `[n, k]`; forward is `x.matmul(w.t())` in `dtype`.
    Dense { w: Tensor, dtype: DType },
    /// GGML quantized weight; forward is `QMatMul::forward` (quantizes activations).
    Quant(QMatMul),
}

impl CandleW {
    pub fn dense(m: &F32Mat, dtype: DType) -> Result<Self> {
        let w = Tensor::from_slice(&m.w, (m.n, m.k), &Device::Cpu)?.to_dtype(dtype)?;
        Ok(Self::Dense { w, dtype })
    }

    pub fn quant_from_bytes(n: usize, k: usize, kind: GgmlKind, bytes: &[u8]) -> Result<Self> {
        let dt = ggml_dtype(kind);
        let st = QStorage::from_data(Cow::Borrowed(bytes), &Device::Cpu, dt)?;
        Ok(Self::Quant(QMatMul::from_qtensor(QTensor::new(st, (n, k))?)?))
    }

    /// `xs` is `[s, k]` f32; returns `[s, n]` f32.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense { w, dtype } => xs.to_dtype(*dtype)?.matmul(&w.t()?)?.to_dtype(DType::F32),
            Self::Quant(q) => q.forward(xs),
        }
    }
}

pub fn ggml_dtype(kind: GgmlKind) -> GgmlDType {
    match kind {
        GgmlKind::Q8_0 => GgmlDType::Q8_0,
        GgmlKind::Q4K => GgmlDType::Q4K,
    }
}

/// Quantizes with Candle's (GGML-equivalent) quantizer and returns the raw blocks.
pub fn quantize_bytes(m: &F32Mat, kind: GgmlKind) -> Result<Vec<u8>> {
    let t = Tensor::from_slice(&m.w, (m.n, m.k), &Device::Cpu)?;
    let q = QTensor::quantize(&t, ggml_dtype(kind))?;
    Ok(q.data()?.into_owned())
}

/// Candle's dequantization of raw blocks, for cross-checking our decoder.
pub fn dequantize(n: usize, k: usize, kind: GgmlKind, bytes: &[u8]) -> Result<Vec<f32>> {
    let st = QStorage::from_data(Cow::Borrowed(bytes), &Device::Cpu, ggml_dtype(kind))?;
    QTensor::new(st, (n, k))?.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1()
}
