//! Built-in architectures.

mod common;
mod transformer_dense;
mod transformer_moe;

pub use transformer_dense::{DenseConfig, DenseTensorNames, TransformerDense};
pub use transformer_moe::{
    ExpertLayout, MoeConfig, QkNorm, RouterScoring, TensorNames, TransformerMoe,
};
