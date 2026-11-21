//! Built-in architectures.

mod common;
mod transformer_dense;
mod transformer_moe;

pub use common::RopeScaling;

pub use transformer_dense::{DenseConfig, DenseTensorNames, TransformerDense};
pub(crate) use transformer_moe::names as moe_names;
pub use transformer_moe::{
    ExpertActivation, ExpertLayout, MoeConfig, QkNorm, RouterScoring, TensorNames, TransformerMoe,
};
