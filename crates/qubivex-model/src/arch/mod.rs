//! Built-in architectures.

mod transformer_moe;

pub use transformer_moe::{
    ExpertLayout, MoeConfig, QkNorm, RouterScoring, TensorNames, TransformerMoe,
};
