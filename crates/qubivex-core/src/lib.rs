//! Core types shared by every QUBIVEX crate.
//!
//! This crate has no knowledge of models, devices or I/O. It defines:
//!
//! - newtype identifiers ([`LayerIdx`], [`ExpertIdx`], [`ModelKey`], [`ExpertId`],
//!   [`DeviceId`]),
//! - byte quantities ([`Bytes`]),
//! - tensor element types and their storage sizes ([`DType`], [`GgmlType`]),
//! - tensor shapes with overflow-checked arithmetic ([`Shape`]),
//! - tensor descriptors with byte strides and offsets, and zero-copy views ([`TensorDesc`]),
//! - the core error type ([`CoreError`]),
//! - typed configuration ([`config::QubivexConfig`]).
//!
//! All sizes that may be derived from untrusted model files are computed with checked
//! arithmetic and return [`CoreError`] on overflow (RUST_GUIDELINES §2).

mod bytes;
pub mod config;
mod dtype;
mod error;
mod ids;
mod shape;
mod tensor_desc;

pub use bytes::Bytes;
pub use dtype::{DType, GgmlType, StorageLayout};
pub use error::CoreError;
pub use ids::{DeviceId, DeviceKind, ExpertId, ExpertIdx, LayerIdx, ModelKey};
pub use shape::{MAX_RANK, Shape};
pub use tensor_desc::TensorDesc;

/// The QUBIVEX version string, taken from the workspace package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
