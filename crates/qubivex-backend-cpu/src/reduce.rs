//! The one reduction shape every dot product uses; it lives in
//! [`qubivex_compute::reduce`] so the forward pass's norms share it.

pub use qubivex_compute::reduce::{LANES, dot, reduce};
