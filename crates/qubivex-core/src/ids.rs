//! Newtype identifiers. Bare integers and strings never cross crate boundaries
//! (RUST_GUIDELINES §6).

use std::fmt;

use serde::{Deserialize, Serialize};

/// Index of a transformer layer within a model, starting at 0.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerIdx(pub u32);

/// Index of a routed expert within one MoE layer, starting at 0.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExpertIdx(pub u32);

/// Key of a loaded model, interned by the runtime.
///
/// It is cheap to copy and hash, unlike a `String` model id, because it is hashed on
/// every expert lookup (QUBIVEX_DESIGN §4.1).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelKey(pub u32);

/// Globally unique identifier of one routed expert of one loaded model.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct ExpertId {
    /// The model the expert belongs to.
    pub model: ModelKey,
    /// The layer the expert belongs to.
    pub layer: LayerIdx,
    /// The expert's index within the layer.
    pub expert: ExpertIdx,
}

// ExpertId is hashed layers × top-K times per token. Keep growth deliberate
// (RUST_GUIDELINES §5, mem-assert-type-size).
const _: () = assert!(size_of::<ExpertId>() == 12);

impl ExpertId {
    /// Builds an expert id.
    #[must_use]
    pub const fn new(model: ModelKey, layer: LayerIdx, expert: ExpertIdx) -> Self {
        Self {
            model,
            layer,
            expert,
        }
    }
}

/// The kind of compute device.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum DeviceKind {
    /// The host CPU.
    Cpu,
    /// An NVIDIA GPU through CUDA.
    Cuda,
    /// A GPU through Vulkan.
    Vulkan,
    /// An Apple GPU through Metal.
    Metal,
}

impl DeviceKind {
    /// The lowercase name used in logs and configuration.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
        }
    }
}

/// A compute device: a kind and an ordinal within that kind.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct DeviceId {
    /// The device kind.
    pub kind: DeviceKind,
    /// The ordinal among devices of the same kind, starting at 0.
    pub ordinal: u32,
}

impl DeviceId {
    /// The host CPU.
    pub const CPU: Self = Self {
        kind: DeviceKind::Cpu,
        ordinal: 0,
    };

    /// Builds a device id.
    #[must_use]
    pub const fn new(kind: DeviceKind, ordinal: u32) -> Self {
        Self { kind, ordinal }
    }
}

macro_rules! display_as_inner {
    ($($t:ty),*) => {$(
        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
    )*};
}
display_as_inner!(LayerIdx, ExpertIdx, ModelKey);

impl fmt::Display for ExpertId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m{}/L{}/E{}", self.model, self.layer, self.expert)
    }
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind, self.ordinal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_id_display() {
        let id = ExpertId::new(ModelKey(1), LayerIdx(15), ExpertIdx(63));
        assert_eq!(id.to_string(), "m1/L15/E63");
    }

    #[test]
    fn device_id_display() {
        assert_eq!(DeviceId::CPU.to_string(), "cpu:0");
        assert_eq!(DeviceId::new(DeviceKind::Cuda, 1).to_string(), "cuda:1");
    }

    #[test]
    fn ids_serialize_transparently() {
        assert_eq!(serde_json::to_string(&LayerIdx(3)).unwrap(), "3");
        let id = ExpertId::new(ModelKey(0), LayerIdx(2), ExpertIdx(5));
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, r#"{"model":0,"layer":2,"expert":5}"#);
        assert_eq!(serde_json::from_str::<ExpertId>(&json).unwrap(), id);
    }

    #[test]
    fn device_kind_serializes_lowercase() {
        let d = DeviceId::new(DeviceKind::Vulkan, 0);
        assert_eq!(
            serde_json::to_string(&d).unwrap(),
            r#"{"kind":"vulkan","ordinal":0}"#
        );
    }
}
