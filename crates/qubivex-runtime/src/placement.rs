//! Where each matmul runs: the execution plan's device placement (ADR-007, ADR-011 §5).
//!
//! Placement is fixed when the engine loads, by weight: the dense matmuls of the kinds
//! [`MetalOps`] selects run on Metal, over the dense arena wrapped in place (no copy);
//! everything else, routed experts included, runs on the CPU. Live load never moves an
//! operation. Metal gives the CPU's bits (ADR-011 §1), so placement changes speed only.

use std::collections::HashMap;
use std::sync::Arc;

use qubivex_arch::DenseMatmul;
use qubivex_backend_cpu::CpuBackend;
use qubivex_backend_metal::{MetalBackend, MetalWeights, ResidentMatMul};
use qubivex_compute::host::PageAligned;
use qubivex_compute::{ComputeBackend, ComputeError, MatMul, WeightView};
use qubivex_core::DType;

/// The compute backend an engine uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendKind {
    /// Every operation on the CPU.
    #[default]
    Cpu,
    /// The dense matmuls [`MetalOps`] selects on the Metal GPU, the rest on the CPU.
    Metal,
}

impl BackendKind {
    /// The name used on the command line and in records.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

/// What runs on Metal with [`BackendKind::Metal`].
///
/// The default is whole layers ([`MetalOps::layers`]): one submission per layer, with
/// the norms, attention and the routed experts on the GPU (ADR-011 amendment 2). That
/// is the only placement that beats the CPU: a submission costs about 110 µs on the M3
/// Pro, so placing single dense matmuls (attention, router, head) only adds
/// submissions while the GPU idles in between (amendment 1). Those placements remain
/// for comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalOps {
    /// The q, k, v and output projections.
    pub attention: bool,
    /// The routers.
    pub router: bool,
    /// The output head.
    pub head: bool,
    /// Whole layers: every dense weight, the norms, attention and the routed experts,
    /// one submission per layer (ADR-011 amendment 2). Implies the other three.
    pub layers: bool,
}

impl Default for MetalOps {
    fn default() -> Self {
        Self {
            attention: false,
            router: false,
            head: false,
            layers: true,
        }
    }
}

impl MetalOps {
    /// Whether matmuls of `kind` are selected.
    #[must_use]
    pub const fn selects(self, kind: DenseMatmul) -> bool {
        self.layers
            || match kind {
                DenseMatmul::Attention => self.attention,
                DenseMatmul::Router => self.router,
                DenseMatmul::Head => self.head,
                // A dense FFN is placed with whole layers only.
                DenseMatmul::Ffn => false,
            }
    }

    /// The selected kinds, comma-separated (`head`, `attention,router,head`, `layers`
    /// or `none`).
    #[must_use]
    pub fn describe(self) -> String {
        if self.layers {
            return "layers".to_owned();
        }
        let v: Vec<&str> = [
            (self.attention, "attention"),
            (self.router, "router"),
            (self.head, "head"),
        ]
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, n)| *n)
        .collect();
        if v.is_empty() {
            "none".to_owned()
        } else {
            v.join(",")
        }
    }
}

/// What the plan placed on a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementReport {
    /// The backend asked for.
    pub backend: BackendKind,
    /// The device name, for a GPU backend.
    pub device: Option<String>,
    /// The kinds of matmul on the device.
    pub ops: Option<MetalOps>,
    /// Weight matrices on the device.
    pub tensors: usize,
    /// Their bytes (host bytes used in place: no copy and no extra ledger charge).
    pub bytes: u64,
    /// Why the plan ran the CPU pass although a device was asked for, if it did.
    pub fallback: Option<String>,
}

struct OnMetal {
    backend: MetalBackend,
    /// By the address of the weight's first byte in the dense arena.
    weights: HashMap<usize, MetalWeights>,
}

/// The engine's backend: the CPU, plus the weights the plan placed on Metal.
pub(crate) struct Placed {
    cpu: CpuBackend,
    metal: Option<OnMetal>,
    report: PlacementReport,
}

impl std::fmt::Debug for Placed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Placed")
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

impl Placed {
    /// Every operation on the CPU.
    pub(crate) fn cpu() -> Self {
        Self {
            cpu: CpuBackend::new(),
            metal: None,
            report: PlacementReport {
                backend: BackendKind::Cpu,
                device: None,
                ops: None,
                tensors: 0,
                bytes: 0,
                fallback: None,
            },
        }
    }

    /// Every operation on the CPU although Metal was asked for, because the plan found
    /// the CPU faster; `reason` says why.
    pub(crate) fn fallback(device: String, ops: MetalOps, reason: String) -> Self {
        let mut p = Self::cpu();
        p.report = PlacementReport {
            backend: BackendKind::Metal,
            device: Some(device),
            ops: Some(ops),
            tensors: 0,
            bytes: 0,
            fallback: Some(reason),
        };
        p
    }

    /// A Metal report for the whole-layer pass, which places its own weights; every
    /// matmul this backend is asked for (the embedding lookup) stays on the CPU.
    pub(crate) fn layers(device: String, tensors: usize, bytes: u64, ops: MetalOps) -> Self {
        Self {
            cpu: CpuBackend::new(),
            metal: None,
            report: PlacementReport {
                backend: BackendKind::Metal,
                device: Some(device),
                ops: Some(ops),
                tensors,
                bytes,
                fallback: None,
            },
        }
    }

    /// The `weights` (views into `arena`) of the selected kinds on Metal, the arena
    /// wrapped in place. Weights whose dtype has no Metal kernel stay on the CPU.
    pub(crate) fn metal<'a>(
        arena: &Arc<PageAligned>,
        weights: impl IntoIterator<Item = (DenseMatmul, WeightView<'a>)>,
        ops: MetalOps,
    ) -> Result<Self, ComputeError> {
        let backend = MetalBackend::new().map_err(ComputeError::from)?;
        let host = backend.map_host(Arc::clone(arena))?;
        let mut on = HashMap::new();
        let mut bytes = 0u64;
        for (kind, w) in weights {
            if !ops.selects(kind) || !qubivex_backend_metal::supported(w.dtype()) {
                continue;
            }
            let key = w.bytes().as_ptr() as usize;
            if on.contains_key(&key) {
                continue;
            }
            let r = host.weights(&w)?.ok_or_else(|| ComputeError::Device {
                backend: "metal",
                detail: "a placed weight is outside the dense arena".into(),
            })?;
            bytes += w.bytes().len() as u64;
            on.insert(key, r);
        }
        let report = PlacementReport {
            backend: BackendKind::Metal,
            device: Some(backend.info().name),
            ops: Some(ops),
            tensors: on.len(),
            bytes,
            fallback: None,
        };
        Ok(Self {
            cpu: CpuBackend::new(),
            metal: Some(OnMetal {
                backend,
                weights: on,
            }),
            report,
        })
    }

    pub(crate) const fn report(&self) -> &PlacementReport {
        &self.report
    }

    /// The device and resident weights for `w`, if the plan placed it.
    fn placed(&self, w: &WeightView<'_>) -> Option<(&MetalBackend, &MetalWeights)> {
        let m = self.metal.as_ref()?;
        let r = m.weights.get(&(w.bytes().as_ptr() as usize))?;
        (r.dtype() == w.dtype() && r.rows() == w.rows() && r.cols() == w.cols())
            .then_some((&m.backend, r))
    }
}

impl ComputeBackend for Placed {
    fn name(&self) -> &'static str {
        if self.metal.is_some() {
            "cpu+metal"
        } else {
            "cpu"
        }
    }

    fn supports(&self, dtype: DType) -> bool {
        self.cpu.supports(dtype)
    }

    fn matmul(
        &self,
        w: &WeightView<'_>,
        xs: &[f32],
        s: usize,
        ys: &mut [f32],
    ) -> Result<(), ComputeError> {
        match self.placed(w) {
            Some((m, r)) => m.matmul_resident(r, xs, s, ys),
            None => self.cpu.matmul(w, xs, s, ys),
        }
    }

    /// One Metal submission when every op is placed there; otherwise op by op.
    fn matmul_many(&self, ops: &mut [MatMul<'_>]) -> Result<(), ComputeError> {
        let placed: Option<Vec<_>> = ops.iter().map(|op| self.placed(&op.w)).collect();
        match placed {
            Some(p) if !p.is_empty() => {
                let m = p[0].0;
                let mut batch: Vec<ResidentMatMul<'_>> = ops
                    .iter_mut()
                    .zip(&p)
                    .map(|(op, (_, r))| ResidentMatMul {
                        w: r,
                        xs: op.xs,
                        s: op.s,
                        ys: &mut *op.ys,
                    })
                    .collect();
                m.matmul_many_resident(&mut batch)
            }
            _ => {
                for op in ops {
                    self.matmul(&op.w, op.xs, op.s, op.ys)?;
                }
                Ok(())
            }
        }
    }

    fn dequant_row(
        &self,
        w: &WeightView<'_>,
        r: usize,
        out: &mut [f32],
    ) -> Result<(), ComputeError> {
        self.cpu.dequant_row(w, r, out)
    }
}
