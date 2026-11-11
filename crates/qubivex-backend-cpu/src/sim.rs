//! A simulated discrete GPU (ADR-018): a device with its own, limited memory, which
//! computes with this crate's kernels.
//!
//! It exists so the code around a discrete device (VRAM planning, the ledger's VRAM
//! tier, uploads, a device expert cache and its eviction) can be built and tested on a
//! machine without CUDA or Vulkan. What it simulates:
//!
//! - **Separate memory.** Every upload copies the host bytes into a new allocation that
//!   only this device reads. A matmul on a buffer of another device is an error, and
//!   nothing reads host weights, so a missing upload fails instead of passing silently.
//! - **A hard capacity.** An allocation that would exceed it fails with
//!   [`ComputeError::OutOfDeviceMemory`], as `cudaMalloc` would.
//! - **Transfer cost.** Copies are counted, and their time is modelled from a bandwidth
//!   and a per-copy latency. With [`SimConfig::delay`] the device also sleeps for that
//!   time, so overlap and prefetch can be measured.
//!
//! - **Arenas.** [`DiscreteDevice::alloc`] and [`DiscreteDevice::write`] keep many copies
//!   in one allocation, as a device expert cache's fixed-slot arenas do.
//! - **A fused expert block**, when [`SimConfig::fuse_rows`] asks for one: a layer's
//!   gated feed-forward blocks are one submission, so the fused path above the device
//!   is exercised and tested here. Off by default.
//! - **Routing on the device**, when [`SimConfig::route_rows`] asks for it: a MoE layer's
//!   router, routing, experts and combine are one submission (Step 3b), so the routed
//!   path above the device is tested here. Off by default.
//!
//! What it does not simulate: kernel speed (the kernels are the CPU's), asynchronous
//! submission, and alignment or page-size rules of a real driver.
//!
//! Bits are the CPU's by construction: the same kernels on the same bytes.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use qubivex_compute::discrete::{
    DeviceBuffer, DeviceGatedFfn, DeviceMatMul, DeviceRoutedMoe, DeviceUsage, DiscreteDevice,
    Guard, TransferStats,
};
use qubivex_compute::host::PageAligned;
use qubivex_compute::{ComputeBackend, ComputeError, WeightView};
use qubivex_core::DType;

use crate::{CpuBackend, formats};

/// How a [`SimDevice`] behaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimConfig {
    /// Device memory in bytes.
    pub capacity: u64,
    /// Host-to-device bandwidth in bytes per second (0: copies take no modelled time).
    pub bandwidth: u64,
    /// Fixed cost of each copy.
    pub latency: Duration,
    /// Sleep for the modelled transfer time, instead of only counting it.
    pub delay: bool,
    /// Fuse a layer's expert blocks into one submission when no block has more than
    /// this many activation rows ([`DiscreteDevice::fuses_gated_ffn`]); 0: never.
    pub fuse_rows: usize,
    /// Route MoE layers of at most this many tokens on the device, as one submission
    /// ([`DiscreteDevice::routes_moe`]); 0: never.
    pub route_rows: usize,
    /// Route those layers with their prelude, the O projection, residual and norm, in
    /// the same submission ([`DiscreteDevice::routes_moe_prelude`], Step 3c).
    pub route_prelude: bool,
}

impl SimConfig {
    /// PCIe 4.0 x16 in practice: about 25 GB/s and 10 µs per copy.
    pub const PCIE4_BANDWIDTH: u64 = 25_000_000_000;

    /// A device of `capacity` bytes with PCIe 4.0 transfer costs, counted but not
    /// slept.
    #[must_use]
    pub const fn new(capacity: u64) -> Self {
        Self {
            capacity,
            bandwidth: Self::PCIE4_BANDWIDTH,
            latency: Duration::from_micros(10),
            delay: false,
            fuse_rows: 0,
            route_rows: 0,
            route_prelude: false,
        }
    }

    /// The modelled time to copy `bytes`.
    #[must_use]
    pub fn transfer_time(&self, bytes: u64) -> Duration {
        let wire = if self.bandwidth == 0 {
            0
        } else {
            u64::try_from(u128::from(bytes) * 1_000_000_000 / u128::from(self.bandwidth))
                .unwrap_or(u64::MAX)
        };
        self.latency.saturating_add(Duration::from_nanos(wire))
    }
}

#[derive(Debug, Default)]
struct Counters {
    used: AtomicU64,
    peak: AtomicU64,
    refused: AtomicU64,
    uploads: AtomicU64,
    upload_bytes: AtomicU64,
    activations_in: AtomicU64,
    activations_out: AtomicU64,
    submissions: AtomicU64,
    transfer_ns: AtomicU64,
}

/// One allocation's memory. Dropping it returns the bytes to the device.
struct SimBuffer {
    device: usize,
    len: usize,
    bytes: Memory,
    counters: Arc<Counters>,
}

/// An upload's bytes never change. An arena's are written later, so they sit behind a
/// lock, held only to copy bytes in or out: never while computing, because the CPU
/// kernels run on rayon, and a thread waiting inside them may pick up a task that
/// writes the same arena.
enum Memory {
    Fixed(PageAligned),
    Arena(RwLock<PageAligned>),
}

impl Drop for SimBuffer {
    fn drop(&mut self) {
        self.counters
            .used
            .fetch_sub(self.len as u64, Ordering::Relaxed);
    }
}

/// A simulated discrete GPU. See the module documentation.
#[derive(Debug)]
pub struct SimDevice {
    id: usize,
    config: SimConfig,
    counters: Arc<Counters>,
    cpu: CpuBackend,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

impl SimDevice {
    /// A device configured by `config`.
    #[must_use]
    pub fn new(config: SimConfig) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            config,
            counters: Arc::default(),
            cpu: CpuBackend::new(),
        }
    }

    /// The configuration.
    #[must_use]
    pub const fn config(&self) -> &SimConfig {
        &self.config
    }

    /// Reserves `bytes` of capacity, or refuses.
    fn reserve(&self, bytes: u64) -> Result<(), ComputeError> {
        let c = &self.counters;
        let mut used = c.used.load(Ordering::Relaxed);
        loop {
            let next = used
                .checked_add(bytes)
                .filter(|&n| n <= self.config.capacity);
            let Some(next) = next else {
                c.refused.fetch_add(1, Ordering::Relaxed);
                return Err(ComputeError::OutOfDeviceMemory {
                    backend: "sim",
                    requested: bytes,
                    used,
                    capacity: self.config.capacity,
                });
            };
            match c
                .used
                .compare_exchange_weak(used, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    c.peak.fetch_max(next, Ordering::Relaxed);
                    return Ok(());
                }
                Err(now) => used = now,
            }
        }
    }

    /// Counts a copy of `bytes`, and sleeps for it if configured to.
    fn transfer(&self, bytes: u64) {
        let t = self.config.transfer_time(bytes);
        self.counters.transfer_ns.fetch_add(
            u64::try_from(t.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        if self.config.delay && !t.is_zero() {
            std::thread::sleep(t);
        }
    }

    fn buffer_of<'b>(&self, b: &'b DeviceBuffer) -> Result<&'b SimBuffer, ComputeError> {
        match b.state::<SimBuffer>() {
            Some(s) if s.device == self.id => Ok(s),
            _ => Err(ComputeError::Device {
                backend: "sim",
                detail: "the buffer belongs to another device".into(),
            }),
        }
    }

    /// New zeroed memory of `len` bytes, as an upload's or an arena's.
    fn allocate(
        &self,
        len: usize,
        guard: Option<Guard>,
        arena: bool,
    ) -> Result<DeviceBuffer, ComputeError> {
        self.reserve(len as u64)?;
        let bytes = match PageAligned::try_zeroed(len) {
            Ok(b) => b,
            Err(e) => {
                self.counters.used.fetch_sub(len as u64, Ordering::Relaxed);
                return Err(ComputeError::Device {
                    backend: "sim",
                    detail: format!("cannot allocate {len} bytes of host memory: {e}"),
                });
            }
        };
        let state = SimBuffer {
            device: self.id,
            len,
            bytes: if arena {
                Memory::Arena(RwLock::new(bytes))
            } else {
                Memory::Fixed(bytes)
            },
            counters: Arc::clone(&self.counters),
        };
        Ok(DeviceBuffer::new(len, state, guard))
    }

    /// Counts one copy to the device of `len` bytes.
    fn uploaded(&self, len: u64) {
        self.counters.uploads.fetch_add(1, Ordering::Relaxed);
        self.counters.upload_bytes.fetch_add(len, Ordering::Relaxed);
        self.transfer(len);
    }

    /// One matmul on the device's copy of its weights.
    fn matmul_one(&self, op: &mut DeviceMatMul<'_>) -> Result<(), ComputeError> {
        let range = op.w.offset()..op.w.offset() + op.w.len();
        let copy;
        let data = match &self.buffer_of(op.w.buffer())?.bytes {
            Memory::Fixed(b) => &b.as_slice()[range],
            Memory::Arena(lock) => {
                copy = lock
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .as_slice()[range]
                    .to_vec();
                &copy[..]
            }
        };
        let w = WeightView::new(op.w.dtype(), op.w.rows(), op.w.cols(), data)?;
        self.cpu.matmul(&w, op.xs, op.s, op.ys)
    }

    /// Counts activations copied in and out by a submission, as one copy each way.
    fn moved(&self, inb: u64, outb: u64) {
        let c = &self.counters;
        c.activations_in.fetch_add(inb, Ordering::Relaxed);
        c.activations_out.fetch_add(outb, Ordering::Relaxed);
        if inb + outb > 0 {
            self.transfer(inb + outb);
        }
    }
}

impl DiscreteDevice for SimDevice {
    fn name(&self) -> &'static str {
        "sim"
    }

    /// The simulator groups, so the grouped expert path stays exercised and tested.
    fn groups_matmuls(&self) -> bool {
        true
    }

    fn describe(&self) -> String {
        #[allow(clippy::cast_precision_loss)]
        let gb = |b: u64| b as f64 / 1e9;
        format!(
            "simulated discrete GPU ({}, {:.1} GB/s, {} µs per copy{})",
            qubivex_core::Bytes(self.config.capacity),
            gb(self.config.bandwidth),
            self.config.latency.as_micros(),
            if self.config.delay { ", slept" } else { "" }
        )
    }

    fn supports(&self, dtype: DType) -> bool {
        formats::supported(dtype)
    }

    fn upload(&self, host: &[u8], guard: Option<Guard>) -> Result<DeviceBuffer, ComputeError> {
        let len = host.len() as u64;
        self.reserve(len)?;
        let mut bytes = match PageAligned::try_zeroed(host.len()) {
            Ok(b) => b,
            Err(e) => {
                self.counters.used.fetch_sub(len, Ordering::Relaxed);
                return Err(ComputeError::Device {
                    backend: "sim",
                    detail: format!("cannot allocate {len} bytes of host memory: {e}"),
                });
            }
        };
        bytes.as_mut_slice().copy_from_slice(host);
        let state = SimBuffer {
            device: self.id,
            len: host.len(),
            bytes: Memory::Fixed(bytes),
            counters: Arc::clone(&self.counters),
        };
        self.uploaded(len);
        Ok(DeviceBuffer::new(host.len(), state, guard))
    }

    fn alloc(&self, len: usize, guard: Option<Guard>) -> Result<DeviceBuffer, ComputeError> {
        self.allocate(len, guard, true)
    }

    fn write(&self, buffer: &DeviceBuffer, offset: usize, host: &[u8]) -> Result<(), ComputeError> {
        let b = self.buffer_of(buffer)?;
        let Memory::Arena(lock) = &b.bytes else {
            return Err(ComputeError::Device {
                backend: "sim",
                detail: "only an arena (alloc) can be written".into(),
            });
        };
        let mut bytes = lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dst = offset
            .checked_add(host.len())
            .and_then(|end| bytes.as_mut_slice().get_mut(offset..end))
            .ok_or_else(|| ComputeError::Device {
                backend: "sim",
                detail: format!(
                    "{} bytes at {offset} are outside a {}-byte buffer",
                    host.len(),
                    b.len
                ),
            })?;
        dst.copy_from_slice(host);
        drop(bytes);
        self.uploaded(host.len() as u64);
        Ok(())
    }

    fn matmul_many(&self, ops: &mut [DeviceMatMul<'_>]) -> Result<(), ComputeError> {
        self.counters.submissions.fetch_add(1, Ordering::Relaxed);
        let (mut inb, mut outb) = (0u64, 0u64);
        for op in ops.iter_mut() {
            inb += size_of_val(op.xs) as u64;
            outb += size_of_val(op.ys) as u64;
            self.matmul_one(op)?;
        }
        self.moved(inb, outb);
        Ok(())
    }

    fn fuses_gated_ffn(&self, blocks: usize, rows: usize) -> bool {
        blocks > 0 && rows <= self.config.fuse_rows
    }

    /// One submission: the intermediates stay on the device, so only each block's
    /// inputs and outputs are copied.
    fn gated_ffn_many(&self, ops: &mut [DeviceGatedFfn<'_>]) -> Result<(), ComputeError> {
        self.counters.submissions.fetch_add(1, Ordering::Relaxed);
        let (mut inb, mut outb) = (0u64, 0u64);
        for op in ops.iter_mut() {
            inb += size_of_val(op.xs) as u64;
            outb += size_of_val(op.ys) as u64;
            let ffn = op.gate.rows();
            let mut g = vec![0f32; op.s * ffn];
            let mut u = vec![0f32; op.s * ffn];
            for (w, ys) in [(op.gate, &mut g), (op.up, &mut u)] {
                self.matmul_one(&mut DeviceMatMul {
                    w,
                    xs: op.xs,
                    s: op.s,
                    ys,
                })?;
            }
            qubivex_compute::math::swiglu(&mut g, &u);
            self.matmul_one(&mut DeviceMatMul {
                w: op.down,
                xs: &g,
                s: op.s,
                ys: op.ys,
            })?;
        }
        self.moved(inb, outb);
        Ok(())
    }

    fn routes_moe(&self, experts: usize, k: usize, s: usize) -> bool {
        k > 0 && k <= experts && s > 0 && s <= self.config.route_rows
    }

    fn routes_moe_prelude(&self, experts: usize, k: usize, s: usize) -> bool {
        self.config.route_prelude && self.routes_moe(experts, k, s)
    }

    /// One submission: the router logits, the routing and the expert intermediates stay
    /// on the device; the inputs, the combined outputs and the routes are copied.
    fn routed_moe(&self, op: &mut DeviceRoutedMoe<'_>) -> Result<bool, ComputeError> {
        if !self.routes_moe(op.experts.len(), op.k, op.s) {
            return Ok(false);
        }
        if op.prelude.is_some() && !self.config.route_prelude {
            return Ok(false);
        }
        let (ne, cols, s, k) = (op.router.rows(), op.router.cols(), op.s, op.k);
        let rows = op.ys.len().checked_div(s).unwrap_or(0);
        let xs_len = if op.prelude.is_some() { 0 } else { s * cols };
        if op.experts.len() != ne
            || op.xs.len() != xs_len
            || op.ids.len() != s * k
            || op.prelude.as_ref().is_some_and(|p| {
                p.o.rows() != cols
                    || p.att.len() != s * p.o.cols()
                    || p.hs.len() != s * cols
                    || p.norm.len() != cols
            })
        {
            return Err(ComputeError::Shape {
                op: "routed moe",
                detail: format!("{ne}-row router, {} experts, {s} tokens", op.experts.len()),
            });
        }
        self.counters.submissions.fetch_add(1, Ordering::Relaxed);
        // The prelude: the O projection, residual and norm, in the same submission.
        let (normed, pre_in, pre_out) = match op.prelude.as_mut() {
            Some(p) => {
                let mut o = vec![0f32; s * cols];
                self.matmul_one(&mut DeviceMatMul {
                    w: p.o,
                    xs: p.att,
                    s,
                    ys: &mut o,
                })?;
                let x = qubivex_compute::residual_norm(&o, p.hs, p.norm, p.eps);
                let inb = size_of_val(p.att) + size_of_val(&*p.hs) + size_of_val(p.norm);
                (Some(x), inb as u64, size_of_val(&*p.hs) as u64)
            }
            None => (None, 0, 0),
        };
        let xs: &[f32] = normed.as_deref().unwrap_or(op.xs);
        let mut logits = vec![0f32; s * ne];
        self.matmul_one(&mut DeviceMatMul {
            w: op.router,
            xs,
            s,
            ys: &mut logits,
        })?;
        qubivex_compute::route_tokens(&logits, s, k, op.renormalise, op.ids, op.weights);
        // Entry-major, as a device does it: one row per (token, rank).
        let mut contrib = vec![0f32; s * k * rows];
        for (j, &e) in op.ids.iter().enumerate() {
            let t = j / k;
            let [gate, up, down] = &op.experts[e as usize];
            let ffn = gate.rows();
            let (mut g, mut u) = (vec![0f32; ffn], vec![0f32; ffn]);
            let x = &xs[t * cols..(t + 1) * cols];
            for (w, ys) in [(gate, &mut g), (up, &mut u)] {
                self.matmul_one(&mut DeviceMatMul { w, xs: x, s: 1, ys })?;
            }
            qubivex_compute::math::swiglu(&mut g, &u);
            self.matmul_one(&mut DeviceMatMul {
                w: down,
                xs: &g,
                s: 1,
                ys: &mut contrib[j * rows..(j + 1) * rows],
            })?;
        }
        qubivex_compute::combine_ranks(&contrib, op.weights, k, op.ys);
        self.moved(
            size_of_val(op.xs) as u64 + pre_in,
            (size_of_val(op.ys) + size_of_val(op.ids) + size_of_val(op.weights)) as u64 + pre_out,
        );
        Ok(true)
    }

    fn usage(&self) -> DeviceUsage {
        let c = &self.counters;
        DeviceUsage {
            capacity: self.config.capacity,
            used: c.used.load(Ordering::Relaxed),
            peak: c.peak.load(Ordering::Relaxed),
            refused: c.refused.load(Ordering::Relaxed),
        }
    }

    fn transfers(&self) -> TransferStats {
        let c = &self.counters;
        TransferStats {
            uploads: c.uploads.load(Ordering::Relaxed),
            upload_bytes: c.upload_bytes.load(Ordering::Relaxed),
            activations_in: c.activations_in.load(Ordering::Relaxed),
            activations_out: c.activations_out.load(Ordering::Relaxed),
            submissions: c.submissions.load(Ordering::Relaxed),
            transfer_ns: c.transfer_ns.load(Ordering::Relaxed),
            ..TransferStats::default()
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::unwrap_used)]
mod tests {
    use super::*;
    use qubivex_compute::discrete::DeviceWeights;
    use qubivex_core::GgmlType;

    /// Deterministic bytes for a weight of `dtype`, and its view.
    fn weights(dtype: DType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
        let n = dtype
            .storage_bytes(&qubivex_core::Shape::new(&[rows as u64, cols as u64]).unwrap())
            .unwrap() as usize;
        let mut x = seed.wrapping_mul(2_654_435_761).max(1);
        let mut b: Vec<u8> = (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 24) as u8
            })
            .collect();
        // Keep f16 scales of the block formats finite: clear exponent bits.
        if matches!(dtype, DType::Ggml(_)) {
            for v in b.iter_mut().skip(1).step_by(2) {
                *v &= 0x3b;
            }
        }
        b
    }

    fn acts(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect()
    }

    #[test]
    fn matmul_matches_the_cpu_bits() {
        let dev = SimDevice::new(SimConfig::new(1 << 24));
        for dtype in [
            DType::F32,
            DType::F16,
            DType::Ggml(GgmlType::Q8_0),
            DType::Ggml(GgmlType::Q4_K),
        ] {
            let (rows, cols, s) = (19, 512, 3);
            let host = weights(dtype, rows, cols, 7);
            let wv = WeightView::new(dtype, rows, cols, &host).unwrap();
            let xs = acts(s * cols);
            let mut want = vec![0.0; s * rows];
            CpuBackend::new().matmul(&wv, &xs, s, &mut want).unwrap();

            let buf = dev.upload(&host, None).unwrap();
            let dw = DeviceWeights::like(buf, 0, &wv).unwrap();
            let mut got = vec![0.0; s * rows];
            dev.matmul_many(&mut [DeviceMatMul {
                w: &dw,
                xs: &xs,
                s,
                ys: &mut got,
            }])
            .unwrap();
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{dtype}");
        }
    }

    #[test]
    fn capacity_is_enforced_and_freed_on_drop() {
        let dev = SimDevice::new(SimConfig::new(1000));
        let a = dev.upload(&[1; 600], None).unwrap();
        let clone = a.clone();
        assert!(matches!(
            dev.upload(&[2; 500], None),
            Err(ComputeError::OutOfDeviceMemory {
                requested: 500,
                used: 600,
                capacity: 1000,
                ..
            })
        ));
        drop(a);
        assert_eq!(dev.usage().used, 600, "a clone keeps the memory");
        drop(clone);
        let b = dev.upload(&[2; 900], None).unwrap();
        let u = dev.usage();
        assert_eq!((u.used, u.peak, u.refused), (900, 900, 1));
        drop(b);
        assert_eq!(dev.usage().used, 0);
    }

    #[test]
    fn guards_live_as_long_as_the_buffer() {
        let dev = SimDevice::new(SimConfig::new(100));
        let live = Arc::new(());
        let b = dev
            .upload(&[0; 10], Some(Box::new(Arc::clone(&live))))
            .unwrap();
        assert_eq!(Arc::strong_count(&live), 2);
        drop(b);
        assert_eq!(Arc::strong_count(&live), 1);
    }

    #[test]
    fn buffers_of_another_device_are_refused() {
        let a = SimDevice::new(SimConfig::new(1 << 20));
        let b = SimDevice::new(SimConfig::new(1 << 20));
        let host = weights(DType::F32, 2, 4, 1);
        let wv = WeightView::new(DType::F32, 2, 4, &host).unwrap();
        let dw = DeviceWeights::like(a.upload(&host, None).unwrap(), 0, &wv).unwrap();
        let mut ys = [0.0; 2];
        let e = b.matmul_many(&mut [DeviceMatMul {
            w: &dw,
            xs: &[1.0; 4],
            s: 1,
            ys: &mut ys,
        }]);
        assert!(matches!(e, Err(ComputeError::Device { .. })), "{e:?}");
    }

    #[test]
    fn transfers_are_counted_and_modelled() {
        let mut cfg = SimConfig::new(1 << 20);
        cfg.bandwidth = 1_000_000_000; // 1 GB/s: 1 ns per byte
        cfg.latency = Duration::from_micros(1);
        let dev = SimDevice::new(cfg);
        let host = weights(DType::F32, 4, 8, 3);
        let wv = WeightView::new(DType::F32, 4, 8, &host).unwrap();
        let dw = DeviceWeights::like(dev.upload(&host, None).unwrap(), 0, &wv).unwrap();
        let mut ys = [0.0; 8];
        dev.matmul_many(&mut [DeviceMatMul {
            w: &dw,
            xs: &acts(16),
            s: 2,
            ys: &mut ys,
        }])
        .unwrap();
        let t = dev.transfers();
        assert_eq!((t.uploads, t.upload_bytes), (1, 128));
        assert_eq!(
            (t.activations_in, t.activations_out, t.submissions),
            (64, 32, 1)
        );
        // Two copies: 128 bytes + 1 µs, then 96 bytes + 1 µs.
        assert_eq!(t.transfer_ns, 128 + 1000 + 96 + 1000);
    }

    #[test]
    fn weights_outside_the_buffer_are_refused() {
        let dev = SimDevice::new(SimConfig::new(1 << 20));
        let buf = dev.upload(&[0; 64], None).unwrap();
        assert!(DeviceWeights::new(buf.clone(), 0, DType::F32, 4, 4).is_ok());
        assert!(DeviceWeights::new(buf.clone(), 4, DType::F32, 4, 4).is_err());
        assert!(DeviceWeights::new(buf, usize::MAX, DType::F32, 1, 1).is_err());
    }
}
