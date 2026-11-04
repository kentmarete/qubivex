//! The objc2 FFI layer: device, queue, pipelines, buffers and dispatch.
//!
//! All unsafe code of the crate is here. Metal devices, command queues, pipeline
//! states and buffers are thread-safe objects (objc2-metal marks their protocols
//! `Send + Sync`). Command buffers and encoders are created, used and dropped inside
//! one call, on one thread.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLLibrary, MTLMathMode,
    MTLResourceOptions, MTLResourceUsage, MTLSize,
};

use qubivex_compute::host::PageAligned;

use crate::{DeviceInfo, MetalError};

/// Rows per threadgroup; must equal `ROWS` in `kernels.metal`.
pub(crate) const ROWS: usize = 8;
/// Rows per SIMD group of the K-quant matmul kernels; must equal `NR` in
/// `kernels.metal`.
pub(crate) const KQ_ROWS: usize = 2;

/// Activation rows per threadgroup of the token-tiled kernels; must equal `TILE` in
/// `kernels.metal`.
pub(crate) const TILE: usize = 8;
/// Activation rows from which a matmul uses the token-tiled kernel (ADR-011 amendment 5).
/// Measured on OLMoE (M3 Pro): from 4 rows the tiles win (6 tokens: 68.5 → 62.3 ms),
/// below they lose a little.
pub(crate) const TILE_MIN: usize = 4;

/// The pipeline to run a matmul of `s` activation rows with (`pipeline` is the
/// one-token kernel's `DTYPES` index), and its threadgroup rows: the token-tiled kernel
/// from [`TILE_MIN`] rows. Both give the same bits.
pub(crate) fn matmul_pipeline(pipeline: usize, s: usize) -> (usize, usize) {
    if s >= TILE_MIN {
        (
            crate::program::op("matmul_f32_t") + pipeline,
            s.div_ceil(TILE),
        )
    } else {
        (pipeline, s)
    }
}

/// Output rows one threadgroup of matmul `pipeline` (a `DTYPES` index) covers.
pub(crate) fn rows_per_group(pipeline: usize) -> usize {
    use qubivex_core::{DType, GgmlType};
    match crate::DTYPES.get(pipeline) {
        Some(DType::Ggml(GgmlType::Q4_K | GgmlType::Q6_K)) => ROWS * KQ_ROWS,
        _ => ROWS,
    }
}

/// Threads per SIMD group on Apple GPUs; the kernels' reduction assumes it.
pub(crate) const SIMD_WIDTH: usize = 32;

fn ns_error(e: &NSError) -> String {
    e.localizedDescription().to_string()
}

/// Kernel parameters; the layout matches `struct Params` in `kernels.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    n: u32,
    k: u32,
    s: u32,
    row_bytes: u32,
}

impl Params {
    /// Checked conversion of the matmul sizes to the kernel's 32-bit fields.
    pub(crate) fn new(n: usize, k: usize, s: usize, row_bytes: usize) -> Result<Self, MetalError> {
        let f = |v: usize, what: &'static str| {
            u32::try_from(v).map_err(|_| MetalError::Create {
                what: "dispatch",
                detail: Some(format!("{what} {v} exceeds 32 bits")),
            })
        };
        Ok(Self {
            n: f(n, "rows")?,
            k: f(k, "columns")?,
            s: f(s, "activation rows")?,
            row_bytes: f(row_bytes, "row bytes")?,
        })
    }
}

/// A shared-storage device buffer. A buffer that wraps host memory in place holds that
/// memory (`_host`) for as long as the buffer, or any clone of it, lives (ADR-011 §3).
#[derive(Clone, Debug)]
pub(crate) struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    _host: Option<Arc<PageAligned>>,
}

// SAFETY: an MTLBuffer object may be retained, released and bound from any thread
// (Metal resources are thread-safe objects; objc2-metal only leaves the protocol
// unmarked). Its contents are touched by the CPU only in `Device::run` under the
// scratch lock, after the GPU's last use has completed, and wrapped host memory
// (`_host`) is never written through a buffer.
unsafe impl Send for Buffer {}
// SAFETY: as for `Send`; `&Buffer` only allows binding and reading the length.
unsafe impl Sync for Buffer {}

impl Buffer {
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// The buffer's GPU virtual address (for tables of weights a kernel indexes).
    pub(crate) fn address(&self) -> u64 {
        self.raw.gpuAddress()
    }
}

/// One kernel dispatch of a batch: weights `w` from byte `w_offset`, activations `xs`
/// (`s × k`) and outputs `ys` (`s × n`). The caller has checked the shapes and that
/// `w` holds `n × row_bytes` bytes from `w_offset`.
pub(crate) struct Dispatch<'a> {
    pub(crate) pipeline: usize,
    pub(crate) w: &'a Buffer,
    pub(crate) w_offset: usize,
    pub(crate) xs: &'a [f32],
    pub(crate) ys: &'a mut [f32],
    pub(crate) params: Params,
}

/// Activation and output staging buffers, kept between submissions so a dispatch does
/// not allocate. Each holds the largest size asked for so far.
#[derive(Debug, Default)]
struct Scratch {
    inputs: Option<Buffer>,
    outputs: Option<Buffer>,
}

/// Offsets inside staging buffers are rounded to this many bytes.
const STAGE_ALIGN: usize = 256;

/// The device, its queue and one pipeline per kernel.
#[derive(Debug)]
pub(crate) struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// Submissions are serialised: one batch owns the staging buffers at a time.
    scratch: Mutex<Scratch>,
    /// Held while a recorded submission is in flight and while the host reads or writes
    /// a device vector, so the host never touches memory the GPU is using.
    gpu: Mutex<()>,
    /// Asynchronous submissions not yet waited for (see [`Device::submit_async`]).
    outstanding: AtomicUsize,
}

impl Device {
    /// The default device, with `source` compiled and a pipeline per entry of `kernels`.
    pub(crate) fn new(source: &str, kernels: &[&str]) -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| MetalError::Unavailable("no default Metal device".into()))?;
        let queue = device.newCommandQueue().ok_or(MetalError::Create {
            what: "command queue",
            detail: None,
        })?;
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Safe);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&opts))
            .map_err(|e| MetalError::Compile(ns_error(&e)))?;
        let mut pipelines = Vec::with_capacity(kernels.len());
        for name in kernels {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| MetalError::Create {
                    what: "kernel function",
                    detail: Some((*name).to_owned()),
                })?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| MetalError::Create {
                    what: "compute pipeline",
                    detail: Some(format!("{name}: {}", ns_error(&e))),
                })?;
            if p.threadExecutionWidth() != SIMD_WIDTH
                || p.maxTotalThreadsPerThreadgroup() < ROWS * SIMD_WIDTH
            {
                return Err(MetalError::Unavailable(format!(
                    "{name}: SIMD width {} and {} threads per threadgroup; the kernels need {SIMD_WIDTH} and {}",
                    p.threadExecutionWidth(),
                    p.maxTotalThreadsPerThreadgroup(),
                    ROWS * SIMD_WIDTH
                )));
            }
            pipelines.push(p);
        }
        Ok(Self {
            device,
            queue,
            pipelines,
            scratch: Mutex::new(Scratch::default()),
            gpu: Mutex::new(()),
            outstanding: AtomicUsize::new(0),
        })
    }

    pub(crate) fn info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name().to_string(),
            unified_memory: self.device.hasUnifiedMemory(),
            recommended_working_set: self.device.recommendedMaxWorkingSetSize(),
        }
    }

    /// A shared buffer holding a copy of `bytes` (which must not be empty).
    pub(crate) fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer, MetalError> {
        let ptr =
            NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>()).ok_or(MetalError::Create {
                what: "buffer",
                detail: Some("empty source".into()),
            })?;
        if bytes.is_empty() {
            return Err(MetalError::Create {
                what: "buffer",
                detail: Some("empty source".into()),
            });
        }
        // SAFETY: `ptr` is valid for reads of `bytes.len()` bytes for the duration of
        // the call, and Metal copies them (newBufferWithBytes never retains `ptr`).
        let raw = unsafe {
            self.device.newBufferWithBytes_length_options(
                ptr,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::Create {
            what: "buffer",
            detail: Some(format!("{} bytes", bytes.len())),
        })?;
        Ok(Buffer {
            raw,
            len: bytes.len(),
            _host: None,
        })
    }

    /// A buffer over `host`'s padded region, without a copy. The buffer holds `host`.
    pub(crate) fn wrap(&self, host: Arc<PageAligned>) -> Result<Buffer, MetalError> {
        let region = host.padded();
        let ptr = NonNull::new(region.as_ptr().cast_mut().cast::<c_void>()).ok_or(
            MetalError::Create {
                what: "buffer",
                detail: Some("null region".into()),
            },
        )?;
        // SAFETY: `region` is page-aligned and a whole number of pages (PageAligned's
        // invariant), as newBufferWithBytesNoCopy requires. The memory stays allocated
        // and unchanged while the buffer exists: the returned Buffer (and every clone)
        // holds the Arc, PageAligned never reallocates, and no `&mut` can be taken
        // through a shared Arc. With no deallocator, Metal never frees it.
        let raw = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    region.len(),
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or(MetalError::Create {
            what: "no-copy buffer",
            detail: Some(format!("{} bytes", region.len())),
        })?;
        Ok(Buffer {
            len: region.len(),
            raw,
            _host: Some(host),
        })
    }

    pub(crate) fn buffer_zeroed(&self, len: usize) -> Result<Buffer, MetalError> {
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Create {
                what: "buffer",
                detail: Some(format!("{len} bytes")),
            })?;
        Ok(Buffer {
            raw,
            len,
            _host: None,
        })
    }

    /// A staging buffer of at least `need` bytes, reusing `slot`'s if it is large enough.
    fn stage(&self, slot: &mut Option<Buffer>, need: usize) -> Result<Buffer, MetalError> {
        if let Some(b) = slot.as_ref().filter(|b| b.len >= need) {
            return Ok(b.clone());
        }
        let b = self.buffer_zeroed(need.max(64 * 1024).next_power_of_two())?;
        *slot = Some(b.clone());
        Ok(b)
    }

    /// Runs `ops` in one command buffer and waits for it (ADR-011 §2). Activations that
    /// are the same slice (for example the shared input of the q, k and v projections)
    /// are staged once.
    pub(crate) fn run(&self, ops: &mut [Dispatch<'_>]) -> Result<(), MetalError> {
        if ops.is_empty() {
            return Ok(());
        }
        let round = |b: usize| b.div_ceil(STAGE_ALIGN) * STAGE_ALIGN;
        // Staging layout: each distinct input once, each output in order.
        let mut x_at: Vec<usize> = Vec::with_capacity(ops.len());
        let mut x_seen: Vec<(*const f32, usize, usize)> = Vec::new();
        let mut x_len = 0;
        let mut y_at: Vec<usize> = Vec::with_capacity(ops.len());
        let mut y_len = 0;
        for op in ops.iter() {
            let key = (op.xs.as_ptr(), op.xs.len());
            let at = if let Some(&(_, _, at)) = x_seen.iter().find(|(p, l, _)| (*p, *l) == key) {
                at
            } else {
                let at = x_len;
                x_seen.push((key.0, key.1, at));
                x_len += round(core::mem::size_of_val(op.xs));
                at
            };
            x_at.push(at);
            y_at.push(y_len);
            y_len += round(core::mem::size_of_val(op.ys));
        }
        let mut scratch = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let xbuf = self.stage(&mut scratch.inputs, x_len)?;
        let ybuf = self.stage(&mut scratch.outputs, y_len)?;
        let xbase = xbuf.raw.contents().as_ptr().cast::<u8>();
        for &(p, l, at) in &x_seen {
            // SAFETY: no command buffer uses the staging buffers (the scratch lock is
            // held and every earlier submission was waited for). `at + 4l <= x_len <=
            // xbuf.len`, and `p` points to `l` f32 of a distinct host allocation.
            unsafe {
                core::ptr::copy_nonoverlapping(p.cast::<u8>(), xbase.add(at), 4 * l);
            }
        }
        let cmd = self.queue.commandBuffer().ok_or(MetalError::Create {
            what: "command buffer",
            detail: None,
        })?;
        let enc = cmd.computeCommandEncoder().ok_or(MetalError::Create {
            what: "compute encoder",
            detail: None,
        })?;
        for (i, op) in ops.iter().enumerate() {
            let (pipeline, height) = matmul_pipeline(op.pipeline, op.params.s as usize);
            enc.setComputePipelineState(&self.pipelines[pipeline]);
            let params = NonNull::from(&op.params).cast::<c_void>();
            // SAFETY: indices 0..=3 match the kernel's [[buffer(n)]] bindings and types
            // (bytes, f32, f32, Params). All three buffers live until
            // `waitUntilCompleted` returns below; `params` is copied by setBytes. The
            // kernel reads `n × row_bytes` weight bytes from `w_offset` (checked by the
            // caller to be inside `w`), `s × k` f32 from `x_at[i]` and writes `s × n`
            // f32 from `y_at[i]`, both inside the staging buffers by construction.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&op.w.raw), op.w_offset, 0);
                enc.setBuffer_offset_atIndex(Some(&xbuf.raw), x_at[i], 1);
                enc.setBuffer_offset_atIndex(Some(&ybuf.raw), y_at[i], 2);
                enc.setBytes_length_atIndex(params, core::mem::size_of::<Params>(), 3);
            }
            let groups = MTLSize {
                width: (op.params.n as usize).div_ceil(rows_per_group(op.pipeline)),
                height,
                depth: 1,
            };
            let threads = MTLSize {
                width: ROWS * SIMD_WIDTH,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if cmd.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::Execution(cmd.error().map_or_else(
                || format!("status {:?}", cmd.status()),
                |e| ns_error(&e),
            )));
        }
        let ybase = ybuf.raw.contents().as_ptr().cast::<u8>();
        for (op, &at) in ops.iter_mut().zip(&y_at) {
            // SAFETY: the command buffer has completed, so the GPU no longer writes the
            // staging buffer; `at + size_of_val(ys) <= y_len <= ybuf.len`, written by the
            // kernel, and `ys` is a distinct host allocation.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    ybase.add(at).cast::<f32>(),
                    op.ys.as_mut_ptr(),
                    op.ys.len(),
                );
            }
        }
        drop(scratch);
        Ok(())
    }
}

/// The bytes of a kernel parameter block.
///
/// `T` must be a `#[repr(C)]` struct of 4-byte fields (u32 or f32) with no padding, as
/// every parameter block of this crate is.
pub(crate) fn bytes_of<T: Copy>(v: &T) -> Vec<u8> {
    let n = core::mem::size_of::<T>();
    debug_assert_eq!(n % 4, 0);
    let mut out = vec![0u8; n];
    // SAFETY: `v` is valid for `n` bytes, and with no padding every byte is initialised.
    unsafe {
        core::ptr::copy_nonoverlapping(core::ptr::from_ref(v).cast::<u8>(), out.as_mut_ptr(), n);
    }
    out
}

/// One argument of a recorded dispatch.
pub(crate) enum Arg<'a> {
    /// A buffer bound from a byte offset.
    Buf(&'a Buffer, usize),
    /// Small constant data, copied when recorded.
    Bytes(&'a [u8]),
}

/// Records dispatches into one command buffer (see [`Device::submit`]).
pub(crate) struct Recorder<'a> {
    enc: &'a ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipelines: &'a [Retained<ProtocolObject<dyn MTLComputePipelineState>>],
    /// Every wrapped host buffer bound, so an asynchronous submission keeps the host
    /// memory alive until the GPU is done with it (the command buffer retains the
    /// Metal objects, but not the Rust allocation they wrap).
    keep: Vec<Buffer>,
    /// Buffers declared with `useResource` on this encoder, by object address.
    used: Vec<*const c_void>,
    /// Whether anything has been dispatched yet.
    any: bool,
    /// Inside a concurrent group ([`Recorder::group`]): dispatches after the group's
    /// first are not separated by barriers.
    grouped: Option<usize>,
}

impl Recorder<'_> {
    /// Declares that later dispatches read `b` through an address table (not a bound
    /// argument), so the GPU makes it resident; wrapped host memory is kept alive as
    /// for a bound buffer.
    pub(crate) fn use_read(&mut self, b: &Buffer) {
        let ptr = Retained::as_ptr(&b.raw).cast::<c_void>();
        if self.used.contains(&ptr) {
            return;
        }
        self.used.push(ptr);
        self.keep_host(b);
        self.enc
            .useResource_usage(ProtocolObject::from_ref(&*b.raw), MTLResourceUsage::Read);
    }

    /// Records what `f` records with no barriers between its dispatches, so the GPU may
    /// run them at once: they must not read what another of them writes, nor write
    /// the same memory. The group as a whole stays ordered after earlier dispatches and
    /// before later ones.
    pub(crate) fn group<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let outer = self.grouped.replace(0);
        let r = f(self);
        self.grouped = outer;
        r
    }

    fn keep_host(&mut self, b: &Buffer) {
        if b._host.is_some()
            && !self
                .keep
                .iter()
                .any(|k| Retained::as_ptr(&k.raw) == Retained::as_ptr(&b.raw))
        {
            self.keep.push(b.clone());
        }
    }

    /// Records `pipeline` over `groups` threadgroups of `threads` threads, with `args`
    /// bound at indices `0..args.len()`.
    ///
    /// The caller (this crate's typed operations) has checked that every element the
    /// kernel reads or writes for these sizes lies inside its buffer, from its offset,
    /// and that the arguments match the kernel's bindings. Each dispatch starts after
    /// every earlier one has finished, except that dispatches of one
    /// [`Recorder::group`] may overlap each other.
    pub(crate) fn dispatch(
        &mut self,
        pipeline: usize,
        args: &[Arg<'_>],
        groups: [usize; 3],
        threads: [usize; 3],
    ) {
        // The encoder is concurrent: a barrier before each dispatch orders it after
        // every earlier one, except within a group.
        let first_in_group = match &mut self.grouped {
            Some(n) => {
                *n += 1;
                *n == 1
            }
            None => true,
        };
        if self.any && first_in_group {
            self.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        self.any = true;
        self.enc.setComputePipelineState(&self.pipelines[pipeline]);
        for (i, a) in args.iter().enumerate() {
            match a {
                Arg::Buf(b, off) => {
                    self.keep_host(b);
                    // SAFETY: the Metal buffer is retained by the command buffer until it
                    // completes, and wrapped host memory by `keep` (held by the
                    // submission until it completes); the caller has checked the
                    // kernel's accesses against the buffer's length.
                    unsafe { self.enc.setBuffer_offset_atIndex(Some(&b.raw), *off, i) };
                }
                Arg::Bytes(d) => {
                    if let Some(ptr) = NonNull::new(d.as_ptr().cast_mut().cast::<c_void>()) {
                        // SAFETY: `d` is valid for `d.len()` bytes; setBytes copies them.
                        unsafe { self.enc.setBytes_length_atIndex(ptr, d.len(), i) };
                    }
                }
            }
        }
        let size = |v: [usize; 3]| MTLSize {
            width: v[0],
            height: v[1],
            depth: v[2],
        };
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
    }
}

type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

/// An asynchronous submission (see [`Device::submit_async`]). Dropping it waits for the
/// GPU, so the host memory it keeps is never released while the GPU may read it.
pub(crate) struct Pending<'d> {
    dev: &'d Device,
    cmd: CommandBuffer,
    _keep: Vec<Buffer>,
    done: bool,
}

impl Pending<'_> {
    /// Whether the GPU has finished (successfully or not).
    pub(crate) fn is_complete(&self) -> bool {
        self.done
            || matches!(
                self.cmd.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
    }

    /// Waits for the GPU and reports whether the submission succeeded.
    pub(crate) fn wait(&mut self) -> Result<(), MetalError> {
        if !self.done {
            self.cmd.waitUntilCompleted();
            self.done = true;
            self.dev.outstanding.fetch_sub(1, Ordering::AcqRel);
        }
        status(&self.cmd)
    }
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        // The status was the caller's to check; Drop only must not release early.
        let _ = self.wait();
    }
}

fn status(cmd: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    if cmd.status() == MTLCommandBufferStatus::Completed {
        Ok(())
    } else {
        Err(MetalError::Execution(cmd.error().map_or_else(
            || format!("status {:?}", cmd.status()),
            |e| ns_error(&e),
        )))
    }
}

impl Device {
    /// Encodes what `record` records into a new command buffer (not committed).
    fn encode<E: From<MetalError>>(
        &self,
        record: impl FnOnce(&mut Recorder<'_>) -> Result<(), E>,
    ) -> Result<(CommandBuffer, Vec<Buffer>), E> {
        let cmd = self.queue.commandBuffer().ok_or(MetalError::Create {
            what: "command buffer",
            detail: None,
        })?;
        let enc = cmd
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or(MetalError::Create {
                what: "compute encoder",
                detail: None,
            })?;
        let mut rec = Recorder {
            enc: &enc,
            pipelines: &self.pipelines,
            keep: Vec::new(),
            used: Vec::new(),
            any: false,
            grouped: None,
        };
        let recorded = record(&mut rec);
        let keep = rec.keep;
        enc.endEncoding();
        recorded?;
        Ok((cmd, keep))
    }

    /// Runs the dispatches `record` records in one command buffer and waits for it.
    /// Nothing is committed if `record` fails.
    pub(crate) fn submit<E: From<MetalError>>(
        &self,
        record: impl FnOnce(&mut Recorder<'_>) -> Result<(), E>,
    ) -> Result<(), E> {
        let _gpu = self
            .gpu
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (cmd, keep) = self.encode(record)?;
        cmd.commit();
        cmd.waitUntilCompleted();
        drop(keep);
        status(&cmd)?;
        Ok(())
    }

    /// Commits what `record` records without waiting. Submissions run in commit order
    /// (one queue), so later ones see this one's writes.
    pub(crate) fn submit_async<E: From<MetalError>>(
        &self,
        record: impl FnOnce(&mut Recorder<'_>) -> Result<(), E>,
    ) -> Result<Pending<'_>, E> {
        let _gpu = self
            .gpu
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (cmd, keep) = self.encode(record)?;
        cmd.commit();
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        Ok(Pending {
            dev: self,
            cmd,
            _keep: keep,
            done: false,
        })
    }

    /// Waits until no asynchronous submission is running: an empty command buffer
    /// completes only after every one committed before it. The gpu lock is held.
    fn drain(&self) {
        if self.outstanding.load(Ordering::Acquire) == 0 {
            return;
        }
        if let Some(cmd) = self.queue.commandBuffer() {
            cmd.commit();
            cmd.waitUntilCompleted();
        }
    }

    /// Copies `data` into `buf` from f32 index `at`. The caller has checked the range.
    pub(crate) fn write_f32(&self, buf: &Buffer, at: usize, data: &[f32]) {
        let _gpu = self
            .gpu
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            buf._host.is_none() && (at + data.len()) * 4 <= buf.len,
            "write out of range"
        );
        self.drain();
        let base = buf.raw.contents().as_ptr().cast::<f32>();
        // SAFETY: the range is inside the buffer (checked above); no submission is in
        // flight (the gpu lock is held, synchronous submissions wait for completion
        // and `drain` waited for asynchronous ones); the
        // buffer is device-allocated, not wrapped host memory, so nothing else aliases
        // it on the host.
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), base.add(at), data.len()) };
    }

    /// Copies `out.len()` f32 from `buf` at index `at`. The caller has checked the range.
    pub(crate) fn read_f32(&self, buf: &Buffer, at: usize, out: &mut [f32]) {
        let _gpu = self
            .gpu
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!((at + out.len()) * 4 <= buf.len, "read out of range");
        self.drain();
        let base = buf.raw.contents().as_ptr().cast::<f32>();
        // SAFETY: as for `write_f32`; `out` is a distinct host allocation.
        unsafe { core::ptr::copy_nonoverlapping(base.add(at), out.as_mut_ptr(), out.len()) };
    }
}
