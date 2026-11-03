//! The objc2 FFI layer: device, queue, pipelines, buffers and dispatch.
//!
//! All unsafe code of the crate is here. Metal devices, command queues, pipeline
//! states and buffers are thread-safe objects (objc2-metal marks their protocols
//! `Send + Sync`). Command buffers and encoders are created, used and dropped inside
//! one call, on one thread.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

use qubivex_compute::host::PageAligned;

use crate::{DeviceInfo, MetalError};

/// Rows per threadgroup; must equal `ROWS` in `kernels.metal`.
const ROWS: usize = 8;
/// Threads per SIMD group on Apple GPUs; the kernels' reduction assumes it.
const SIMD_WIDTH: usize = 32;

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

    fn buffer_zeroed(&self, len: usize) -> Result<Buffer, MetalError> {
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
            enc.setComputePipelineState(&self.pipelines[op.pipeline]);
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
                width: (op.params.n as usize).div_ceil(ROWS),
                height: op.params.s as usize,
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
