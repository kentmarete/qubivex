//! The objc2 FFI layer: device, queue, pipelines, buffers and dispatch.
//!
//! All unsafe code of the crate is here. Metal devices, command queues, pipeline
//! states and buffers are thread-safe objects (objc2-metal marks their protocols
//! `Send + Sync`). Command buffers and encoders are created, used and dropped inside
//! one call, on one thread.

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

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

/// A shared-storage device buffer.
#[derive(Debug)]
pub(crate) struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

/// The device, its queue and one pipeline per kernel.
#[derive(Debug)]
pub(crate) struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
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
        Ok(Buffer { raw, len })
    }

    /// Runs kernel `pipeline` on resident weights `w`, synchronously. The caller has
    /// checked the shapes (`xs` is `s × k`, `ys` is `s × n`, both non-empty) and that
    /// `w` holds `n × row_bytes` bytes.
    pub(crate) fn matmul(
        &self,
        pipeline: usize,
        w: &Buffer,
        xs: &[f32],
        ys: &mut [f32],
        p: Params,
    ) -> Result<(), MetalError> {
        let x_bytes = f32_bytes(xs);
        let xbuf = self.buffer_from(x_bytes)?;
        let ylen = core::mem::size_of_val(ys);
        let ybuf = self.buffer_zeroed(ylen)?;
        let cmd = self.queue.commandBuffer().ok_or(MetalError::Create {
            what: "command buffer",
            detail: None,
        })?;
        let enc = cmd.computeCommandEncoder().ok_or(MetalError::Create {
            what: "compute encoder",
            detail: None,
        })?;
        enc.setComputePipelineState(&self.pipelines[pipeline]);
        let params = NonNull::from(&p).cast::<c_void>();
        // SAFETY: indices 0..=3 match the kernel's [[buffer(n)]] bindings and types
        // (bytes, f32, f32, Params). The buffers live until `waitUntilCompleted`
        // returns below; `params` is copied by setBytes. The kernel reads at most
        // `n × row_bytes` weight bytes, `s × k` activations and writes `s × n`
        // outputs, which are the buffers' sizes.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&w.raw), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&xbuf.raw), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&ybuf.raw), 0, 2);
            enc.setBytes_length_atIndex(params, core::mem::size_of::<Params>(), 3);
        }
        let groups = MTLSize {
            width: (p.n as usize).div_ceil(ROWS),
            height: p.s as usize,
            depth: 1,
        };
        let threads = MTLSize {
            width: ROWS * SIMD_WIDTH,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if cmd.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalError::Execution(cmd.error().map_or_else(
                || format!("status {:?}", cmd.status()),
                |e| ns_error(&e),
            )));
        }
        debug_assert_eq!(ybuf.len, ylen);
        // SAFETY: the command buffer has completed, so the GPU no longer writes the
        // shared buffer; `contents` points to `ylen` bytes of f32 written by the kernel
        // (or zero), and `ys` is a distinct host allocation of the same length.
        unsafe {
            core::ptr::copy_nonoverlapping(
                ybuf.raw.contents().as_ptr().cast::<f32>(),
                ys.as_mut_ptr(),
                ys.len(),
            );
        }
        Ok(())
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns as bytes, u8 has alignment 1,
    // and the length covers exactly the same memory.
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), core::mem::size_of_val(v)) }
}
