//! The Vulkan device: instance, queue, memory, pipelines and submissions.
//!
//! All `unsafe` in the crate is here. Every Vulkan object is created and destroyed by
//! this module, and each one is owned by exactly one Rust value whose `Drop` destroys
//! it, after everything that uses it:
//! - [`Shared`] owns the instance, the device and the pipelines. Every allocation holds
//!   an `Arc<Shared>`, so the device outlives its memory.
//! - [`Allocation`] owns one buffer and its memory.
//! - [`Submitter`] owns the command pool, the fence, the descriptor pool and the
//!   staging and activation buffers. The device keeps several, each behind its own
//!   mutex, so concurrent callers record, submit and wait at the same time. The queue
//!   is externally synchronised by [`Shared`]'s queue lock, held only around
//!   `vkQueueSubmit`.
//!
//! Every submission waits for its fence before returning, so no GPU work is in flight
//! when control returns to safe code: an allocation can only be dropped when no
//! command buffer reads it.

use std::ffi::{CStr, c_char};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use ash::vk;
use qubivex_compute::ComputeError;
use qubivex_compute::discrete::{
    DeviceBuffer, DeviceGatedFfn, DeviceMatMul, DeviceMoeAttention, DeviceRoutedMoe, DeviceUsage,
    DeviceWeights, Guard, KV_BLOCK, TransferStats, kv_block_words,
};
use qubivex_core::{DType, GgmlType};

use crate::{BACKEND, VulkanConfig};

/// The matmul shader (`shaders/matmul.comp`, compiled by `shaders/build.sh`, which also
/// marks every `fma` NoContraction).
pub(crate) const MATMUL_SPV: &[u8] = include_bytes!("matmul.spv");

/// The fused matmul (`shaders/matmul_fused.comp`): several matrices per dispatch.
pub(crate) const MATMUL_FUSED_SPV: &[u8] = include_bytes!("matmul_fused.spv");

/// SwiGLU (`shaders/swiglu.comp`, built -O0).
pub(crate) const SWIGLU_SPV: &[u8] = include_bytes!("swiglu.spv");

/// The most experts a layer routed on the device may have (`route.comp`'s
/// `MAX_EXPERTS`: its shared arrays).
pub(crate) const ROUTED_EXPERTS: usize = 256;

/// `shaders/route.comp`: routing on the device (Step 3b).
pub(crate) const ROUTE_SPV: &[u8] = include_bytes!("route.spv");

/// `shaders/combine.comp`: the rank-order combine on the device (Step 3b).
pub(crate) const COMBINE_SPV: &[u8] = include_bytes!("combine.spv");

/// `shaders/resnorm.comp`: the attention residual and the FFN RMSNorm on the device
/// (audit §18).
pub(crate) const RESNORM_SPV: &[u8] = include_bytes!("resnorm.spv");

/// `shaders/rms_scale.comp`: the RMSNorm scale alone, for the conformance tests.
pub(crate) const RMS_SCALE_SPV: &[u8] = include_bytes!("rms_scale.spv");

/// `shaders/kv_sync.comp`: host rows into the device KV pool (audit §21).
pub(crate) const KV_SYNC_SPV: &[u8] = include_bytes!("kv_sync.spv");

/// `shaders/attn_prelude.comp`: the clip, Q and K norms, RoPE and KV store for decode
/// (audit §21).
pub(crate) const ATTN_PRELUDE_SPV: &[u8] = include_bytes!("attn_prelude.spv");

/// `shaders/attn_decode.comp`: decode attention over the paged KV pool (audit §21).
pub(crate) const ATTN_DECODE_SPV: &[u8] = include_bytes!("attn_decode.spv");

/// The most positions one token's attention sees on the device (`attn_decode.comp`'s
/// `MAXP`: its shared scores); past it, attention runs on the host.
pub(crate) const ATTN_POSITIONS: usize = 4096;

/// Invocations per SwiGLU workgroup. Must equal `local_size_x` in the shader.
const SWIGLU_GROUP: u32 = 256;

/// Output rows per workgroup. Must equal `ROWS` in the shader.
const ROWS: u32 = 8;

/// Bytes of the staging buffer for uploads; larger uploads go in chunks.
const STAGING: u64 = 64 << 20;

/// Shader dtype codes, in pipeline order. Must match `DT_*` in the shader.
const DTYPES: [DType; 6] = [
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::Ggml(GgmlType::Q8_0),
    DType::Ggml(GgmlType::Q4_K),
    DType::Ggml(GgmlType::Q6_K),
];

pub(crate) fn dtype_index(dtype: DType) -> Option<usize> {
    DTYPES.iter().position(|d| *d == dtype)
}

fn err(what: &str, e: impl std::fmt::Display) -> ComputeError {
    ComputeError::Device {
        backend: BACKEND,
        detail: format!("{what}: {e}"),
    }
}

fn u32_of(v: impl TryInto<u32>, what: &str) -> Result<u32, ComputeError> {
    v.try_into()
        .map_err(|_| err("matmul", format!("{what} do not fit in 32 bits")))
}

/// One matmul of a submission, checked and laid out.
struct Planned {
    buffer: vk::Buffer,
    base: u64,
    range: u64,
    dtype: usize,
    params: Params,
    groups: u32,
    tokens: u32,
}

/// The push constants, as the shader's `Params`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n: u32,
    k: u32,
    row_bytes: u32,
    w_base: u32,
    x_base: u32,
    y_base: u32,
}

impl Params {
    fn bytes(&self) -> [u8; 24] {
        let mut b = [0; 24];
        for (i, v) in [
            self.n,
            self.k,
            self.row_bytes,
            self.w_base,
            self.x_base,
            self.y_base,
        ]
        .iter()
        .enumerate()
        {
            b[4 * i..4 * i + 4].copy_from_slice(&v.to_ne_bytes());
        }
        b
    }
}

/// The Vulkan library, its loader entry points, which file it came from.
fn load_entry() -> Result<ash::Entry, ComputeError> {
    let mut tried = Vec::new();
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("QUBIVEX_VULKAN_LIBRARY") {
        candidates.push(p);
    }
    if cfg!(target_os = "macos") {
        // `dlopen` does not search Homebrew's prefixes.
        candidates.extend(
            [
                "libvulkan.1.dylib",
                "/opt/homebrew/lib/libvulkan.1.dylib",
                "/usr/local/lib/libvulkan.1.dylib",
            ]
            .map(String::from),
        );
    } else if cfg!(windows) {
        candidates.push("vulkan-1.dll".into());
    } else {
        candidates.extend(["libvulkan.so.1", "libvulkan.so"].map(String::from));
    }
    for c in candidates {
        // SAFETY: loading the Vulkan loader runs its initialisers; it is the library
        // this crate exists to call, from a path the user or the platform names.
        match unsafe { ash::Entry::load_from(&c) } {
            Ok(e) => return Ok(e),
            Err(e) => tried.push(format!("{c}: {e}")),
        }
    }
    Err(ComputeError::Device {
        backend: BACKEND,
        detail: format!(
            "no Vulkan loader found (set QUBIVEX_VULKAN_LIBRARY to its path); tried {}",
            tried.join("; ")
        ),
    })
}

/// The instance, device, queue and pipelines.
pub(crate) struct Shared {
    entry: ash::Entry,
    instance: ash::Instance,
    pub(crate) device: ash::Device,
    queue: vk::Queue,
    /// Held around every `vkQueueSubmit` (the queue needs external synchronisation).
    queue_lock: Mutex<()>,
    queue_family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
    pub(crate) limits: vk::PhysicalDeviceLimits,
    /// The valid bits of the queue's timestamps; 0 when it has none.
    timestamp_mask: u64,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    /// The fused expert block's pipelines; `None` if the driver could not build them
    /// (the device then runs expert blocks as separate submissions).
    pub(crate) fused: Option<Fused>,
    /// Routing and combine on the device (Step 3b), on the fused layout; `None` without
    /// fused pipelines or if the driver could not build them.
    pub(crate) routed: Option<Routed>,
    /// Decode attention on the device (audit §21), on the fused layout; `None` without
    /// the routed pipelines or if the driver could not build them.
    pub(crate) attn: Option<Attn>,
    /// Experimental (feature `spike`, audit §20): steps the next submission records ahead
    /// of its own, and the last submission's timing.
    #[cfg(feature = "spike")]
    spike: Mutex<SpikeHook>,
}

/// The `spike` feature's hook into every submission (audit §20).
#[cfg(feature = "spike")]
#[derive(Default)]
struct SpikeHook {
    /// Recorded at the head of the next command buffer, then moved to `done`.
    pending: Option<SpikePrefix>,
    done: Option<SpikePrefix>,
    /// The last submission's (fence wait, GPU) nanoseconds.
    last: Option<SpikeTiming>,
}

/// The routed layer's own pipelines, on [`Fused`]'s layout.
pub(crate) struct Routed {
    route: vk::Pipeline,
    combine: vk::Pipeline,
    /// The residual and RMSNorm (audit §18) and the scale probe.
    resnorm: vk::Pipeline,
    rms_scale: vk::Pipeline,
}

/// Decode attention's pipelines (audit §21), on [`Fused`]'s layout.
pub(crate) struct Attn {
    kv_sync: vk::Pipeline,
    prelude: vk::Pipeline,
    decode: vk::Pipeline,
}

/// The fused matmul (one pipeline per dtype) and SwiGLU, on one layout: binding 0 the
/// weights, 1 the inputs, 2 the outputs, 3 the table; 16 bytes of push constants.
pub(crate) struct Fused {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    matmul: Vec<vk::Pipeline>,
    swiglu: vk::Pipeline,
}

impl Fused {
    /// Destroys every object.
    ///
    /// # Safety
    /// Nothing may use them now or later.
    unsafe fn destroy(&self, d: &ash::Device) {
        // SAFETY: as the caller promises.
        unsafe {
            for &p in self.matmul.iter().chain([&self.swiglu]) {
                d.destroy_pipeline(p, None);
            }
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: every allocation and the submitter hold an `Arc<Shared>`, so this
        // runs after all of them are destroyed, and no work is in flight (every
        // submission waits for its fence). Objects are destroyed child first.
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(a) = &self.attn {
                self.device.destroy_pipeline(a.kv_sync, None);
                self.device.destroy_pipeline(a.prelude, None);
                self.device.destroy_pipeline(a.decode, None);
            }
            if let Some(r) = &self.routed {
                self.device.destroy_pipeline(r.route, None);
                self.device.destroy_pipeline(r.combine, None);
                self.device.destroy_pipeline(r.resnorm, None);
                self.device.destroy_pipeline(r.rms_scale, None);
            }
            if let Some(f) = &self.fused {
                f.destroy(&self.device);
            }
            for &p in &self.pipelines {
                self.device.destroy_pipeline(p, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry;
    }
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared").finish_non_exhaustive()
    }
}

/// A physical device as listed by [`list`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalInfo {
    /// Index in the driver's enumeration order.
    pub index: usize,
    /// The device name.
    pub name: String,
    /// Discrete, integrated, virtual, CPU or other.
    pub kind: &'static str,
    /// The driver's name and version, when the driver reports them.
    pub driver: String,
    /// Bytes of the largest device-local memory heap.
    pub device_local: u64,
    /// Whether it has a compute queue.
    pub compute: bool,
}

fn kind_name(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

fn cstr(chars: &[c_char]) -> String {
    // SAFETY: Vulkan fills these fixed arrays with a NUL-terminated UTF-8 string.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(chars.as_ptr().cast(), chars.len()) };
    CStr::from_bytes_until_nul(bytes)
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn has_extension(props: &[vk::ExtensionProperties], name: &CStr) -> bool {
    props
        .iter()
        .any(|p| p.extension_name_as_c_str().is_ok_and(|n| n == name))
}

/// An instance with the extensions this backend needs.
fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, ComputeError> {
    // SAFETY: a plain query on a loaded entry.
    let exts = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(|e| err("cannot list instance extensions", e))?;
    let mut names: Vec<*const c_char> = Vec::new();
    let mut flags = vk::InstanceCreateFlags::empty();
    // MoltenVK and other layered drivers are only listed with portability
    // enumeration.
    if has_extension(&exts, vk::KHR_PORTABILITY_ENUMERATION_NAME) {
        names.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
        flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
    }
    let layer_settings = has_extension(&exts, vk::EXT_LAYER_SETTINGS_NAME);
    if layer_settings {
        names.push(vk::EXT_LAYER_SETTINGS_NAME.as_ptr());
    }
    let app = vk::ApplicationInfo::default()
        .application_name(c"qubivex")
        .engine_name(c"qubivex")
        .api_version(vk::API_VERSION_1_1);
    // MoltenVK compiles shaders with Metal's fast math unless told otherwise; fast
    // math may reassociate and fuse, which breaks the CPU's bits. 0: never.
    let never: i32 = 0;
    let mut setting = vk::LayerSettingEXT::default()
        .layer_name(c"MoltenVK")
        .setting_name(c"MVK_CONFIG_FAST_MATH_ENABLED")
        .ty(vk::LayerSettingTypeEXT::INT32);
    // `values` takes bytes and would count them as values: set the one value here.
    setting.value_count = 1;
    setting.p_values = (&raw const never).cast();
    let settings = [setting];
    let mut layer_info = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
    let mut info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_extension_names(&names)
        .flags(flags);
    if layer_settings {
        info = info.push_next(&mut layer_info);
    }
    // SAFETY: `info` and everything it points to live until the call returns.
    unsafe { entry.create_instance(&info, None) }.map_err(|e| err("cannot create an instance", e))
}

/// The devices the Vulkan loader lists, for reports and `--vulkan-device`.
///
/// # Errors
/// [`ComputeError::Device`] when no Vulkan loader or instance is available.
pub fn list() -> Result<Vec<PhysicalInfo>, ComputeError> {
    let entry = load_entry()?;
    let instance = create_instance(&entry)?;
    let out = describe_all(&instance);
    // SAFETY: nothing was created from the instance.
    unsafe { instance.destroy_instance(None) };
    out
}

fn describe_all(instance: &ash::Instance) -> Result<Vec<PhysicalInfo>, ComputeError> {
    // SAFETY: queries on a live instance.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| err("cannot list devices", e))?;
    Ok(devices
        .iter()
        .enumerate()
        .map(|(index, &pd)| describe(instance, pd, index))
        .collect())
}

fn describe(instance: &ash::Instance, pd: vk::PhysicalDevice, index: usize) -> PhysicalInfo {
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
    // SAFETY: queries on a device of a live instance; `props2` chains a struct that
    // Vulkan 1.1 (KHR_driver_properties core in 1.2) may leave untouched.
    let (props, mem, queues) = unsafe {
        instance.get_physical_device_properties2(pd, &mut props2);
        (
            props2.properties,
            instance.get_physical_device_memory_properties(pd),
            instance.get_physical_device_queue_family_properties(pd),
        )
    };
    let device_local = mem.memory_heaps[..mem.memory_heap_count as usize]
        .iter()
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size)
        .max()
        .unwrap_or(0);
    let dname = cstr(&driver.driver_name);
    let dinfo = cstr(&driver.driver_info);
    PhysicalInfo {
        index,
        name: cstr(&props.device_name),
        kind: kind_name(props.device_type),
        driver: [dname, dinfo]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        device_local,
        compute: queues
            .iter()
            .any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)),
    }
}

/// Discrete first, then integrated, virtual, CPU; ties in enumeration order.
fn rank(kind: &str) -> u8 {
    match kind {
        "discrete" => 0,
        "integrated" => 1,
        "virtual" => 2,
        "cpu" => 3,
        _ => 4,
    }
}

/// What [`Shared::new`] chose.
pub(crate) struct Opened {
    pub(crate) shared: Arc<Shared>,
    pub(crate) info: PhysicalInfo,
}

impl Shared {
    pub(crate) fn open(config: &VulkanConfig) -> Result<Opened, ComputeError> {
        let entry = load_entry()?;
        let instance = create_instance(&entry)?;
        match Self::open_on(entry, &instance, config) {
            Ok(o) => Ok(o),
            Err(e) => {
                // SAFETY: `open_on` destroyed whatever it created on failure.
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    fn open_on(
        entry: ash::Entry,
        instance: &ash::Instance,
        config: &VulkanConfig,
    ) -> Result<Opened, ComputeError> {
        // SAFETY: a query on a live instance.
        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| err("cannot list devices", e))?;
        let infos = describe_all(instance)?;
        let info = match config.device {
            Some(i) => infos.get(i).cloned().ok_or_else(|| ComputeError::Device {
                backend: BACKEND,
                detail: format!("no Vulkan device {i}: {} listed", infos.len()),
            })?,
            None => infos
                .iter()
                .filter(|d| d.compute)
                .min_by_key(|d| rank(d.kind))
                .cloned()
                .ok_or_else(|| ComputeError::Device {
                    backend: BACKEND,
                    detail: "no Vulkan device with a compute queue".into(),
                })?,
        };
        let pd = devices[info.index];
        // SAFETY: queries on a device of a live instance.
        let (queues, memory, props, dev_exts) = unsafe {
            (
                instance.get_physical_device_queue_family_properties(pd),
                instance.get_physical_device_memory_properties(pd),
                instance.get_physical_device_properties(pd),
                instance
                    .enumerate_device_extension_properties(pd)
                    .map_err(|e| err("cannot list device extensions", e))?,
            )
        };
        if props.api_version < vk::API_VERSION_1_1 {
            return Err(err(
                "device",
                format!("{} supports only Vulkan 1.0", info.name),
            ));
        }
        // A compute-only family if there is one (often an async compute queue), else
        // any family with compute.
        let family = queues
            .iter()
            .position(|q| {
                q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .or_else(|| {
                queues
                    .iter()
                    .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            })
            .ok_or_else(|| err("device", format!("{} has no compute queue", info.name)))?;
        let valid_bits = queues[family].timestamp_valid_bits;
        let timestamp_mask = match valid_bits {
            0 => 0,
            64.. => u64::MAX,
            b => (1u64 << b) - 1,
        };
        let family = u32::try_from(family).map_err(|e| err("queue family", e))?;
        let mut names = Vec::new();
        // Required wherever it is listed (MoltenVK).
        if has_extension(&dev_exts, vk::KHR_PORTABILITY_SUBSET_NAME) {
            names.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }
        let prio = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&prio)];
        let dinfo = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qinfo)
            .enabled_extension_names(&names);
        // SAFETY: `dinfo` and what it points to live until the call returns.
        let device = unsafe { instance.create_device(pd, &dinfo, None) }
            .map_err(|e| err("cannot create a device", e))?;
        // SAFETY: the queue family and index were requested above.
        let queue = unsafe { device.get_device_queue(family, 0) };
        let (set_layout, layout, pipelines) = match create_pipelines(&device) {
            Ok(v) => v,
            Err(e) => {
                // SAFETY: nothing else was created on the device.
                unsafe { device.destroy_device(None) };
                return Err(e);
            }
        };
        let fused = create_fused(&device).ok();
        let routed = fused
            .as_ref()
            .and_then(|f| create_routed(&device, f.layout).ok());
        let attn = match (&fused, &routed) {
            (Some(f), Some(_)) => create_on(
                &device,
                f.layout,
                &[KV_SYNC_SPV, ATTN_PRELUDE_SPV, ATTN_DECODE_SPV],
                "attention pipelines",
            )
            .ok()
            .map(|p| Attn {
                kv_sync: p[0],
                prelude: p[1],
                decode: p[2],
            }),
            _ => None,
        };
        let shared = Arc::new(Self {
            entry,
            instance: instance.clone(),
            device,
            queue,
            queue_lock: Mutex::new(()),
            queue_family: family,
            memory,
            limits: props.limits,
            timestamp_mask,
            set_layout,
            layout,
            pipelines,
            fused,
            routed,
            attn,
            #[cfg(feature = "spike")]
            spike: Mutex::default(),
        });
        Ok(Opened { shared, info })
    }

    /// A memory type allowed by `bits` with all of `want`, preferring one with
    /// `prefer` as well.
    fn memory_type(
        &self,
        bits: u32,
        want: vk::MemoryPropertyFlags,
        prefer: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        let ok = |i: usize, f: vk::MemoryPropertyFlags| {
            bits & (1 << i) != 0 && types[i].property_flags.contains(f)
        };
        (0..types.len())
            .find(|&i| ok(i, want | prefer))
            .or_else(|| (0..types.len()).find(|&i| ok(i, want)))
            .and_then(|i| u32::try_from(i).ok())
    }
}

fn create_pipelines(
    device: &ash::Device,
) -> Result<
    (
        vk::DescriptorSetLayout,
        vk::PipelineLayout,
        Vec<vk::Pipeline>,
    ),
    ComputeError,
> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(MATMUL_SPV))
        .map_err(|e| err("bad SPIR-V", e))?;
    let bindings: Vec<_> = (0..3)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let push = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(24)];
    // SAFETY: each create-info and what it points to live until its call returns;
    // on failure, the objects created so far are destroyed before returning.
    unsafe {
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(|e| err("descriptor set layout", e))?;
        let layouts = [set_layout];
        let layout = match device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push),
            None,
        ) {
            Ok(l) => l,
            Err(e) => {
                device.destroy_descriptor_set_layout(set_layout, None);
                return Err(err("pipeline layout", e));
            }
        };
        let module = match device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
        {
            Ok(m) => m,
            Err(e) => {
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_set_layout(set_layout, None);
                return Err(err("shader module", e));
            }
        };
        let values: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let entries = [vk::SpecializationMapEntry::default()
            .constant_id(0)
            .offset(0)
            .size(4)];
        let specs: Vec<_> = values
            .iter()
            .map(|v| {
                vk::SpecializationInfo::default()
                    .map_entries(&entries)
                    .data(std::slice::from_raw_parts((v as *const u32).cast(), 4))
            })
            .collect();
        let infos: Vec<_> = specs
            .iter()
            .map(|s| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main")
                            .specialization_info(s),
                    )
                    .layout(layout)
            })
            .collect();
        let made = device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None);
        device.destroy_shader_module(module, None);
        match made {
            Ok(p) => Ok((set_layout, layout, p)),
            Err((p, e)) => {
                for p in p.into_iter().filter(|p| *p != vk::Pipeline::null()) {
                    device.destroy_pipeline(p, None);
                }
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_set_layout(set_layout, None);
                Err(err("compute pipelines", e))
            }
        }
    }
}

/// The fused expert block's layout and pipelines.
fn create_fused(device: &ash::Device) -> Result<Fused, ComputeError> {
    let spv = |b: &[u8]| {
        ash::util::read_spv(&mut std::io::Cursor::new(b)).map_err(|e| err("bad SPIR-V", e))
    };
    let (fused_code, swiglu_code) = (spv(MATMUL_FUSED_SPV)?, spv(SWIGLU_SPV)?);
    let bindings: Vec<_> = (0..4)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let push = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(16)];
    // SAFETY: each create-info and what it points to live until its call returns;
    // on failure, the objects created so far are destroyed before returning.
    unsafe {
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(|e| err("descriptor set layout", e))?;
        let layouts = [set_layout];
        let layout = match device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push),
            None,
        ) {
            Ok(l) => l,
            Err(e) => {
                device.destroy_descriptor_set_layout(set_layout, None);
                return Err(err("pipeline layout", e));
            }
        };
        let undo = || {
            device.destroy_pipeline_layout(layout, None);
            device.destroy_descriptor_set_layout(set_layout, None);
        };
        let module = |code: &[u32]| {
            device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
                .map_err(|e| err("shader module", e))
        };
        let fm = match module(&fused_code) {
            Ok(m) => m,
            Err(e) => {
                undo();
                return Err(e);
            }
        };
        let sm = match module(&swiglu_code) {
            Ok(m) => m,
            Err(e) => {
                device.destroy_shader_module(fm, None);
                undo();
                return Err(e);
            }
        };
        let values: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let entries = [vk::SpecializationMapEntry::default()
            .constant_id(0)
            .offset(0)
            .size(4)];
        let specs: Vec<_> = values
            .iter()
            .map(|v| {
                vk::SpecializationInfo::default()
                    .map_entries(&entries)
                    .data(std::slice::from_raw_parts((v as *const u32).cast(), 4))
            })
            .collect();
        let mut infos: Vec<_> = specs
            .iter()
            .map(|s| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(fm)
                            .name(c"main")
                            .specialization_info(s),
                    )
                    .layout(layout)
            })
            .collect();
        infos.push(
            vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(sm)
                        .name(c"main"),
                )
                .layout(layout),
        );
        let made = device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None);
        device.destroy_shader_module(fm, None);
        device.destroy_shader_module(sm, None);
        match made {
            Ok(mut p) => {
                let swiglu = p.pop().ok_or_else(|| err("compute pipelines", "missing"))?;
                Ok(Fused {
                    set_layout,
                    layout,
                    matmul: p,
                    swiglu,
                })
            }
            Err((p, e)) => {
                for p in p.into_iter().filter(|p| *p != vk::Pipeline::null()) {
                    device.destroy_pipeline(p, None);
                }
                undo();
                Err(err("fused pipelines", e))
            }
        }
    }
}

/// The route, combine, residual-norm and scale-probe pipelines on the fused `layout`.
fn create_routed(device: &ash::Device, layout: vk::PipelineLayout) -> Result<Routed, ComputeError> {
    let p = create_on(
        device,
        layout,
        &[ROUTE_SPV, COMBINE_SPV, RESNORM_SPV, RMS_SCALE_SPV],
        "routed pipelines",
    )?;
    Ok(Routed {
        route: p[0],
        combine: p[1],
        resnorm: p[2],
        rms_scale: p[3],
    })
}

/// Compute pipelines from `spvs`, in order, on `layout`; all or none.
fn create_on(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    spvs: &[&[u8]],
    what: &str,
) -> Result<Vec<vk::Pipeline>, ComputeError> {
    let codes = spvs
        .iter()
        .map(|b| {
            ash::util::read_spv(&mut std::io::Cursor::new(b)).map_err(|e| err("bad SPIR-V", e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // SAFETY: each create-info and what it points to live until its call returns;
    // modules are destroyed after pipeline creation, pipelines on failure.
    unsafe {
        let mut modules = Vec::with_capacity(codes.len());
        for code in &codes {
            match device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
            {
                Ok(m) => modules.push(m),
                Err(e) => {
                    for m in modules {
                        device.destroy_shader_module(m, None);
                    }
                    return Err(err("shader module", e));
                }
            }
        }
        let infos: Vec<_> = modules
            .iter()
            .map(|&m| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(m)
                            .name(c"main"),
                    )
                    .layout(layout)
            })
            .collect();
        let made = device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None);
        for m in modules {
            device.destroy_shader_module(m, None);
        }
        match made {
            Ok(p) if p.len() == codes.len() => Ok(p),
            Ok(p) | Err((p, _)) => {
                for p in p.into_iter().filter(|p| *p != vk::Pipeline::null()) {
                    device.destroy_pipeline(p, None);
                }
                Err(err(what, "not created"))
            }
        }
    }
}

/// One buffer and its memory. Destroyed on drop.
pub(crate) struct Allocation {
    shared: Arc<Shared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    mapped: Option<NonNull<u8>>,
    /// Device memory to return to the device's count on drop.
    counted: Option<Arc<Counters>>,
}

// SAFETY: the handles are plain identifiers, and `mapped` points into memory owned by
// this value. All writes through it go through `&mut self` or the submitter's mutex.
unsafe impl Send for Allocation {}
// SAFETY: as above; shared references never write through `mapped`.
unsafe impl Sync for Allocation {}

impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: no submission is in flight when safe code runs (see the module
        // documentation), so nothing uses the buffer; the memory is unmapped by
        // `free_memory`.
        unsafe {
            self.shared.device.destroy_buffer(self.buffer, None);
            self.shared.device.free_memory(self.memory, None);
        }
        if let Some(c) = &self.counted {
            c.used.fetch_sub(self.size, Ordering::Relaxed);
        }
    }
}

impl Allocation {
    fn new(
        shared: &Arc<Shared>,
        size: u64,
        usage: vk::BufferUsageFlags,
        host: bool,
    ) -> Result<Self, ComputeError> {
        let d = &shared.device;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `info` is valid for the call.
        let buffer =
            unsafe { d.create_buffer(&info, None) }.map_err(|e| oom_or("buffer", e, size))?;
        // SAFETY: `buffer` was just created on `d`.
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let (want, prefer) = if host {
            (
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::HOST_CACHED,
            )
        } else {
            (
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            )
        };
        let Some(ty) = shared.memory_type(req.memory_type_bits, want, prefer) else {
            // SAFETY: unused buffer of `d`.
            unsafe { d.destroy_buffer(buffer, None) };
            return Err(err(
                "memory",
                format!("no memory type with flags {:#x}", want.as_raw()),
            ));
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(ty);
        // SAFETY: `alloc` is valid for the call; on failure the buffer is destroyed.
        let memory = match unsafe { d.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: unused buffer of `d`.
                unsafe { d.destroy_buffer(buffer, None) };
                return Err(oom_or("allocation", e, size));
            }
        };
        let mut a = Self {
            shared: Arc::clone(shared),
            buffer,
            memory,
            size,
            mapped: None,
            counted: None,
        };
        // SAFETY: memory of a matching type and size, bound once; `a` frees both on
        // any error from here on.
        unsafe { d.bind_buffer_memory(buffer, memory, 0) }.map_err(|e| err("bind", e))?;
        if host {
            // SAFETY: host-visible memory, mapped once for its whole life.
            let p = unsafe { d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
                .map_err(|e| err("map", e))?;
            a.mapped = NonNull::new(p.cast());
        }
        Ok(a)
    }

    /// The mapped memory of a host-visible allocation as floats.
    fn floats_mut(&mut self) -> &mut [f32] {
        match self.mapped {
            // SAFETY: as `bytes_mut`; a mapping is aligned to at least
            // `minMemoryMapAlignment` (64 or more), so to 4, and every host buffer is a
            // power of two of at least 4096 bytes, so a whole number of floats.
            Some(p) => unsafe {
                std::slice::from_raw_parts_mut(
                    p.as_ptr().cast(),
                    usize::try_from(self.size / 4).unwrap_or(0),
                )
            },
            None => &mut [],
        }
    }

    /// The mapped bytes of a host-visible allocation.
    fn bytes_mut(&mut self) -> &mut [u8] {
        match self.mapped {
            // SAFETY: the mapping covers `size` bytes and lives as long as `self`;
            // `&mut self` makes this the only access, and no GPU work is in flight.
            Some(p) => unsafe {
                std::slice::from_raw_parts_mut(p.as_ptr(), usize::try_from(self.size).unwrap_or(0))
            },
            None => &mut [],
        }
    }
}

fn oom_or(what: &str, e: vk::Result, size: u64) -> ComputeError {
    if matches!(
        e,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY
    ) {
        err(what, format!("the driver refused {size} bytes: {e}"))
    } else {
        err(what, e)
    }
}

/// Device memory and copy counters.
#[derive(Debug, Default)]
pub(crate) struct Counters {
    used: AtomicU64,
    peak: AtomicU64,
    refused: AtomicU64,
    uploads: AtomicU64,
    upload_bytes: AtomicU64,
    activations_in: AtomicU64,
    activations_out: AtomicU64,
    submissions: AtomicU64,
    transfer_ns: AtomicU64,
    fence_waits: AtomicU64,
    submit_ns: AtomicU64,
    wait_ns: AtomicU64,
    timed: AtomicU64,
    gpu_ns: AtomicU64,
    busy: Mutex<Busy>,
}

/// The union of GPU busy intervals, in device nanoseconds.
///
/// Concurrent submissions finish out of order, so intervals arrive a little out of
/// order. The most recent [`Busy::WINDOW`] are kept, and the older half is folded
/// into the total, in start order, once the window is full. This is exact while no
/// interval arrives more than half a window late; at most [`crate::SUBMITTERS`] can be
/// in flight, far fewer than that.
#[derive(Debug, Default)]
struct Busy {
    pending: Vec<(u64, u64)>,
    /// The end of the folded intervals' union.
    frontier: u64,
    folded_ns: u64,
}

impl Busy {
    const WINDOW: usize = 64;

    /// Adds `[start, end)` to the union `(frontier, total)`.
    fn fold(frontier: &mut u64, total: &mut u64, (start, end): (u64, u64)) {
        if end > *frontier {
            *total += end - start.max(*frontier);
            *frontier = end;
        }
    }

    fn add(&mut self, interval: (u64, u64)) {
        self.pending.push(interval);
        if self.pending.len() >= Self::WINDOW {
            self.pending.sort_unstable();
            let old: Vec<_> = self.pending.drain(..Self::WINDOW / 2).collect();
            for i in old {
                Self::fold(&mut self.frontier, &mut self.folded_ns, i);
            }
        }
    }

    fn total(&self) -> u64 {
        let mut pending = self.pending.clone();
        pending.sort_unstable();
        let (mut frontier, mut total) = (self.frontier, self.folded_ns);
        for i in pending {
            Self::fold(&mut frontier, &mut total, i);
        }
        total
    }
}

impl Counters {
    fn reserve(&self, bytes: u64, capacity: u64) -> Result<(), ComputeError> {
        let mut used = self.used.load(Ordering::Relaxed);
        loop {
            let Some(next) = used.checked_add(bytes).filter(|&n| n <= capacity) else {
                self.refused.fetch_add(1, Ordering::Relaxed);
                return Err(ComputeError::OutOfDeviceMemory {
                    backend: BACKEND,
                    requested: bytes,
                    used,
                    capacity,
                });
            };
            match self
                .used
                .compare_exchange_weak(used, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::Relaxed);
                    return Ok(());
                }
                Err(now) => used = now,
            }
        }
    }

    fn time(&self, since: Instant) {
        let ns = u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.transfer_ns.fetch_add(ns, Ordering::Relaxed);
    }

    pub(crate) fn usage(&self, capacity: u64) -> DeviceUsage {
        DeviceUsage {
            capacity,
            used: self.used.load(Ordering::Relaxed),
            peak: self.peak.load(Ordering::Relaxed),
            refused: self.refused.load(Ordering::Relaxed),
        }
    }

    /// Counts one completed matmul submission.
    fn ran(&self, prepared: Instant, ran: &Ran) {
        let ns = |d: std::time::Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        self.fence_waits.fetch_add(1, Ordering::Relaxed);
        self.submit_ns.fetch_add(
            ns(ran.submitted.saturating_duration_since(prepared)),
            Ordering::Relaxed,
        );
        self.wait_ns.fetch_add(ran.wait_ns, Ordering::Relaxed);
        if let Some((start, end)) = ran.gpu {
            self.timed.fetch_add(1, Ordering::Relaxed);
            self.gpu_ns.fetch_add(end - start, Ordering::Relaxed);
            self.busy
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .add((start, end));
        }
    }

    pub(crate) fn transfers(&self) -> TransferStats {
        TransferStats {
            uploads: self.uploads.load(Ordering::Relaxed),
            upload_bytes: self.upload_bytes.load(Ordering::Relaxed),
            activations_in: self.activations_in.load(Ordering::Relaxed),
            activations_out: self.activations_out.load(Ordering::Relaxed),
            submissions: self.submissions.load(Ordering::Relaxed),
            transfer_ns: self.transfer_ns.load(Ordering::Relaxed),
            fence_waits: self.fence_waits.load(Ordering::Relaxed),
            submit_ns: self.submit_ns.load(Ordering::Relaxed),
            wait_ns: self.wait_ns.load(Ordering::Relaxed),
            timed: self.timed.load(Ordering::Relaxed),
            gpu_ns: self.gpu_ns.load(Ordering::Relaxed),
            gpu_busy_ns: self
                .busy
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .total(),
        }
    }
}

/// A weight allocation's state inside a [`DeviceBuffer`].
pub(crate) struct WeightBuffer {
    alloc: Allocation,
}

/// One completed submission's timing.
pub(crate) struct Ran {
    /// When `vkQueueSubmit` returned.
    submitted: Instant,
    /// Host time blocked on the fence.
    wait_ns: u64,
    /// GPU start and end, in device nanoseconds, when timed.
    gpu: Option<(u64, u64)>,
}

/// The command pool, fence, descriptor pool, and reusable host buffers.
pub(crate) struct Submitter {
    shared: Arc<Shared>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    descriptors: vk::DescriptorPool,
    sets: u32,
    /// Two timestamp queries (start, end), when submissions are timed.
    queries: Option<vk::QueryPool>,
    staging: Option<Allocation>,
    xs: Option<Allocation>,
    ys: Option<Allocation>,
    /// The fused block's descriptor pool (null without fused pipelines) and its size.
    fused_pool: vk::DescriptorPool,
    fused_sets: u32,
    /// The fused block's intermediates (device-local) and its table (host-visible).
    mid: Option<Allocation>,
    table: Option<Allocation>,
    /// The routed layer's router logits and expert outputs (device-local), and its
    /// routes (host-visible, read back).
    logits: Option<Allocation>,
    contrib: Option<Allocation>,
    routes: Option<Allocation>,
    /// The prelude's attention outputs and residual stream (host-visible, the residual
    /// read back) and the O projection's outputs (device-local), Step 3c.
    att: Option<Allocation>,
    hs: Option<Allocation>,
    oproj: Option<Allocation>,
    /// Decode attention (audit §21): the attention-normed activations, the rows to copy
    /// into the KV pool and the new keys and values (host-visible), and q, k and v
    /// (device-local).
    axs: Option<Allocation>,
    sync: Option<Allocation>,
    knew: Option<Allocation>,
    qkv: Option<Allocation>,
}

// SAFETY: the handles are plain identifiers, used only through `&mut self`.
unsafe impl Send for Submitter {}

impl Drop for Submitter {
    fn drop(&mut self) {
        // SAFETY: no submission is in flight (each waits for its fence); the command
        // buffer is freed with its pool.
        unsafe {
            let d = &self.shared.device;
            if let Some(q) = self.queries {
                d.destroy_query_pool(q, None);
            }
            d.destroy_descriptor_pool(self.descriptors, None);
            if self.fused_pool != vk::DescriptorPool::null() {
                d.destroy_descriptor_pool(self.fused_pool, None);
            }
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
        }
    }
}

fn descriptor_pool(d: &ash::Device, sets: u32) -> Result<vk::DescriptorPool, ComputeError> {
    descriptor_pool_of(d, sets, 3)
}

/// A pool of `sets` sets of `per` storage buffers.
fn descriptor_pool_of(
    d: &ash::Device,
    sets: u32,
    per: u32,
) -> Result<vk::DescriptorPool, ComputeError> {
    let sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(per * sets)];
    // SAFETY: the create-info is valid for the call.
    unsafe {
        d.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(sets)
                .pool_sizes(&sizes),
            None,
        )
    }
    .map_err(|e| err("descriptor pool", e))
}

impl Submitter {
    pub(crate) fn new(shared: &Arc<Shared>, timestamps: bool) -> Result<Self, ComputeError> {
        let d = &shared.device;
        // SAFETY: each create-info is valid for its call; objects created before a
        // failure are destroyed.
        unsafe {
            let pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(shared.queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| err("command pool", e))?;
            let cleanup = |e: ComputeError| {
                d.destroy_command_pool(pool, None);
                e
            };
            let cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| cleanup(err("command buffer", e)))?[0];
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| cleanup(err("fence", e)))?;
            let descriptors = match descriptor_pool(d, 64) {
                Ok(p) => p,
                Err(e) => {
                    d.destroy_fence(fence, None);
                    return Err(cleanup(e));
                }
            };
            let fused_pool = if shared.fused.is_some() {
                match descriptor_pool_of(d, 16, 4) {
                    Ok(p) => p,
                    Err(e) => {
                        d.destroy_descriptor_pool(descriptors, None);
                        d.destroy_fence(fence, None);
                        return Err(cleanup(e));
                    }
                }
            } else {
                vk::DescriptorPool::null()
            };
            let queries = if timestamps && shared.timestamp_mask != 0 {
                match d.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                ) {
                    Ok(q) => Some(q),
                    Err(e) => {
                        if fused_pool != vk::DescriptorPool::null() {
                            d.destroy_descriptor_pool(fused_pool, None);
                        }
                        d.destroy_descriptor_pool(descriptors, None);
                        d.destroy_fence(fence, None);
                        return Err(cleanup(err("query pool", e)));
                    }
                }
            } else {
                None
            };
            Ok(Self {
                shared: Arc::clone(shared),
                pool,
                cmd,
                fence,
                descriptors,
                sets: 64,
                queries,
                staging: None,
                xs: None,
                ys: None,
                fused_pool,
                fused_sets: 16,
                mid: None,
                table: None,
                logits: None,
                contrib: None,
                routes: None,
                att: None,
                hs: None,
                oproj: None,
                axs: None,
                sync: None,
                knew: None,
                qkv: None,
            })
        }
    }

    /// A host buffer of at least `bytes` in `slot`, grown by doubling.
    fn host_buffer<'s>(
        shared: &Arc<Shared>,
        slot: &'s mut Option<Allocation>,
        bytes: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<&'s mut Allocation, ComputeError> {
        Self::scratch(shared, slot, bytes, usage, true)
    }

    /// A host or device buffer of at least `bytes` in `slot`, grown by doubling.
    fn scratch<'s>(
        shared: &Arc<Shared>,
        slot: &'s mut Option<Allocation>,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        host: bool,
    ) -> Result<&'s mut Allocation, ComputeError> {
        if slot.as_ref().is_none_or(|a| a.size < bytes) {
            *slot = None;
            let size = bytes.max(4096).next_power_of_two();
            *slot = Some(Allocation::new(shared, size, usage, host)?);
        }
        slot.as_mut()
            .ok_or_else(|| err("scratch buffer", "missing"))
    }

    /// Whether this submitter times its submissions on the GPU.
    pub(crate) const fn timed(&self) -> bool {
        self.queries.is_some()
    }

    /// Records with `record`, submits, and waits.
    fn run(
        &mut self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<Ran, ComputeError> {
        let d = &self.shared.device;
        let cmds = [self.cmd];
        let queries = self.queries;
        // SAFETY: the command buffer is not in flight (the last submission was waited
        // for), its pool allows reset, and the queue is externally synchronised by the
        // queue lock.
        unsafe {
            d.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| err("reset", e))?;
            d.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| err("begin", e))?;
            if let Some(q) = queries {
                d.cmd_reset_query_pool(self.cmd, q, 0, 2);
                d.cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0);
            }
            #[cfg(feature = "spike")]
            let prefix = lock_hook(&self.shared.spike).pending.take();
            #[cfg(feature = "spike")]
            if let Some(p) = &prefix {
                p.record(d, self.cmd);
            }
            record(d, self.cmd);
            if let Some(q) = queries {
                d.cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1);
            }
            d.end_command_buffer(self.cmd).map_err(|e| err("end", e))?;
            let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
            {
                let _queue = self
                    .shared
                    .queue_lock
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                d.queue_submit(self.shared.queue, &submit, self.fence)
                    .map_err(|e| err("submit", e))?;
            }
            let submitted = Instant::now();
            let waited = d.wait_for_fences(&[self.fence], true, u64::MAX);
            let wait_ns = u64::try_from(submitted.elapsed().as_nanos()).unwrap_or(u64::MAX);
            d.reset_fences(&[self.fence])
                .map_err(|e| err("fence reset", e))?;
            waited.map_err(|e| err("wait", e))?;
            let gpu = match queries {
                None => None,
                Some(q) => {
                    let mut t = [0u64; 2];
                    // The fence has signalled, so both results are available.
                    d.get_query_pool_results(
                        q,
                        0,
                        &mut t,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .map_err(|e| err("timestamps", e))?;
                    let period = f64::from(self.shared.limits.timestamp_period);
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::cast_precision_loss
                    )]
                    let ns = |v: u64| ((v & self.shared.timestamp_mask) as f64 * period) as u64;
                    let (start, end) = (ns(t[0]), ns(t[1]));
                    Some((start, end.max(start)))
                }
            };
            #[cfg(feature = "spike")]
            {
                let mut hook = lock_hook(&self.shared.spike);
                hook.last = Some(SpikeTiming {
                    wait_ns,
                    gpu_ns: gpu.map(|(a, b)| b - a),
                });
                if prefix.is_some() {
                    hook.done = prefix;
                }
            }
            Ok(Ran {
                submitted,
                wait_ns,
                gpu,
            })
        }
    }

    /// New device-local memory for `len` bytes, padded to whole 32-bit words (the
    /// shaders read words), counted against `capacity`.
    fn allocate(
        &self,
        len: usize,
        counters: &Arc<Counters>,
        capacity: u64,
    ) -> Result<Allocation, ComputeError> {
        let size = (len as u64).max(1).next_multiple_of(4);
        counters.reserve(size, capacity)?;
        let mut dst = match Allocation::new(
            &self.shared,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            false,
        ) {
            Ok(a) => a,
            Err(e) => {
                counters.used.fetch_sub(size, Ordering::Relaxed);
                return Err(e);
            }
        };
        dst.counted = Some(Arc::clone(counters));
        Ok(dst)
    }

    /// Copies `host`, then zeros up to `size` bytes, into `dst` at `offset`, through
    /// the staging buffer, in chunks.
    fn copy_in(
        &mut self,
        dst: vk::Buffer,
        offset: u64,
        host: &[u8],
        size: u64,
    ) -> Result<(), ComputeError> {
        let shared = Arc::clone(&self.shared);
        let mut done = 0u64;
        while done < size {
            let n = (size - done).min(STAGING);
            let stage = Self::host_buffer(
                &shared,
                &mut self.staging,
                STAGING.min(size),
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            let bytes = stage.bytes_mut();
            let (from, to) = (
                usize::try_from(done).unwrap_or(usize::MAX),
                usize::try_from(done + n).unwrap_or(usize::MAX),
            );
            let real = host.len().min(to).saturating_sub(from);
            bytes[..real].copy_from_slice(&host[from..from + real]);
            bytes[real..usize::try_from(n).unwrap_or(0)].fill(0);
            let src = stage.buffer;
            let _ = self.run(|d, cmd| {
                let region = [vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(offset + done)
                    .size(n)];
                // SAFETY: both buffers are live and large enough for the region.
                unsafe { d.cmd_copy_buffer(cmd, src, dst, &region) };
            })?;
            done += n;
        }
        Ok(())
    }

    /// Copies `host` into a new device-local buffer.
    pub(crate) fn upload(
        &mut self,
        host: &[u8],
        counters: &Arc<Counters>,
        capacity: u64,
        guard: Option<Guard>,
    ) -> Result<DeviceBuffer, ComputeError> {
        let dst = self.allocate(host.len(), counters, capacity)?;
        let start = Instant::now();
        self.copy_in(dst.buffer, 0, host, dst.size)?;
        counters.time(start);
        counters.uploads.fetch_add(1, Ordering::Relaxed);
        counters
            .upload_bytes
            .fetch_add(host.len() as u64, Ordering::Relaxed);
        Ok(DeviceBuffer::new(
            host.len(),
            WeightBuffer { alloc: dst },
            guard,
        ))
    }

    /// New device-local memory of `len` bytes, its contents unspecified (an arena).
    pub(crate) fn alloc(
        &self,
        len: usize,
        counters: &Arc<Counters>,
        capacity: u64,
        guard: Option<Guard>,
    ) -> Result<DeviceBuffer, ComputeError> {
        let dst = self.allocate(len, counters, capacity)?;
        Ok(DeviceBuffer::new(len, WeightBuffer { alloc: dst }, guard))
    }

    /// The allocation behind `b`, if it belongs to this device.
    fn allocation_of<'b>(&self, b: &'b DeviceBuffer) -> Result<&'b Allocation, ComputeError> {
        match b.state::<WeightBuffer>() {
            Some(w) if Arc::ptr_eq(&w.alloc.shared, &self.shared) => Ok(&w.alloc),
            _ => Err(ComputeError::Device {
                backend: BACKEND,
                detail: "the buffer belongs to another device".into(),
            }),
        }
    }

    /// Copies `host` into `buffer` at `offset`, counted as an upload.
    pub(crate) fn write(
        &mut self,
        buffer: &DeviceBuffer,
        offset: usize,
        host: &[u8],
        counters: &Counters,
    ) -> Result<(), ComputeError> {
        let (dst, size) = {
            let a = self.allocation_of(buffer)?;
            (a.buffer, a.size)
        };
        let (off, len) = (offset as u64, host.len() as u64);
        if off.checked_add(len).is_none_or(|end| end > size) {
            return Err(err(
                "write",
                format!("{len} bytes at {off} are outside a {size}-byte buffer"),
            ));
        }
        if len == 0 {
            return Ok(());
        }
        let start = Instant::now();
        self.copy_in(dst, off, host, len)?;
        counters.time(start);
        counters.uploads.fetch_add(1, Ordering::Relaxed);
        counters.upload_bytes.fetch_add(len, Ordering::Relaxed);
        Ok(())
    }

    /// Runs `ops` as one submission.
    pub(crate) fn matmul_many(
        &mut self,
        ops: &mut [DeviceMatMul<'_>],
        counters: &Counters,
    ) -> Result<(), ComputeError> {
        let lim = self.shared.limits;
        // Check every op and lay out the activations before touching the GPU.
        let mut plan = Vec::with_capacity(ops.len());
        let (mut xn, mut yn) = (0usize, 0usize);
        // Ops that read the same activations (QKV, gate and up) share one copy: by
        // address and length, so only an identical slice is shared. `Some(base)`: this
        // op reads the copy an earlier op made; `None`: its inputs are copied here.
        let mut shared_x: Vec<Option<usize>> = Vec::with_capacity(ops.len());
        let mut x_bases: Vec<(*const f32, usize, usize)> = Vec::with_capacity(ops.len());
        for op in ops.iter() {
            let w = op.w;
            let Some(dt) = dtype_index(w.dtype()) else {
                return Err(ComputeError::Unsupported {
                    backend: BACKEND,
                    op: "matmul",
                    dtype: w.dtype(),
                });
            };
            let alloc = match w.buffer().state::<WeightBuffer>() {
                Some(b) if Arc::ptr_eq(&b.alloc.shared, &self.shared) => &b.alloc,
                _ => {
                    return Err(ComputeError::Device {
                        backend: BACKEND,
                        detail: "the buffer belongs to another device".into(),
                    });
                }
            };
            let (n, k, s) = (w.rows(), w.cols(), op.s);
            if op.xs.len() != s * k || op.ys.len() != s * n {
                return Err(ComputeError::Shape {
                    op: "matmul",
                    detail: format!(
                        "{n} x {k} weights, {s} rows: {} inputs, {} outputs",
                        op.xs.len(),
                        op.ys.len()
                    ),
                });
            }
            let row_bytes = w.len().checked_div(n).unwrap_or(0);
            let align = lim.min_storage_buffer_offset_alignment.max(4);
            let off = w.offset() as u64;
            let base = off / align * align;
            let end = (off + w.len() as u64).next_multiple_of(4).min(alloc.size);
            let range = end.saturating_sub(base).max(4);
            let too_big = || {
                err(
                    "matmul",
                    format!(
                        "a {n} x {k} {} matrix is beyond the device's storage buffer range",
                        w.dtype()
                    ),
                )
            };
            if range > u64::from(lim.max_storage_buffer_range)
                || u32::try_from(off - base + w.len() as u64).is_err()
            {
                return Err(too_big());
            }
            let groups = (n as u64).div_ceil(u64::from(ROWS));
            let limits = || {
                err(
                    "matmul",
                    format!("{n} rows x {s} tokens exceed the dispatch limits"),
                )
            };
            let groups = u32::try_from(groups)
                .ok()
                .filter(|&g| g <= lim.max_compute_work_group_count[0])
                .ok_or_else(limits)?;
            let tokens = u32::try_from(s)
                .ok()
                .filter(|&t| t <= lim.max_compute_work_group_count[1])
                .ok_or_else(limits)?;
            let params = Params {
                n: u32_of(n, "rows")?,
                k: u32_of(k, "columns")?,
                row_bytes: u32_of(row_bytes, "row bytes")?,
                w_base: u32_of(off - base, "offset")?,
                x_base: u32_of(
                    match x_bases
                        .iter()
                        .find(|(p, l, _)| *p == op.xs.as_ptr() && *l == op.xs.len())
                    {
                        Some(&(_, _, b)) => {
                            shared_x.push(Some(b));
                            b
                        }
                        None => {
                            shared_x.push(None);
                            x_bases.push((op.xs.as_ptr(), op.xs.len(), xn));
                            xn += op.xs.len();
                            xn - op.xs.len()
                        }
                    },
                    "activations",
                )?,
                y_base: u32_of(yn, "outputs")?,
            };
            plan.push(Planned {
                buffer: alloc.buffer,
                base,
                range,
                dtype: dt,
                params,
                groups,
                tokens,
            });
            yn += op.ys.len();
        }
        let (xb, yb) = (4 * xn.max(1) as u64, 4 * yn.max(1) as u64);
        let max = u64::from(lim.max_storage_buffer_range);
        if xb > max || yb > max {
            return Err(err(
                "matmul",
                "the submission's activations exceed the storage buffer range",
            ));
        }
        counters.submissions.fetch_add(1, Ordering::Relaxed);
        if xn == 0 && yn == 0 {
            return Ok(());
        }

        // Activations in.
        let start = Instant::now();
        let shared = Arc::clone(&self.shared);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let xbuf = {
            let x = Self::host_buffer(&shared, &mut self.xs, xb, usage)?;
            let floats = x.floats_mut();
            for ((op, p), shared) in ops.iter().zip(&plan).zip(&shared_x) {
                if shared.is_none() {
                    let base = p.params.x_base as usize;
                    floats[base..base + op.xs.len()].copy_from_slice(op.xs);
                }
            }
            x.buffer
        };
        let ybuf = Self::host_buffer(&shared, &mut self.ys, yb, usage)?.buffer;
        counters.time(start);
        counters
            .activations_in
            .fetch_add(4 * xn as u64, Ordering::Relaxed);

        // Descriptor sets, one per op.
        let prepared = Instant::now();
        let d = &shared.device;
        let count = u32::try_from(plan.len()).map_err(|e| err("ops", e))?;
        if count > self.sets {
            let sets = count.next_power_of_two();
            let pool = descriptor_pool(d, sets)?;
            // SAFETY: no set from the old pool is in use (nothing is in flight).
            unsafe { d.destroy_descriptor_pool(self.descriptors, None) };
            self.descriptors = pool;
            self.sets = sets;
        }
        let layouts = vec![shared.set_layout; plan.len()];
        // SAFETY: the pool has room for `count` sets of this layout after a reset, and
        // none of its sets is in use.
        let sets = unsafe {
            d.reset_descriptor_pool(self.descriptors, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| err("descriptor reset", e))?;
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptors)
                    .set_layouts(&layouts),
            )
            .map_err(|e| err("descriptor sets", e))?
        };
        for (set, p) in sets.iter().zip(&plan) {
            let wi = [vk::DescriptorBufferInfo::default()
                .buffer(p.buffer)
                .offset(p.base)
                .range(p.range)];
            let xi = [vk::DescriptorBufferInfo::default()
                .buffer(xbuf)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let yi = [vk::DescriptorBufferInfo::default()
                .buffer(ybuf)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&wi),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&xi),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&yi),
            ];
            // SAFETY: the set is not in use, and the buffers are live.
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }

        let (layout, pipelines) = (shared.layout, shared.pipelines.clone());
        let ran = self.run(|d, cmd| {
            // SAFETY: recording into a command buffer in the recording state, with
            // objects that outlive the submission (it is waited for in `run`).
            unsafe {
                for (set, p) in sets.iter().zip(&plan) {
                    let (dt, params, groups, s) = (p.dtype, p.params, p.groups, p.tokens);
                    if groups == 0 || s == 0 {
                        continue;
                    }
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines[dt]);
                    d.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &[*set],
                        &[],
                    );
                    d.cmd_push_constants(
                        cmd,
                        layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &params.bytes(),
                    );
                    d.cmd_dispatch(cmd, groups, s, 1);
                }
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
            }
        })?;
        counters.ran(prepared, &ran);

        // Outputs back.
        let start = Instant::now();
        let y = self.ys.as_mut().ok_or_else(|| err("outputs", "missing"))?;
        let floats = y.floats_mut();
        for (op, p) in ops.iter_mut().zip(&plan) {
            let base = p.params.y_base as usize;
            op.ys.copy_from_slice(&floats[base..base + op.ys.len()]);
        }
        counters.time(start);
        counters
            .activations_out
            .fetch_add(4 * yn as u64, Ordering::Relaxed);
        Ok(())
    }
}

/// One matrix of a fused dispatch, laid out: `ys[y + t·n ..] = W · xs[x + t·k ..]` for
/// `t < tokens`, offsets in floats.
struct FusedMat<'w> {
    w: &'w DeviceWeights,
    x: usize,
    y: usize,
    tokens: usize,
}

/// One fused dispatch: matrices of one dtype, one row count and one buffer.
struct FusedGroup {
    buffer: vk::Buffer,
    base: u64,
    range: u64,
    dtype: usize,
    blocks: u32,
    count: u32,
    tokens: u32,
    /// Word offset of the group's first table entry.
    tab: u32,
}

/// Words per table entry. Must match the shader's `8u * qm`.
const ENTRY_WORDS: usize = 8;

impl Submitter {
    /// Splits `mats` into fused dispatches and appends their table entries to `table`:
    /// consecutive matrices with the same buffer, dtype and row count share one, within
    /// the device's dispatch and storage-buffer limits.
    fn fused_groups(
        &self,
        mats: &[FusedMat<'_>],
        table: &mut Vec<u32>,
    ) -> Result<Vec<FusedGroup>, ComputeError> {
        let lim = self.shared.limits;
        let align = lim.min_storage_buffer_offset_alignment.max(4);
        let max_range = u64::from(lim.max_storage_buffer_range);
        let mut groups: Vec<FusedGroup> = Vec::new();
        // Each new entry's weight offset in its buffer, rebased on its group below.
        let first_entry = table.len();
        let mut offsets: Vec<u64> = Vec::with_capacity(mats.len());
        // The first and last byte of the open group's weights.
        let mut span = (0u64, 0u64);
        for m in mats {
            let w = m.w;
            let Some(dt) = dtype_index(w.dtype()) else {
                return Err(ComputeError::Unsupported {
                    backend: BACKEND,
                    op: "matmul",
                    dtype: w.dtype(),
                });
            };
            let alloc = self.allocation_of(w.buffer())?;
            let (n, k) = (w.rows(), w.cols());
            let blocks = u32_of((n as u64).div_ceil(u64::from(ROWS)), "row blocks")?;
            let tokens = u32_of(m.tokens, "tokens")?;
            if tokens > lim.max_compute_work_group_count[1] {
                return Err(err(
                    "matmul",
                    format!("{tokens} tokens exceed the dispatch limits"),
                ));
            }
            let (off, end) = (w.offset() as u64, (w.offset() + w.len()) as u64);
            let joins = groups.last().is_some_and(|g| {
                let (lo, hi) = (span.0.min(off), span.1.max(end));
                let base = lo / align * align;
                g.buffer == alloc.buffer
                    && g.dtype == dt
                    && g.blocks == blocks
                    && u64::from(g.count + 1) * u64::from(blocks)
                        <= u64::from(lim.max_compute_work_group_count[0])
                    && hi.next_multiple_of(4).min(alloc.size) - base <= max_range
                    && hi - base <= u64::from(u32::MAX)
            });
            if joins {
                span = (span.0.min(off), span.1.max(end));
            } else {
                if u64::from(blocks) > u64::from(lim.max_compute_work_group_count[0]) {
                    return Err(err(
                        "matmul",
                        format!("{n} rows exceed the dispatch limits"),
                    ));
                }
                span = (off, end);
                groups.push(FusedGroup {
                    buffer: alloc.buffer,
                    base: 0,
                    range: 0,
                    dtype: dt,
                    blocks,
                    count: 0,
                    tokens: 0,
                    tab: u32_of(table.len(), "table")?,
                });
            }
            let g = groups.last_mut().expect("a group");
            g.count += 1;
            g.tokens = g.tokens.max(tokens);
            g.base = span.0 / align * align;
            g.range = (span.1.next_multiple_of(4).min(alloc.size) - g.base).max(4);
            if g.range > max_range {
                return Err(err(
                    "matmul",
                    format!(
                        "a {n} x {k} {} matrix is beyond the device's storage buffer range",
                        w.dtype()
                    ),
                ));
            }
            // `w_base` is relative to the group's base, which may still move down as
            // matrices join: it is filled in below.
            offsets.push(off);
            table.extend([
                u32_of(n, "rows")?,
                u32_of(k, "columns")?,
                u32_of(w.len().checked_div(n).unwrap_or(0), "row bytes")?,
                0,
                u32_of(m.x, "activations")?,
                u32_of(m.y, "outputs")?,
                tokens,
                0,
            ]);
        }
        // Every entry's weight offset relative to its group's final base.
        let mut i = 0;
        for g in &groups {
            for _ in 0..g.count {
                let e = first_entry + i * ENTRY_WORDS;
                table[e + 3] = u32_of(offsets[i] - g.base, "offset")?;
                i += 1;
            }
        }
        Ok(groups)
    }

    /// The fused descriptor pool with room for `count` sets, reset.
    fn fused_sets(&mut self, count: u32) -> Result<Vec<vk::DescriptorSet>, ComputeError> {
        let shared = Arc::clone(&self.shared);
        let d = &shared.device;
        let fused = shared
            .fused
            .as_ref()
            .ok_or_else(|| err("fused", "no pipelines"))?;
        if count > self.fused_sets {
            let sets = count.next_power_of_two();
            let pool = descriptor_pool_of(d, sets, 4)?;
            // SAFETY: no set from the old pool is in use (nothing is in flight).
            unsafe { d.destroy_descriptor_pool(self.fused_pool, None) };
            self.fused_pool = pool;
            self.fused_sets = sets;
        }
        let layouts = vec![fused.set_layout; count as usize];
        // SAFETY: the pool has room for `count` sets of this layout after a reset, and
        // none of its sets is in use.
        unsafe {
            d.reset_descriptor_pool(self.fused_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| err("descriptor reset", e))?;
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.fused_pool)
                    .set_layouts(&layouts),
            )
            .map_err(|e| err("descriptor sets", e))
        }
    }

    /// Points `set`'s four bindings at the given ranges (`None`: the whole buffer).
    fn write_set(&self, set: vk::DescriptorSet, bindings: [(vk::Buffer, u64, Option<u64>); 4]) {
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bindings
            .iter()
            .map(|&(b, off, range)| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b)
                    .offset(off)
                    .range(range.unwrap_or(vk::WHOLE_SIZE))]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .zip(0u32..)
            .map(|(info, binding)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        // SAFETY: the set is not in use, and the buffers are live.
        unsafe { self.shared.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Runs `ops` as one submission: every gate and up as fused dispatches, SwiGLU, then
    /// every down as fused dispatches, with the intermediates on the device. Each
    /// output has the bits of the separate steps.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn gated_ffn_many(
        &mut self,
        ops: &mut [DeviceGatedFfn<'_>],
        counters: &Counters,
    ) -> Result<(), ComputeError> {
        let lim = self.shared.limits;
        // Check and lay out before touching the GPU. Inputs: shared slices share one
        // copy (as in `matmul_many`). Intermediates: every gate output, then every up
        // output, in the same order, so SwiGLU is one elementwise pass.
        let (mut xn, mut gn, mut yn) = (0usize, 0usize, 0usize);
        let mut x_at: Vec<(usize, bool)> = Vec::with_capacity(ops.len());
        let mut seen: Vec<(*const f32, usize, usize)> = Vec::new();
        let mut g_at = Vec::with_capacity(ops.len());
        let mut y_at = Vec::with_capacity(ops.len());
        for op in ops.iter() {
            let (g, u, d) = (op.gate, op.up, op.down);
            if g.rows() != u.rows()
                || g.cols() != u.cols()
                || d.cols() != g.rows()
                || op.xs.len() != op.s * g.cols()
                || op.ys.len() != op.s * d.rows()
            {
                return Err(ComputeError::Shape {
                    op: "gated ffn",
                    detail: format!(
                        "gate {} x {}, up {} x {}, down {} x {}, {} rows: {} inputs, {} outputs",
                        g.rows(),
                        g.cols(),
                        u.rows(),
                        u.cols(),
                        d.rows(),
                        d.cols(),
                        op.s,
                        op.xs.len(),
                        op.ys.len()
                    ),
                });
            }
            match seen
                .iter()
                .find(|(p, l, _)| *p == op.xs.as_ptr() && *l == op.xs.len())
            {
                Some(&(_, _, b)) => x_at.push((b, false)),
                None => {
                    seen.push((op.xs.as_ptr(), op.xs.len(), xn));
                    x_at.push((xn, true));
                    xn += op.xs.len();
                }
            }
            g_at.push(gn);
            gn += op.s * g.rows();
            y_at.push(yn);
            yn += op.ys.len();
        }
        let max = u64::from(lim.max_storage_buffer_range);
        let (xb, mb, yb) = (
            4 * xn.max(1) as u64,
            8 * gn.max(1) as u64,
            4 * yn.max(1) as u64,
        );
        if xb > max || mb > max || yb > max || u32::try_from(2 * gn).is_err() {
            return Err(err(
                "gated ffn",
                "the submission's activations exceed the storage buffer range",
            ));
        }
        let first: Vec<FusedMat<'_>> = ops
            .iter()
            .zip(&x_at)
            .zip(&g_at)
            .flat_map(|((op, &(x, _)), &g)| {
                [
                    FusedMat {
                        w: op.gate,
                        x,
                        y: g,
                        tokens: op.s,
                    },
                    FusedMat {
                        w: op.up,
                        x,
                        y: gn + g,
                        tokens: op.s,
                    },
                ]
            })
            .filter(|m| m.tokens > 0)
            .collect();
        let second: Vec<FusedMat<'_>> = ops
            .iter()
            .zip(&g_at)
            .zip(&y_at)
            .map(|((op, &g), &y)| FusedMat {
                w: op.down,
                x: g,
                y,
                tokens: op.s,
            })
            .filter(|m| m.tokens > 0)
            .collect();
        let mut table = Vec::new();
        let groups1 = self.fused_groups(&first, &mut table)?;
        let groups2 = self.fused_groups(&second, &mut table)?;
        counters.submissions.fetch_add(1, Ordering::Relaxed);
        if gn == 0 {
            // No rows: every output is empty.
            return Ok(());
        }
        let swiglu_groups = u32_of(gn.div_ceil(SWIGLU_GROUP as usize), "SwiGLU groups")?;
        if swiglu_groups > lim.max_compute_work_group_count[0] {
            return Err(err("gated ffn", "SwiGLU exceeds the dispatch limits"));
        }

        // Inputs and the table in.
        let start = Instant::now();
        let shared = Arc::clone(&self.shared);
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let xbuf = {
            let x = Self::host_buffer(&shared, &mut self.xs, xb, storage)?;
            let floats = x.floats_mut();
            for (op, &(base, copy)) in ops.iter().zip(&x_at) {
                if copy {
                    floats[base..base + op.xs.len()].copy_from_slice(op.xs);
                }
            }
            x.buffer
        };
        let tbuf = {
            let t = Self::host_buffer(
                &shared,
                &mut self.table,
                4 * table.len().max(1) as u64,
                storage,
            )?;
            let words = t.floats_mut();
            for (dst, &v) in words.iter_mut().zip(&table) {
                *dst = f32::from_bits(v);
            }
            t.buffer
        };
        let ybuf = Self::host_buffer(&shared, &mut self.ys, yb, storage)?.buffer;
        let mbuf = Self::scratch(&shared, &mut self.mid, mb, storage, false)?.buffer;
        counters.time(start);
        counters
            .activations_in
            .fetch_add(4 * xn as u64, Ordering::Relaxed);

        // One descriptor set per fused dispatch.
        let prepared = Instant::now();
        let total = u32_of(groups1.len() + groups2.len(), "dispatches")?;
        let sets = self.fused_sets(total.max(1))?;
        for (set, g) in sets.iter().zip(&groups1) {
            self.write_set(
                *set,
                [
                    (g.buffer, g.base, Some(g.range)),
                    (xbuf, 0, None),
                    (mbuf, 0, None),
                    (tbuf, 0, None),
                ],
            );
        }
        for (set, g) in sets[groups1.len()..].iter().zip(&groups2) {
            self.write_set(
                *set,
                [
                    (g.buffer, g.base, Some(g.range)),
                    (mbuf, 0, None),
                    (ybuf, 0, None),
                    (tbuf, 0, None),
                ],
            );
        }
        let fused = shared
            .fused
            .as_ref()
            .ok_or_else(|| err("fused", "no pipelines"))?;
        let (layout, matmul, swiglu) = (fused.layout, &fused.matmul, fused.swiglu);
        let swiglu_set = sets[0];
        let gn32 = u32_of(gn, "intermediates")?;
        let ran = self.run(|d, cmd| {
            let push = |words: [u32; 4]| {
                let mut b = [0u8; 16];
                for (i, w) in words.iter().enumerate() {
                    b[4 * i..4 * i + 4].copy_from_slice(&w.to_ne_bytes());
                }
                b
            };
            let between = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            // SAFETY: recording into a command buffer in the recording state, with
            // objects that outlive the submission (it is waited for in `run`).
            unsafe {
                let dispatch = |set: vk::DescriptorSet, g: &FusedGroup| {
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, matmul[g.dtype]);
                    d.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &[set],
                        &[],
                    );
                    d.cmd_push_constants(
                        cmd,
                        layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &push([g.blocks, g.count, g.tab, 0]),
                    );
                    d.cmd_dispatch(cmd, g.count * g.blocks, g.tokens, 1);
                };
                let barrier = |dst: vk::PipelineStageFlags, m: &[vk::MemoryBarrier<'_>]| {
                    d.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        dst,
                        vk::DependencyFlags::empty(),
                        m,
                        &[],
                        &[],
                    );
                };
                for (set, g) in sets.iter().zip(&groups1) {
                    dispatch(*set, g);
                }
                barrier(vk::PipelineStageFlags::COMPUTE_SHADER, &between);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, swiglu);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[swiglu_set],
                    &[],
                );
                d.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push([gn32, 0, gn32, 0]),
                );
                d.cmd_dispatch(cmd, swiglu_groups, 1, 1);
                barrier(vk::PipelineStageFlags::COMPUTE_SHADER, &between);
                for (set, g) in sets[groups1.len()..].iter().zip(&groups2) {
                    dispatch(*set, g);
                }
                let to_host = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                barrier(vk::PipelineStageFlags::HOST, &to_host);
            }
        })?;
        counters.ran(prepared, &ran);

        // Outputs back.
        let start = Instant::now();
        let y = self.ys.as_mut().ok_or_else(|| err("outputs", "missing"))?;
        let floats = y.floats_mut();
        for (op, &base) in ops.iter_mut().zip(&y_at) {
            op.ys.copy_from_slice(&floats[base..base + op.ys.len()]);
        }
        counters.time(start);
        counters
            .activations_out
            .fetch_add(4 * yn as u64, Ordering::Relaxed);
        Ok(())
    }

    /// The SwiGLU shader alone, `g = silu(g) · u` in place, for the conformance tests.
    pub(crate) fn swiglu(&mut self, g: &mut [f32], u: &[f32]) -> Result<(), ComputeError> {
        let n = g.len();
        if u.len() != n {
            return Err(ComputeError::Shape {
                op: "swiglu",
                detail: format!("{n} gate values, {} up values", u.len()),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let groups = u32_of(n.div_ceil(SWIGLU_GROUP as usize), "SwiGLU groups")?;
        if groups > self.shared.limits.max_compute_work_group_count[0]
            || 8 * n as u64 > u64::from(self.shared.limits.max_storage_buffer_range)
        {
            return Err(err(
                "swiglu",
                format!("{n} values exceed the device's limits"),
            ));
        }
        let shared = Arc::clone(&self.shared);
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let ybuf = {
            let y = Self::host_buffer(&shared, &mut self.ys, 8 * n as u64, storage)?;
            let floats = y.floats_mut();
            floats[..n].copy_from_slice(g);
            floats[n..2 * n].copy_from_slice(u);
            y.buffer
        };
        let sets = self.fused_sets(1)?;
        self.write_set(sets[0], [(ybuf, 0, None); 4]);
        let fused = shared
            .fused
            .as_ref()
            .ok_or_else(|| err("fused", "no pipelines"))?;
        let (layout, pipeline) = (fused.layout, fused.swiglu);
        let n32 = u32_of(n, "values")?;
        let set = sets[0];
        self.run(|d, cmd| {
            let mut push = [0u8; 16];
            for (i, w) in [n32, 0, n32, 0].iter().enumerate() {
                push[4 * i..4 * i + 4].copy_from_slice(&w.to_ne_bytes());
            }
            let to_host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            // SAFETY: as in `gated_ffn_many`.
            unsafe {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                d.cmd_dispatch(cmd, groups, 1, 1);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &to_host,
                    &[],
                    &[],
                );
            }
        })?;
        let y = self.ys.as_mut().ok_or_else(|| err("outputs", "missing"))?;
        g.copy_from_slice(&y.floats_mut()[..n]);
        Ok(())
    }
}

/// One dispatch of a routed layer: pipeline, descriptor set, push constants, groups.
type Step = (vk::Pipeline, vk::DescriptorSet, [u32; 4], (u32, u32));

/// One kind of matrix (gate, up or down) across every expert of a routed layer: one
/// buffer, one dtype, one shape.
struct Kind {
    buffer: vk::Buffer,
    size: u64,
    dtype: usize,
    n: usize,
    k: usize,
    row_bytes: usize,
}

/// Decode attention's shape, checked against the device (audit §21).
struct AttnShape {
    d: usize,
    heads: usize,
    kv_heads: usize,
    q_out: usize,
    kv_out: usize,
    sync_rows: usize,
    pool: vk::Buffer,
    /// The QKV weights' buffer, the binding's base and range, and each matrix's dtype,
    /// rows, row bytes and offset from the base.
    wbuf: vk::Buffer,
    w_base: u64,
    w_range: u64,
    mats: [(usize, usize, usize, u64); 3],
}

/// Decode attention's section of the routed layer's table (audit §21), from word `start`:
/// the QKV entries (8 words each, as `FusedEntry`), the header `attn_prelude.comp` and
/// `attn_decode.comp` read, the Q and K norm weights, the RoPE rows, two words per token
/// (position, the table word of its block ids), the block ids, and each sync row's pool
/// position.
struct AttnTable {
    entries: usize,
    hdr: usize,
    qn: usize,
    kn: usize,
    cs: usize,
    tok: usize,
    blk: usize,
    sync_at: usize,
    end: usize,
    /// QKV dispatches: dtype, row blocks per matrix, the first entry's word, matrices.
    groups: Vec<(usize, usize, usize, usize)>,
}

/// The header's marker for "no Q and K norms".
const NO_NORM: u32 = u32::MAX;

impl AttnTable {
    fn new(a: &DeviceMoeAttention<'_>, sh: &AttnShape, s: usize, start: usize) -> Self {
        let entries = start;
        let hdr = entries + 24;
        let qn = hdr + 9;
        let (kn, cs) = if a.qk_norm.is_some() {
            (qn + sh.q_out, qn + sh.q_out + sh.kv_out)
        } else {
            (qn, qn)
        };
        let tok = cs + s * sh.d;
        let blk = tok + 2 * s;
        let sync_at = blk + a.blocks.len();
        let end = sync_at + a.sync_to.len();
        // Consecutive matrices with one dtype and one row-block count share a dispatch.
        let rows_of = |n: usize| n.div_ceil(ROWS as usize);
        let mut groups: Vec<(usize, usize, usize, usize)> = Vec::new();
        for (i, &(dt, n, _, _)) in sh.mats.iter().enumerate() {
            match groups.last_mut() {
                Some(g) if g.0 == dt && g.1 == rows_of(n) => g.3 += 1,
                _ => groups.push((dt, rows_of(n), entries + 8 * i, 1)),
            }
        }
        Self {
            entries,
            hdr,
            qn,
            kn,
            cs,
            tok,
            blk,
            sync_at,
            end,
            groups,
        }
    }

    fn fill(
        &self,
        table: &mut [u32],
        a: &DeviceMoeAttention<'_>,
        sh: &AttnShape,
        s: usize,
    ) -> Result<(), ComputeError> {
        let w32 = |v: usize, what: &str| u32_of(v, what);
        let cols = a.xs.len() / s;
        let y_base = [0, s * sh.q_out, s * (sh.q_out + sh.kv_out)];
        for (i, &(_, n, row_bytes, off)) in sh.mats.iter().enumerate() {
            let e = self.entries + 8 * i;
            table[e..e + 8].copy_from_slice(&[
                w32(n, "rows")?,
                w32(cols, "columns")?,
                w32(row_bytes, "row bytes")?,
                u32_of(off, "offset")?,
                0,
                w32(y_base[i], "outputs")?,
                w32(s, "tokens")?,
                0,
            ]);
        }
        // `eps as f32`, as `math::rms_norm` converts it.
        #[allow(clippy::cast_possible_truncation)]
        let eps = a.eps as f32;
        let (qn, kn) = match a.qk_norm {
            Some(_) => (w32(self.qn, "table")?, w32(self.kn, "table")?),
            None => (NO_NORM, NO_NORM),
        };
        table[self.hdr..self.hdr + 9].copy_from_slice(&[
            w32(sh.d, "head dim")?,
            w32(sh.heads, "heads")?,
            w32(sh.kv_heads, "KV heads")?,
            eps.to_bits(),
            a.clip.map_or(0, f32::to_bits),
            qn,
            kn,
            w32(self.cs, "table")?,
            w32(self.tok, "table")?,
        ]);
        if let Some((qw, kw)) = a.qk_norm {
            for (dst, w) in table[self.qn..].iter_mut().zip(qw.iter().chain(kw)) {
                *dst = w.to_bits();
            }
        }
        for (dst, v) in table[self.cs..self.tok].iter_mut().zip(a.rope) {
            *dst = v.to_bits();
        }
        for t in 0..s {
            table[self.tok + 2 * t] = a.pos[t];
            table[self.tok + 2 * t + 1] = w32(self.blk + a.table_at[t] as usize, "table")?;
        }
        table[self.blk..self.sync_at].copy_from_slice(a.blocks);
        table[self.sync_at..self.end].copy_from_slice(a.sync_to);
        Ok(())
    }
}

impl Submitter {
    /// Decode attention's shape on this device: `Ok(None)` when the kernels cannot run
    /// it (the caller then runs attention on the host), an error when the inputs do not
    /// fit together.
    fn attention_shape(
        &self,
        a: &DeviceMoeAttention<'_>,
        s: usize,
        cols: usize,
        att_cols: usize,
    ) -> Result<Option<AttnShape>, ComputeError> {
        let lim = self.shared.limits;
        let (d, heads, kvh) = (a.head_dim, a.heads, a.kv_heads);
        let (q_out, kv_out) = (heads * d, kvh * d);
        let rows = a.sync_to.len();
        let bad = |detail: String| ComputeError::Shape {
            op: "routed moe attention",
            detail,
        };
        if d == 0
            || d % 2 != 0
            || kvh == 0
            || heads % kvh != 0
            || q_out != att_cols
            || a.q.rows() != q_out
            || a.k.rows() != kv_out
            || a.v.rows() != kv_out
            || [a.q, a.k, a.v].iter().any(|w| w.cols() != cols)
            || a.xs.len() != s * cols
            || a.rope.len() != s * d
            || a.pos.len() != s
            || a.table_at.len() != s
            || a.k_new.len() != s * kv_out
            || a.v_new.len() != s * kv_out
            || a.sync.len() != rows * 2 * kv_out
            || a.qk_norm
                .is_some_and(|(qn, kn)| qn.len() != q_out || kn.len() != kv_out)
        {
            return Err(bad(format!(
                "{heads} heads of {d}, {kvh} KV heads, {s} tokens of {cols}, {att_cols} \
                 attention outputs"
            )));
        }
        for t in 0..s {
            let need = a.pos[t] as usize / KV_BLOCK + 1;
            if a.table_at[t] as usize + need > a.blocks.len() {
                return Err(bad(format!(
                    "token {t} has no block for position {}",
                    a.pos[t]
                )));
            }
        }
        let pa = self.allocation_of(a.pool)?;
        let held = pa.size / (4 * kv_block_words(kv_out) as u64).max(1);
        if a.blocks.iter().any(|&b| u64::from(b) >= held)
            || a.sync_to
                .iter()
                .any(|&p| u64::from(p) / KV_BLOCK as u64 >= held)
        {
            return Err(bad(format!("a block past the pool's {held}")));
        }
        let maxx = u64::from(lim.max_compute_work_group_count[0]);
        if d > 512
            || a.pos.iter().any(|&p| p as usize >= ATTN_POSITIONS)
            || heads as u64 > maxx
            || rows as u64 > maxx
            || pa.size > u64::from(lim.max_storage_buffer_range)
        {
            return Ok(None);
        }
        // The three projections: kernels for their dtypes, one buffer, one binding.
        let mut mats = [(0usize, 0usize, 0usize, 0u64); 3];
        let mut buf = None;
        for (i, w) in [a.q, a.k, a.v].into_iter().enumerate() {
            let Some(dt) = dtype_index(w.dtype()) else {
                return Ok(None);
            };
            let wa = self.allocation_of(w.buffer())?;
            if buf.is_some_and(|(b, _)| b != wa.buffer) {
                return Ok(None);
            }
            buf = Some((wa.buffer, wa.size));
            mats[i] = (dt, w.rows(), w.len() / w.rows().max(1), w.offset() as u64);
        }
        let Some((wbuf, wsize)) = buf else {
            return Ok(None);
        };
        let align = lim.min_storage_buffer_offset_alignment.max(4);
        let (lo, hi) = [a.q, a.k, a.v].iter().fold((u64::MAX, 0), |(lo, hi), w| {
            (
                lo.min(w.offset() as u64),
                hi.max((w.offset() + w.len()) as u64),
            )
        });
        let w_base = lo / align * align;
        let w_range = (hi.next_multiple_of(4).min(wsize) - w_base).max(4);
        if w_range > u64::from(lim.max_storage_buffer_range)
            || (ROWS as usize) == 0
            || mats
                .iter()
                .any(|&(_, n, _, _)| (n.div_ceil(ROWS as usize) as u64) * 3 > maxx)
        {
            return Ok(None);
        }
        for m in &mut mats {
            m.3 -= w_base;
        }
        Ok(Some(AttnShape {
            d,
            heads,
            kv_heads: kvh,
            q_out,
            kv_out,
            sync_rows: rows,
            pool: pa.buffer,
            wbuf,
            w_base,
            w_range,
            mats,
        }))
    }
}

impl Submitter {
    /// The kind of matrix `i` of every expert, if they all share a buffer, dtype and
    /// shape; `None` otherwise (the layer is then routed on the host).
    fn kind_of(&self, op: &DeviceRoutedMoe<'_>, i: usize) -> Result<Option<Kind>, ComputeError> {
        let mut kind: Option<Kind> = None;
        for e in op.experts {
            let w = &e[i];
            let Some(dt) = dtype_index(w.dtype()) else {
                return Ok(None);
            };
            let a = self.allocation_of(w.buffer())?;
            let (n, k) = (w.rows(), w.cols());
            let row_bytes = w.len().checked_div(n).unwrap_or(0);
            match &kind {
                None => {
                    kind = Some(Kind {
                        buffer: a.buffer,
                        size: a.size,
                        dtype: dt,
                        n,
                        k,
                        row_bytes,
                    });
                }
                Some(c) => {
                    if c.buffer != a.buffer
                        || c.dtype != dt
                        || c.n != n
                        || c.k != k
                        || c.row_bytes != row_bytes
                    {
                        return Ok(None);
                    }
                }
            }
        }
        Ok(kind)
    }

    /// Runs a MoE layer as one submission (Step 3b, audit §17.5): the router as a fused
    /// dispatch, routing on the device, which writes the routed entries of the table,
    /// every (token, rank) entry's gate and up, SwiGLU, every entry's down, and the
    /// rank-order combine. Returns `Ok(false)`, with nothing written, when the layer's
    /// experts are not all in one buffer per matrix kind with one dtype and shape, or the
    /// shape is past the device's limits.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn routed_moe(
        &mut self,
        op: &mut DeviceRoutedMoe<'_>,
        counters: &Counters,
    ) -> Result<bool, ComputeError> {
        let shared = Arc::clone(&self.shared);
        let (Some(fused), Some(routed)) = (shared.fused.as_ref(), shared.routed.as_ref()) else {
            return Ok(false);
        };
        let lim = shared.limits;
        let (ne, cols, s, k) = (op.router.rows(), op.router.cols(), op.s, op.k);
        let rows = op.ys.len().checked_div(s).unwrap_or(0);
        let xs_len = if op.prelude.is_some() { 0 } else { s * cols };
        if s == 0
            || k == 0
            || k > ne
            || op.experts.len() != ne
            || op.xs.len() != xs_len
            || op.ys.len() != s * rows
            || op.ids.len() != s * k
            || op.weights.len() != s * k
        {
            return Err(ComputeError::Shape {
                op: "routed moe",
                detail: format!(
                    "{ne}-row router, {} experts, top {k}, {s} tokens, {} outputs",
                    op.experts.len(),
                    op.ys.len()
                ),
            });
        }
        let (Some(g), Some(u), Some(d)) = (
            self.kind_of(op, 0)?,
            self.kind_of(op, 1)?,
            self.kind_of(op, 2)?,
        ) else {
            return Ok(false);
        };
        let ffn = g.n;
        if g.buffer != u.buffer
            || g.dtype != u.dtype
            || g.n != u.n
            || g.row_bytes != u.row_bytes
            || g.k != cols
            || d.k != ffn
            || d.n != rows
        {
            return Ok(false);
        }
        let Some(rdt) = dtype_index(op.router.dtype()) else {
            return Ok(false);
        };
        let ra = self.allocation_of(op.router.buffer())?;
        let (rbuf, rsize) = (ra.buffer, ra.size);
        // The prelude's O projection (Step 3c): one dtype with a kernel, `cols` rows.
        let pre = match &op.prelude {
            None => None,
            Some(p) => {
                let Some(odt) = dtype_index(p.o.dtype()) else {
                    return Ok(false);
                };
                let (on, ok) = (p.o.rows(), p.o.cols());
                if on != cols
                    || ok == 0
                    || p.att.len() != if op.attention.is_some() { 0 } else { s * ok }
                    || p.hs.len() != s * cols
                    || p.norm.len() != cols
                {
                    return Err(ComputeError::Shape {
                        op: "routed moe prelude",
                        detail: format!(
                            "O {on} x {ok}, {} attention outputs, {} residuals, {} norm \
                             weights, for {s} tokens of {cols}",
                            p.att.len(),
                            p.hs.len(),
                            p.norm.len()
                        ),
                    });
                }
                let oa = self.allocation_of(p.o.buffer())?;
                Some((odt, ok, p.o.len() / on, oa.buffer, oa.size))
            }
        };

        // Each binding's base (aligned down to the first matrix it reads) and range.
        let align = lim.min_storage_buffer_offset_alignment.max(4);
        let max_range = u64::from(lim.max_storage_buffer_range);
        let bind = |offs: &mut dyn Iterator<Item = (u64, u64)>, size: u64| {
            let (lo, hi) = offs.fold((u64::MAX, 0), |(lo, hi), (o, l)| (lo.min(o), hi.max(o + l)));
            let base = lo / align * align;
            let range = (hi.next_multiple_of(4).min(size) - base).max(4);
            (range <= max_range && hi - base <= u64::from(u32::MAX)).then_some((base, range))
        };
        let span = |i: usize| -> Vec<(u64, u64)> {
            op.experts
                .iter()
                .map(|e| (e[i].offset() as u64, e[i].len() as u64))
                .collect()
        };
        let (gu_spans, d_spans) = ([span(0), span(1)].concat(), span(2));
        let Some((gu_base, gu_range)) = bind(&mut gu_spans.iter().copied(), g.size) else {
            return Ok(false);
        };
        let Some((d_base, d_range)) = bind(&mut d_spans.iter().copied(), d.size) else {
            return Ok(false);
        };
        let Some((r_base, r_range)) = bind(
            &mut std::iter::once((op.router.offset() as u64, op.router.len() as u64)),
            rsize,
        ) else {
            return Ok(false);
        };
        let o_bind = match (&op.prelude, pre) {
            (Some(p), Some((_, _, _, _, osize))) => {
                let Some(b) = bind(
                    &mut std::iter::once((p.o.offset() as u64, p.o.len() as u64)),
                    osize,
                ) else {
                    return Ok(false);
                };
                Some(b)
            }
            _ => None,
        };

        // Decode attention (audit §21): its pipelines, the prelude, and shapes its kernels
        // take; past `ATTN_POSITIONS` positions it runs on the host.
        let attn = match (&op.attention, pre) {
            (None, _) => None,
            (Some(_), None) => return Ok(false),
            (Some(a), Some((_, ok, ..))) => match self.attention_shape(a, s, cols, ok)? {
                Some(shape) => Some(shape),
                None => return Ok(false),
            },
        };

        // Dispatch sizes, all known here: the GPU only picks which expert each entry
        // reads.
        let rows_of = |n: usize| (n as u64).div_ceil(u64::from(ROWS));
        let (rb, gb, db) = (rows_of(ne), rows_of(ffn), rows_of(rows));
        let sk = s * k;
        let gn = sk * ffn;
        let maxx = u64::from(lim.max_compute_work_group_count[0]);
        if s as u64 > u64::from(lim.max_compute_work_group_count[1])
            || rb > maxx
            || 2 * sk as u64 * gb > maxx
            || sk as u64 * db > maxx
            || (gn as u64).div_ceil(u64::from(SWIGLU_GROUP)) > maxx
            || ((s * rows) as u64).div_ceil(256) > maxx
            || s as u64 > maxx
            || ne > ROUTED_EXPERTS
        {
            return Ok(false);
        }
        let (xb, lb, mb, cb, rtb, yb) = (
            4 * (s * cols) as u64,
            4 * (s * ne) as u64,
            8 * gn as u64,
            4 * (sk * rows) as u64,
            8 * sk as u64,
            4 * (s * rows) as u64,
        );
        if [xb, lb, mb, cb, rtb, yb].iter().any(|&b| b > max_range) {
            return Ok(false);
        }
        // The prelude's buffers: attention outputs, residual stream, O outputs.
        let (ab, hb) = match pre {
            Some((_, ok, ..)) => (4 * (s * ok) as u64, 4 * (s * cols) as u64),
            None => (0, 0),
        };
        if ab > max_range || hb > max_range || (pre.is_some() && rows_of(cols) > maxx) {
            return Ok(false);
        }

        // The table: the router's entry, the header, the templates (3 per expert), and
        // room for the routed entries (2 per (token, rank) for gate and up, then 1 for
        // down), which the route shader writes.
        let hdr = 8usize;
        let tpl = 16usize;
        let ent = tpl + 24 * ne;
        // With a prelude: the O projection's entry, then the norm weights as bits.
        let (o_at, norm_at) = (ent + 24 * sk, ent + 24 * sk + 8);
        let pre_end = if pre.is_some() { norm_at + cols } else { o_at };
        // With attention, its section follows (`AttnTable`).
        let at = match (&op.attention, &attn) {
            (Some(a), Some(sh)) => Some(AttnTable::new(a, sh, s, pre_end)),
            _ => None,
        };
        let words = at.as_ref().map_or(pre_end, |t| t.end);
        let mut table = vec![0u32; words];
        if let (Some(a), Some(sh), Some(t)) = (&op.attention, &attn, &at) {
            t.fill(&mut table, a, sh, s)?;
        }
        let w32 = |v: u64, what: &str| u32_of(v, what);
        if let (Some(p), Some((_, ok, orb, ..)), Some((o_base, _))) = (&op.prelude, pre, o_bind) {
            table[o_at..o_at + 8].copy_from_slice(&[
                u32_of(cols, "rows")?,
                u32_of(ok, "columns")?,
                u32_of(orb, "row bytes")?,
                w32(p.o.offset() as u64 - o_base, "offset")?,
                0,
                0,
                u32_of(s, "tokens")?,
                0,
            ]);
            for (dst, w) in table[norm_at..].iter_mut().zip(p.norm) {
                *dst = w.to_bits();
            }
        }
        table[..8].copy_from_slice(&[
            u32_of(ne, "experts")?,
            u32_of(cols, "columns")?,
            u32_of(op.router.len() / ne, "row bytes")?,
            w32(op.router.offset() as u64 - r_base, "offset")?,
            0,
            0,
            u32_of(s, "tokens")?,
            0,
        ]);
        table[hdr..hdr + 6].copy_from_slice(&[
            u32::from(op.renormalise),
            u32_of(tpl, "table")?,
            u32_of(ent, "table")?,
            u32_of(cols, "columns")?,
            u32_of(ffn, "ffn")?,
            u32_of(rows, "rows")?,
        ]);
        for (e, ws) in op.experts.iter().enumerate() {
            for (i, (kind, base)) in [(&g, gu_base), (&u, gu_base), (&d, d_base)]
                .into_iter()
                .enumerate()
            {
                let at = tpl + 24 * e + 8 * i;
                table[at..at + 4].copy_from_slice(&[
                    u32_of(kind.n, "rows")?,
                    u32_of(kind.k, "columns")?,
                    u32_of(kind.row_bytes, "row bytes")?,
                    w32(ws[i].offset() as u64 - base, "offset")?,
                ]);
            }
        }
        counters.submissions.fetch_add(1, Ordering::Relaxed);

        // Inputs and the table in.
        let start = Instant::now();
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let xbuf = {
            let x = Self::host_buffer(&shared, &mut self.xs, xb, storage)?;
            x.floats_mut()[..op.xs.len()].copy_from_slice(op.xs);
            x.buffer
        };
        // The prelude's inputs: attention outputs and the residual stream.
        let pre_bufs = match &op.prelude {
            Some(p) => {
                let abuf = {
                    let a = Self::host_buffer(&shared, &mut self.att, ab, storage)?;
                    a.floats_mut()[..p.att.len()].copy_from_slice(p.att);
                    a.buffer
                };
                let hbuf = {
                    let h = Self::host_buffer(&shared, &mut self.hs, hb, storage)?;
                    h.floats_mut()[..p.hs.len()].copy_from_slice(p.hs);
                    h.buffer
                };
                let obuf = Self::scratch(&shared, &mut self.oproj, hb, storage, false)?.buffer;
                Some((abuf, hbuf, obuf))
            }
            None => None,
        };
        // Attention's inputs: the activations and the rows to copy in; q, k and v stay
        // on the device; the new keys and values come back.
        let attn_bufs = match (&op.attention, &attn) {
            (Some(a), Some(sh)) => {
                let axbuf = {
                    let x =
                        Self::host_buffer(&shared, &mut self.axs, 4 * a.xs.len() as u64, storage)?;
                    x.floats_mut()[..a.xs.len()].copy_from_slice(a.xs);
                    x.buffer
                };
                let sybuf = {
                    let y = Self::host_buffer(
                        &shared,
                        &mut self.sync,
                        4 * a.sync.len().max(1) as u64,
                        storage,
                    )?;
                    y.floats_mut()[..a.sync.len()].copy_from_slice(a.sync);
                    y.buffer
                };
                let knbuf = Self::host_buffer(
                    &shared,
                    &mut self.knew,
                    8 * (s * sh.kv_out) as u64,
                    storage,
                )?
                .buffer;
                let qkvbuf = Self::scratch(
                    &shared,
                    &mut self.qkv,
                    4 * (s * (sh.q_out + 2 * sh.kv_out)) as u64,
                    storage,
                    false,
                )?
                .buffer;
                counters
                    .activations_in
                    .fetch_add(4 * (a.xs.len() + a.sync.len()) as u64, Ordering::Relaxed);
                Some((axbuf, sybuf, knbuf, qkvbuf))
            }
            _ => None,
        };
        let tbuf = {
            let t = Self::host_buffer(&shared, &mut self.table, 4 * words as u64, storage)?;
            for (dst, &v) in t.floats_mut().iter_mut().zip(&table) {
                *dst = f32::from_bits(v);
            }
            t.buffer
        };
        let ybuf = Self::host_buffer(&shared, &mut self.ys, yb, storage)?.buffer;
        let rtbuf = Self::host_buffer(&shared, &mut self.routes, rtb, storage)?.buffer;
        let lbuf = Self::scratch(&shared, &mut self.logits, lb, storage, false)?.buffer;
        let mbuf = Self::scratch(&shared, &mut self.mid, mb, storage, false)?.buffer;
        let cbuf = Self::scratch(&shared, &mut self.contrib, cb, storage, false)?.buffer;
        counters.time(start);
        counters
            .activations_in
            .fetch_add(if pre.is_some() { ab + hb } else { xb }, Ordering::Relaxed);

        let prepared = Instant::now();
        let sets = self.fused_sets(match (pre.is_some(), attn.is_some()) {
            (_, true) => 11,
            (true, false) => 7,
            (false, false) => 5,
        })?;
        let whole = |b: vk::Buffer| (b, 0, None);
        // Router, route, gate/up (and SwiGLU, which reads binding 2), down, combine.
        self.write_set(
            sets[0],
            [
                (rbuf, r_base, Some(r_range)),
                whole(xbuf),
                whole(lbuf),
                whole(tbuf),
            ],
        );
        self.write_set(
            sets[1],
            [whole(tbuf), whole(lbuf), whole(rtbuf), whole(tbuf)],
        );
        self.write_set(
            sets[2],
            [
                (g.buffer, gu_base, Some(gu_range)),
                whole(xbuf),
                whole(mbuf),
                whole(tbuf),
            ],
        );
        self.write_set(
            sets[3],
            [
                (d.buffer, d_base, Some(d_range)),
                whole(mbuf),
                whole(cbuf),
                whole(tbuf),
            ],
        );
        self.write_set(
            sets[4],
            [whole(rtbuf), whole(cbuf), whole(ybuf), whole(tbuf)],
        );
        let layout = fused.layout;
        let n32 = |v: usize, what: &str| u32_of(v, what);
        let (rb, gb, db) = (
            u32_of(rb, "row blocks")?,
            u32_of(gb, "row blocks")?,
            u32_of(db, "row blocks")?,
        );
        let (s32, ne32, k32, sk32, gn32, rows32) = (
            n32(s, "tokens")?,
            n32(ne, "experts")?,
            n32(k, "top k")?,
            n32(sk, "entries")?,
            n32(gn, "intermediates")?,
            n32(rows, "rows")?,
        );
        let (hdr32, ent32) = (n32(hdr, "table")?, n32(ent, "table")?);
        // The prelude (Step 3c, audit §18.3): the O projection, then the residual and
        // norm, which writes the activations the router and the experts read.
        let mut steps: Vec<Step> = Vec::with_capacity(14);
        // Decode attention (audit §21): the host rows into the pool, the QKV projections,
        // the prelude (clip, Q and K norms, RoPE, the store) and attention, whose outputs
        // the O projection below reads.
        if let (
            Some(sh),
            Some(t),
            Some((axbuf, sybuf, knbuf, qkvbuf)),
            Some(ap),
            Some((abuf, _, _)),
        ) = (&attn, &at, attn_bufs, shared.attn.as_ref(), pre_bufs)
        {
            let n32 = |v: usize, what: &str| u32_of(v, what);
            if sh.sync_rows > 0 {
                self.write_set(
                    sets[7],
                    [whole(sybuf), (sh.pool, 0, None), whole(sybuf), whole(tbuf)],
                );
                steps.push((
                    ap.kv_sync,
                    sets[7],
                    [n32(sh.kv_out, "KV width")?, n32(t.sync_at, "table")?, 0, 0],
                    (n32(sh.sync_rows, "rows")?, 1),
                ));
            }
            self.write_set(
                sets[8],
                [
                    (sh.wbuf, sh.w_base, Some(sh.w_range)),
                    whole(axbuf),
                    whole(qkvbuf),
                    whole(tbuf),
                ],
            );
            for &(dt, blocks, first, count) in &t.groups {
                let b = n32(blocks, "row blocks")?;
                let c = n32(count, "matrices")?;
                steps.push((
                    fused.matmul[dt],
                    sets[8],
                    [b, c, n32(first, "table")?, 0],
                    (b * c, n32(s, "tokens")?),
                ));
            }
            self.write_set(
                sets[9],
                [whole(qkvbuf), (sh.pool, 0, None), whole(knbuf), whole(tbuf)],
            );
            steps.push((
                ap.prelude,
                sets[9],
                [n32(t.hdr, "table")?, 0, 0, 0],
                (3, n32(s, "tokens")?),
            ));
            self.write_set(
                sets[10],
                [whole(qkvbuf), (sh.pool, 0, None), whole(abuf), whole(tbuf)],
            );
            steps.push((
                ap.decode,
                sets[10],
                [n32(t.hdr, "table")?, 0, 0, 0],
                (n32(sh.heads, "heads")?, n32(s, "tokens")?),
            ));
        }
        if let (
            Some((odt, _, _, wbuf, _)),
            Some((o_base, o_range)),
            Some((abuf, hbuf, obuf)),
            Some(p),
        ) = (pre, o_bind, pre_bufs, &op.prelude)
        {
            let ob = u32_of(rows_of(cols), "row blocks")?;
            // `eps as f32`, as `math::rms_scale` converts it.
            #[allow(clippy::cast_possible_truncation)]
            let eps = p.eps as f32;
            self.write_set(
                sets[5],
                [
                    (wbuf, o_base, Some(o_range)),
                    whole(abuf),
                    whole(obuf),
                    whole(tbuf),
                ],
            );
            self.write_set(
                sets[6],
                [whole(hbuf), whole(obuf), whole(xbuf), whole(tbuf)],
            );
            steps.push((
                fused.matmul[odt],
                sets[5],
                [ob, 1, n32(o_at, "table")?, 0],
                (ob, s32),
            ));
            steps.push((
                routed.resnorm,
                sets[6],
                [
                    n32(cols, "columns")?,
                    n32(norm_at, "table")?,
                    eps.to_bits(),
                    0,
                ],
                (s32, 1),
            ));
        }
        steps.extend([
            (fused.matmul[rdt], sets[0], [rb, 1, 0, 0], (rb, s32)),
            (routed.route, sets[1], [s32, ne32, k32, hdr32], (s32, 1)),
            (
                fused.matmul[g.dtype],
                sets[2],
                [gb, 2 * sk32, ent32, 0],
                (2 * sk32 * gb, 1),
            ),
            (
                fused.swiglu,
                sets[2],
                [gn32, 0, gn32, 0],
                (gn32.div_ceil(SWIGLU_GROUP), 1),
            ),
            (
                fused.matmul[d.dtype],
                sets[3],
                [db, sk32, ent32 + 16 * sk32, 0],
                (sk32 * db, 1),
            ),
            (
                routed.combine,
                sets[4],
                [s32, k32, rows32, 0],
                ((s32 * rows32).div_ceil(256), 1),
            ),
        ]);
        let ran = self.run(|dev, cmd| {
            let between = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            let to_host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            // SAFETY: recording into a command buffer in the recording state, with
            // objects that outlive the submission (it is waited for in `run`).
            unsafe {
                for (i, (pipeline, set, words, (x, y))) in steps.iter().enumerate() {
                    if i > 0 {
                        dev.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &between,
                            &[],
                            &[],
                        );
                    }
                    let mut push = [0u8; 16];
                    for (j, w) in words.iter().enumerate() {
                        push[4 * j..4 * j + 4].copy_from_slice(&w.to_ne_bytes());
                    }
                    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, *pipeline);
                    dev.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &[*set],
                        &[],
                    );
                    dev.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                    dev.cmd_dispatch(cmd, *x, *y, 1);
                }
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &to_host,
                    &[],
                    &[],
                );
            }
        })?;
        counters.ran(prepared, &ran);

        // Outputs and routes back.
        let start = Instant::now();
        let y = self.ys.as_mut().ok_or_else(|| err("outputs", "missing"))?;
        op.ys.copy_from_slice(&y.floats_mut()[..op.ys.len()]);
        let r = self
            .routes
            .as_mut()
            .ok_or_else(|| err("routes", "missing"))?;
        let rt = r.floats_mut();
        for (id, v) in op.ids.iter_mut().zip(&rt[..sk]) {
            *id = v.to_bits();
        }
        op.weights.copy_from_slice(&rt[sk..2 * sk]);
        if let Some(p) = op.prelude.as_mut() {
            let h = self
                .hs
                .as_mut()
                .ok_or_else(|| err("residuals", "missing"))?;
            p.hs.copy_from_slice(&h.floats_mut()[..p.hs.len()]);
        }
        if let (Some(a), Some(sh)) = (op.attention.as_mut(), &attn) {
            let n = s * sh.kv_out;
            let kn = self
                .knew
                .as_mut()
                .ok_or_else(|| err("new keys", "missing"))?;
            let f = kn.floats_mut();
            a.k_new.copy_from_slice(&f[..n]);
            a.v_new.copy_from_slice(&f[n..2 * n]);
            counters
                .activations_out
                .fetch_add(8 * n as u64, Ordering::Relaxed);
        }
        counters.time(start);
        counters
            .activations_out
            .fetch_add(yb + rtb + hb, Ordering::Relaxed);
        Ok(true)
    }
}

impl Submitter {
    /// The route shader alone over `s` tokens' logits (`s × ne`), for the numerical
    /// conformance tests, which check it against [`qubivex_compute::math::route`].
    pub(crate) fn route(
        &mut self,
        logits: &[f32],
        s: usize,
        k: usize,
        renormalise: bool,
        ids: &mut [u32],
        weights: &mut [f32],
    ) -> Result<(), ComputeError> {
        let shared = Arc::clone(&self.shared);
        let (Some(fused), Some(routed)) = (shared.fused.as_ref(), shared.routed.as_ref()) else {
            return Err(err("route", "no routing pipelines"));
        };
        let ne = logits.len().checked_div(s).unwrap_or(0);
        if s == 0
            || k == 0
            || k > ne
            || ne > ROUTED_EXPERTS
            || ids.len() != s * k
            || weights.len() != s * k
        {
            return Err(ComputeError::Shape {
                op: "route",
                detail: format!("{} logits, {s} tokens, top {k}", logits.len()),
            });
        }
        let sk = s * k;
        let (hdr, tpl) = (0usize, 8usize);
        let ent = tpl + 24 * ne;
        let words = ent + 24 * sk;
        let mut table = vec![0u32; words];
        table[hdr..hdr + 6].copy_from_slice(&[
            u32::from(renormalise),
            u32_of(tpl, "table")?,
            u32_of(ent, "table")?,
            1,
            1,
            1,
        ]);
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let lbuf = {
            let x = Self::host_buffer(&shared, &mut self.xs, 4 * logits.len() as u64, storage)?;
            x.floats_mut()[..logits.len()].copy_from_slice(logits);
            x.buffer
        };
        let tbuf = {
            let t = Self::host_buffer(&shared, &mut self.table, 4 * words as u64, storage)?;
            for (dst, &v) in t.floats_mut().iter_mut().zip(&table) {
                *dst = f32::from_bits(v);
            }
            t.buffer
        };
        let rbuf = Self::host_buffer(&shared, &mut self.routes, 8 * sk as u64, storage)?.buffer;
        let sets = self.fused_sets(1)?;
        let set = sets[0];
        self.write_set(
            set,
            [
                (tbuf, 0, None),
                (lbuf, 0, None),
                (rbuf, 0, None),
                (tbuf, 0, None),
            ],
        );
        let (layout, pipeline) = (fused.layout, routed.route);
        let words4 = [
            u32_of(s, "tokens")?,
            u32_of(ne, "experts")?,
            u32_of(k, "top k")?,
            u32_of(hdr, "table")?,
        ];
        let groups = u32_of(s, "groups")?;
        self.run(|d, cmd| {
            let mut push = [0u8; 16];
            for (i, w) in words4.iter().enumerate() {
                push[4 * i..4 * i + 4].copy_from_slice(&w.to_ne_bytes());
            }
            let to_host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            // SAFETY: as in `gated_ffn_many`.
            unsafe {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                d.cmd_dispatch(cmd, groups, 1, 1);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &to_host,
                    &[],
                    &[],
                );
            }
        })?;
        let r = self
            .routes
            .as_mut()
            .ok_or_else(|| err("routes", "missing"))?;
        let rt = r.floats_mut();
        for (id, v) in ids.iter_mut().zip(&rt[..sk]) {
            *id = v.to_bits();
        }
        weights.copy_from_slice(&rt[sk..2 * sk]);
        Ok(())
    }
}

impl Submitter {
    /// The RMSNorm scale shader alone: `y[i] = 1 / sqrt(a[i] / n + eps)` (`rms.glsl`),
    /// for the conformance tests, which check every f32 input against the CPU.
    pub(crate) fn rms_scale(
        &mut self,
        a: &[f32],
        n: f32,
        eps: f32,
        y: &mut [f32],
    ) -> Result<(), ComputeError> {
        let shared = Arc::clone(&self.shared);
        let (Some(fused), Some(routed)) = (shared.fused.as_ref(), shared.routed.as_ref()) else {
            return Err(err("rms scale", "no routing pipelines"));
        };
        if a.len() != y.len() || a.is_empty() {
            return Err(ComputeError::Shape {
                op: "rms scale",
                detail: format!("{} inputs, {} outputs", a.len(), y.len()),
            });
        }
        let bytes = 4 * a.len() as u64;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let abuf = {
            let x = Self::host_buffer(&shared, &mut self.xs, bytes, storage)?;
            x.floats_mut()[..a.len()].copy_from_slice(a);
            x.buffer
        };
        let ybuf = Self::host_buffer(&shared, &mut self.ys, bytes, storage)?.buffer;
        let sets = self.fused_sets(1)?;
        let set = sets[0];
        self.write_set(
            set,
            [
                (abuf, 0, None),
                (abuf, 0, None),
                (ybuf, 0, None),
                (abuf, 0, None),
            ],
        );
        let (layout, pipeline) = (fused.layout, routed.rms_scale);
        let count = u32_of(a.len(), "inputs")?;
        let words4 = [count, n.to_bits(), eps.to_bits(), 0];
        let groups = count.div_ceil(256);
        self.run(|d, cmd| {
            let mut push = [0u8; 16];
            for (i, w) in words4.iter().enumerate() {
                push[4 * i..4 * i + 4].copy_from_slice(&w.to_ne_bytes());
            }
            let to_host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            // SAFETY: as in `gated_ffn_many`.
            unsafe {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                d.cmd_dispatch(cmd, groups, 1, 1);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &to_host,
                    &[],
                    &[],
                );
            }
        })?;
        let out = self.ys.as_mut().ok_or_else(|| err("outputs", "missing"))?;
        y.copy_from_slice(&out.floats_mut()[..a.len()]);
        Ok(())
    }
}

/// Locks the submitter, recovering from a poisoned lock (a panic while recording
/// leaves no submission in flight: `run` waits before it can return or unwind).
pub(crate) fn lock(m: &Mutex<Submitter>) -> MutexGuard<'_, Submitter> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// An experimental compute pipeline on the fused layout (4 storage bindings, 16 bytes
/// of push constants), for the investigation harness (audit §19). Destroyed on drop.
#[cfg(feature = "spike")]
pub struct SpikeKernel {
    shared: Arc<Shared>,
    pipeline: vk::Pipeline,
}

#[cfg(feature = "spike")]
impl Drop for SpikeKernel {
    fn drop(&mut self) {
        // SAFETY: no submission using it is in flight (`spike_run` waits).
        unsafe { self.shared.device.destroy_pipeline(self.pipeline, None) };
    }
}

#[cfg(feature = "spike")]
impl std::fmt::Debug for SpikeKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpikeKernel").finish_non_exhaustive()
    }
}

/// One step of a spike run: a kernel, its push constants and its grid.
#[cfg(feature = "spike")]
pub type SpikeStep<'a> = (&'a SpikeKernel, [u32; 4], [u32; 3]);

/// A spike run's timing: the host's wait on the fence, and the GPU time between the
/// command buffer's timestamps when timed.
#[cfg(feature = "spike")]
#[derive(Clone, Copy, Debug)]
pub struct SpikeTiming {
    /// Nanoseconds blocked on the fence.
    pub wait_ns: u64,
    /// GPU nanoseconds, when the device was opened with timestamps.
    pub gpu_ns: Option<u64>,
}

#[cfg(feature = "spike")]
impl Submitter {
    /// Builds `spv` as a pipeline on the fused layout.
    pub(crate) fn spike_kernel(&self, spv: &[u8]) -> Result<SpikeKernel, ComputeError> {
        let shared = Arc::clone(&self.shared);
        let fused = shared
            .fused
            .as_ref()
            .ok_or_else(|| err("spike", "no fused layout"))?;
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))
            .map_err(|e| err("bad SPIR-V", e))?;
        let d = &shared.device;
        // SAFETY: the create-infos live until their calls return; the module is
        // destroyed after pipeline creation.
        unsafe {
            let module = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .map_err(|e| err("shader module", e))?;
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(c"main"),
                )
                .layout(fused.layout)];
            let made = d.create_compute_pipelines(vk::PipelineCache::null(), &info, None);
            d.destroy_shader_module(module, None);
            let pipeline = made.map_err(|(_, e)| err("spike pipeline", e))?[0];
            Ok(SpikeKernel {
                shared: Arc::clone(&shared),
                pipeline,
            })
        }
    }

    /// Copies `bufs` into four host-visible buffers (bindings 0 to 3), records `steps`
    /// `repeat` times with a barrier between every two dispatches, submits once, waits,
    /// and copies every buffer back.
    pub(crate) fn spike_run(
        &mut self,
        steps: &[SpikeStep<'_>],
        bufs: [&mut [f32]; 4],
        repeat: u32,
    ) -> Result<SpikeTiming, ComputeError> {
        let shared = Arc::clone(&self.shared);
        let fused = shared
            .fused
            .as_ref()
            .ok_or_else(|| err("spike", "no fused layout"))?;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut allocs = Vec::with_capacity(4);
        for b in &bufs {
            let mut a = Allocation::new(&shared, (4 * b.len() as u64).max(16), storage, true)?;
            a.floats_mut()[..b.len()].copy_from_slice(b);
            allocs.push(a);
        }
        let sets = self.fused_sets(1)?;
        let whole = |a: &Allocation| (a.buffer, 0, None);
        self.write_set(
            sets[0],
            [
                whole(&allocs[0]),
                whole(&allocs[1]),
                whole(&allocs[2]),
                whole(&allocs[3]),
            ],
        );
        let (layout, set) = (fused.layout, sets[0]);
        let ran = self.run(|dev, cmd| {
            let between = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            let to_host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            // SAFETY: recording into a command buffer in the recording state, with
            // objects that outlive the submission (`run` waits for it).
            unsafe {
                dev.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                let mut first = true;
                for _ in 0..repeat {
                    for (k, words, [x, y, z]) in steps {
                        if !first {
                            dev.cmd_pipeline_barrier(
                                cmd,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::DependencyFlags::empty(),
                                &between,
                                &[],
                                &[],
                            );
                        }
                        first = false;
                        let mut push = [0u8; 16];
                        for (j, w) in words.iter().enumerate() {
                            push[4 * j..4 * j + 4].copy_from_slice(&w.to_ne_bytes());
                        }
                        dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
                        dev.cmd_push_constants(
                            cmd,
                            layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            &push,
                        );
                        dev.cmd_dispatch(cmd, *x, *y, *z);
                    }
                }
                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &to_host,
                    &[],
                    &[],
                );
            }
        })?;
        for (b, a) in bufs.into_iter().zip(allocs.iter_mut()) {
            let n = b.len();
            b.copy_from_slice(&a.floats_mut()[..n]);
        }
        Ok(SpikeTiming {
            wait_ns: ran.wait_ns,
            gpu_ns: ran.gpu.map(|(a, b)| b - a),
        })
    }
}

#[cfg(feature = "spike")]
fn lock_hook(m: &Mutex<SpikeHook>) -> MutexGuard<'_, SpikeHook> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Experimental steps (feature `spike`, audit §20) over their own four host-visible
/// buffers, recorded at the head of the next submission (whatever records it), then a
/// barrier, so that submission's own dispatches follow them in one command buffer.
#[cfg(feature = "spike")]
pub struct SpikePrefix {
    shared: Arc<Shared>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    allocs: Vec<Allocation>,
    steps: Vec<(vk::Pipeline, [u32; 4], [u32; 3])>,
}

#[cfg(feature = "spike")]
impl Drop for SpikePrefix {
    fn drop(&mut self) {
        // SAFETY: a prefix is only recorded into a submission that is waited for before
        // it is handed back, so its set is not in use.
        unsafe { self.shared.device.destroy_descriptor_pool(self.pool, None) };
    }
}

#[cfg(feature = "spike")]
impl std::fmt::Debug for SpikePrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpikePrefix")
            .field("steps", &self.steps.len())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "spike")]
impl SpikePrefix {
    /// Binding `i`'s contents now (after the submission that recorded the prefix).
    pub fn read(&mut self, i: usize, len: usize) -> Vec<f32> {
        self.allocs[i].floats_mut()[..len].to_vec()
    }

    /// Records the steps, a barrier between every two, and one after the last.
    ///
    /// # Safety
    /// `cmd` is recording, and is waited for before the prefix is dropped.
    unsafe fn record(&self, d: &ash::Device, cmd: vk::CommandBuffer) {
        let Some(fused) = self.shared.fused.as_ref() else {
            return;
        };
        let between = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        // SAFETY: as the caller promises; the set and pipelines outlive the submission.
        unsafe {
            for (pipeline, words, [x, y, z]) in &self.steps {
                let mut push = [0u8; 16];
                for (j, w) in words.iter().enumerate() {
                    push[4 * j..4 * j + 4].copy_from_slice(&w.to_ne_bytes());
                }
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, *pipeline);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    fused.layout,
                    0,
                    &[self.set],
                    &[],
                );
                d.cmd_push_constants(cmd, fused.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                d.cmd_dispatch(cmd, *x, *y, *z);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &between,
                    &[],
                    &[],
                );
            }
        }
    }
}

#[cfg(feature = "spike")]
impl Submitter {
    /// A [`SpikePrefix`] of `steps` over copies of `bufs`.
    pub(crate) fn spike_prefix(
        &self,
        steps: &[SpikeStep<'_>],
        bufs: [&[f32]; 4],
    ) -> Result<SpikePrefix, ComputeError> {
        let shared = Arc::clone(&self.shared);
        if shared.fused.is_none() {
            return Err(err("spike", "no fused layout"));
        }
        let mut allocs = Vec::with_capacity(4);
        for b in bufs {
            let mut a = Allocation::new(
                &shared,
                (4 * b.len() as u64).max(16),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                true,
            )?;
            a.floats_mut()[..b.len()].copy_from_slice(b);
            allocs.push(a);
        }
        let d = &shared.device;
        let pool = descriptor_pool_of(d, 1, 4)?;
        let layouts = [shared
            .fused
            .as_ref()
            .map_or(vk::DescriptorSetLayout::null(), |f| f.set_layout)];
        // SAFETY: a new pool with room for one set of this layout.
        let set = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        };
        let set = match set {
            Ok(v) => v[0],
            Err(e) => {
                // SAFETY: nothing was allocated from it.
                unsafe { d.destroy_descriptor_pool(pool, None) };
                return Err(err("descriptor sets", e));
            }
        };
        let whole = |a: &Allocation| (a.buffer, 0, None);
        self.write_set(
            set,
            [
                whole(&allocs[0]),
                whole(&allocs[1]),
                whole(&allocs[2]),
                whole(&allocs[3]),
            ],
        );
        Ok(SpikePrefix {
            shared: Arc::clone(&shared),
            pool,
            set,
            allocs,
            steps: steps.iter().map(|(k, w, g)| (k.pipeline, *w, *g)).collect(),
        })
    }

    /// Arms `prefix` for the next submission.
    pub(crate) fn spike_arm(&self, prefix: SpikePrefix) {
        let mut hook = lock_hook(&self.shared.spike);
        hook.pending = Some(prefix);
        hook.done = None;
    }

    /// The prefix the last armed submission recorded, if it has run.
    pub(crate) fn spike_done(&self) -> Option<SpikePrefix> {
        lock_hook(&self.shared.spike).done.take()
    }

    /// The last submission's timing.
    pub(crate) fn spike_last(&self) -> Option<SpikeTiming> {
        lock_hook(&self.shared.spike).last
    }
}

#[cfg(test)]
mod tests {
    use super::Busy;

    /// The union against a direct computation, with intervals arriving out of order
    /// (up to a few late, as concurrent submissions finish) across several folds.
    #[test]
    fn busy_is_the_union_of_intervals() {
        let mut want = Vec::new();
        for i in 0u64..500 {
            // Overlapping pairs, gaps every fifth interval.
            let start = i * 100 + if i % 5 == 0 { 30 } else { 0 };
            want.push((start, start + 150));
        }
        let mut busy = Busy::default();
        for chunk in want.chunks(4) {
            for &iv in chunk.iter().rev() {
                busy.add(iv);
            }
        }
        let mut sorted = want.clone();
        sorted.sort_unstable();
        let (mut frontier, mut total) = (0, 0);
        for iv in sorted {
            Busy::fold(&mut frontier, &mut total, iv);
        }
        assert_eq!(busy.total(), total);
        // A simple check of the arithmetic: [0, 10) and [5, 20) and [30, 40) cover 30.
        let mut b = Busy::default();
        for iv in [(30, 40), (0, 10), (5, 20)] {
            b.add(iv);
        }
        assert_eq!(b.total(), 30);
    }
}
