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
use qubivex_compute::discrete::{DeviceBuffer, DeviceMatMul, DeviceUsage, Guard, TransferStats};
use qubivex_core::{DType, GgmlType};

use crate::{BACKEND, VulkanConfig};

/// The matmul shader (`shaders/matmul.comp`, compiled by `shaders/build.sh`, which also
/// marks every `fma` NoContraction).
pub(crate) const MATMUL_SPV: &[u8] = include_bytes!("matmul.spv");

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
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: every allocation and the submitter hold an `Arc<Shared>`, so this
        // runs after all of them are destroyed, and no work is in flight (every
        // submission waits for its fence). Objects are destroyed child first.
        unsafe {
            let _ = self.device.device_wait_idle();
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
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
        }
    }
}

fn descriptor_pool(d: &ash::Device, sets: u32) -> Result<vk::DescriptorPool, ComputeError> {
    let sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3 * sets)];
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
            let queries = if timestamps && shared.timestamp_mask != 0 {
                match d.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                ) {
                    Ok(q) => Some(q),
                    Err(e) => {
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
        if slot.as_ref().is_none_or(|a| a.size < bytes) {
            *slot = None;
            let size = bytes.max(4096).next_power_of_two();
            *slot = Some(Allocation::new(shared, size, usage, true)?);
        }
        slot.as_mut().ok_or_else(|| err("host buffer", "missing"))
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
            Ok(Ran {
                submitted,
                wait_ns,
                gpu,
            })
        }
    }

    /// Copies `host` into a new device-local buffer.
    pub(crate) fn upload(
        &mut self,
        host: &[u8],
        counters: &Arc<Counters>,
        capacity: u64,
        guard: Option<Guard>,
    ) -> Result<DeviceBuffer, ComputeError> {
        let len = host.len() as u64;
        // The shader reads 32-bit words: pad to a multiple of 4 (at least one word).
        let size = len.max(1).next_multiple_of(4);
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
        let start = Instant::now();
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
            let (src, dbuf) = (stage.buffer, dst.buffer);
            let _ = self.run(|d, cmd| {
                let region = [vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(done)
                    .size(n)];
                // SAFETY: both buffers are live and large enough for the region.
                unsafe { d.cmd_copy_buffer(cmd, src, dbuf, &region) };
            })?;
            done += n;
        }
        counters.time(start);
        counters.uploads.fetch_add(1, Ordering::Relaxed);
        counters.upload_bytes.fetch_add(len, Ordering::Relaxed);
        Ok(DeviceBuffer::new(
            host.len(),
            WeightBuffer { alloc: dst },
            guard,
        ))
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

/// Locks the submitter, recovering from a poisoned lock (a panic while recording
/// leaves no submission in flight: `run` waits before it can return or unwind).
pub(crate) fn lock(m: &Mutex<Submitter>) -> MutexGuard<'_, Submitter> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
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
