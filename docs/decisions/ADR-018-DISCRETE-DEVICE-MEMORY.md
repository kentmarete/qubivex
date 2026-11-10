# ADR-018: Discrete device memory, and a simulated device to build it on

- Status: **Proposed**
- Date: 2026-11-11
- Related: ADR-005 (memory planner), ADR-006 (expert store), ADR-007 (backends),
  ADR-008 (expert cache), ADR-011 (device execution and parity).

## Context
CUDA and Vulkan GPUs have memory of their own. Metal does not: it wraps host memory in
place, and the planner counts its buffers against the RAM budget (ADR-011 §4). Nothing
built so far handles a separate device memory. The ledger has a `Vram(n)` tier with no
charges, `MemoryPlan::vram` has always been empty, nothing copies weights to a device,
and no cache decides which experts stay there.

The development machine has no NVIDIA GPU (blocker B2) and no Vulkan driver of its own.
That code still has to exist before either backend can be useful, and it can be written
and tested without one if something stands in for the device.

## Decision
1. **A `DiscreteDevice` trait** in `qubivex-compute::discrete`. A device has an
   `upload` that copies host bytes into a `DeviceBuffer`, and a `matmul_many` over
   `DeviceWeights` (a matrix at an offset in a buffer) as one submission. It also
   reports `usage` (capacity, used, peak, refused) and `transfers` (uploads, bytes,
   activation bytes each way, submissions, copy time). `DeviceBuffer` is an `Arc` of
   backend state, as ADR-011 §4 asked. An operation in flight holds a clone of every
   buffer it reads, and memory is freed when the last clone drops (ADR-011 §3). An
   upload carries an optional guard that lives exactly as long as the allocation; the
   runtime passes the ledger charge there. Activations stay in host memory at this
   interface: each submission copies its inputs in and its outputs back, and the device
   counts both. CUDA and Vulkan implement this trait. `ComputeBackend` is unchanged.
2. **`SimDevice`** (`qubivex-backend-cpu::sim`) is a simulated discrete GPU:
   - every upload is a real copy into an allocation only that device reads;
   - it has a hard capacity, and an upload past it fails with
     `ComputeError::OutOfDeviceMemory`;
   - a buffer from another device is refused;
   - copy time is modelled from a bandwidth and a per-copy latency (default PCIe 4.0:
     25 GB/s and 10 µs) and is only counted unless `delay` is set, in which case the
     device sleeps for it.

   It computes with the CPU kernels on its own copy of the bytes, so its bits are the
   CPU's by construction. It does not simulate kernel speed, asynchronous submission or
   driver alignment rules.
3. **The planner plans VRAM** when given a `VramRequest` (device, available bytes, an
   optional explicit budget). The budget is the profile's fraction of the available
   memory, or the explicit one. From it the planner takes the dense weights' copy and
   a 256 MiB device reserve (`VRAM_ACTIVATION_RESERVE`). The rest is the device expert
   cache, capped at the RAM expert cache at full context, because the device cache only
   holds copies of experts the RAM cache holds. A budget below dense + reserve + one
   expert per layer is `DoesNotFit` on that device. Placements gain `Vram(d)` and
   `VramCache(d)`. RAM and VRAM budgets stay separate and are never summed (ADR-005).
4. **The runtime runs on a discrete device** with `BackendKind::Sim`
   (`--backend sim`):
   - **Dense weights.** The dense arena is copied once at load and charged to
     `Tier::Vram(0)` as dense.
   - **The expert cache.** The first time a forward pass uses an expert, its slab is
     copied to the device and charged as expert cache. Copies are evicted least
     recently used when the cache is over budget.
   - **Inclusive with the RAM cache.** Each copy is tied to its RAM slab through the
     slab's attachment, so when the RAM cache reuses a slab, the device copy goes too.
   - **Pins.** A forward pass holds an expert's buffer while it computes the expert,
     and eviction skips held copies. When every copy is held, a new copy goes over the
     budget rather than failing; the device's capacity is the hard limit.
   - **Lookup.** Every matmul finds its weights on the device by host address.
   - **What runs on the device.** Every matmul is a device submission. Norms,
     attention, routing and sampling stay on the host.
   - **No silent fallback.** A matmul whose weights have no device copy is an error,
     not a CPU fallback. On the simulated device, this is what catches a missing upload.
5. **Parity.** A discrete device must give the CPU's bits under ADR-011 §1. On the sim
   that holds by construction. The tests check it anyway, because they also prove that
   every weight takes the device path:
   - on all four fixtures;
   - with a device cache of one expert;
   - with a RAM cache small enough that slabs are reused;
   - with four concurrent requests;
   - on the real OLMoE Q4_K_M model, env-gated.
6. **Reporting.** The ledger's `Vram(0)` tier equals the device's allocated memory.
   `qubivex run` prints the device's memory, its expert cache and its copies.
   `/metrics` adds `qubivex_vram_*` and `qubivex_device_*` series for a model on a
   discrete device. `qubivex plan --vram SIZE [--vram-budget SIZE]` plans a device
   without loading anything.

## What was measured
OLMoE Q4_K_M on the M3 Pro, `--ram-budget 3GiB --backend sim --sim-vram 1500MiB`, 48
greedy tokens:
- **Output.** The same text as the CPU run.
- **Expert traffic.** The device expert cache got 767 MiB, and the RAM cache 2.15 GiB.
  It made 4572 copies against 2065 hits, with 4245 evictions, and 116 copies were
  dropped because the RAM cache dropped their expert.
- **Copy volume.** 16.5 GiB of weights were copied in all, which models as 0.93 s at
  PCIe 4.0 speeds.

So a device cache smaller than the routed working set does not cut host-to-device
traffic much. It mostly churns. That is the case the planner should steer away from,
and the number a real device has to be measured on.

## Consequences
- The VRAM tier, VRAM planning, device copies, the device expert cache and its
  accounting are built and tested before any real discrete GPU. A CUDA or Vulkan
  backend implements `DiscreteDevice`, and the runtime and planner stay as they are.
- `--backend sim` is a test and planning tool, not a way to run faster. Its speed says
  nothing about a real GPU, and its copies cost real host memory.
- The host keeps its copy of the dense weights, because embedding, norms and attention
  run there. Moving whole layers to the device (as Metal amendment 2 did) is a later
  step. It needs device activations, which this interface does not have yet.

## Not decided here
- The parity contract for CUDA and Vulkan. ADR-011 §1 sets the target. Each backend
  gets its own amendment once it is measured.
- Pinned host memory (`Tier::PinnedHost`) for faster copies, asynchronous copies that
  overlap compute, and prefetching into the device cache.
- More than one device.
- Whether the host copy of the dense weights can be dropped once whole layers run on
  the device.
