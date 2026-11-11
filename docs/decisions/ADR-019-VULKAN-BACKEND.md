# ADR-019: The Vulkan backend, a discrete device with the CPU's bits

- Status: **Accepted (2026-11-11)** by owner decision
- Date: 2026-11-11
- Related: ADR-007 (backends), ADR-010 (lossless by default), ADR-011 (device execution
  and parity), ADR-018 (discrete device memory).

## Context
ADR-018 built the `DiscreteDevice` interface, VRAM planning, the device expert cache and
its accounting on a simulated GPU, so that a real one would only need to implement the
trait. ADR-007 named Vulkan as the last backend, through `ash` and precompiled SPIR-V.
CUDA is still blocked on NVIDIA hardware (B2), but Vulkan can be run on this machine:
MoltenVK translates it to Metal on the M3 Pro's GPU, and Mesa's lavapipe runs it on the
CPU in a Linux container. ADR-011 §1 left the Vulkan parity contract to be set once
measured.

## Decision
1. **A new crate, `qubivex-backend-vulkan`**, with `VulkanDevice` implementing
   `DiscreteDevice`. Its one new dependency is `ash` 0.38 (MIT or Apache-2.0), with the
   `loaded` feature: the Vulkan loader is opened at run time, so the crate builds on every
   platform with no Vulkan SDK, and on a machine without a loader or device,
   `VulkanDevice::new` returns an error. Like Metal, it is an unconditional dependency of
   the runtime; no Cargo feature is needed. The loader is found by its platform name, then
   Homebrew's paths on macOS (`dlopen` does not search them), or
   `QUBIVEX_VULKAN_LIBRARY`.
2. **Device and memory.** It needs Vulkan 1.1 and nothing optional: no 8- or 16-bit
   storage, no subgroup operations. It enables portability enumeration and the
   portability subset where the driver lists them (MoltenVK). It picks the first discrete
   GPU, else integrated, else any device with a compute queue; `--vulkan-device N` picks by
   the index `qubivex system` shows. Weights are copied through a 64 MiB host staging
   buffer into device-local memory, one allocation per upload. Capacity is the largest
   device-local heap, or `--vulkan-vram`. The device counts its own allocations against
   it and refuses past it with `OutOfDeviceMemory`, as the simulated device does.
   Allocations are padded to 4 bytes.
3. **Submission.** Each `matmul_many` is one command buffer on one queue, behind a mutex.
   It copies every op's activations into one host-visible buffer, has one descriptor set
   and one dispatch per op, and waits on a fence. Outputs are read back from a second
   host-visible buffer. Host buffers and the descriptor pool grow by doubling and are
   reused. Every submission completes before the call returns, so a buffer can only be
   freed when nothing reads it (ADR-011 §3).
4. **One shader, `shaders/matmul.comp`,** compiled by `shaders/build.sh` (glslc) and
   checked in as SPIR-V. The dtype is a specialization constant, so one module gives six
   pipelines: f32, f16, bf16, Q8_0, Q4_K and Q6_K, the CPU's set. glslang marks `precise`
   arithmetic `NoContraction` but not the `Fma` instructions, so the script adds the
   decoration to each one (SPIR-V lets a driver split an undecorated `Fma`) and validates
   the result. A unit test reruns the script when glslc and SPIRV-Tools are installed,
   and fails if the checked-in file is stale.
5. **Parity contract: the CPU's bits, as for Metal (ADR-011 §1).**
   - Each weight is dequantised with the CPU's operations in its order. f16 is converted
     by integer bit manipulation, so it is exact, subnormals included.
   - The dot product has the CPU's 64-lane shape. Invocation `lane` of a 32-wide group
     holds lanes `lane` and `lane + 32`. The halving tree runs through shared memory with
     barriers, not subgroup shuffles, so the result does not depend on the driver's
     subgroup size.
   - Every intermediate is `precise`, so the compiler may not fuse or reassociate it.
     Only the explicit `fma` is fused.
   - On MoltenVK, fast math is also turned off at instance creation, through
     `VK_EXT_layer_settings` (`MVK_CONFIG_FAST_MATH_ENABLED = 0`). This setting overrides
     the environment variable.
   - The one exception is Metal's: Apple GPUs flush f32 subnormals.
   - **A device that splits `fma` is refused.** `VulkanDevice::new` runs one probe
     output, `fma(1 + 2^-12, 1 + 2^-12, -1)` in a lane, through the real pipeline. A
     single rounding gives 2^-11 + 2^-24; a multiply then an add gives 2^-11. A device
     that gives the second fails to open, with that reason. It is never used with
     different bits. The parity tests are the full check; the probe stops such a device
     from passing silently at run time.
6. **Runtime.** `BackendKind::Vulkan` (`--backend vulkan`) runs exactly as
   `BackendKind::Sim` does under ADR-018 §4. The device is opened before planning,
   because the plan's `VramRequest` needs its capacity. Placement, the ledger's
   `Vram(0)` tier, `qubivex run`'s device line and the `/metrics` series are the same.
   The copy time in `TransferStats` is measured wall time, not modelled.

## What was measured
On the M3 Pro through MoltenVK 1.4.2:
- **Kernels.** All six dtypes match the CPU's bits:
  - shapes with tails (k = 1 to 2048);
  - 1, 9 and 33 rows;
  - batches of 1, 3 and 8 activation rows;
  - all six dtypes at unaligned offsets in one buffer, in one submission;
  - 100 ops in one submission.
- **Positive control.** Replacing the first lane's `fma` with a separate multiply and add
  fails the tests across dtypes. With the layer setting removed and fast math forced on
  (MoltenVK logs "MathMode Fast"), they still pass. So `precise` alone holds the bits,
  and the setting is a second guard.
- **Fixtures.** On all four fixtures, the prompt logits and 12 greedy tokens equal the
  CPU's. The ledger's VRAM tier equals the device's memory. A one-expert device cache
  evicts with the same bits, and 4 concurrent requests match.
- **Real model.** OLMoE-1B-7B-0125-Instruct Q4_K_M with a 64-expert device cache: the CPU's
  prompt logits bit for bit, and Colibri's 12 greedy tokens.
- **Speed.** `qubivex run`, 48 tokens, `--ram-budget 3GiB --vulkan-vram 1500MiB`: the
  same text as the CPU. Decode was 5.7 tok/s against 31.6 on the CPU and 28.1 on the
  simulated device. The device cache churned as ADR-018 found: 4186 copies and 15.2 GiB
  copied, 1.46 s of it copying. 15242 submissions each waited for the GPU.

On Mesa's lavapipe (llvmpipe) in a Linux container, `linux/arm64`, Mesa 22.3.6 (Debian
bookworm) and Mesa 25.0.7 (trixie):
- **Correct except one thing.** Memory, capacity, guards, refusals, counters, the
  fixtures with 4 concurrent requests, and the OLMoE skip path all pass.
- **The one thing: lavapipe computes every `fma` as a multiply and an add.** That holds
  even with `NoContraction` on each `Fma`. A diagnostic compared its outputs with CPU
  references computed four ways. The "every `fma` unfused" reference matched all 99 f32
  outputs and all 16 Q4_K outputs; the CPU's fused one did not. Dequantisation and the
  reduction order were already exact. So the differences are in the last bits (for
  example -90.48816 against -90.48822), and on Q4_K most outputs differ (789 of 1024).
- **It is refused.** With the probe, lavapipe fails to open: "splits fma into a
  multiply and an add (4.8828125e-4 where one rounding gives 4.8834085e-4)". The
  tests skip there, as with no device.

## Consequences
- The discrete-device path now runs on a real driver, and the runtime and planner did
  not change beyond opening the device before planning. CUDA can follow the same shape.
- `--backend vulkan` is correct but slow, about 5.5 times slower than the CPU at decode
  on this machine. Like the first Metal slice, it is not a way to run faster yet. The
  slow parts are known:
  - weights are dequantised per element, with scales recomputed for each one;
  - each activation row reads the weights again;
  - each submission waits synchronously, about 15 submissions per token;
  - activations go through host memory;
  - a device cache below the working set churns.
- On an integrated GPU, device-local memory is host memory. MoltenVK reports all 18 GiB
  of RAM as the heap, so by default the RAM and VRAM budgets overlap. Cap the device with
  `--vulkan-vram` there. Metal remains the backend for Apple GPUs.
- The checked-in SPIR-V is a build artifact in the repository. The staleness test guards
  it only where glslc and SPIRV-Tools are installed (locally, not in CI).
- Bit-identical Vulkan needs a driver whose `fma` is one rounding. Hardware drivers on
  GPUs with an FMA unit are expected to keep it fused; MoltenVK does. That is unmeasured
  on any discrete GPU. A driver that does not is refused, not trusted.

## Not decided here
- Faster kernels with the same bits: block-wise K-quant dequantisation, token tiles and
  several rows per group, as Metal's slices 3 to 5 did.
- Whole layers and activations on the device, asynchronous copies, overlapped
  submissions and more than one queue.
- Whether integrated GPUs should default to a smaller capacity, or be planned as RAM as
  ADR-011 §4 plans Metal.
- Measuring a discrete Vulkan GPU.
- Lavapipe as a parity target, for CI. Two ways to get there:
  - an exact `fma` emulated in the shader for devices that split it, as a separate
    pipeline: products are exact in f64, and a sum rounded to odd then rounded to f32
    is correctly rounded. Lavapipe has `shaderFloat64`; Apple GPUs do not, and do not
    need it;
  - or a documented tolerance for such devices, as a lossy opt-in under ADR-010.
