# ADR-019: The Vulkan backend, a discrete device with the CPU's bits

- Status: **Accepted (2026-11-11)** by owner decision; amendment 1 (2026-11-11, kernel
  speed) accepted 2026-11-11 by owner decision, after its parity and regression checks
  were rerun on `main` at 9721f58 (§ Amendment 1, "Acceptance"). Amendment 2 (the shared
  GPU pass) is to be written separately. Amendment 3 (2026-11-11, the fused expert block,
  Step 3a) records the implementation the owner accepted on 2026-11-11, and the parity
  contract's wording for NaNs.
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
- Faster kernels with the same bits: block-wise K-quant dequantisation is done
  (amendment 1); token tiles remain, and several rows per group was measured slower.
- Whole layers and activations on the device, asynchronous copies and more than one
  queue (overlapped submissions: amendment 1).
- Whether integrated GPUs should default to a smaller capacity, or be planned as RAM as
  ADR-011 §4 plans Metal.
- Measuring a discrete Vulkan GPU.
- Lavapipe as a parity target, for CI. Two ways to get there:
  - an exact `fma` emulated in the shader for devices that split it, as a separate
    pipeline: products are exact in f64, and a sum rounded to odd then rounded to f32
    is correctly rounded. Lavapipe has `shaderFloat64`; Apple GPUs do not, and do not
    need it;
  - or a documented tolerance for such devices, as a lossy opt-in under ADR-010.

## Amendment 1 (2026-11-11): kernel speed with the same bits

### What changed
- **Block-wise K-quants.** The shader decodes Q4_K, Q6_K and Q8_0 a block at a time:
  the scales once per block, and each quant byte once for every weight it holds (in
  Q4_K, lanes `l` and `l + 32` of a chunk are the low and high nibbles of one byte). A
  Q4_K block's 16-byte header is read as four words. Each lane gets the same weights,
  from the same operations, in the same order as before; the per-element `weight`
  function remains for f32, f16, bf16 and tails.
- **How contraction is forbidden (replaces "every intermediate is `precise`" in §5).**
  SPIRV-Cross, inside MoltenVK, turns each `NoContraction` operation into a call to a
  `[[clang::optnone]]` helper (a multiply becomes `fma(l, r, 0)` in an unoptimised
  function), and `precise` marked every multiply. The shader now uses no `precise`.
  `shaders/build.sh` decorates every `OpFAdd`, `OpFSub` and `Fma` with
  `NoContraction`: that alone forbids fusing a multiply into the add or subtract that
  consumes it. Multiplies compile to plain `*`.
- **Q4_K's subtract is an `fma`.** `y = t - m` is written `fma(t, 1, -m)`: `t · 1` is
  exact, so it is the same single rounding, and an `fma` compiles to one instruction
  where a decorated subtract became a helper call. `t` passes through an integer OR
  with `n >> 31` (zero, since n < 2^31, but not provably so), so the multiply producing
  it is rounded on its own and cannot be folded into the `fma` by any compiler.
- **Concurrent submissions.** The device keeps 8 submitters, each with its own command
  buffer, fence, descriptor pool and host buffers. A caller takes a free one, so a
  layer's experts, computed on several threads, record, submit and wait at the same
  time. The queue is locked only around `vkQueueSubmit`. Every submission still
  completes before its call returns (§3 unchanged).

### Measured (M3 Pro, MoltenVK 1.4.2, mains power)
- **Bits.** All parity tests, the fixtures and the OLMoE test (the CPU's prompt logits,
  Colibri's tokens) pass after each step; `bench` tokens match the starting record.
- **Positive controls.**
  - Fast math forced on (layer setting removed, `MVK_CONFIG_FAST_MATH_ENABLED=1`,
    "MathMode Fast" logged): the tests pass. The decorations hold the bits, as
    `precise` did; turning fast math off stays a second guard.
  - Fast math on and only `Fma` decorated: 3 tests fail (254 of 1024 outputs differ in
    one). So the add and subtract decorations are what holds the bits there.
  - With fast math off, neither the add/subtract decorations nor the integer OR are
    needed on Metal: both are guards for other compilers, not this one.
- **End to end**, OLMoE-1B-7B-0125-Instruct Q4_K_M, 10-token prompt, 64 generated, 5
  threads (records `docs/performance/baselines/2026-11-11-m3pro-vulkan-*.json`):

  | Step | Decode ms/token | tok/s | Prefill tok/s |
  |---|---|---|---|
  | Start | 153.3 | 6.52 | 8.2 |
  | Block-wise decoding | 100.3 | 9.97 | 16.7 |
  | 8 submitters | 57.9 | 17.26 | 26.8 |
  | Q4_K header as four words | 51.5 | 19.42 | 31.3 |
  | No `precise`; adds and subtracts decorated | 41.9 | 23.89 | 42.7 |
  | Q4_K subtract as `fma` | 36.6 | 27.36 | 48.9 |

  The CPU in the same session: 23.0 ms (43.50 tok/s), prefill 77.0 tok/s.
- **Where the time goes now.** About 297 submissions per token, not the 15 that
  Consequences gives: each expert is two. Per submission the GPU is busy about 74 µs,
  `vkQueueSubmit` takes about 34 µs on the host, and the caller waits about 320 µs.
  The GPU is idle more than half the time: the cost is the chain of about 7 dependent
  round trips per layer (attention, router, then experts in waves), not the kernels.
- **Tried and not kept.**
  - Four (or two) rows per invocation group, reusing each activation load: faster in the
    kernel timing test, but 89 ms per token end to end against 51.5. One row per group.
  - Polling the fence instead of `vkWaitForFences`: no change (MoltenVK signals the fence
    from Metal's completion handler either way).
  - 8 or 10 compute threads instead of 5: 35.0 and 35.6 ms against 36.5. The default is
    unchanged.
  - The shared-memory tree: a build without it was no faster, so it stays (no subgroup
    operations).

### Acceptance (2026-11-11)
The owner accepted this amendment, subject to the existing parity and regression
checks. They were rerun on `main` at 9721f58 (release build, `QUBIVEX_REQUIRE_VULKAN=1`,
MoltenVK 1.4.2), and all passed:
- `qubivex-backend-vulkan` parity: 8 passed, including every dtype, shape and batch and
  many matrices at offsets in one submission. The ignored timing test was not run.
- `qubivex-runtime` `tests/vulkan.rs`: 4 passed. These are the fixture bits, concurrent
  requests, a small device cache that evicts, and real OLMoE (the CPU's prompt logits
  and Colibri's tokens).

The speed figures above are unchanged: they come from the mains-power records.

### Consequences
- The kernel timing test (`cargo test --release -p qubivex-backend-vulkan --test parity
  -- --ignored --nocapture speed`) is a guide only: one change it favoured was slower
  end to end. Kernel changes are judged by `qubivex bench`.
- Further speed on this path needs fewer dependent round trips: whole layers on the
  device, as Metal amendment 2 did, with the CPU's bits for norms, attention, softmax,
  routing and SwiGLU. That is its own decision. Token tiles for prefill and batched
  decode remain open.

## Amendment 3 (2026-11-11): the fused expert block on two fixed-slot arenas (Step 3a)

The owner accepted Step 3a on 2026-11-11 (audit `VULKAN_EXECUTION_AUDIT.md` §16). This
amendment records what was built and the rules it adds. Amendment 2 is still reserved for
the shared GPU-pass contract; nothing here depends on it.

### The fused expert block
- **A generic block op.** `qubivex-compute` gains `GatedFfn` (`ys = down · swiglu(gate ·
  xs, up · xs)`) with `ComputeBackend::{fuses_gated_ffn, gated_ffn_many}` and
  `DiscreteDevice::{fuses_gated_ffn, gated_ffn_many}`. The defaults run the separate steps
  (two `matmul_many` with `math::swiglu` on the host between them), so a backend changes
  nothing until it opts in. The MoE stage asks the backend by shape (number of blocks,
  largest rows per expert) and, when every routed expert of the layer is resident, runs
  them as one `gated_ffn_many`. It applies to any SwiGLU MoE; there is no model-specific
  path.
- **One submission per layer's experts on Vulkan:** fused gate and up dispatch(es),
  barrier, SwiGLU, barrier, fused down dispatch(es), in one command buffer. The gate and
  up outputs stay in a device-local working buffer; only the inputs and the block's
  outputs cross host memory. Matrices are grouped into dispatches by (buffer, dtype,
  rows), so experts outside an arena, or of mixed dtypes, still run, as extra dispatches
  in the same submission.
- **Shaders.** `matmul.comp`'s body is in `matmul_body.glsl`, shared by `matmul.comp`
  (its SPIR-V is byte-identical, so the unfused pipeline did not change) and
  `matmul_fused.comp` (each workgroup finds its matrix in a table, `wg.x / blocks`, and
  skips rows past that matrix's tokens uniformly). `swiglu.comp` is the port of
  `math::swiglu` with `math::exp`.
- **Shape selection.** `--vulkan-fuse-rows N` (default 8, 0 turns fusion off): a layer
  fuses when no expert has more than N rows. 8 was measured (audit §16.4): equal to 4
  below 8 concurrent sequences, better at 8. A 128-token prompt never fuses, so prefill
  is unchanged.

### Two fixed-slot arenas, and memory accounting
- **The device expert cache is one arena per expert size class**, divided into equal
  slots (OLMoE Q4_K_M: Q4_K-down experts of 3 538 944 bytes and Q6_K-down experts of
  4 079 616 bytes, so two arenas). A copy is a write into a free slot, never an
  allocation; there is no fragmentation; every expert of a layer sits in one buffer.
  New device methods: `alloc`, `write`, `max_arena`.
- **Slot counts** follow the budget's distribution over layers (the RAM cache's rule,
  `qubivex_memory::distribute`), capped per layer by the RAM cache's slots (the device
  cache is inclusive, so it can never use more) and per arena by `max_arena`, the
  largest buffer one binding can address (`maxStorageBufferRange`, at most 4 GiB). Slots
  start at multiples of 256 bytes.
- **Pins are per slot.** A forward pass holds a slot's pin while computing that expert;
  eviction (least recently used within the class) and overwriting skip pinned slots. If
  every slot of a class is pinned, the copy gets its own allocation over the budget
  (`over_budget` counts it); the device's capacity stays the hard limit.
- **No arenas** when the device has none (the default `alloc` refuses) or the budget
  cannot give every layer one slot: each copy is its own allocation, as before.
- **Accounting.** An arena is allocated on its class's first copy and charged to the
  ledger's `Vram(0)` tier whole, at that moment. So the device cache's reported numbers
  split:
  - `experts.used`: device bytes the cache occupies, allocated arenas plus copies with
    their own allocation. This is the ledger's VRAM expert-cache charge.
  - `experts.held`: bytes of the expert copies it holds now.
  - `experts.arena_bytes` and `arena_slots`: the planned arenas, allocated or not.

  `used` can exceed `held` (an arena is charged before it fills), never the budget except
  through counted `over_budget` copies.
- **Working buffers are not charged to the ledger.** Each of the 8 submitters keeps a
  device-local intermediate buffer (8 bytes per gate output) and its host buffers, grown
  by doubling and reused, as `matmul_many`'s host buffers already were. The 8-row
  default bounds them: on OLMoE, 8 bytes × 1024 × (tokens × 8 routed) is 64 KiB at one
  decode token, 512 KiB at 8 sequences and at most 4 MiB (64 experts × 8 rows) per
  submitter. They come out of the plan's 256 MiB device reserve (ADR-018).

### SwiGLU compilation requirements
- **`glslc -O` breaks the CPU's bits for any shader with a rounding idiom.** glslc's
  optimiser runs before `build.sh` adds `NoContraction`, so it folds `exp`'s
  `(x · log2e + SHIFT) − SHIFT` into `x · log2e`; `int(kf)` then truncates instead of
  rounding (472 million of 2^32 inputs 1 ulp off; audit §15.1).
- **Rule:** `build.sh` has a per-shader optimisation level. A shader with a rounding idiom
  (today `swiglu.comp`) is built `-O0`, or uses `precise` on that idiom; the matmuls have
  none and stay `-O`, proven by their parity tests. Every new shader is gated by an
  exhaustive or targeted test against the CPU, and `spirv_is_current` keeps the checked-in
  SPIR-V in step with `build.sh`.
- `-O0` costs nothing measurable here: SwiGLU and its two barriers add at most about
  3 µs of GPU time per layer.
- The division in `silu` must be correctly rounded. MoltenVK's Safe math mode is on this
  device (exhaustive test); other drivers are not assumed to be, and the tests are the
  check on each.

### Numerical parity contract
The contract stays §5's: the CPU's bits. Step 3a states three points precisely; they apply
to every GPU backend (Metal already behaves this way; ADR-011 amendment 7 records it in
the cross-backend contract):
1. **Subnormals: ADR-011 §1 unchanged.** A device that flushes f32 subnormals may differ
   wherever an input, product, partial sum, intermediate (here `silu(g)`) or the exact
   result is below `f32::MIN_POSITIVE` in magnitude. That includes a subnormal exact
   result that IEEE rounding would round up to `f32::MIN_POSITIVE` but the device flushes
   first. A flush can also turn a CPU infinity into a NaN (a flushed `silu(g)` times an
   infinite `u`); that difference is the subnormal exception too, and is counted as such.
2. **NaN payloads and NaN signs are outside the contract.** Where the CPU's output is
   NaN, the device's output must be NaN, but its payload and sign bit are not compared.
   The device's SwiGLU returns the canonical `0x7fc00000`; the CPU propagates the input
   payload.
3. **NaN classification is inside the contract.** A NaN where the CPU has a number, or a
   number where the CPU has a NaN, fails, unless point 1 explains it.

The tests follow it: exact bits everywhere else; the SwiGLU tests classify every
difference as subnormal, NaN payload (both NaN) or other, print the counts, and fail on
any other.

### Measured
Audit §16, every number recorded on mains power, untimed, alternated with the permanent
Step 1a baseline (tag `bench/vulkan-step1a`, driver `scripts/bench-vulkan-baseline`):
- **Correctness.** The fused block equals the CPU's separate steps bit for bit for
  Q4_K/Q4_K, Q4_K/Q6_K, Q6_K/Q6_K, Q8_0, f32/f16 and bf16/f32 at 1, 3 and 8 experts × 1,
  2, 4 and 10 rows, in an arena and in separate buffers, and at OLMoE's shapes. Device
  SwiGLU over all 2 × 2^32 inputs: 0 differences outside the subnormal and NaN-payload
  classes. Real OLMoE: the CPU's prompt logits and Colibri's tokens, with a device cache
  that evicts. A `-O` build of SwiGLU fails the targeted test (positive control).
- **Single stream** (10-token prompt, 128 generated, 5 threads): 35.78 → 33.00 ms per token
  (27.95 → 30.30 tok/s), 305 → 65 submissions per token, prefill unchanged. The arenas
  alone change nothing; the gain is the single submission.
- **Concurrent** (the batch example): +11% at 1–2 sequences, +9.5% at 4, +5.7% at 8.

### Consequences
- Step 1a stays the permanent benchmark baseline, and Step 3a (`--vulkan-fuse-rows 8`) is
  the reference that later Vulkan steps are measured against as well.
- The arena layout is what lets a later step find an expert from its index alone: an
  expert's device address is (class, slot), so a GPU-side table can map routed indices to
  weights (Step 3b).
- The working buffers are outside the ledger; a later step that keeps more on the device
  (activations, KV) must charge them.
