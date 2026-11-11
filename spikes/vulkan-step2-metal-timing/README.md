# Metal timing harness for the Vulkan expert matmuls (audit §13)

A measurement tool, not product code. It is not in the workspace and nothing depends on it.

It runs MoltenVK's own MSL translation of `matmul.comp` (Q4_K), taken from MoltenVK's debug
log, in the shapes the Vulkan backend submits for an OLMoE layer's experts. It times them with
Metal's own clocks:
- `GPUStartTime`/`GPUEndTime` per command buffer;
- stage-boundary counter samples per encoder (the only sampling point an M3 offers).

Neither clock goes through Vulkan timestamp queries. With `TS=1` it also reproduces exactly
how MoltenVK 1.4.2 encodes `vkCmdWriteTimestamp` on Apple GPUs, so the two can be compared.

```
spikes/vulkan-step2-metal-timing/run.sh focus     # per-expert vs grouped vs concurrent, 8 experts
spikes/vulkan-step2-metal-timing/run.sh shapes    # every shape at 1/4/8 experts × 1/4 tokens
spikes/vulkan-step2-metal-timing/run.sh dispatch  # per-dispatch intervals inside one command buffer
spikes/vulkan-step2-metal-timing/run.sh scale     # one dispatch of 128 … 16384 rows
TS=1 THREADS=1 spikes/vulkan-step2-metal-timing/run.sh focus
spikes/vulkan-step2-metal-timing/run.sh fused     # + the fused multi-matrix dispatch (audit §14)
spikes/vulkan-step2-metal-timing/run.sh check     # fused vs unfused vs the CPU, bit for bit
SEARCH=1 spikes/vulkan-step2-metal-timing/run.sh fused   # the table-search lookup instead of O(1)
spikes/vulkan-step2-metal-timing/run.sh swiglu-check   # SwiGLU over all 2^32 g, twice (audit §15.1; ~2.5 min)
spikes/vulkan-step2-metal-timing/run.sh swiglu-time    # SwiGLU on the GPU and on the host, at a layer's sizes
spikes/vulkan-step2-metal-timing/run.sh binding-check  # bindings and the one-submission expert block, bit for bit
spikes/vulkan-step2-metal-timing/run.sh binding        # arena / array / bda (/ bda-all) timing, 64-expert pool
```

Audit §15 adds:
- `swiglu/`: a GLSL SwiGLU built as `matmul.comp` is, plus `-O0` and `precise` builds.
  `mvkmsl.c` makes MoltenVK translate it to MSL, and `check.m` runs the exhaustive comparison
  with a C reference.
- `binding.m`: the weight-binding variants, and the expert block in one command buffer.
- `cpucheck`'s `rusthash`: the C reference against `ops::swiglu`, by hash over every input.
- `cpucheck`'s `blockcheck`: the expert block against the CPU.
- `cpucheck`'s `swiglutime`: host SwiGLU time.

The fused prototype (audit §14) is in two parts:
- `fuse.py` derives `fused0` from MoltenVK's kernel: the same body, with the push constants
  taken from a per-matrix table.
- `cpucheck/` is a standalone crate, outside the workspace. It compares the `check` dump
  with `CpuBackend::matmul`.

Needs only the Command Line Tools: clang, plus the Metal runtime compiler for the MSL. The CLT
Swift toolchain on this machine does not match its SDK, hence Objective-C. Shapes are fixed to
OLMoE: hidden 2048, ffn 1024, 8 experts. Weights and activations follow the parity tests (random bytes with every odd byte & 0x3b; `acts`).
