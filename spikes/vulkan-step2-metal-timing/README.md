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
```

Needs only the Command Line Tools: clang, plus the Metal runtime compiler for the MSL. The CLT
Swift toolchain on this machine does not match its SDK, hence Objective-C. Shapes are fixed to
OLMoE: hidden 2048, ffn 1024, 8 experts. Weights are random quants with plausible f16 scales.
