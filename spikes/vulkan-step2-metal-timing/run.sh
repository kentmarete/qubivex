#!/bin/zsh
# Builds the Metal timing harness and runs it on MoltenVK's own MSL for matmul.comp.
# Usage: spikes/vulkan-step2-metal-timing/run.sh [shapes|focus|dispatch|scale|all]
# Env: REPS (default 200), TS=1 (add MoltenVK-style timestamps), THREADS (per-expert
# threads, default 5), UNTRACKED=1 (untracked hazard tracking).
set -e
here=${0:A:h}
repo=${here:h:h}
out=${TMPDIR:-/tmp}/qvx-metal-timing
mkdir -p $out
# MoltenVK logs the MSL it generates; take the kernel and bind plain buffers instead of
# its argument buffer (the body is unchanged).
(cd $repo && MVK_CONFIG_DEBUG=1 MVK_CONFIG_LOG_LEVEL=4 cargo test --release -q \
  -p qubivex-backend-vulkan --test parity every_dtype_shape_and_batch_matches_the_cpu) > $out/mvk.log 2>&1
start=$(grep -n '#include <metal_stdlib>' $out/mvk.log | head -1 | cut -d: -f1)
awk -v s=$start 'NR>=s' $out/mvk.log | awk '{print} /^}$/ && seen {exit} /kernel void main0/{seen=1}' \
  | sed -e 's/kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 \[\[buffer(0)\]\], /kernel void main0(device _27* w28 [[buffer(0)]], device _32* x33 [[buffer(1)]], device _35* y36 [[buffer(2)]], /' \
        -e 's/spvDescriptorSet0\.m_28/w28/g; s/spvDescriptorSet0\.m_33/x33/g; s/spvDescriptorSet0\.m_36/y36/g' > $out/kernel.metal
clang -O2 -fobjc-arc -framework Metal -framework Foundation -framework QuartzCore $here/main.m -o $out/harness
$out/harness $out/kernel.metal ${1:-all}
