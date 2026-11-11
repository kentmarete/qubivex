#!/bin/sh
# SPIR-V for swiglu.comp, exactly as crates/qubivex-backend-vulkan/shaders/build.sh builds
# matmul.comp (glslc -O, then NoContraction on every OpFAdd, OpFSub and Fma), then the MSL
# MoltenVK generates from it (mvkmsl, which only creates the pipeline).
# Usage: build.sh OUT_DIR [NAME] [GLSLC_OPT] [SOURCE]
#   defaults: swiglu, -O (as matmul.comp), swiglu.comp. Writes OUT_DIR/NAME.metal.
# Also writes OUT_DIR/control.metal: the native Metal backend's qexp and swiglu kernel,
# extracted from crates/qubivex-backend-metal/src/kernels.metal (the parity-tested reference).
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
out="$1"
name="${2:-swiglu}"
opt="${3:--O}"
src="${4:-$here/swiglu.comp}"
mkdir -p "$out"
glslc --target-env=vulkan1.1 "$opt" -o "$out/$name.raw.spv" "$src"
spirv-dis --raw-id "$out/$name.raw.spv" > "$out/$name.spvasm"
awk '
    / = OpExtInst .* Fma / || / = OpF(Add|Sub) / { fma[n++] = $1 }
    /OpDecorate .* NoContraction/ { have[$2] = 1 }
    { line[m++] = $0 }
    END {
        done = 0
        for (i = 0; i < m; i++) {
            if (!done && line[i] ~ /^ *OpDecorate /) {
                for (j = 0; j < n; j++) if (!(fma[j] in have)) print "OpDecorate " fma[j] " NoContraction"
                done = 1
            }
            print line[i]
        }
        if (n == 0 || !done) exit 1
    }
' "$out/$name.spvasm" > "$out/$name.fixed.spvasm"
spirv-as --target-env vulkan1.1 -o "$out/$name.spv" "$out/$name.fixed.spvasm"
spirv-val --target-env vulkan1.1 "$out/$name.spv"
clang -O1 -I/opt/homebrew/include -L/opt/homebrew/lib -lvulkan "$here/mvkmsl.c" -o "$out/mvkmsl"
MVK_CONFIG_DEBUG=1 MVK_CONFIG_LOG_LEVEL=4 MVK_CONFIG_FAST_MATH_ENABLED=0 \
    "$out/mvkmsl" "$out/$name.spv" > "$out/mvk-$name.log" 2>&1
start=$(grep -n '#include <metal_stdlib>' "$out/mvk-$name.log" | head -1 | cut -d: -f1)
awk -v s="$start" 'NR>=s' "$out/mvk-$name.log" | awk '{print} /^}$/ && seen {exit} /kernel void main0/{seen=1}' > "$out/$name.metal"
grep -c 'kernel void main0' "$out/$name.metal" > /dev/null
k="$here/../../../crates/qubivex-backend-metal/src/kernels.metal"
{
    printf '#include <metal_stdlib>\nusing namespace metal;\n#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n'
    printf 'struct EwParams {\n    uint n;\n    float f;\n};\n'
    awk '/^constant float QEXP_LOG2E/{p=1} p{print} p && /^}$/ && seen{exit} /static inline float qexp/{seen=1}' "$k"
    awk '/^kernel void swiglu\(/{p=1} p{print} p && /^}$/{exit}' "$k"
} > "$out/control.metal"
