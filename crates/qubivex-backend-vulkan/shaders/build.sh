#!/bin/sh
# Compiles the Vulkan shaders to the SPIR-V checked in under src/ (ADR-019).
# Usage: build.sh [OUT_DIR] (default: ../src). Needs glslc, spirv-dis, spirv-as and
# spirv-val (shaderc and SPIRV-Tools). The unit test `spirv_is_current` runs this into
# a temporary directory and fails when the checked-in SPIR-V differs, if the tools are
# installed.
#
# The shaders do not use `precise` (see matmul.comp): every OpFAdd, OpFSub and
# GLSL.std.450 Fma result is decorated NoContraction here instead, so no driver fuses
# a multiply into an add or subtract, and none may split an fma into a multiply and
# an add (lavapipe does). Multiplies stay undecorated.
#
# The optimisation level is per shader. glslc's optimiser runs before the decoration,
# so it treats float arithmetic as algebra: a shader with a rounding idiom such as
# exp's `(x * LOG2E + SHIFT) - SHIFT` must be built -O0 (swiglu.comp; audit §15.1).
# The matmuls have none and are built -O; their parity tests are the proof.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/../src}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
build() {
    name="$1"
    opt="$2"
    glslc --target-env=vulkan1.1 "$opt" -o "$tmp/$name.spv" "$here/$name.comp"
    spirv-dis --raw-id "$tmp/$name.spv" > "$tmp/$name.spvasm"
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
    ' "$tmp/$name.spvasm" > "$tmp/$name.fixed.spvasm"
    spirv-as --target-env vulkan1.1 -o "$tmp/$name.fixed.spv" "$tmp/$name.fixed.spvasm"
    spirv-val --target-env vulkan1.1 "$tmp/$name.fixed.spv"
    cp "$tmp/$name.fixed.spv" "$out/$name.spv"
}
build matmul -O
build matmul_fused -O
build swiglu -O0
build route -O0
build combine -O0
