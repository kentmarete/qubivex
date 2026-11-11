#!/bin/sh
# Compiles the Vulkan shaders to the SPIR-V checked in under src/ (ADR-019).
# Usage: build.sh [OUT_DIR] (default: ../src). Needs glslc, spirv-dis, spirv-as and
# spirv-val (shaderc and SPIRV-Tools). The unit test `spirv_is_current` runs this into
# a temporary directory and fails when the checked-in SPIR-V differs, if the tools are
# installed.
#
# The shader does not use `precise` (see matmul.comp): every OpFAdd, OpFSub and
# GLSL.std.450 Fma result is decorated NoContraction here instead, so no driver fuses
# a multiply into an add or subtract, and none may split an fma into a multiply and
# an add (lavapipe does). Multiplies stay undecorated.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/../src}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
glslc --target-env=vulkan1.1 -O -o "$tmp/matmul.spv" "$here/matmul.comp"
spirv-dis --raw-id "$tmp/matmul.spv" > "$tmp/matmul.spvasm"
awk '
    / = OpExtInst .* Fma / || / = OpF(Add|Sub) / { fma[n++] = $1 }
    { line[m++] = $0 }
    END {
        done = 0
        for (i = 0; i < m; i++) {
            if (!done && line[i] ~ /^ *OpDecorate /) {
                for (j = 0; j < n; j++) print "OpDecorate " fma[j] " NoContraction"
                done = 1
            }
            print line[i]
        }
        if (n == 0 || !done) exit 1
    }
' "$tmp/matmul.spvasm" > "$tmp/fixed.spvasm"
spirv-as --target-env vulkan1.1 -o "$tmp/fixed.spv" "$tmp/fixed.spvasm"
spirv-val --target-env vulkan1.1 "$tmp/fixed.spv"
cp "$tmp/fixed.spv" "$out/matmul.spv"
