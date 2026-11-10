#!/bin/sh
# Compiles the Vulkan shaders to the SPIR-V checked in under src/ (ADR-019).
# Needs glslc (shaderc). The unit test `spirv_is_current` fails when the checked-in
# SPIR-V differs from what this produces, if glslc is installed.
set -eu
cd "$(dirname "$0")"
glslc --target-env=vulkan1.1 -O -o ../src/matmul.spv matmul.comp
if command -v spirv-val >/dev/null; then spirv-val --target-env vulkan1.1 ../src/matmul.spv; fi
