#!/bin/sh
# Builds the investigation harness's experimental shaders (audit §19) into this
# directory, exactly as ../shaders/build.sh builds production ones: glslc at the given
# level, then NoContraction on every add, subtract and fma. Not production code: only the
# `spike` feature's tests load these.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
build() {
    name="$1"
    opt="$2"
    glslc --target-env=vulkan1.1 "$opt" -I "$here/../shaders" -o "$tmp/$name.spv" "$here/$name.comp"
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
    cp "$tmp/$name.fixed.spv" "$here/$name.spv"
}
build attn_exact -O0
build attn_exact_sh -O0
build rope -O0
build attn_scores_t -O0
build attn_softmax -O0
build attn_values -O0
build attn_values_u -O0
build attn_softmax_u -O0
