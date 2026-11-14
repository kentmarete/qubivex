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
    src="${3:-$name}"
    shift 2
    [ $# -gt 0 ] && shift
    glslc --target-env=vulkan1.1 "$opt" "$@" -I "$here/../shaders" -o "$tmp/$name.spv" "$here/$src.comp"
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
# Audit §20. Softmax and values fused (attn_sv_*: unroll depth U; attn_sv4_*: four chains
# per invocation); unrolled scores (attn_scores_u*); scores, softmax and values in one
# dispatch (attn_one*; attn_oneb_*: each token its own sequence); and the clip, Q/K norm,
# RoPE and KV store prelude.
for u in 8 16 32; do
    build "attn_sv_u$u" -O0 attn_sv -DU=$u
    build "attn_sv4_u$u" -O0 attn_sv -DU=$u -DVEC4
    build "attn_scores_u$u" -O0 attn_scores_u -DU=$u
done
for u in 8 16; do
    build "attn_one_u$u" -O0 attn_sv -DU=$u -DONE
    build "attn_one4_u$u" -O0 attn_sv -DU=$u -DONE -DVEC4
done
build attn_oneb_u16 -O0 attn_sv -DU=16 -DONE -DBATCH
build qk_prelude -O0
# Audit §23. The matmul's K-quant and Q8_0 bodies (mm.comp) by dtype, reduction (t0 shared
# memory, t1 subgroup shuffles), rows per group and workgroup size, built -O as
# production's matmul is; `_nd` variants drop the decoding (timing controls only).
for dt in 3:q8 4:q4k 5:q6k; do
    n="${dt#*:}"
    v="${dt%%:*}"
    for spec in t0_r1_w256 t1_r1_w256 t1_r2_w256 t1_r4_w256 t1_r1_w64 t1_r2_w128 t0_r2_w256; do
        tr="${spec%%_*}"
        rest="${spec#*_}"
        r="${rest%%_*}"
        wg="${rest#*_}"
        build "mm_${n}_$spec" -O mm -DDT="$v" -DTREE="${tr#t}" -DNR="${r#r}" -DWG="${wg#w}"
    done
    build "mm_${n}_t0_r1_w256_nd" -O mm -DDT="$v" -DTREE=0 -DNR=1 -DWG=256 -DNODEQ
    build "mm_${n}_t1_r2_w256_nd" -O mm -DDT="$v" -DTREE=1 -DNR=2 -DWG=256 -DNODEQ
done
# Q4_K decode ablations (timing only): scales from fixed bytes, quant byte not loaded,
# no `opaque`.
for a in SCALE BYTE OPAQUE; do
    l="$(echo "$a" | tr 'A-Z' 'a-z')"
    build "mm_q4k_t1_r2_w256_a$l" -O mm -DDT=4 -DTREE=1 -DNR=2 -DWG=256 -DABL_$a
done
# Q4_K rewrites: groups written out (u), and quant bytes by shuffle (q).
build mm_q4k_t1_r2_w256_u -O mm -DDT=4 -DTREE=1 -DNR=2 -DWG=256 -DK4V2
build mm_q4k_t1_r2_w256_uq -O mm -DDT=4 -DTREE=1 -DNR=2 -DWG=256 -DK4V2 -DK4Q
build mm_q4k_t1_r1_w256_uq -O mm -DDT=4 -DTREE=1 -DNR=1 -DWG=256 -DK4V2 -DK4Q
build mm_q4k_t1_r4_w256_uq -O mm -DDT=4 -DTREE=1 -DNR=4 -DWG=256 -DK4V2 -DK4Q
# Scales once per block by lanes 0..7, shared by shuffle (s).
build mm_q4k_t1_r2_w256_us -O mm -DDT=4 -DTREE=1 -DNR=2 -DWG=256 -DK4V2 -DK4S
build mm_q4k_t1_r2_w256_uqs -O mm -DDT=4 -DTREE=1 -DNR=2 -DWG=256 -DK4V2 -DK4Q -DK4S
build mm_q4k_t1_r4_w256_uqs -O mm -DDT=4 -DTREE=1 -DNR=4 -DWG=256 -DK4V2 -DK4Q -DK4S
build mm_q4k_t1_r1_w256_uqs -O mm -DDT=4 -DTREE=1 -DNR=1 -DWG=256 -DK4V2 -DK4Q -DK4S
# Q6_K rewrites: written out (u), scales once per block shared by shuffle (s).
build mm_q6k_t1_r2_w256_u -O mm -DDT=5 -DTREE=1 -DNR=2 -DWG=256 -DK6V2
build mm_q6k_t1_r2_w256_us -O mm -DDT=5 -DTREE=1 -DNR=2 -DWG=256 -DK6V2 -DK6S
build mm_q6k_t1_r4_w256_us -O mm -DDT=5 -DTREE=1 -DNR=4 -DWG=256 -DK6V2 -DK6S
build mm_q6k_t1_r1_w256_us -O mm -DDT=5 -DTREE=1 -DNR=1 -DWG=256 -DK6V2 -DK6S
# The shared products through shared memory (h): no subgroup operation at all with t0.
for r in 1 2 4; do
    build "mm_q4k_t0_r${r}_w256_ush" -O mm -DDT=4 -DTREE=0 -DNR=$r -DWG=256 -DK4V2 -DK4S -DSH
    build "mm_q6k_t0_r${r}_w256_ush" -O mm -DDT=5 -DTREE=0 -DNR=$r -DWG=256 -DK6V2 -DK6S -DSH
done
build mm_q4k_t1_r2_w256_ush -O mm -DDT=4 -DTREE=1 -DNR=2 -DWG=256 -DK4V2 -DK4S -DSH
build mm_q6k_t1_r2_w256_ush -O mm -DDT=5 -DTREE=1 -DNR=2 -DWG=256 -DK6V2 -DK6S -DSH
