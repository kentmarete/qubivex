"""Turns the unfused harness kernel (MoltenVK's MSL for matmul.comp, plain bindings) into
a multi-matrix kernel: one dispatch covers several independent matrices.

The grid's x axis is the concatenation of every matrix's row blocks. Each threadgroup
finds its matrix in a table (the first row block of each matrix, ascending), then runs
the unchanged body with that matrix's push constants and its row block within the
matrix. Every output therefore goes through exactly the same instructions with the same
operands as in its own dispatch, so it has the same bits.

Usage: python3 fuse.py kernel.metal [--equal] > fused.metal

--equal: every matrix in the dispatch has the same number of row blocks (a stage's
gate/up matrices, or its downs), so the matrix index is the row block divided by that
count (table entry 0's `pad`), not a search of the table.

--binding arena|array|bda (audit §15; implies a 48-byte table entry with `wi` and `w`):
  arena  the weights are one buffer at buffer(0); `w_base` is the matrix's offset in it.
  array  each expert is its own buffer; the matrix's buffer is element `wi` of a pointer
         array at buffer(10) (what MoltenVK makes of a Vulkan descriptor array on Metal 3).
  bda    the matrix's buffer address is in its table entry (`w`), as a Vulkan
         buffer-device-address shader would carry it.
  For array and bda, `w_base` is the matrix's offset within its expert's buffer.
Kernel name: fused_<binding>.
"""

import sys

src = open(sys.argv[1]).read()
equal = "--equal" in sys.argv[2:]
binding = None
if "--binding" in sys.argv:
    binding = sys.argv[sys.argv.index("--binding") + 1]
    assert binding in ("arena", "array", "bda"), binding


def once(old: str, new: str) -> None:
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)


# Table entry: the matrix's push constants, then its first row block in the grid.
if binding is None:
    once(
        "kernel void main0(",
        "struct QvxFused\n{\n    _29 p;\n    uint first;\n    uint pad;\n};\n\n"
        "kernel void fused0(",
    )
else:
    once(
        "kernel void main0(",
        "struct QvxFused\n{\n    _29 p;\n    uint first;\n    uint pad;\n    uint wi;\n    uint pad2;\n"
        "    device _27* w;\n};\n\nstruct QvxWArr\n{\n    device _27* w[64];\n};\n\n"
        f"kernel void fused_{binding}(",
    )
    if binding != "arena":
        once("device _27* w28 [[buffer(0)]], ", "")

once(
    "constant _29& _78 [[buffer(8)]]",
    "constant QvxFused* qvx_tab [[buffer(8)]], constant uint& qvx_count [[buffer(9)]]",
)
once(
    "uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])\n{\n",
    "uint3 qvx_wg [[threadgroup_position_in_grid]])\n{\n"
    # Uniform across the threadgroup, so the body's barriers stay uniform.
    + (
        "    uint qvx_m = min(qvx_wg.x / qvx_tab[0].pad, qvx_count - 1u);\n"
        if equal
        else "    uint qvx_m = 0u;\n"
        "    while (qvx_m + 1u < qvx_count && qvx_wg.x >= qvx_tab[qvx_m + 1u].first)\n"
        "    {\n"
        "        qvx_m++;\n"
        "    }\n"
    )
    +
    "    const _29 _78 = qvx_tab[qvx_m].p;\n"
    "    const uint3 gl_WorkGroupID = uint3(qvx_wg.x - qvx_tab[qvx_m].first, qvx_wg.y, qvx_wg.z);\n"
    + (
        "    device _27* w28 = qvx_warr.w[qvx_tab[qvx_m].wi];\n" if binding == "array"
        else "    device _27* w28 = qvx_tab[qvx_m].w;\n" if binding == "bda"
        else ""
    ),
)
if binding == "array":
    once("constant uint& qvx_count [[buffer(9)]]",
         "constant uint& qvx_count [[buffer(9)]], constant QvxWArr& qvx_warr [[buffer(10)]]")
sys.stdout.write(src)
