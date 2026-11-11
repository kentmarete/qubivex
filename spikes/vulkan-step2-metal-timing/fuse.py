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
"""

import sys

src = open(sys.argv[1]).read()
equal = "--equal" in sys.argv[2:]


def once(old: str, new: str) -> None:
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)


# Table entry: the matrix's push constants, then its first row block in the grid.
once(
    "kernel void main0(",
    "struct QvxFused\n{\n    _29 p;\n    uint first;\n    uint pad;\n};\n\n"
    "kernel void fused0(",
)
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
    "    const uint3 gl_WorkGroupID = uint3(qvx_wg.x - qvx_tab[qvx_m].first, qvx_wg.y, qvx_wg.z);\n",
)
sys.stdout.write(src)
