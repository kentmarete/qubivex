"""Reference dequantization of random GGML blocks (Q8_0, Q4_K, Q6_K, MXFP4), from gguf-py (llama.cpp's Python package).

Run with the oracle venv (see fixtures/oracle/README.md):
    ~/.qubivex/oracle/bin/python -I fixtures/oracle/ggml_dequant.py > fixtures/oracle/ggml_dequant.json

The Rust test `qubivex-backend-cpu/tests/oracle.rs` checks its dequantizers against
these values bit for bit.
"""
import json
import struct
import sys

import gguf
import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

rng = np.random.default_rng(20261103)

def f16_bytes(x):
    return struct.pack("<e", x)

def blocks(qtype, n_blocks):
    size, scale_offsets = {
        T.Q8_0: (34, [0]),
        T.Q4_K: (144, [0, 2]),
        T.Q6_K: (210, [208]),
    }[qtype]
    raw = bytearray(rng.integers(0, 256, size * n_blocks, dtype=np.uint8).tobytes())
    for b in range(n_blocks):
        for off in scale_offsets:
            raw[b * size + off: b * size + off + 2] = f16_bytes(float(rng.uniform(1e-4, 2e-2)))
    return bytes(raw)

cases = []
for name, qtype, per_block, n_blocks in [("q8_0", T.Q8_0, 32, 8), ("q4_k", T.Q4_K, 256, 4), ("q6_k", T.Q6_K, 256, 4)]:
    raw = blocks(qtype, n_blocks)
    vals = dequantize(np.frombuffer(raw, dtype=np.uint8), qtype).astype(np.float32).reshape(-1)
    assert vals.size == per_block * n_blocks
    cases.append({
        "dtype": name,
        "cols": per_block * n_blocks,
        "bytes_hex": raw.hex(),
        "f32_bits": [int(b) for b in vals.view(np.uint32)],
    })

# MXFP4: one block per E8M0 scale byte (0 to 255, subnormal halves and overflow
# included), random nibbles. Drawn after the cases above so their bytes are unchanged.
raw = bytearray(rng.integers(0, 256, 17 * 256, dtype=np.uint8).tobytes())
for b in range(256):
    raw[17 * b] = b
raw = bytes(raw)
with np.errstate(over="ignore"):
    vals = dequantize(np.frombuffer(raw, dtype=np.uint8), T.MXFP4).astype(np.float32).reshape(-1)
assert vals.size == 32 * 256
cases.append({
    "dtype": "mxfp4",
    "cols": 32 * 256,
    "bytes_hex": raw.hex(),
    "f32_bits": [int(b) for b in vals.view(np.uint32)],
})

json.dump({
    "generator": "fixtures/oracle/ggml_dequant.py",
    "gguf": "0.19.0",
    "numpy": np.__version__,
    "cases": cases,
}, sys.stdout, indent=1)
sys.stdout.write("\n")
