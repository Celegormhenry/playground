"""
Method 10: Lossless Encoding (LE)
───────────────────────────────────
Role   : Final encoding stage — every compressor uses this.
Idea   : The quantization codes from earlier stages are sparse integers (mostly 0,
         small ±values). Lossless encoding exploits this structure for final compression.
Types  : Huffman (entropy), Zlib/Zstd (dictionary), Run-Length (RLE), Fixed-length.
Used by:
  Huffman         → SZ1/2/3, FAZ, MGARD
  Arithmetic      → TTHRESH
  Zlib/Zstd       → SZ (after Huffman), Bit Grooming
  Run-length      → cuSZp, TTHRESH
  Fixed-length    → cuSZp (fast on GPU)
  Embedded (BPC)  → ZFP
"""

import numpy as np
import zlib
import heapq
from collections import Counter
from utils import header, subheader, huffman_build

header("10. LOSSLESS ENCODING")

# ════════════════════════════════════════════════════════════════════════════
# 10a. Huffman encoding
# ════════════════════════════════════════════════════════════════════════════

subheader("10a. Huffman Encoding")

print("""
  Assign shorter bit-codes to more frequent symbols.
  Optimal for symbol-by-symbol encoding (minimum average code length).
  Used as the primary encoder in SZ1/2/3 before Zstd.
""")

# Typical Lorenzo qcodes: heavily concentrated at 0, ±1, ±2
qcodes = np.array([0,0,0,1,0,0,-1,0,0,0,1,0,0,0,0,-1,2,0,0,0,
                   0,0,0,1,0,0,0,0,-1,0], dtype=np.int32)

freq = dict(sorted(Counter(qcodes.tolist()).items()))
print(f"INPUT  (30 Lorenzo qcodes): {qcodes}")
print(f"Frequency table: {freq}")

table = huffman_build(qcodes.tolist())

print(f"\nHUFFMAN CODE TABLE (sorted by code length):")
print(f"  {'symbol':>8}  {'freq':>6}  {'code':>12}  {'bits':>6}  {'contribution':>14}")
total_bits = 0
for sym, code in sorted(table.items(), key=lambda x: len(x[1])):
    f    = freq.get(sym, 0)
    bits = len(code) * f
    total_bits += bits
    print(f"  {sym:>8}  {f:>6}  {code:>12}  {len(code):>6}  {bits:>10} bits")

naive_bits   = len(qcodes) * 8
huffman_bits = total_bits

print(f"\n  Naive    (8 bits/symbol) : {naive_bits:>5} bits = {naive_bits//8} bytes")
print(f"  Huffman                  : {huffman_bits:>5} bits = {huffman_bits//8} bytes")
print(f"  Savings                  : {(1 - huffman_bits/naive_bits)*100:.1f}%")
print(f"\nEntropy lower bound: {sum(-f/len(qcodes)*np.log2(f/len(qcodes)) for f in freq.values() if f>0):.2f} bits/symbol")
print(f"Huffman achieved   : {huffman_bits/len(qcodes):.2f} bits/symbol")

# ════════════════════════════════════════════════════════════════════════════
# 10b. Zlib / Zstd (dictionary encoding)
# ════════════════════════════════════════════════════════════════════════════

subheader("10b. Zlib / Zstd  (SZ uses Huffman → Zstd in sequence)")

print("""
  Dictionary encoder: finds repeated patterns across the whole stream,
  replaces them with back-references. Complements Huffman well.
  Zstd (used in SZ3) is faster than zlib with similar or better ratio.
""")

# Simulate three types of data that LE operates on
cases = {
    "smooth signal qcodes (mostly 0)":
        np.array([0]*20 + [1,0,-1,0,0,0,1,0,0,-1], dtype=np.int8),
    "noisy signal qcodes (spread)":
        np.random.default_rng(2).integers(-10, 11, 30).astype(np.int8),
    "constant region (all 0)":
        np.zeros(30, dtype=np.int8),
}

print(f"\n  {'data type':>35}  {'raw':>6}  {'zlib':>6}  {'ratio':>7}")
for name, data in cases.items():
    raw_bytes  = data.tobytes()
    compressed = zlib.compress(raw_bytes, level=9)
    ratio      = len(raw_bytes) / len(compressed)
    print(f"  {name:>35}  {len(raw_bytes):>6}  {len(compressed):>6}  {ratio:>6.2f}×")

# ════════════════════════════════════════════════════════════════════════════
# 10c. Run-Length Encoding (RLE)
# ════════════════════════════════════════════════════════════════════════════

subheader("10c. Run-Length Encoding (cuSZp / TTHRESH)")

print("""
  RLE encodes runs of repeated symbols as (value, count) pairs.
  Very effective when many consecutive qcodes are 0
  (common for cuSZp's constant blocks and TTHRESH's sparse coefficients).
""")

def rle_encode(arr: np.ndarray):
    out = []
    i   = 0
    while i < len(arr):
        val = arr[i]
        run = 0
        while i < len(arr) and arr[i] == val:
            run += 1; i += 1
        out.append((int(val), run))
    return out

def rle_decode(encoded: list, dtype=np.int32) -> np.ndarray:
    return np.array([v for val, run in encoded for v in [val]*run], dtype=dtype)

for name, arr in [
    ("sparse (many zeros)", np.array([0,0,0,0,0,3,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,1,0,0,0,0])),
    ("dense (alternating)",  np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1])),
    ("block zeros",          np.array([0]*12 + [5] + [0]*12)),
]:
    enc = rle_encode(arr)
    dec = rle_decode(enc)
    tokens = len(enc)
    print(f"\n  {name}:")
    print(f"    input : {arr}")
    print(f"    RLE   : {enc}")
    print(f"    {len(arr)} symbols → {tokens} tokens  ({len(arr)/tokens:.1f}× token reduction)")
    print(f"    verify: {np.array_equal(dec, arr)}")

# ════════════════════════════════════════════════════════════════════════════
# 10d. Fixed-length encoding (cuSZp GPU style)
# ════════════════════════════════════════════════════════════════════════════

subheader("10d. Fixed-length encoding (cuSZp GPU optimisation)")

print("""
  Huffman requires a tree traversal — inherently serial per symbol.
  On GPU, cuSZp uses fixed-length encoding with a small number of bits/symbol.
  If qcodes fit in N bits (e.g. N=4 for values -7 to +7), pack them tightly.
  This is much faster on GPU (fully parallel) at a modest ratio cost.
""")

qcodes2 = np.array([0,1,-1,2,0,0,3,-2,0,1,0,0,-1,0,2,0], dtype=np.int32)
n_bits   = 4   # e.g. 4-bit signed = range [-8, 7]

in_range  = np.all(np.abs(qcodes2) < 2**(n_bits-1))
packed_sz = len(qcodes2) * n_bits / 8   # bytes if packed at n_bits/symbol
naive_sz  = len(qcodes2) * 4            # int32 = 4 bytes/symbol

print(f"  qcodes       : {qcodes2}")
print(f"  All fit in {n_bits} bits (signed): {in_range}")
print(f"  Naive storage (int32) : {naive_sz} bytes")
print(f"  Fixed {n_bits}-bit packing  : {packed_sz:.1f} bytes  ({naive_sz/packed_sz:.1f}× over naive)")
print(f"  Huffman would save more, but fixed-length is GPU-parallel with no tree lookup.")
