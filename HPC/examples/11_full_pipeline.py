"""
Method 11: Full End-to-End Pipeline (SZ3-style)
────────────────────────────────────────────────
Demonstrates how the modular techniques chain together into a complete compressor.

Pipeline: Lorenzo (PDP) → Quantization (QT) → Huffman (LE) → Zlib (LE)

This mirrors SZ3's core pipeline (SZ3 uses spline interpolation instead of
Lorenzo for better ratio, but Lorenzo is simpler to demonstrate step-by-step).
"""

import numpy as np
import zlib
from utils import header, subheader, lorenzo_1d, huffman_build

header("11. FULL PIPELINE (SZ3-style)")

# ════════════════════════════════════════════════════════════════════════════
# Setup
# ════════════════════════════════════════════════════════════════════════════

t    = np.linspace(0, 4*np.pi, 100)
data = (np.sin(t) + 0.3*np.cos(2*t)).astype(np.float64)
eb   = 1e-3

print(f"\nINPUT signal: sin(t) + 0.3·cos(2t),  100 points")
print(f"  dtype : {data.dtype}")
print(f"  range : [{data.min():.4f}, {data.max():.4f}]")
print(f"  size  : {data.nbytes} bytes  ({data.nbytes*8} bits)")
print(f"  EB    : {eb}")

# ════════════════════════════════════════════════════════════════════════════
# Step 1: Lorenzo prediction → quantization codes
# ════════════════════════════════════════════════════════════════════════════

subheader("STEP 1 — Lorenzo Prediction + Quantization")

qcodes, recon = lorenzo_1d(data, eb)

print(f"\n  qcodes range    : [{qcodes.min()}, {qcodes.max()}]")
print(f"  unique qcodes   : {len(np.unique(qcodes))}")
print(f"  zero fraction   : {(qcodes==0).mean()*100:.1f}%")
print(f"  max error (recon vs original): {np.abs(recon - data).max():.2e}  ≤ EB={eb}")

print(f"\n  First 20 qcodes: {qcodes[:20]}")
print(f"  Histogram of codes (top 10 by frequency):")
from collections import Counter
top = Counter(qcodes.tolist()).most_common(10)
for sym, cnt in top:
    bar = '█' * int(cnt * 40 / top[0][1])
    print(f"    {sym:>5} : {bar:<40} ({cnt})")

# ════════════════════════════════════════════════════════════════════════════
# Step 2: Huffman encoding
# ════════════════════════════════════════════════════════════════════════════

subheader("STEP 2 — Huffman Encoding")

table   = huffman_build(qcodes.tolist())
bitstr  = ''.join(table[s] for s in qcodes.tolist())

code_lengths = sorted(set(len(c) for c in table.values()))
print(f"\n  Code length range: {min(code_lengths)}–{max(code_lengths)} bits")
print(f"  Most frequent codes:")
top5 = sorted(table.items(), key=lambda x: len(x[1]))[:5]
for sym, code in top5:
    freq = (qcodes == sym).sum()
    print(f"    symbol {sym:>5}  freq={freq:>3}  code={code:<12} ({len(code)} bits)")

huffman_bytes = len(bitstr) / 8
print(f"\n  Total Huffman bits  : {len(bitstr)}")
print(f"  Total Huffman bytes : {huffman_bytes:.1f}")
print(f"  vs naive (8b/sym)   : {len(qcodes)*8} bits = {len(qcodes)} bytes")
print(f"  Huffman savings     : {(1 - len(bitstr)/(len(qcodes)*8))*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# Step 3: Zlib compression
# ════════════════════════════════════════════════════════════════════════════

subheader("STEP 3 — Zlib Dictionary Compression")

n_bytes   = (len(bitstr) + 7) // 8
bit_bytes = int(bitstr + '0'*(n_bytes*8 - len(bitstr)), 2).to_bytes(n_bytes, 'big')
compressed = zlib.compress(bit_bytes, level=9)

print(f"\n  Huffman bytes → Zlib bytes: {n_bytes} → {len(compressed)}")
print(f"  Additional savings: {(1 - len(compressed)/n_bytes)*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

subheader("COMPRESSION SUMMARY")

print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  Stage               Size (bytes)   Ratio over raw  │
  ├─────────────────────────────────────────────────────┤
  │  Original (float64)  {data.nbytes:>12}                   │
  │  After Huffman       {int(huffman_bytes):>12}   {data.nbytes/huffman_bytes:>7.2f}×          │
  │  After Zlib          {len(compressed):>12}   {data.nbytes/len(compressed):>7.2f}×          │
  ├─────────────────────────────────────────────────────┤
  │  Final CR            {data.nbytes/len(compressed):>12.2f}×                       │
  │  Max absolute error  {np.abs(recon-data).max():>12.2e}   ≤ EB={eb}      │
  │  Error bound held?   {'         YES ✓' if np.all(np.abs(recon-data) <= eb) else '          NO ✗'}                  │
  └─────────────────────────────────────────────────────┘
""")

# ════════════════════════════════════════════════════════════════════════════
# EB sweep: compression ratio vs. error bound
# ════════════════════════════════════════════════════════════════════════════

subheader("EB SWEEP — compression ratio vs. error bound tradeoff")

print(f"\n  {'EB':>10}  {'CR':>8}  {'max_err':>10}  {'unique_codes':>14}")
for eb_sweep in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    qc, rc    = lorenzo_1d(data, eb_sweep)
    ht        = huffman_build(qc.tolist())
    bs        = ''.join(ht[s] for s in qc.tolist())
    nb        = (len(bs) + 7) // 8
    bb        = int(bs + '0'*(nb*8-len(bs)), 2).to_bytes(nb, 'big')
    comp      = zlib.compress(bb, level=9)
    cr        = data.nbytes / len(comp)
    merr      = np.abs(rc - data).max()
    print(f"  {eb_sweep:>10.0e}  {cr:>7.2f}×  {merr:>10.2e}  {len(np.unique(qc)):>14}")

print(f"\nKey insight: tighter EB → more unique qcodes → worse Huffman efficiency → lower CR.")
