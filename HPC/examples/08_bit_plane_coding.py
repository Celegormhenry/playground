"""
Method 8: Bit-Plane Coding (BPC)
──────────────────────────────────
Role   : Module — used in both preprocessing and encoding stages.
Idea   : Process data bit-plane by bit-plane (from most to least significant).
         Transmit/keep only the top N bit-planes → error decreases with each plane.
         This enables progressive precision: stop when error is small enough.
Key property: dropping the k least-significant bit-planes introduces an error
              proportional to the data value (scale-dependent).
Used by: ZFP (embedded coding), SZx, cuSZp, SZ series, Bit Grooming, TTHRESH.

IEEE 754 float32:
  bit 31    : sign
  bits 30-23: exponent (biased by 127)
  bits 22-0 : mantissa (implicit leading 1)
  value = (-1)^sign × 2^(exp-127) × 1.mantissa
"""

import numpy as np
import struct
from utils import header, subheader

header("8. BIT-PLANE CODING (BPC)")

# ════════════════════════════════════════════════════════════════════════════
# 8a. Float32 bit layout
# ════════════════════════════════════════════════════════════════════════════

subheader("8a. IEEE 754 float32 anatomy")

for val in [3.14159265, -2.71828, 0.001, 1000.0]:
    v      = np.float32(val)
    packed = struct.pack('f', v)
    as_int = struct.unpack('I', packed)[0]
    bits   = format(as_int, '032b')
    exp_val = int(bits[1:9], 2) - 127
    print(f"\n  value    : {v}")
    print(f"  bits     : {bits[0]} {bits[1:9]} {bits[9:]}")
    print(f"  meaning  : sign={bits[0]}  exp={int(bits[1:9],2)}-127={exp_val}  mantissa=1.{bits[9:]}")

# ════════════════════════════════════════════════════════════════════════════
# 8b. Progressive precision (ZFP-style)
# ════════════════════════════════════════════════════════════════════════════

subheader("8b. Progressive bit-plane transmission (ZFP embedded coding concept)")

val    = np.float32(3.14159265)
packed = struct.pack('f', val)
as_int = struct.unpack('I', packed)[0]
bits   = format(as_int, '032b')

print(f"\nValue: {val}  bits: {bits}")
print(f"\nTransmitting bit-planes one by one (most significant first):")
print(f"  {'planes kept':>12}  {'reconstructed':>16}  {'abs_err':>10}  {'rel_err':>10}")

for keep in [32, 28, 24, 20, 16, 12, 8]:
    n_drop    = 32 - keep
    mask      = np.uint32(0xFFFFFFFF) << np.uint32(n_drop)
    truncated = struct.unpack('f', struct.pack('I', as_int & int(mask)))[0]
    abs_err   = abs(float(truncated) - float(val))
    rel_err   = abs_err / abs(float(val)) * 100
    print(f"  {keep:>12}  {truncated:>16.8f}  {abs_err:>10.6f}  {rel_err:>9.4f}%")

# ════════════════════════════════════════════════════════════════════════════
# 8c. BPC on an array — exponent alignment (ZFP block step)
# ════════════════════════════════════════════════════════════════════════════

subheader("8c. Exponent alignment (ZFP preprocessing step)")

print("""
  ZFP aligns all values in a block to the same exponent before BPC.
  This ensures the bit-planes are comparable across values in the block.
  Step: multiply each value by 2^(max_exp - val_exp) to align exponents.
""")

block = np.array([3.14, 0.00628, 314.0, 0.314], dtype=np.float32)
print(f"Block before alignment: {block}")

def get_exponent(v: np.float32) -> int:
    packed = struct.pack('f', float(v))
    bits   = format(struct.unpack('I', packed)[0], '032b')
    return int(bits[1:9], 2) - 127

exponents  = [get_exponent(v) for v in block]
max_exp    = max(exponents)
print(f"Exponents: {exponents}  →  max_exp={max_exp}")

aligned = block * (2.0 ** (max_exp - np.array(exponents, dtype=np.float32)))
print(f"Block after alignment : {aligned}")
print(f"Now all values share the same exponent → bit-planes are meaningful across values.")

# ════════════════════════════════════════════════════════════════════════════
# 8d. BPC vs quantization — which controls error better?
# ════════════════════════════════════════════════════════════════════════════

subheader("8d. BPC vs. Linear Quantization — error comparison")

data = np.array([0.001, 0.1, 1.0, 10.0, 100.0], dtype=np.float32)
print(f"\nInput: {data}")
print(f"\n  {'method':>30}  {'max_abs_err':>12}  {'max_rel_err':>12}")

# BPC: drop 12 bits
n_drop = 12
mask   = np.uint32(0xFFFFFFFF) << np.uint32(n_drop)
bpc    = data.view(np.uint32).copy()
bpc   &= mask
bpc_out = bpc.view(np.float32)
bpc_abs = np.abs(bpc_out - data).max()
bpc_rel = (np.abs(bpc_out - data) / np.abs(data)).max() * 100
print(f"  {'BPC (drop 12 bits)':>30}  {bpc_abs:>12.4f}  {bpc_rel:>11.4f}%")

# Linear QT: abs EB = 0.001
eb  = 0.001
qt  = np.round(data.astype(np.float64) / (2*eb)).astype(np.int32) * (2*eb)
qt  = qt.astype(np.float32)
qt_abs = np.abs(qt - data).max()
qt_rel = (np.abs(qt - data) / np.abs(data)).max() * 100
print(f"  {'Linear QT (abs EB=0.001)':>30}  {qt_abs:>12.4f}  {qt_rel:>11.4f}%")

print(f"\nBPC: relative error is roughly constant (scale-invariant error).")
print(f"Linear QT: absolute error is bounded but relative error is bad for small values.")
