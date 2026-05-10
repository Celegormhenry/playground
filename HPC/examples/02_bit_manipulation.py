"""
Method 2: Bit Manipulation
───────────────────────────
Model  : Bit-manipulation-based
Idea   : IEEE 754 floats have sign/exponent/mantissa. Drop the least significant
         mantissa bits — they contribute least to the value's accuracy.
Pro    : Extremely fast (pure bitwise ops); trivially GPU-parallelisable.
Con    : Low compression ratio — ignores inter-sample correlation.
Used by: SZx, Bit Grooming (NCO), Digit Rounding (HDF5).

IEEE 754 float32 layout (32 bits total):
  [sign 1b][exponent 8b][mantissa 23b]
   bit 31   bits 30-23   bits 22-0
"""

import numpy as np
import struct
from utils import header, subheader

header("2. BIT MANIPULATION")

# ── Input ────────────────────────────────────────────────────────────────────
data = np.array([3.14159265, 2.71828182, 1.41421356,
                 0.57721566, 1.61803398], dtype=np.float32)

print("\nINPUT (float32 values with IEEE 754 bit layout):")
print(f"  {'value':>14}  sign  exponent      mantissa")
for v in data:
    packed = struct.pack('f', v)
    bits   = format(struct.unpack('I', packed)[0], '032b')
    print(f"  {v:>14.8f}    {bits[0]}   {bits[1:9]}   {bits[9:]}")

# ── Bit Grooming ──────────────────────────────────────────────────────────────
subheader("2a. Bit Grooming — zero out N least-significant mantissa bits")

print(f"\n  {'keep bits':>10}  {'output':>50}  {'max_err':>10}  {'mean_err':>10}")
for keep in [23, 20, 16, 12, 8, 6]:
    n_drop  = 32 - keep
    mask    = np.uint32(0xFFFFFFFF) << np.uint32(n_drop)
    ui      = data.view(np.uint32).copy()
    ui     &= mask
    groomed = ui.view(np.float32)
    errs    = np.abs(groomed - data)
    print(f"  {keep:>10}  {str(groomed):>50}  {errs.max():>10.2e}  {errs.mean():>10.2e}")

# ── Digit Rounding ────────────────────────────────────────────────────────────
subheader("2b. Digit Rounding — keep N significant decimal digits")

data_dr = np.array([3.14159265, 2.71828182, 1.41421356], dtype=np.float32)
print(f"\nINPUT : {data_dr}")
print(f"\n  {'nsd':>5}  {'output':>40}  {'max_err':>10}")
for nsd in [7, 6, 4, 3, 2]:
    scale   = 10 ** nsd
    rounded = (np.round(data_dr.astype(np.float64) * scale) / scale).astype(np.float32)
    errs    = np.abs(rounded - data_dr)
    print(f"  {nsd:>5}  {str(rounded):>40}  {errs.max():>10.2e}")

print(f"\nNOTE: Bit Grooming ~ Digit Rounding, but Digit Rounding works in decimal")
print(f"      (used by NCO tools for NetCDF/climate files).")
