"""
Method 3: Transformation-based (Haar Wavelet)
──────────────────────────────────────────────
Model  : Transformation-based
Idea   : Apply a linear transform to decorrelate data. Transformed coefficients
         are sparser and easier to compress. Apply thresholding / BPC there.
Pro    : High reconstruction quality; GPU-friendly (matrix multiply).
Con    : Hard to control pointwise error precisely; transform has a cost.
Used by: ZFP (near-orthogonal block transform), SPERR (CDF9/7), FAZ (Sym13).

Haar wavelet — the simplest wavelet:
  forward:  avg[i] = (x[2i] + x[2i+1]) / 2     (low-frequency)
            diff[i]= (x[2i] - x[2i+1]) / 2     (high-frequency detail)
  inverse:  x[2i]  = avg[i] + diff[i]
            x[2i+1]= avg[i] - diff[i]
"""

import numpy as np
from utils import header, subheader

header("3. TRANSFORMATION-BASED (Haar Wavelet, 2-level)")

def haar_fwd(x: np.ndarray):
    avg  = (x[0::2] + x[1::2]) / 2.0
    diff = (x[0::2] - x[1::2]) / 2.0
    return avg, diff

def haar_inv(avg: np.ndarray, diff: np.ndarray):
    out       = np.empty(len(avg) * 2)
    out[0::2] = avg + diff
    out[1::2] = avg - diff
    return out

# ── Input ─────────────────────────────────────────────────────────────────────
data = np.array([1.0, 1.2, 0.9, 1.1, 4.0, 3.8, 4.2, 3.9], dtype=np.float64)
print(f"\nINPUT  (8-point signal: two smooth regions ~1.0 and ~4.0):")
print(f"  {data}")

# ── Forward transform (2 levels) ──────────────────────────────────────────────
avg1, diff1 = haar_fwd(data)
avg2, diff2 = haar_fwd(avg1)

print(f"\nFORWARD TRANSFORM (2-level Haar decomposition):")
print(f"  Level-1 avg  (4 values) : {avg1}")
print(f"  Level-1 diff (4 values) : {diff1}  ← small for smooth data")
print(f"  Level-2 avg  (2 values) : {avg2}   ← coarse representation")
print(f"  Level-2 diff (2 values) : {diff2}   ← small for smooth data")

# ── Thresholding (compression step) ───────────────────────────────────────────
subheader("Thresholding — zero out detail coefficients below threshold")

for threshold in [0.05, 0.2, 0.5]:
    d1_t  = np.where(np.abs(diff1) < threshold, 0.0, diff1)
    d2_t  = np.where(np.abs(diff2) < threshold, 0.0, diff2)
    kept  = np.count_nonzero(d1_t) + np.count_nonzero(d2_t) + len(avg2)

    recon = haar_inv(haar_inv(avg2, d2_t), d1_t)
    err   = np.abs(recon - data)

    print(f"\n  threshold={threshold}:")
    print(f"    diff1 kept : {d1_t}  ({np.count_nonzero(d1_t)}/4 non-zero)")
    print(f"    diff2 kept : {d2_t}  ({np.count_nonzero(d2_t)}/2 non-zero)")
    print(f"    coeffs kept: {kept}/{len(data)}  ({kept/len(data)*100:.0f}%)")
    print(f"    recon      : {recon}")
    print(f"    max error  : {err.max():.4f}")

# ── Perfect reconstruction (no threshold) ─────────────────────────────────────
subheader("Perfect reconstruction (threshold=0, lossless)")
recon_perfect = haar_inv(haar_inv(avg2, diff2), diff1)
print(f"\nRECONSTRUCTED: {recon_perfect}")
print(f"Max error     : {np.abs(recon_perfect - data).max():.2e}  (machine precision)")
