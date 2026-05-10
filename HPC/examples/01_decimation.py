"""
Method 1: Decimation / Sampling
─────────────────────────────────
Model  : Decimation-based
Idea   : Keep every K-th snapshot; reconstruct missing points via interpolation.
Pro    : Fastest possible compression (just skip writes).
Con    : Expensive decompression; quality depends on data smoothness.
Used by: AMR simulation codes (HACC, Flash-X), compressed sensing.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from utils import header

header("1. DECIMATION / SAMPLING")

# ── Input ────────────────────────────────────────────────────────────────────
t    = np.linspace(0, 2*np.pi, 20)
data = np.sin(t).round(4)

print("\nINPUT  (20 time-steps of sin(t)):")
print(data)

# ── Compression ──────────────────────────────────────────────────────────────
keep_every = 4
idx_kept   = np.arange(0, len(data), keep_every)
vals_kept  = data[idx_kept]

print(f"\nCOMPRESSED (keep 1 of every {keep_every} snapshots → {len(vals_kept)} values):")
print(f"  kept indices : {idx_kept}")
print(f"  kept values  : {vals_kept}")
print(f"  Compression ratio: {len(data)}/{len(vals_kept)} = {len(data)/len(vals_kept):.1f}×")

# ── Decompression ─────────────────────────────────────────────────────────────
cs    = CubicSpline(idx_kept, vals_kept)
recon = cs(np.arange(len(data)))

print(f"\nRECONSTRUCTED (cubic spline interpolation):")
print(recon.round(4))

# ── Error ─────────────────────────────────────────────────────────────────────
err = np.abs(recon - data)
print(f"\nERROR (per point):")
print(err.round(5))
print(f"  Max error : {err.max():.5f}")
print(f"  Mean error: {err.mean():.5f}")
print(f"\nNOTE: Error spikes at boundaries — cubic spline extrapolation breaks down.")
print(f"      Interior points are much more accurate ({err[2:-2].max():.5f} max).")
