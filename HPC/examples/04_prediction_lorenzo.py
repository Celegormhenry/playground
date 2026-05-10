"""
Method 4: Prediction-based (Lorenzo Predictor)
───────────────────────────────────────────────
Model  : Prediction-based  ← most common SOTA model
Idea   : Predict each value from already-reconstructed neighbors.
         Quantize the (small) residual. Encode the integer codes losslessly.
Pro    : High compression ratio, tunable error bound.
Con    : Sequential prediction; harder to parallelise naively.
Used by: SZ1, SZ2, SZ3, QoZ, FAZ, FPZIP, cuSZ, cuSZp, MDZ, ...

Two critical constraints:
  1. Reconstructed-Data-Driven Policy:
       Always predict from the RECONSTRUCTED value (not original),
       because decompression only has reconstructed values.
  2. Recoverable Recursive-Scanning Policy:
       Scan order must allow every point to be reconstructed sequentially.
"""

import numpy as np
from utils import header, subheader

header("4. PREDICTION-BASED (Lorenzo Predictor)")

# ════════════════════════════════════════════════════════════════════════════
# 4a. 1D Lorenzo
# ════════════════════════════════════════════════════════════════════════════

subheader("4a. 1D Lorenzo  —  pred[i] = recon[i-1]")

data = np.array([1.00, 1.05, 1.12, 1.18, 1.25, 1.31, 1.40, 1.48,
                 1.55, 1.60], dtype=np.float64)
eb    = 0.02
delta = 2.0 * eb

print(f"\nINPUT  (smooth increasing signal):")
print(f"  {data}")
print(f"  EB = {eb}   bin_width = 2×EB = {delta}")

# ── Compression step-by-step ──────────────────────────────────────────────────
print(f"\nCOMPRESSION (step-by-step):")
print(f"  {'i':>3}  {'original':>10}  {'pred':>10}  {'residual':>10}  {'qcode':>6}  {'recon':>10}  {'error':>8}")

recon_1d   = np.zeros_like(data)
qcodes_1d  = np.zeros(len(data), dtype=np.int32)

for i, x in enumerate(data):
    pred         = recon_1d[i-1] if i > 0 else 0.0   # ← use RECON, not original
    residual     = x - pred
    q            = int(np.round(residual / delta))
    r            = pred + q * delta
    qcodes_1d[i] = q
    recon_1d[i]  = r
    print(f"  {i:>3}  {x:>10.4f}  {pred:>10.4f}  {residual:>10.4f}  {q:>6}  {r:>10.4f}  {abs(r-x):>8.4f}")

print(f"\nCOMPRESSED  (only these qcodes are stored):")
print(f"  {qcodes_1d}")
print(f"  Unique symbols : {sorted(np.unique(qcodes_1d))}  ← small alphabet → compressible")
print(f"  Storage        : {len(qcodes_1d)} integers  (vs {len(data)*8} bytes raw)")

# ── Decompression ─────────────────────────────────────────────────────────────
print(f"\nDECOMPRESSION (from qcodes only — no original data needed):")
check = np.zeros_like(data)
for i in range(len(qcodes_1d)):
    pred     = check[i-1] if i > 0 else 0.0
    check[i] = pred + qcodes_1d[i] * delta
print(f"  {check}")
print(f"  Max error : {np.abs(check - data).max():.4f}  ≤ EB={eb}  ✓")

# ════════════════════════════════════════════════════════════════════════════
# 4b. 2D Lorenzo
# ════════════════════════════════════════════════════════════════════════════

subheader("4b. 2D Lorenzo  —  pred[i,j] = recon[i-1,j] + recon[i,j-1] - recon[i-1,j-1]")

print("""
  Inclusion-exclusion on the three already-reconstructed neighbors:

    recon[i-1,j-1]  recon[i-1,j]
                        ↑
    recon[i,  j-1] →  data[i,j]  ← to predict

  pred = right + below_right - corner  (cancels double-counting)
""")

data_2d = np.array([[1.0, 1.1, 1.2, 1.3],
                     [1.1, 1.2, 1.3, 1.4],
                     [1.2, 1.3, 1.4, 1.5],
                     [1.3, 1.4, 1.5, 1.6]], dtype=np.float64)
eb2    = 0.02
delta2 = 2.0 * eb2

print(f"INPUT (4×4 smooth gradient field):")
print(data_2d)
print(f"EB = {eb2}")

rows, cols = data_2d.shape
qcodes_2d  = np.zeros((rows, cols), dtype=np.int32)
recon_2d   = np.zeros((rows, cols))

for i in range(rows):
    for j in range(cols):
        r    = recon_2d[i-1, j]   if i > 0 else 0.0
        d    = recon_2d[i,   j-1] if j > 0 else 0.0
        rd   = recon_2d[i-1, j-1] if (i > 0 and j > 0) else 0.0
        pred = r + d - rd
        res  = data_2d[i, j] - pred
        q    = int(np.round(res / delta2))
        qcodes_2d[i, j] = q
        recon_2d[i, j]  = pred + q * delta2

print(f"\nCOMPRESSED (quantization codes):")
print(qcodes_2d)
print(f"  Unique values  : {sorted(np.unique(qcodes_2d))}")
print(f"  Zero fraction  : {(qcodes_2d==0).mean()*100:.0f}%  ← smooth field → mostly zeros")

print(f"\nRECONSTRUCTED:")
print(recon_2d.round(4))
print(f"Max error: {np.abs(recon_2d - data_2d).max():.4f}  ≤ EB={eb2}  ✓")

# ════════════════════════════════════════════════════════════════════════════
# 4c. Predictor comparison
# ════════════════════════════════════════════════════════════════════════════

subheader("4c. Predictor comparison on a harder signal")

t     = np.linspace(0, 2*np.pi, 16)
data3 = np.sin(t).astype(np.float64)
eb3   = 0.01

print(f"\nINPUT: sin(t), 16 points, EB={eb3}")
print(f"  {data3.round(4)}")

# Lorenzo (1 neighbor)
def compress_and_measure(data, eb, predictor_name, predictor_fn):
    delta  = 2.0 * eb
    recon  = np.zeros_like(data)
    qcodes = np.zeros(len(data), dtype=np.int32)
    for i in range(len(data)):
        pred      = predictor_fn(recon, i)
        qcodes[i] = int(np.round((data[i] - pred) / delta))
        recon[i]  = pred + qcodes[i] * delta
    n_unique = len(np.unique(qcodes))
    max_err  = np.abs(recon - data).max()
    print(f"  {predictor_name:<25}  unique_codes={n_unique:>3}  max_err={max_err:.4f}")

compress_and_measure(data3, eb3, "Lorenzo (1 neighbor)",
                     lambda r, i: r[i-1] if i > 0 else 0.0)
compress_and_measure(data3, eb3, "Linear extrap (2 neighbors)",
                     lambda r, i: 2*r[i-1]-r[i-2] if i > 1 else (r[i-1] if i > 0 else 0.0))
compress_and_measure(data3, eb3, "Quadratic (3 neighbors)",
                     lambda r, i: 3*r[i-1]-3*r[i-2]+r[i-3] if i > 2 else
                                  (2*r[i-1]-r[i-2] if i > 1 else
                                   (r[i-1] if i > 0 else 0.0)))

print(f"\nFewer unique codes → smaller Huffman tree → better lossless compression.")
