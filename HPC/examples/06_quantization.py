"""
Method 6: Quantization (QT)
────────────────────────────
Role   : Module (used by almost every compressor after the prediction/transform step)
Idea   : Map continuous residuals to discrete integer codes (bins).
         The bin width determines the error bound.
         Smaller bins → tighter error → more codes → less compressible.
Types  : Linear-scale, Log-scale, Vector, Multi-interval.
Used by: SZ (linear), NUMARCK (log), MDZ (vector), Cons-SZ (multi-interval).
"""

import numpy as np
from utils import header, subheader

header("6. QUANTIZATION TYPES")

# ════════════════════════════════════════════════════════════════════════════
# 6a. Linear-scale quantization
# ════════════════════════════════════════════════════════════════════════════

subheader("6a. Linear-scale quantization (SZ1/2/3)")

print("""
  Uniform bins of width 2×EB.
  qcode = round(residual / (2×EB))
  recon  = qcode × (2×EB)
  Guarantees |recon - residual| ≤ EB.
""")

residuals = np.array([0.0012, -0.0008, 0.0025, -0.0003, 0.0019,
                      0.0045, -0.0031,  0.0007, -0.0015,  0.0002])
print(f"INPUT (prediction residuals): {residuals}")

for eb in [0.004, 0.002, 0.001]:
    delta   = 2.0 * eb
    qcodes  = np.round(residuals / delta).astype(np.int32)
    decoded = qcodes * delta
    err     = np.abs(decoded - residuals)
    print(f"\n  EB={eb}  bin_width={delta}")
    print(f"  qcodes : {qcodes}")
    print(f"  decoded: {decoded.round(5)}")
    print(f"  errors : {err.round(5)}")
    print(f"  max_err: {err.max():.2e}  ≤ EB={eb}  ✓   unique_codes={len(np.unique(qcodes))}")

# ════════════════════════════════════════════════════════════════════════════
# 6b. Log-scale quantization
# ════════════════════════════════════════════════════════════════════════════

subheader("6b. Log-scale quantization (NUMARCK)")

print("""
  Variable-width bins: smaller near zero, larger at extremes.
  Advantage: balanced histogram for heavy-tailed data.
  Useful when data spans many orders of magnitude.
""")

pos_data = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
n_bins   = 8
lo, hi   = np.log10(pos_data.min()), np.log10(pos_data.max())
edges    = np.linspace(lo, hi, n_bins + 1)
mids     = 0.5 * (edges[:-1] + edges[1:])
qcodes   = np.digitize(np.log10(pos_data), edges[1:-1])
decoded  = 10.0 ** mids[qcodes]
rel_err  = np.abs(decoded - pos_data) / pos_data

print(f"INPUT  : {pos_data}  (spans 6 orders of magnitude)")
print(f"\nLog10 bin edges : {edges.round(2)}")
print(f"Bin midpoints   : {mids.round(3)}")
print(f"\n  {'value':>8}  {'qcode':>6}  {'decoded':>10}  {'rel_err':>10}")
for v, q, d, e in zip(pos_data, qcodes, decoded, rel_err):
    print(f"  {v:>8.3f}  {q:>6}  {d:>10.4f}  {e*100:>9.2f}%")

# ════════════════════════════════════════════════════════════════════════════
# 6c. Multi-interval quantization
# ════════════════════════════════════════════════════════════════════════════

subheader("6c. Multi-interval quantization (Cons-SZ)")

print("""
  Different error bounds for different value ranges.
  Useful when the user cares more about certain value ranges (QoI-based).
  E.g. climate scientists care more about values near freezing point (0°C).
""")

data_mi = np.array([-5.0, -4.0, -0.5, -0.1, 0.0, 0.1, 0.5, 4.0, 5.0])

intervals = [
    (-0.2,  0.2,  0.005),  # near zero: tight EB=0.005
    (-1.0, -0.2,  0.05),   # medium negative
    ( 0.2,  1.0,  0.05),   # medium positive
    (-10., -1.0,  0.5),    # far negative: loose EB=0.5
    ( 1.0, 10.0,  0.5),    # far positive: loose EB=0.5
]

print(f"INPUT : {data_mi}")
print(f"\nInterval rules:")
for lo, hi, eb in intervals:
    print(f"  [{lo:>5}, {hi:>5})  →  EB={eb}")

qcodes_mi = np.zeros(len(data_mi), dtype=np.int32)
recon_mi  = np.zeros(len(data_mi))
applied_eb = np.zeros(len(data_mi))

for i, v in enumerate(data_mi):
    for lo, hi, eb in intervals:
        if lo <= v < hi:
            delta          = 2 * eb
            qcodes_mi[i]   = int(np.round(v / delta))
            recon_mi[i]    = qcodes_mi[i] * delta
            applied_eb[i]  = eb
            break

print(f"\n  {'value':>6}  {'EB used':>8}  {'qcode':>6}  {'decoded':>8}  {'error':>8}")
for v, eb, q, d in zip(data_mi, applied_eb, qcodes_mi, recon_mi):
    print(f"  {v:>6.2f}  {eb:>8.3f}  {q:>6}  {d:>8.4f}  {abs(d-v):>8.4f}")
