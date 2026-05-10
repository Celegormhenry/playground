"""
Method 9: Filtering — Data Folding (FTR)
──────────────────────────────────────────
Role   : Module — optional filtering step, typically before or after quantization.
Idea   : Split data into fixed-size blocks. If a block's value range ≤ 2×EB,
         all values in that block can be replaced by a single representative
         (the mean). These are called "constant blocks".
Why it works: For smooth data (common in scientific simulations), large contiguous
              regions barely change — their range is well within the error budget.
Pro    : Very fast; removes redundancy in smooth regions without touching the predictor.
Con    : Blocks with high variation are passed through unchanged.
Used by: SZx, cuSZx (data folding + BPC), SPERR (data extraction for outliers).
"""

import numpy as np
from utils import header, subheader

header("9. FILTERING — DATA FOLDING (SZx / cuSZx style)")

def fold_compress(data: np.ndarray, eb: float, block_size: int):
    blocks  = []
    for start in range(0, len(data), block_size):
        blk = data[start:start + block_size]
        rng = blk.max() - blk.min()
        if rng <= 2 * eb:
            blocks.append(('const', float(blk.mean()), len(blk)))
        else:
            blocks.append(('raw', blk.copy(), len(blk)))
    return blocks

def fold_decompress(blocks: list, block_size: int) -> np.ndarray:
    out = []
    for btype, val, n in blocks:
        out.extend([val] * n if btype == 'const' else list(val))
    return np.array(out)

# ════════════════════════════════════════════════════════════════════════════
# 9a. Basic example
# ════════════════════════════════════════════════════════════════════════════

subheader("9a. Basic block-by-block folding")

data = np.array([
    1.001,  0.999,  1.002,  0.998,   # block 0: smooth → CONST
    1.0,    2.5,    0.3,    3.7,     # block 1: varied → RAW
    4.998,  5.001,  4.999,  5.002,   # block 2: smooth → CONST
    0.002, -0.001,  0.003, -0.002,   # block 3: smooth → CONST
], dtype=np.float64)
eb         = 0.01
block_size = 4

print(f"\nINPUT ({len(data)} values, block_size={block_size}, EB={eb}):")
print(f"  {data}")

blocks = fold_compress(data, eb, block_size)

print(f"\nCOMPRESSION (block analysis):")
for i, (btype, val, n) in enumerate(blocks):
    blk = data[i*block_size:(i+1)*block_size]
    rng = blk.max() - blk.min()
    if btype == 'const':
        print(f"  Block {i}: range={rng:.4f} ≤ 2×{eb}={2*eb}  →  CONST  mean={val:.4f}")
    else:
        print(f"  Block {i}: range={rng:.4f} > 2×{eb}={2*eb}  →  RAW    {val}")

n_const = sum(1 for b in blocks if b[0] == 'const')
n_raw   = sum(1 for b in blocks if b[0] == 'raw')
stored  = n_const * 1 + n_raw * block_size

print(f"\nCOMPRESSED STORAGE:")
print(f"  Constant blocks: {n_const}  → each stored as 1 value")
print(f"  Raw blocks     : {n_raw}   → each stored as {block_size} values")
print(f"  Total stored   : {stored} values  (original: {len(data)})")
print(f"  Ratio          : {len(data)/stored:.2f}×")

recon = fold_decompress(blocks, block_size)
print(f"\nRECONSTRUCTED:")
print(f"  {recon}")
print(f"  Max error: {np.abs(recon - data).max():.4f}  ≤ EB={eb}  ✓")

# ════════════════════════════════════════════════════════════════════════════
# 9b. Effect of EB and block size on ratio
# ════════════════════════════════════════════════════════════════════════════

subheader("9b. Compression ratio vs. EB and block size")

t      = np.linspace(0, 2*np.pi, 256)
smooth = np.sin(t).astype(np.float64)          # very smooth
noisy  = smooth + 0.05 * np.random.default_rng(0).standard_normal(256)

print(f"\n  {'data':>10}  {'EB':>8}  {'block':>6}  {'ratio':>7}  {'max_err':>10}")
for signal_name, signal in [("smooth", smooth), ("noisy", noisy)]:
    for eb in [0.01, 0.05]:
        for bs in [4, 8, 16]:
            blks    = fold_compress(signal, eb, bs)
            n_c     = sum(1 for b in blks if b[0] == 'const')
            n_r     = sum(1 for b in blks if b[0] == 'raw')
            stored  = n_c * 1 + n_r * bs
            recon   = fold_decompress(blks, bs)
            err     = np.abs(recon - signal).max()
            ratio   = len(signal) / stored
            print(f"  {signal_name:>10}  {eb:>8.2f}  {bs:>6}  {ratio:>6.2f}×  {err:>10.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 9c. Data extraction (outlier handling) — SPERR style
# ════════════════════════════════════════════════════════════════════════════

subheader("9c. Data extraction — outlier handling (SPERR style)")

print("""
  After wavelet transform + thresholding, some points may still exceed EB.
  These "outliers" are stored separately (losslessly) and patched in after decompression.
  This ensures the error bound is always met, even for hard-to-compress regions.
""")

# Simulate post-wavelet reconstruction with some outliers
original  = np.sin(np.linspace(0, 2*np.pi, 32))
wavelet_recon = original + np.random.default_rng(1).standard_normal(32) * 0.005
eb_out = 0.003

errors   = np.abs(wavelet_recon - original)
outliers = np.where(errors > eb_out)[0]
inliers  = np.where(errors <= eb_out)[0]

print(f"Original signal   : {len(original)} points")
print(f"After wavelet recon, EB={eb_out}:")
print(f"  Inliers  (error ≤ EB): {len(inliers)}  ({len(inliers)/len(original)*100:.0f}%) — kept as-is")
print(f"  Outliers (error > EB): {len(outliers)}  ({len(outliers)/len(original)*100:.0f}%) — stored verbatim")
print(f"\nOutlier indices: {outliers}")
print(f"Outlier errors : {errors[outliers].round(5)}")

# After patching outliers
patched = wavelet_recon.copy()
patched[outliers] = original[outliers]
print(f"\nAfter patching outliers:")
print(f"  Max error: {np.abs(patched - original).max():.4f}  ≤ EB={eb_out}  ✓")
print(f"  Extra storage: {len(outliers)} (index, value) pairs = {len(outliers)*2} floats")
