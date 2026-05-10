"""
Input/Output Examples for Error-bounded Lossy Compression Methods
==================================================================
Uses only: numpy, scipy, zlib, struct (standard packages)
Run: python3 io_examples.py
"""

import numpy as np
import struct
import zlib
from collections import Counter
import heapq
from scipy.interpolate import CubicSpline

np.set_printoptions(precision=4, suppress=True, linewidth=100)

SEP  = "─" * 70
SEP2 = "═" * 70

def header(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

def subheader(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def lorenzo_1d(data: np.ndarray, eb: float):
    n = len(data)
    qcodes = np.zeros(n, dtype=np.int32)
    recon  = np.zeros(n, dtype=np.float64)
    delta  = 2.0 * eb
    for i in range(n):
        pred       = recon[i-1] if i > 0 else 0.0
        qcodes[i]  = int(np.round((data[i] - pred) / delta))
        recon[i]   = pred + qcodes[i] * delta
    return qcodes, recon

def huffman_build(symbols):
    freq = Counter(symbols)
    heap = [[w, [sym, ""]] for sym, w in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][1][0]: "0"}
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for p in lo[1:]: p[1] = "0" + p[1]
        for p in hi[1:]: p[1] = "1" + p[1]
        heapq.heappush(heap, [lo[0]+hi[0]] + lo[1:] + hi[1:])
    return {sym: code for sym, code in heap[0][1:]}


# ═════════════════════════════════════════════════════════════════════════════
# 1. DECIMATION / SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

header("1. DECIMATION / SAMPLING")

t    = np.linspace(0, 2*np.pi, 20)
data = np.sin(t).round(4)

print("\nINPUT  (20 time-steps of sin(t)):")
print(data)

# Keep every 4th point
keep_every = 4
idx_kept   = np.arange(0, len(data), keep_every)
vals_kept  = data[idx_kept]

print(f"\nCOMPRESSED (keep 1 of every {keep_every} snapshots → {len(vals_kept)} values):")
print(f"  kept indices : {idx_kept}")
print(f"  kept values  : {vals_kept}")
print(f"  Compression ratio: {len(data)}/{len(vals_kept)} = {len(data)/len(vals_kept):.1f}×")

cs    = CubicSpline(idx_kept, vals_kept)
recon = cs(np.arange(len(data)))

print(f"\nRECONSTRUCTED (cubic spline interpolation):")
print(recon.round(4))
print(f"\nERROR:")
err = np.abs(recon - data)
print(err.round(5))
print(f"  Max error : {err.max():.5f}")
print(f"  Mean error: {err.mean():.5f}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. BIT MANIPULATION
# ═════════════════════════════════════════════════════════════════════════════

header("2. BIT MANIPULATION (Bit Grooming)")

data_bm = np.array([3.14159265, 2.71828182, 1.41421356,
                     0.57721566, 1.61803398], dtype=np.float32)

print("\nINPUT (float32 values):")
for v in data_bm:
    packed = struct.pack('f', v)
    bits   = format(struct.unpack('I', packed)[0], '032b')
    print(f"  {v:>12.8f}  bits: {bits[0]} {bits[1:9]} {bits[9:]}")
    #                            sign  exponent  mantissa

for keep in [20, 12, 6]:
    mask    = np.uint32(0xFFFFFFFF) << np.uint32(32 - keep)
    ui      = data_bm.view(np.uint32).copy()
    ui     &= mask
    groomed = ui.view(np.float32)
    errs    = np.abs(groomed - data_bm)
    print(f"\nBit Grooming — keep {keep}/32 mantissa bits:")
    print(f"  output     : {groomed}")
    print(f"  max error  : {errs.max():.2e}")
    print(f"  mean error : {errs.mean():.2e}")

subheader("Digit Rounding (keep N significant decimal digits)")

data_dr = np.array([3.14159265, 2.71828182, 1.41421356], dtype=np.float32)
print(f"\nINPUT : {data_dr}")
for nsd in [6, 4, 2]:
    scale   = 10 ** nsd
    rounded = (np.round(data_dr.astype(np.float64) * scale) / scale).astype(np.float32)
    errs    = np.abs(rounded - data_dr)
    print(f"  nsd={nsd}  output={rounded}  max_err={errs.max():.2e}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. HAAR WAVELET TRANSFORM (Transformation-based)
# ═════════════════════════════════════════════════════════════════════════════

header("3. TRANSFORMATION-BASED (Haar Wavelet, 2-level)")

data_w = np.array([1.0, 1.2, 0.9, 1.1, 4.0, 3.8, 4.2, 3.9], dtype=np.float64)
print(f"\nINPUT  (8-point signal with two regions: ~1.0 and ~4.0):")
print(f"  {data_w}")

def haar_fwd(x):
    n    = len(x)
    avg  = (x[0::2] + x[1::2]) / 2.0
    diff = (x[0::2] - x[1::2]) / 2.0
    return avg, diff

def haar_inv(avg, diff):
    out       = np.empty(len(avg)*2)
    out[0::2] = avg + diff
    out[1::2] = avg - diff
    return out

avg1, diff1 = haar_fwd(data_w)
avg2, diff2 = haar_fwd(avg1)

print(f"\nCOMPRESSED (2-level Haar decomposition):")
print(f"  Level-1 avg  (4 values): {avg1}")
print(f"  Level-1 diff (4 values): {diff1}  ← near-zero for smooth regions")
print(f"  Level-2 avg  (2 values): {avg2}")
print(f"  Level-2 diff (2 values): {diff2}  ← near-zero")

# Threshold: zero out detail coefficients below 0.2
threshold = 0.2
diff1_t   = np.where(np.abs(diff1) < threshold, 0.0, diff1)
diff2_t   = np.where(np.abs(diff2) < threshold, 0.0, diff2)
kept       = np.count_nonzero(diff1_t) + np.count_nonzero(diff2_t) + len(avg2)

print(f"\nAFTER THRESHOLDING (zero out |coeff| < {threshold}):")
print(f"  diff1 (thresholded): {diff1_t}")
print(f"  diff2 (thresholded): {diff2_t}")
print(f"  Non-zero coefficients kept: {kept}/{len(data_w)}")

recon_l1 = haar_inv(avg2, diff2_t)
recon    = haar_inv(recon_l1, diff1_t)
print(f"\nRECONSTRUCTED:")
print(f"  {recon}")
print(f"  max error: {np.abs(recon - data_w).max():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. PREDICTION-BASED (Lorenzo Predictor)
# ═════════════════════════════════════════════════════════════════════════════

header("4. PREDICTION-BASED (Lorenzo Predictor)")

subheader("4a. 1D Lorenzo")

data_1d = np.array([1.00, 1.05, 1.12, 1.18, 1.25, 1.31, 1.40, 1.48,
                     1.55, 1.60], dtype=np.float64)
eb = 0.02

print(f"\nINPUT  (smooth increasing signal):")
print(f"  {data_1d}")
print(f"  Error bound (EB) = {eb}")

print(f"\nCOMPRESSION STEP-BY-STEP:")
print(f"  {'i':>3}  {'original':>10}  {'pred':>10}  {'residual':>10}  {'qcode':>6}  {'recon':>10}  {'error':>8}")
recon_1d = np.zeros_like(data_1d)
qcodes_1d = np.zeros(len(data_1d), dtype=np.int32)
delta = 2.0 * eb
for i, x in enumerate(data_1d):
    pred       = recon_1d[i-1] if i > 0 else 0.0
    residual   = x - pred
    q          = int(np.round(residual / delta))
    r          = pred + q * delta
    qcodes_1d[i] = q
    recon_1d[i]  = r
    print(f"  {i:>3}  {x:>10.4f}  {pred:>10.4f}  {residual:>10.4f}  {q:>6}  {r:>10.4f}  {abs(r-x):>8.4f}")

print(f"\nCOMPRESSED (quantization codes — these are what gets stored):")
print(f"  {qcodes_1d}")
print(f"  Unique values: {sorted(np.unique(qcodes_1d))}  ← small range → very compressible")

print(f"\nDECOMPRESSION (reconstruct from qcodes alone):")
check = np.zeros_like(data_1d)
for i in range(len(qcodes_1d)):
    pred    = check[i-1] if i > 0 else 0.0
    check[i] = pred + qcodes_1d[i] * delta
print(f"  {check}")
print(f"  Max error: {np.abs(check - data_1d).max():.4f}  ≤ EB={eb}")

# ─────────────────────────────────────────────────────────────────────────────

subheader("4b. 2D Lorenzo (surface compression)")

data_2d = np.array([[1.0, 1.1, 1.2, 1.3],
                     [1.1, 1.2, 1.3, 1.4],
                     [1.2, 1.3, 1.4, 1.5],
                     [1.3, 1.4, 1.5, 1.6]], dtype=np.float64)
eb2 = 0.02

print(f"\nINPUT (4×4 smooth 2D field):")
print(data_2d)
print(f"EB = {eb2}")

rows, cols = data_2d.shape
qcodes_2d  = np.zeros((rows, cols), dtype=np.int32)
recon_2d   = np.zeros((rows, cols))
delta2     = 2.0 * eb2

for i in range(rows):
    for j in range(cols):
        r  = recon_2d[i-1, j]   if i > 0 else 0.0
        d  = recon_2d[i,   j-1] if j > 0 else 0.0
        rd = recon_2d[i-1, j-1] if (i > 0 and j > 0) else 0.0
        pred          = r + d - rd       # 2D Lorenzo: inclusion-exclusion
        res           = data_2d[i, j] - pred
        q             = int(np.round(res / delta2))
        qcodes_2d[i,j]= q
        recon_2d[i,j]  = pred + q * delta2

print(f"\nCOMPRESSED (quantization codes):")
print(qcodes_2d)
print(f"  Unique values: {sorted(np.unique(qcodes_2d))}  ← 70%+ are 0 for smooth fields")
print(f"  Zero fraction: {(qcodes_2d==0).mean()*100:.0f}%")
print(f"\nRECONSTRUCTED:")
print(recon_2d.round(4))
print(f"Max error: {np.abs(recon_2d - data_2d).max():.4f}  ≤ EB={eb2}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. HOSVD / TUCKER DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════════════

header("5. HOSVD / TUCKER DECOMPOSITION")

# 3D tensor: 8×8×8 smooth field
x, y, z = np.meshgrid(np.linspace(0, np.pi, 8),
                       np.linspace(0, np.pi, 8),
                       np.linspace(0, np.pi, 8))
T = np.sin(x) * np.cos(y) * np.exp(-z/4)

print(f"\nINPUT tensor shape: {T.shape}  ({T.size} values, {T.nbytes} bytes)")
print(f"  Value range: [{T.min():.4f}, {T.max():.4f}]")

def tucker_compress(tensor, rank):
    G   = tensor.copy()
    Us  = []
    for mode, r in enumerate(rank):
        s    = G.shape
        unf  = np.reshape(np.moveaxis(G, mode, 0), (s[mode], -1))
        U, _, _ = np.linalg.svd(unf, full_matrices=False)
        U_r  = U[:, :r]
        G    = np.tensordot(U_r.T, G, axes=([1], [mode]))
        G    = np.moveaxis(G, 0, mode)
        Us.append(U_r)
    return G, Us

def tucker_decompress(G, Us):
    T = G.copy()
    for mode, U in enumerate(Us):
        T = np.tensordot(U, T, axes=([1], [mode]))
        T = np.moveaxis(T, 0, mode)
    return T

for rank in [(4,4,4), (3,3,3), (2,2,2)]:
    G, Us = tucker_compress(T, rank)
    Tr    = tucker_decompress(G, Us)
    orig  = T.size
    comp  = G.size + sum(U.size for U in Us)
    print(f"\n  rank={rank}:")
    print(f"    Original elements : {orig}")
    print(f"    Stored elements   : {comp} ({orig/comp:.1f}× ratio)")
    print(f"    Max abs error     : {np.abs(Tr - T).max():.4f}")
    print(f"    RMSE              : {np.sqrt(np.mean((Tr-T)**2)):.6f}")

print(f"\nCore tensor (rank 3,3,3) — first slice G[:,:,0]:")
G33, _ = tucker_compress(T, (3,3,3))
print(G33[:,:,0].round(4))


# ═════════════════════════════════════════════════════════════════════════════
# 6. QUANTIZATION TYPES
# ═════════════════════════════════════════════════════════════════════════════

header("6. QUANTIZATION TYPES")

residuals = np.array([0.0012, -0.0008, 0.0025, -0.0003, 0.0019,
                       0.0045, -0.0031, 0.0007, -0.0015, 0.0002],
                     dtype=np.float64)

print(f"\nINPUT (prediction residuals, typical of SZ output):")
print(f"  {residuals}")

subheader("6a. Linear-scale quantization (SZ1/2/3)")
eb_q  = 0.002
delta = 2.0 * eb_q
qlin  = np.round(residuals / delta).astype(np.int32)
dlin  = qlin * delta
print(f"  EB={eb_q}, bin_width={delta}")
print(f"  qcodes   : {qlin}")
print(f"  decoded  : {dlin}")
print(f"  max error: {np.abs(dlin - residuals).max():.2e}  ≤ EB={eb_q}")

subheader("6b. Log-scale quantization (NUMARCK) — for positive data")
pos_data = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
n_bins   = 8
lo, hi   = np.log10(pos_data.min()), np.log10(pos_data.max())
edges    = np.linspace(lo, hi, n_bins+1)
mids     = 0.5*(edges[:-1] + edges[1:])
qlog     = np.digitize(np.log10(pos_data), edges[1:-1])  # bin index
decoded  = 10.0 ** mids[qlog]
print(f"\nINPUT (multi-scale positive data):  {pos_data}")
print(f"  log10 range: [{lo:.1f}, {hi:.1f}]  →  {n_bins} log-spaced bins")
print(f"  bin midpoints (log10): {mids.round(3)}")
print(f"  qcodes  : {qlog}")
print(f"  decoded : {decoded.round(4)}")
rel_err = np.abs(decoded - pos_data) / pos_data
print(f"  relative errors: {rel_err.round(4)}")

subheader("6c. Multi-interval quantization (Cons-SZ) — different EB per range")
data_mi = np.array([-5.0, -4.0, -0.5, -0.1, 0.0, 0.1, 0.5, 4.0, 5.0])
print(f"\nINPUT: {data_mi}")
print("  Strategy: tight EB near zero (high QoI), loose EB far from zero")

def multi_interval_quantize(x, intervals):
    """intervals: list of (lo, hi, eb) tuples"""
    codes  = np.zeros(len(x), dtype=np.int32)
    recon  = np.zeros(len(x))
    for i, v in enumerate(x):
        for lo, hi, eb in intervals:
            if lo <= v < hi:
                delta   = 2*eb
                codes[i] = int(np.round(v / delta))
                recon[i] = codes[i] * delta
                break
    return codes, recon

intervals = [(-0.2, 0.2, 0.005),   # near zero: tight eb=0.005
             (-1.0, -0.2, 0.05),   # medium
             (0.2,  1.0,  0.05),
             (-10,  -1.0, 0.5),    # far from zero: loose eb=0.5
             (1.0,  10.0, 0.5)]

codes_mi, recon_mi = multi_interval_quantize(data_mi, intervals)
print(f"  qcodes : {codes_mi}")
print(f"  decoded: {recon_mi}")
print(f"  errors : {np.abs(recon_mi - data_mi).round(4)}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. DOMAIN TRANSFORM (Log-transform for relative error bound)
# ═════════════════════════════════════════════════════════════════════════════

header("7. DOMAIN TRANSFORM (Relative Error Bound via Log)")

data_dt = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
er      = 0.01   # 1% relative error bound

print(f"\nINPUT (multi-scale data spanning 6 orders of magnitude):")
print(f"  {data_dt}")
print(f"  Relative EB = {er*100}%")

abs_eb_log = np.log(1.0 + er)
print(f"\nSTEP 1: Log-transform data")
log_data = np.log(data_dt)
print(f"  log(data) = {log_data.round(4)}")
print(f"  Absolute EB on log domain = log(1+{er}) = {abs_eb_log:.6f}")

print(f"\nSTEP 2: Compress log_data with absolute EB = {abs_eb_log:.4f}")
qcodes_dt, recon_log = lorenzo_1d(log_data, eb=abs_eb_log)
print(f"  qcodes         : {qcodes_dt}")
print(f"  reconstructed log values: {recon_log.round(4)}")

print(f"\nSTEP 3: Exponentiate back (decompression)")
recon_dt = np.exp(recon_log)
rel_errs  = np.abs(recon_dt - data_dt) / data_dt * 100
print(f"  reconstructed  : {recon_dt.round(4)}")
print(f"  relative errors: {rel_errs.round(3)}%")
print(f"  Max relative err: {rel_errs.max():.3f}%  ≤ {er*100}%")
print(f"\nKEY INSIGHT: Without log-transform, abs EB=0.001 would fail for")
print(f"  the large values (e.g. 1000 → needs EB=1.0 for 0.1% accuracy)")
print(f"  but that EB would be too loose for small values (e.g. 0.001).")


# ═════════════════════════════════════════════════════════════════════════════
# 8. BIT-PLANE CODING (BPC)
# ═════════════════════════════════════════════════════════════════════════════

header("8. BIT-PLANE CODING (BPC)")

val = np.float32(3.14159265)
packed = struct.pack('f', val)
as_int = struct.unpack('I', packed)[0]
bits   = format(as_int, '032b')

print(f"\nINPUT: {val}")
print(f"  Binary (32 bits): {bits}")
print(f"  Sign    : bit 31  = {bits[0]}")
print(f"  Exponent: bits 30-23 = {bits[1:9]}  (= {int(bits[1:9],2)} → actual exp = {int(bits[1:9],2)-127})")
print(f"  Mantissa: bits 22-0  = {bits[9:]}")

print(f"\nBIT-PLANE TRUNCATION (dropping N least-significant bits):")
print(f"  {'N dropped':>10}  {'Result':>14}  {'Error':>12}  {'Bytes saved':>12}")
for n_drop in [0, 4, 8, 12, 16, 20]:
    mask    = np.uint32(0xFFFFFFFF) << np.uint32(n_drop)
    ui      = struct.unpack('I', packed)[0]
    ui     &= int(mask)
    truncated = struct.unpack('f', struct.pack('I', ui))[0]
    err     = abs(float(truncated) - float(val))
    saved   = n_drop / 8.0
    print(f"  {n_drop:>10}  {truncated:>14.8f}  {err:>12.6f}  {saved:>10.2f} bytes")


# ═════════════════════════════════════════════════════════════════════════════
# 9. FILTERING — DATA FOLDING
# ═════════════════════════════════════════════════════════════════════════════

header("9. FILTERING — DATA FOLDING (SZx style)")

data_f = np.array([
    # block 0: all ~1.0 (smooth → foldable)
    1.001, 0.999, 1.002, 0.998,
    # block 1: big variation (not foldable)
    1.0,   2.5,   0.3,   3.7,
    # block 2: all ~5.0 (smooth → foldable)
    4.998, 5.001, 4.999, 5.002,
    # block 3: all ~0.0 (smooth → foldable)
    0.002, -0.001, 0.003, -0.002,
], dtype=np.float64)

eb_f      = 0.01
block_size = 4

print(f"\nINPUT ({len(data_f)} values, block_size={block_size}, EB={eb_f}):")
print(f"  {data_f}")

blocks    = []
n_const   = 0
n_raw     = 0

print(f"\nCOMPRESSION:")
for b in range(len(data_f) // block_size):
    blk = data_f[b*block_size:(b+1)*block_size]
    rng = blk.max() - blk.min()
    if rng <= 2 * eb_f:
        mean_val = blk.mean()
        blocks.append(('const', mean_val))
        n_const += 1
        print(f"  Block {b}: range={rng:.4f} ≤ 2×EB={2*eb_f:.3f}  → CONSTANT block, store mean={mean_val:.4f}")
    else:
        blocks.append(('raw', blk.copy()))
        n_raw += 1
        print(f"  Block {b}: range={rng:.4f}  > 2×EB={2*eb_f:.3f}  → RAW block, store all {block_size} values")

const_storage = n_const * 1          # 1 value per constant block
raw_storage   = n_raw * block_size   # block_size values per raw block
orig_storage  = len(data_f)
print(f"\nCOMPRESSED REPRESENTATION:")
for i, b in enumerate(blocks):
    if b[0] == 'const':
        print(f"  Block {i}: [CONST] {b[1]:.4f}")
    else:
        print(f"  Block {i}: [RAW  ] {b[1]}")
print(f"\n  Original values  : {orig_storage}")
print(f"  Stored values    : {const_storage + raw_storage}  (const={const_storage}, raw={raw_storage})")
print(f"  Compression ratio: {orig_storage/(const_storage + raw_storage):.2f}×")

print(f"\nRECONSTRUCTED:")
recon_f = []
for b in blocks:
    if b[0] == 'const':
        recon_f.extend([b[1]] * block_size)
    else:
        recon_f.extend(b[1])
recon_f = np.array(recon_f)
print(f"  {recon_f}")
print(f"  Max error: {np.abs(recon_f - data_f).max():.4f}  ≤ EB={eb_f}")


# ═════════════════════════════════════════════════════════════════════════════
# 10. LOSSLESS ENCODING
# ═════════════════════════════════════════════════════════════════════════════

header("10. LOSSLESS ENCODING")

subheader("10a. Huffman Encoding")

# Simulate the qcodes you'd get from Lorenzo prediction on smooth data
# Most values are 0 or ±1 because Lorenzo predicts smooth data well
qcodes_le = np.array([0,0,0,1,0,0,-1,0,0,0,1,0,0,0,0,-1,2,0,0,0,
                       0,0,0,1,0,0,0,0,-1,0], dtype=np.int32)

print(f"\nINPUT (Lorenzo qcodes, 30 values):")
print(f"  {qcodes_le}")
print(f"  Frequency: {dict(sorted(Counter(qcodes_le.tolist()).items()))}")

table = huffman_build(qcodes_le.tolist())
print(f"\nHUFFMAN CODE TABLE:")
for sym, code in sorted(table.items(), key=lambda x: len(x[1])):
    freq = (qcodes_le == sym).sum()
    print(f"  symbol {sym:>3}  freq={freq:>2}  code={code:<8} ({len(code)} bits)")

bitstring   = ''.join(table[s] for s in qcodes_le.tolist())
naive_bits  = len(qcodes_le) * 8
huffman_bits= len(bitstring)
print(f"\n  Naive encoding (8 bits/symbol): {naive_bits} bits = {naive_bits//8} bytes")
print(f"  Huffman encoded               : {huffman_bits} bits = {huffman_bits//8} bytes")
print(f"  Savings: {(1 - huffman_bits/naive_bits)*100:.1f}%")

subheader("10b. Zlib (Dictionary Encoding — used after Huffman in SZ)")

raw_data    = qcodes_le.astype(np.int8).tobytes()
zlib_data   = zlib.compress(raw_data, level=9)
print(f"\n  Raw bytes  : {len(raw_data)}")
print(f"  Zlib bytes : {len(zlib_data)}")
print(f"  Ratio      : {len(raw_data)/len(zlib_data):.2f}×")

subheader("10c. Run-Length Encoding (cuSZp / TTHRESH style)")

rle_in  = np.array([0,0,0,0,0,3,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,1,0,0,0,0])
print(f"\nINPUT: {rle_in}")
# RLE: encode runs of zeros
rle_out = []
i = 0
while i < len(rle_in):
    if rle_in[i] == 0:
        run = 0
        while i < len(rle_in) and rle_in[i] == 0:
            run += 1; i += 1
        rle_out.append(('ZERO', run))
    else:
        rle_out.append(('VAL', rle_in[i]))
        i += 1

print(f"RLE OUTPUT: {rle_out}")
print(f"  Original symbols : {len(rle_in)}")
print(f"  RLE tokens       : {len(rle_out)}  ({len(rle_in)/len(rle_out):.1f}× reduction in tokens)")


# ═════════════════════════════════════════════════════════════════════════════
# 11. FULL END-TO-END PIPELINE (SZ3-style)
# ═════════════════════════════════════════════════════════════════════════════

header("11. FULL PIPELINE (SZ3-style: Lorenzo → Quantize → Huffman → Zlib)")

t_full    = np.linspace(0, 4*np.pi, 100)
data_full = (np.sin(t_full) + 0.3*np.cos(2*t_full)).astype(np.float64)
eb_full   = 1e-3

print(f"\nINPUT: 100-point signal  sin(t) + 0.3*cos(2t)")
print(f"  shape: {data_full.shape}   dtype: {data_full.dtype}")
print(f"  range: [{data_full.min():.4f}, {data_full.max():.4f}]")
print(f"  size : {data_full.nbytes} bytes")
print(f"  EB   : {eb_full}")

# Step 1: Lorenzo prediction
qcodes_full, recon_full = lorenzo_1d(data_full, eb_full)
print(f"\nSTEP 1 — Lorenzo prediction:")
print(f"  qcodes range: [{qcodes_full.min()}, {qcodes_full.max()}]")
print(f"  unique qcodes: {len(np.unique(qcodes_full))}")
print(f"  zero fraction: {(qcodes_full==0).mean()*100:.1f}%")

# Step 2: Huffman
h_table  = huffman_build(qcodes_full.tolist())
bitstr   = ''.join(h_table[s] for s in qcodes_full.tolist())
print(f"\nSTEP 2 — Huffman encoding:")
print(f"  Total bits: {len(bitstr)}  ({len(bitstr)/8:.0f} bytes)")
print(f"  Top 5 codes: { {k:v for k,v in sorted(h_table.items(), key=lambda x: len(x[1]))[:5]} }")

# Step 3: Zlib on the Huffman bits (pack to bytes first)
n_bytes   = (len(bitstr) + 7) // 8
bit_bytes = int(bitstr + '0'*(n_bytes*8-len(bitstr)), 2).to_bytes(n_bytes, 'big')
final_compressed = zlib.compress(bit_bytes, level=9)

print(f"\nSTEP 3 — Zlib on Huffman bytes:")
print(f"  Huffman bytes  : {n_bytes}")
print(f"  After zlib     : {len(final_compressed)}")

# Summary
print(f"\n{'─'*50}")
print(f"  COMPRESSION SUMMARY")
print(f"{'─'*50}")
print(f"  Original size      : {data_full.nbytes} bytes")
print(f"  Compressed size    : {len(final_compressed)} bytes")
print(f"  Compression ratio  : {data_full.nbytes / len(final_compressed):.2f}×")
print(f"  Max absolute error : {np.abs(recon_full - data_full).max():.2e}  ≤ EB={eb_full}")
print(f"  Error bound held   : {np.all(np.abs(recon_full - data_full) <= eb_full)}")

print(f"\n{'═'*70}")
print(f"  All examples complete.")
print(f"{'═'*70}\n")
