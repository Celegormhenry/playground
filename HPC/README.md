# Error-bounded Lossy Compression for Scientific Data

A hands-on companion to:
> *A Survey on Error-bounded Lossy Compression for Scientific Datasets*
> Di et al., ACM Computing Surveys, Vol. 57, No. 11, Article 287 (June 2025)
> https://doi.org/10.1145/3733104

## Why This Exists

Scientific simulations produce data at rates that dwarf storage bandwidth:
- Quantum state of a 50-qubit system → **16 PB**
- APS/LCLS X-ray detectors → **250 GB/s**
- A single MD trajectory → **260 TB**

Lossless compression (gzip, zstd) gives ~2–5× on floating-point scientific data.
That is not enough. Error-bounded lossy compression routinely achieves **10×–1000×**
while **mathematically guaranteeing** the reconstruction error per data point stays within
a user-specified bound ε.

```
|reconstructed_value − original_value| ≤ ε    (absolute error bound)
|reconstructed_value − original_value| / |original_value| ≤ εᵣ   (relative error bound)
```

---

## Table of Contents

1. [The 6 Compression Models](#1-the-6-compression-models)
2. [The 10 Modular Techniques](#2-the-10-modular-techniques)
3. [Major Compressor Families](#3-major-compressor-families)
4. [Application-Specific Examples](#4-application-specific-examples)
5. [How to Choose a Compressor](#5-how-to-choose-a-compressor)
6. [Dependencies](#6-dependencies)

---

## 1. The 6 Compression Models

```
Speed / Lower Quality  ←───────────────────────────→  Slower / Higher Quality
  Decimation  Bit-manip  Transform  Prediction  HOSVD   Deep-Learning
```

Each model is a compression strategy. In practice, compressors combine multiple models.

---

### Model 1: Decimation / Filtering-based

**Idea:** Throw away data points (temporal or spatial), reconstruct at decompression via interpolation.

**Pro:** Fastest possible compression (just skip writes).
**Con:** Expensive decompression; interpolation quality depends on smoothness.

```python
import numpy as np
from scipy.interpolate import interp1d

def decimate_compress(data: np.ndarray, keep_every: int = 4):
    """Temporal decimation: keep 1 out of every `keep_every` snapshots."""
    n = len(data)
    kept_indices = np.arange(0, n, keep_every)
    kept_values  = data[kept_indices]
    return kept_indices, kept_values

def decimate_decompress(kept_indices, kept_values, original_length: int):
    """Reconstruct via cubic interpolation."""
    all_indices = np.arange(original_length)
    interp = interp1d(kept_indices, kept_values, kind='cubic',
                      fill_value='extrapolate')
    return interp(all_indices)

# --- demo ---
np.random.seed(0)
t = np.linspace(0, 4 * np.pi, 1000)
signal = np.sin(t) + 0.1 * np.random.randn(1000)   # smooth signal with noise

idx, vals = decimate_compress(signal, keep_every=8)
reconstructed = decimate_decompress(idx, vals, len(signal))

cr   = len(signal) / len(vals)
maxe = np.max(np.abs(reconstructed - signal))
print(f"Compression ratio : {cr:.1f}×")
print(f"Max absolute error: {maxe:.4f}")
# → Compression ratio : 8.0×   Max absolute error: ~0.05 (depends on smoothness)
```

---

### Model 2: Bit-manipulation-based

**Idea:** IEEE 754 floats are stored as bits. The least-significant mantissa bits contribute
least to value accuracy. Truncate them.

```
float32 layout (32 bits):
  [sign 1b][exponent 8b][mantissa 23b]
   ^                        ^
   most significant     least significant
```

**Pro:** Extremely fast — pure bitwise ops, trivially GPU-parallelisable.
**Con:** Low compression ratio because no inter-sample correlation is exploited.

```python
import numpy as np
import struct

def bit_grooming(data: np.ndarray, n_significant_bits: int) -> np.ndarray:
    """
    Retain only the `n_significant_bits` most significant mantissa bits.
    Remaining bits are zeroed out (groomed).
    float32 mantissa = 23 bits.
    """
    assert data.dtype == np.float32
    mask_bits = 23 - n_significant_bits          # bits to zero
    mask = np.uint32(0xFFFFFFFF << mask_bits)    # keep top bits

    # reinterpret float memory as uint32, apply mask, reinterpret back
    as_uint = data.view(np.uint32).copy()
    as_uint &= mask
    return as_uint.view(np.float32)

def digit_rounding(data: np.ndarray, n_decimal_digits: int) -> np.ndarray:
    """
    Round to `n_decimal_digits` significant decimal digits,
    then return as float32.
    """
    scale = 10 ** n_decimal_digits
    return np.round(data * scale).astype(np.float32) / scale

# --- demo ---
rng  = np.random.default_rng(42)
data = rng.uniform(-100.0, 100.0, 10).astype(np.float32)

groomed   = bit_grooming(data, n_significant_bits=10)   # keep 10 of 23 mantissa bits
rounded   = digit_rounding(data, n_decimal_digits=4)

print("Original  :", data)
print("Bit-groomed:", groomed)
print("Digit-round:", rounded)
print(f"Max error (grooming) : {np.max(np.abs(groomed - data)):.6f}")
print(f"Max error (rounding) : {np.max(np.abs(rounded - data)):.6f}")
```

---

### Model 3: Transformation-based

**Idea:** Apply a linear transform (wavelet, DCT) to decorrelate the data.
The transformed coefficients are sparser and easier to compress.
Apply BPC/quantization in the transformed domain.

**Pro:** High quality; GPU-friendly (matrix multiply).
**Con:** Hard to control pointwise error precisely; transform cost.

```python
import numpy as np
import pywt   # pip install PyWavelets

def wavelet_compress_1d(data: np.ndarray, wavelet='db4', level=3,
                        threshold_ratio=0.05):
    """
    Threshold wavelet coefficients below `threshold_ratio * max_coeff`.
    This mimics what SPERR/FAZ do in their SPECK encoding stage.
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # global threshold
    all_coeff = np.concatenate([c.ravel() for c in coeffs])
    threshold  = threshold_ratio * np.max(np.abs(all_coeff))

    coeffs_thresh = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]

    # count non-zero coefficients (these are what gets stored)
    nnz_ratio = np.count_nonzero(np.concatenate([c.ravel() for c in coeffs_thresh])) \
                / data.size
    reconstructed = pywt.waverec(coeffs_thresh, wavelet)[:len(data)]
    return reconstructed, nnz_ratio

# --- demo ---
t    = np.linspace(0, 8*np.pi, 2048)
data = (np.sin(t) + 0.5*np.sin(3*t) + 0.2*np.sin(10*t)).astype(np.float64)

recon, nnz = wavelet_compress_1d(data, threshold_ratio=0.02)
print(f"Non-zero coefficients kept : {nnz*100:.1f}%")
print(f"Max absolute error         : {np.max(np.abs(recon - data)):.6f}")
print(f"PSNR                       : {20*np.log10(data.ptp()/np.sqrt(np.mean((recon-data)**2))):.1f} dB")
```

---

### Model 4: Prediction-based (most common SOTA)

**Idea:** Predict each value from its already-reconstructed neighbors.
Compute the residual (error). Quantize it. Encode the (mostly tiny) integers.

This is the backbone of SZ1, SZ2, SZ3, FPZIP, cuSZ, and many others.

**Two critical rules** (from the paper):
1. **Reconstructed-Data-Driven Policy**: always predict from the *reconstructed* value,
   not the original — because decompression only has reconstructed values.
2. **Recoverable Recursive-Scanning Policy**: the scan order must allow all points to be
   reconstructed one by one.

```python
import numpy as np

def lorenzo_1d_compress(data: np.ndarray, eb: float):
    """
    1D Lorenzo predictor: pred[i] = reconstructed[i-1]
    Returns quantization codes and the reconstructed array.
    """
    n    = len(data)
    qcodes = np.zeros(n, dtype=np.int32)
    recon  = np.zeros(n, dtype=data.dtype)
    delta  = 2.0 * eb   # quantization bin width

    for i in range(n):
        pred       = recon[i-1] if i > 0 else 0.0
        residual   = data[i] - pred
        qcodes[i]  = int(np.round(residual / delta))
        recon[i]   = pred + qcodes[i] * delta           # RECONSTRUCTED value

    return qcodes, recon

def lorenzo_1d_decompress(qcodes: np.ndarray, eb: float):
    """Reconstruct from quantization codes."""
    delta = 2.0 * eb
    recon = np.zeros(len(qcodes))
    for i in range(len(qcodes)):
        pred     = recon[i-1] if i > 0 else 0.0
        recon[i] = pred + qcodes[i] * delta
    return recon

def lorenzo_2d_compress(data: np.ndarray, eb: float):
    """
    2D Lorenzo predictor: pred[i,j] = recon[i-1,j] + recon[i,j-1] - recon[i-1,j-1]
    (inclusion-exclusion on the three already-reconstructed neighbors)
    """
    rows, cols = data.shape
    qcodes = np.zeros_like(data, dtype=np.int32)
    recon  = np.zeros_like(data)
    delta  = 2.0 * eb

    for i in range(rows):
        for j in range(cols):
            r  = recon[i-1, j]   if i > 0 else 0.0
            d  = recon[i,   j-1] if j > 0 else 0.0
            rd = recon[i-1, j-1] if (i > 0 and j > 0) else 0.0
            pred = r + d - rd
            residual     = data[i, j] - pred
            qcodes[i, j] = int(np.round(residual / delta))
            recon[i, j]  = pred + qcodes[i, j] * delta

    return qcodes, recon

# --- demo 1D ---
np.random.seed(7)
t    = np.linspace(0, 2*np.pi, 500)
data = np.sin(t).astype(np.float64)
eb   = 1e-3

qcodes, recon = lorenzo_1d_compress(data, eb)
check         = lorenzo_1d_decompress(qcodes, eb)

print("=== 1D Lorenzo ===")
print(f"Error bound        : {eb}")
print(f"Max absolute error : {np.max(np.abs(recon - data)):.2e}  (should be ≤ {eb})")
print(f"Entropy of qcodes  : {len(np.unique(qcodes))} unique symbols (small → high compressibility)")

# --- demo 2D ---
x, y  = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, 2*np.pi, 100))
data2 = (np.sin(x) * np.cos(y)).astype(np.float64)

qcodes2, recon2 = lorenzo_2d_compress(data2, eb=1e-3)
print("\n=== 2D Lorenzo ===")
print(f"Max absolute error  : {np.max(np.abs(recon2 - data2)):.2e}")
print(f"Fraction of qcodes == 0: {np.mean(qcodes2 == 0)*100:.1f}%  (these compress to near-zero bytes)")
```

---

### Model 5: HOSVD-based (Tucker Decomposition)

**Idea:** A multi-dimensional SVD. Decompose the data tensor T into a small core G
and factor matrices {U₁, U₂, …, Uₙ}. Approximating G at low precision gives high compression.

**Pro:** Captures global structure across all dimensions simultaneously.
**Con:** O(n⁴) for 3D data; does not support pointwise error control (only L² error).

```python
import numpy as np

def hosvd_compress(tensor: np.ndarray, rank: tuple):
    """
    Tucker decomposition via sequential SVD (HOSVD).
    `rank` is the target rank per mode, e.g. (10, 10, 5).
    Returns core tensor G and factor matrices Us.
    """
    assert tensor.ndim == len(rank), "rank must have one entry per dimension"
    G = tensor.copy().astype(np.float64)
    Us = []

    for mode, r in enumerate(rank):
        # unfold tensor along this mode
        shape = G.shape
        n_mode = shape[mode]
        unfolded = np.reshape(np.moveaxis(G, mode, 0), (n_mode, -1))

        # truncated SVD
        U, s, Vt = np.linalg.svd(unfolded, full_matrices=False)
        U_r = U[:, :r]   # keep r left singular vectors

        # project mode onto subspace
        G = np.tensordot(U_r.T, G, axes=([1], [mode]))
        G = np.moveaxis(G, 0, mode)
        Us.append(U_r)

    return G, Us

def hosvd_decompress(G: np.ndarray, Us: list):
    """Reconstruct from core tensor and factor matrices."""
    T = G.copy()
    for mode, U in enumerate(Us):
        T = np.tensordot(U, T, axes=([1], [mode]))
        T = np.moveaxis(T, 0, mode)
    return T

# --- demo ---
rng    = np.random.default_rng(0)
# Simulate a smooth 3D field (e.g. velocity component)
shape  = (32, 32, 32)
tensor = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            tensor[i,j,k] = np.sin(i/4) * np.cos(j/4) * np.exp(-k/16)

rank   = (8, 8, 4)
G, Us  = hosvd_compress(tensor, rank=rank)
recon  = hosvd_decompress(G, Us)

orig_size  = tensor.size
compressed = G.size + sum(U.size for U in Us)
print(f"Original elements  : {orig_size}")
print(f"Compressed elements: {compressed}  ({orig_size/compressed:.1f}× ratio)")
print(f"Max absolute error : {np.max(np.abs(recon - tensor)):.4f}")
print(f"RMSE               : {np.sqrt(np.mean((recon-tensor)**2)):.6f}")
```

---

### Model 6: Deep-Learning-based

**Idea:** Train an autoencoder to map data → latent vector → data.
The latent vector is the compressed representation.
Post-encode: quantize the latent vector; clip values that exceed the error bound.

**Pro:** Can learn highly non-linear correlations; potential for very high ratios.
**Con:** Training is expensive; inference is slow; errors are hard to bound a priori.

```python
# Requires: pip install torch

import numpy as np
import torch
import torch.nn as nn

class ScientificAutoencoder(nn.Module):
    """
    Minimal autoencoder for 1D scientific data segments.
    Real compressors (HAE, AE-SZ) use hierarchical or convolutional variants.
    """
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def ae_compress(model, data_segment: np.ndarray, eb: float):
    """
    Compress one segment with the AE.
    After encoding, outliers (points where error > eb) are stored verbatim.
    This is the key trick in AE-SZ.
    """
    x = torch.FloatTensor(data_segment).unsqueeze(0)
    with torch.no_grad():
        recon, z = model(x)
    recon_np = recon.squeeze(0).numpy()

    # identify outliers that violate the error bound
    errors   = np.abs(recon_np - data_segment)
    outliers = np.where(errors > eb)[0]

    return z.squeeze(0).numpy(), outliers, data_segment[outliers]

# --- demo ---
torch.manual_seed(0)
seg_len = 64
model   = ScientificAutoencoder(input_dim=seg_len, latent_dim=8)

# simulate a smooth data segment
t       = np.linspace(0, 2*np.pi, seg_len).astype(np.float32)
segment = np.sin(t) + 0.05 * np.random.randn(seg_len).astype(np.float32)

# (in reality the model would be trained first)
z, outlier_idx, outlier_vals = ae_compress(model, segment, eb=0.1)

print(f"Latent vector size : {z.shape[0]}  ({seg_len/z.shape[0]:.0f}× before outlier correction)")
print(f"Outliers to store  : {len(outlier_idx)} / {seg_len}")
print("Note: a real AE compressor trains on the dataset first, then compresses.")
```

---

## 2. The 10 Modular Techniques

### 2.1 Point-wise Data Prediction (PDP)

Already shown in Model 4 above. Summary of predictor families:

```python
import numpy as np

def predict_linear_regression(history: np.ndarray) -> float:
    """
    Linear regression predictor: fit a line through k previous
    reconstructed values, predict the next.
    Used in SZ2 (k=216 neighbors in 3D).
    """
    k = len(history)
    if k < 2:
        return history[-1] if k == 1 else 0.0
    x = np.arange(k, dtype=float)
    slope, intercept = np.polyfit(x, history, 1)
    return slope * k + intercept

def predict_spline(history: np.ndarray, positions: np.ndarray,
                   next_pos: float) -> float:
    """
    Cubic spline interpolation predictor.
    Used in SZ3 (multidimensional spline).
    """
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(positions, history)
    return float(cs(next_pos))

# comparison
data   = np.sin(np.linspace(0, np.pi, 20))
actual = np.sin(np.linspace(0, np.pi, 21))[-1]

pred_lorenzo = data[-1]                              # trivial Lorenzo
pred_linreg  = predict_linear_regression(data[-5:])  # linear regression
pred_spline  = predict_spline(data[-8:],
                               np.arange(8, dtype=float), 8.0)

print(f"True value      : {actual:.6f}")
print(f"Lorenzo pred    : {pred_lorenzo:.6f}  err={abs(pred_lorenzo-actual):.6f}")
print(f"LinReg pred     : {pred_linreg:.6f}  err={abs(pred_linreg-actual):.6f}")
print(f"Spline pred     : {pred_spline:.6f}  err={abs(pred_spline-actual):.6f}")
```

---

### 2.2 Quantization (QT)

```python
import numpy as np

def linear_quantize(residuals: np.ndarray, eb: float):
    """
    Linear-scale quantization: uniform bins of width 2*eb.
    Used in SZ1/2/3.
    """
    delta   = 2.0 * eb
    qcodes  = np.round(residuals / delta).astype(np.int32)
    decoded = qcodes * delta
    return qcodes, decoded

def log_scale_quantize(data: np.ndarray, n_bins: int):
    """
    Log-scale quantization: smaller bins near zero, larger near extremes.
    Provides balanced histogram — useful when distribution is heavy-tailed.
    Used in NUMARCK.
    """
    assert np.all(data > 0), "log quantization requires positive data"
    log_data = np.log(data)
    lo, hi   = log_data.min(), log_data.max()
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    qcodes    = np.digitize(log_data, bin_edges[1:-1])   # 0-indexed bin
    # represent each bin by its midpoint in log space
    midpoints  = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    decoded    = np.exp(midpoints[qcodes])
    return qcodes, decoded

def vector_quantize(data: np.ndarray, n_clusters: int):
    """
    Vector quantization via k-means: each data point is represented
    by its nearest cluster centroid.
    Used in MDZ and NUMARCK.
    """
    from sklearn.cluster import KMeans
    km      = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels  = km.fit_predict(data.reshape(-1, 1))
    decoded = km.cluster_centers_[labels, 0]
    return labels, decoded

# --- demo ---
rng       = np.random.default_rng(1)
residuals = rng.normal(0, 0.01, 1000)  # typical SZ residuals (small, near-zero)

qcodes_lin, dec_lin = linear_quantize(residuals, eb=1e-3)
print(f"Linear QT  — unique codes: {len(np.unique(qcodes_lin)):4d}  max err: {np.max(np.abs(dec_lin-residuals)):.2e}")

positive_data = np.abs(rng.normal(1.0, 0.5, 1000)) + 0.01
qcodes_log, dec_log = log_scale_quantize(positive_data, n_bins=256)
print(f"Log-scale  — unique codes: {len(np.unique(qcodes_log)):4d}  max err: {np.max(np.abs(dec_log-positive_data)):.4f}")
```

---

### 2.3 Orthogonal/Wavelet Transform (OWT/DWT)

```python
import numpy as np
import pywt

def haar_transform_1d(data: np.ndarray):
    """
    Haar wavelet: simplest wavelet. Computes running averages and differences.
    Produces near-zero detail coefficients for smooth data.
    """
    n = len(data)
    assert n % 2 == 0
    avg  = (data[0::2] + data[1::2]) / 2.0
    diff = (data[0::2] - data[1::2]) / 2.0
    return avg, diff   # avg → further compress; diff → mostly near zero

def haar_inverse(avg: np.ndarray, diff: np.ndarray):
    n = len(avg)
    out = np.empty(2 * n)
    out[0::2] = avg + diff
    out[1::2] = avg - diff
    return out

# multilevel example (what SPERR/FAZ actually do)
t    = np.linspace(0, 4*np.pi, 512)
data = np.sin(t) + 0.1*np.cos(5*t)

coeffs = pywt.wavedec(data, 'db4', level=5)
sizes  = [c.size for c in coeffs]
nnz    = [np.count_nonzero(np.abs(c) > 0.01) for c in coeffs]

print("Wavelet coefficient distribution (db4, 5 levels):")
print(f"{'Level':>6}  {'Size':>6}  {'NNZ(>0.01)':>12}  {'Sparsity':>10}")
for i, (sz, nz) in enumerate(zip(sizes, nnz)):
    label = f"approx" if i == 0 else f"detail-{i}"
    print(f"{label:>8}  {sz:>6}  {nz:>12}  {(1-nz/sz)*100:>9.1f}%")
```

---

### 2.4 Domain Transform (DT)

```python
import numpy as np

def log_transform_for_relative_eb(data: np.ndarray, er: float):
    """
    Convert relative error bound to absolute error bound via log transform.
    Proof (from paper): enforcing |x̂ - x|/|x| ≤ εᵣ is equivalent to
    enforcing |log(x̂) - log(x)| ≤ log(1 + εᵣ) ≈ εᵣ for small εᵣ.
    So: take log of data, compress with absolute EB = log(1 + εᵣ),
    then exp() during decompression.
    """
    assert np.all(data > 0), "log transform requires strictly positive data"
    absolute_eb  = np.log(1.0 + er)
    log_data     = np.log(data)
    return log_data, absolute_eb

def log_decompress(log_recon: np.ndarray):
    return np.exp(log_recon)

# --- demo ---
data = np.array([1.0, 10.0, 100.0, 1000.0, 0.001])  # multi-scale data
er   = 0.01   # 1% relative error bound

log_data, abs_eb = log_transform_for_relative_eb(data, er)
print(f"Relative EB {er} → Absolute EB on log-data: {abs_eb:.6f}")
print("After compression + decompression (simulated with abs_eb):")

# simulate: compress with absolute eb in log domain, then invert
simulated_log_recon = log_data + np.random.uniform(-abs_eb, abs_eb, data.shape)
recon = log_decompress(simulated_log_recon)

rel_errors = np.abs(recon - data) / np.abs(data)
print(f"Max relative error: {rel_errors.max()*100:.4f}%  (bound: {er*100:.1f}%)")
```

---

### 2.5 Bit-Plane Coding (BPC)

```python
import numpy as np
import struct

def bit_plane_encode_float32(values: np.ndarray, keep_planes: int = 16):
    """
    Encode float32 array by keeping only the top `keep_planes` bit-planes.
    The remaining 32 - keep_planes planes are zeroed (dropped).

    This is the core idea of ZFP's embedded coding:
    transmit the most significant bit-planes first, stop when error is small enough.
    """
    assert values.dtype == np.float32
    n_drop = 32 - keep_planes
    mask   = np.uint32(0xFFFFFFFF) << np.uint32(n_drop)
    ui     = values.view(np.uint32).copy()
    ui    &= mask
    return ui.view(np.float32)

def explain_bit_planes(value: float):
    """Show the bit-plane structure of a float32 value."""
    packed   = struct.pack('f', np.float32(value))
    as_int   = struct.unpack('I', packed)[0]
    bits     = f'{as_int:032b}'
    sign     = bits[0]
    exponent = bits[1:9]
    mantissa = bits[9:]
    print(f"Value    : {value}")
    print(f"Bits     : {bits}")
    print(f"Sign     : {sign}")
    print(f"Exponent : {exponent} = {int(exponent, 2) - 127} (biased)")
    print(f"Mantissa : {mantissa}")
    print(f"Dropping last 8 bits (low-significance):")
    truncated = bit_plane_encode_float32(np.array([value], dtype=np.float32), keep_planes=24)
    print(f"Truncated: {truncated[0]}  (error: {abs(truncated[0]-value):.2e})")

explain_bit_planes(3.14159265)
```

---

### 2.6 Tucker Decomposition / HOSVD

Already shown in Model 5 above. Used by TTHRESH, TuckerMPI, ATC.

---

### 2.7 Decimation / Sampling (DS)

Already shown in Model 1 above. Also includes **compressed sensing**:

```python
import numpy as np

def compressed_sensing_compress(data: np.ndarray, sample_rate: float = 0.3,
                                 seed: int = 0):
    """
    Random sampling (CS compression):
    - Sample `sample_rate` fraction of data points randomly.
    - Decompression solves an underdetermined linear system (e.g., LASSO).
    Very fast compression; slow decompression.
    """
    rng     = np.random.default_rng(seed)
    n       = len(data)
    m       = int(n * sample_rate)
    indices = np.sort(rng.choice(n, size=m, replace=False))
    samples = data[indices]
    return indices, samples

def cs_decompress_interp(indices, samples, n: int):
    """Simple decompression via linear interpolation (real CS uses L1 minimization)."""
    recon = np.interp(np.arange(n), indices, samples)
    return recon

# --- demo ---
t    = np.linspace(0, 2*np.pi, 512)
data = np.sin(t) + 0.3*np.sin(5*t)

idx, samp = compressed_sensing_compress(data, sample_rate=0.2)
recon     = cs_decompress_interp(idx, samp, len(data))

print(f"Samples kept       : {len(samp)}/{len(data)} = {len(samp)/len(data)*100:.0f}%")
print(f"Max absolute error : {np.max(np.abs(recon - data)):.4f}")
print("Note: real CS uses L1-minimization (LASSO/OMP) for much better reconstruction.")
```

---

### 2.8 Filtering (FTR) — Data Folding

```python
import numpy as np

def data_folding_compress(data: np.ndarray, eb: float, block_size: int = 16):
    """
    Data Folding (used in SZx):
    Split data into blocks. If a block's value range ≤ 2*eb,
    all values in that block can be represented by a single number (the mean).
    These are "constant blocks" — huge savings for smooth regions.
    """
    n      = len(data)
    blocks = []
    for start in range(0, n, block_size):
        block = data[start:start+block_size]
        lo, hi = block.min(), block.max()
        if (hi - lo) <= 2 * eb:
            # constant block: store just the mean + a flag
            blocks.append(('const', float(block.mean()), len(block)))
        else:
            # non-constant block: store raw
            blocks.append(('raw', block.copy(), len(block)))

    const_blocks = sum(1 for b in blocks if b[0] == 'const')
    print(f"Constant blocks: {const_blocks}/{len(blocks)} "
          f"({const_blocks/len(blocks)*100:.1f}%)  ← these are compressed to 1 value")
    return blocks

def data_folding_decompress(blocks):
    out = []
    for btype, data, n in blocks:
        if btype == 'const':
            out.append(np.full(n, data))
        else:
            out.append(data)
    return np.concatenate(out)

# --- demo: smooth field has many constant blocks ---
t    = np.linspace(0, 2*np.pi, 512)
data = np.sin(t).astype(np.float64)   # very smooth

print("=== Smooth data ===")
blocks = data_folding_compress(data, eb=0.01, block_size=16)
recon  = data_folding_decompress(blocks)
print(f"Max error: {np.max(np.abs(recon - data)):.4f}\n")

print("=== Noisy data ===")
noisy = data + 0.05 * np.random.randn(512)
blocks2 = data_folding_compress(noisy, eb=0.01, block_size=16)
```

---

### 2.9 Lossless Encoding (LE)

```python
import numpy as np
import heapq
from collections import Counter
import zlib, struct

def huffman_encode(symbols: list):
    """
    Build a Huffman tree for the symbol list.
    Symbols with high frequency get short codes.
    Used in SZ (Huffman+Zstd), MGARD, FAZ.
    """
    freq = Counter(symbols)
    heap = [[w, [sym, ""]] for sym, w in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0]+hi[0]] + lo[1:] + hi[1:])

    code_table = {sym: code for sym, code in sorted(heap[0][1:], key=lambda x: (len(x[1]), x))}
    encoded    = ''.join(code_table[s] for s in symbols)
    return code_table, encoded

def demo_lossless_encoding():
    # simulate quantization codes from Lorenzo prediction (mostly 0s and small integers)
    rng     = np.random.default_rng(0)
    qcodes  = rng.integers(-3, 4, size=1000)   # biased toward 0
    qcodes[:700] = 0                            # 70% are 0 (very common in prediction residuals)

    symbols = list(qcodes)

    # Huffman
    table, bitstring = huffman_encode(symbols)
    huffman_bits     = len(bitstring)
    naive_bits       = len(symbols) * 8   # 8 bits per symbol naively

    # zlib on the raw codes
    raw_bytes  = struct.pack(f'{len(symbols)}b', *np.clip(symbols, -128, 127))
    zlib_bytes = len(zlib.compress(raw_bytes, level=9))

    print(f"Symbol count       : {len(symbols)}")
    print(f"Naive (8 bits/sym) : {naive_bits} bits = {naive_bits//8} bytes")
    print(f"Huffman            : {huffman_bits} bits = {huffman_bits//8} bytes  "
          f"({naive_bits/huffman_bits:.2f}× over naive)")
    print(f"Zlib (dict encode) : {zlib_bytes} bytes  "
          f"({naive_bits//8/zlib_bytes:.2f}× over naive)")
    print(f"Huffman code for 0 : '{table.get(0, '?')}'  (most frequent → shortest code)")

demo_lossless_encoding()
```

---

### 2.10 Deep Neural Network (DNN) — as Predictor

```python
# See Model 6 above for the autoencoder example.
# Here: DNN used as a DATA PREDICTOR (not encoder), as in SRNN-SZ and KD-INR.

import numpy as np

# Concept: instead of Lorenzo prediction, train a small network
# to predict data[i,j,k] from its neighbors.
# The prediction residuals are then quantized normally.
# SRNN-SZ uses a super-resolution network to predict from downsampled data.

# Pseudocode sketch:
#
# Training phase (offline):
#   for each data block:
#       input  = block[neighbors]     # e.g., coarser-resolution version
#       target = block[fine]          # fine-resolution values
#       train DNN to minimize |DNN(input) - target|
#
# Compression phase:
#   coarse  = downsample(data)        # stored losslessly (small)
#   fine_pred = DNN(coarse)           # DNN prediction
#   residuals = data - fine_pred      # small residuals
#   qcodes = quantize(residuals, eb)  # error-bounded quantization
#   store: [coarse, DNN_weights, qcodes]
#
# Decompression phase:
#   fine_pred = DNN(coarse)
#   data_recon = fine_pred + dequantize(qcodes)

print("DNN-as-predictor: see SRNN-SZ (super-resolution) and KD-INR (implicit neural rep.).")
print("Key advantage: DNN can exploit complex non-local patterns Lorenzo misses.")
print("Key cost: inference time per compression/decompression call.")
```

---

## 3. Major Compressor Families

### 3.1 SZ Family

```python
# Install: pip install libpressio  (wraps SZ, ZFP, MGARD, etc.)
# Or: conda install -c conda-forge libpressio-tools

import numpy as np

def sz3_compress_via_libpressio(data: np.ndarray, abs_error_bound: float):
    """
    Use SZ3 via the libpressio Python bindings.
    libpressio is the standard benchmark harness used in the survey.
    """
    try:
        import libpressio
        compressor = libpressio.PressioCompressor.from_config({
            "compressor_id": "sz3",
            "compressor_config": {
                "sz3:error_bound_type": "abs",
                "sz3:abs_error_bound": abs_error_bound
            }
        })
        compressed  = compressor.encode(data)
        decompressed = compressor.decode(compressed, data)

        cr   = data.nbytes / len(compressed)
        merr = np.max(np.abs(decompressed - data))
        print(f"[SZ3] EB={abs_error_bound:.1e}  CR={cr:.1f}×  MaxErr={merr:.2e}")
        return decompressed, compressed
    except ImportError:
        print("libpressio not installed. Install with: pip install libpressio")
        return None, None

# --- CLI equivalent (if SZ3 is installed) ---
# Compress:   sz3 -f -i data.f32 -z data.sz -3 100 100 100 -M ABS -A 1e-4
# Decompress: sz3 -f -z data.sz  -o data_recon.f32 -3 100 100 100


# --- Manually simulate SZ3's core pipeline ---
def sz3_pipeline_demo(data: np.ndarray, eb: float):
    """
    Simplified SZ3 pipeline:
    1. Interpolation prediction (cubic spline, multidimensional)
    2. Quantization
    3. Huffman + Zstd encoding (simulated with zlib)
    """
    import zlib
    # Step 1: use 1D Lorenzo as a stand-in for the spline predictor
    qcodes, recon = lorenzo_1d_compress(data.ravel(), eb)

    # Step 2: pack qcodes as bytes
    raw = qcodes.astype(np.int16).tobytes()

    # Step 3: compress with zlib (stands in for Huffman+Zstd)
    compressed = zlib.compress(raw, level=9)

    cr   = data.nbytes / len(compressed)
    merr = np.max(np.abs(recon.reshape(data.shape) - data))
    print(f"[SZ3-sim] EB={eb:.1e}  CR={cr:.1f}×  MaxErr={merr:.2e}")
    return recon.reshape(data.shape)

# from earlier cell
def lorenzo_1d_compress(data, eb):
    n = len(data); qcodes = np.zeros(n, np.int32); recon = np.zeros(n)
    delta = 2.0 * eb
    for i in range(n):
        pred = recon[i-1] if i > 0 else 0.0
        qcodes[i] = int(np.round((data[i] - pred) / delta))
        recon[i]  = pred + qcodes[i] * delta
    return qcodes, recon

t    = np.linspace(0, 2*np.pi, 10000)
data = (np.sin(t) + 0.5*np.cos(3*t)).astype(np.float32)
sz3_pipeline_demo(data, eb=1e-3)
sz3_pipeline_demo(data, eb=1e-4)
```

---

### 3.2 ZFP

```python
# Install: pip install zfpy

import numpy as np

def zfp_demo(data: np.ndarray, tolerance: float):
    """ZFP fixed-accuracy mode (absolute error bound)."""
    try:
        import zfpy
        compressed   = zfpy.compress_numpy(data, tolerance=tolerance)
        decompressed = zfpy.decompress_numpy(compressed)

        cr   = data.nbytes / len(compressed)
        merr = np.max(np.abs(decompressed - data))
        print(f"[ZFP] tol={tolerance:.1e}  CR={cr:.1f}×  MaxErr={merr:.2e}")
        return decompressed
    except ImportError:
        print("zfpy not installed. Install with: pip install zfpy")

# 3D demo (ZFP works block-by-block in 4×4×4 blocks)
x, y, z = np.meshgrid(np.linspace(0, 2*np.pi, 64),
                       np.linspace(0, 2*np.pi, 64),
                       np.linspace(0, 2*np.pi, 64))
field_3d = (np.sin(x) * np.cos(y) * np.exp(-z/10)).astype(np.float64)

for tol in [1e-2, 1e-4, 1e-6]:
    zfp_demo(field_3d, tolerance=tol)

# --- CLI equivalent ---
# zfp -i data.f64 -z data.zfp -3 64 64 64 -a 1e-4 -d
# (compress 64×64×64 double with abs tolerance 1e-4, double precision)
```

---

### 3.3 MGARD

```python
# Install: build from source or via spack/conda

# CLI usage (after building MGARD):
#   mgard -x compress   -i input.f64 -c output.mgard -n 3 64 64 64 -t double -e 1e-4 -s inf
#   mgard -x decompress -c output.mgard -o output_recon.f64
#
# Python (via libpressio):
#   compressor = libpressio.PressioCompressor.from_config({
#       "compressor_id": "mgard",
#       "compressor_config": {
#           "mgard:s": 0,          # smoothness parameter (0 = Linfinity control)
#           "mgard:tolerance": 1e-4
#       }
#   })

print("""
MGARD strengths:
  - Unstructured and non-uniform grids (unique in this survey)
  - L2 and Linfinity error control
  - Derived-quantity error control (e.g. bound error on divergence, not just raw field)
  - Accelerated GPU version: MGARD+
""")
```

---

### 3.4 SPERR

```python
# Install: https://github.com/NCAR/SPERR   or via conda-forge

# CLI usage:
#   sperr_cli 3d --bitrate 2.0 --input data.f32 --dims 64 64 64 --output data.sperr
#   sperr_cli 3d --decompress --input data.sperr --output data_recon.f32
#
# Or using absolute error bound:
#   sperr_cli 3d --psnr 80 --input data.f32 --dims 64 64 64 --output data.sperr

print("""
SPERR pipeline:
  1. CDF9/7 wavelet transform (DWT)
  2. SPECK encoding of wavelet coefficients (threshold-based, level-by-level)
  3. Outlier correction for points that still exceed error bound
  4. Optional zstd post-processing

SPERR achieves the highest compression ratios of any SOTA compressor
at the cost of slow throughput (~30% of SZ3's speed).
""")
```

---

## 4. Application-Specific Examples

### 4.1 Molecular Dynamics (MDZ approach)

```python
import numpy as np

def md_spatial_cluster_predict(positions: np.ndarray, velocities: np.ndarray,
                                 dt: float):
    """
    MDZ-style prediction for MD trajectories.
    Two-level prediction:
      Level 1 (spatial): predict from neighboring atoms in the same molecule.
      Level 2 (temporal): predict from previous timestep using velocity.
    """
    # Temporal prediction: x(t+dt) ≈ x(t) + v(t)*dt
    temporal_pred = positions + velocities * dt
    return temporal_pred

n_atoms = 1000
pos  = np.random.randn(n_atoms, 3).astype(np.float32)
vel  = np.random.randn(n_atoms, 3).astype(np.float32) * 0.01
dt   = 0.002  # 2 fs timestep (typical MD)

pred = md_spatial_cluster_predict(pos, vel, dt)
print(f"MD temporal prediction: mean residual = {np.mean(np.abs(pred - pos)):.4f} Å")
print("MDZ adds spatial clustering to further reduce residuals for crystalline materials.")
```

---

### 4.2 Quantum Chemistry ERIs (PaSTRi approach)

```python
import numpy as np

def pastri_pattern_scaling_predict(eri_block: np.ndarray):
    """
    PaSTRi exploits the scaled repeated pattern (SRP) feature of ERI tensors:
    different sub-blocks of the ERI 4D tensor are related by a scaling factor.
    
    ERI[μ,ν,λ,σ] ≈ scale * ERI[μ',ν',λ',σ']  for related index groups.
    
    SZ achieves 7.2× on ERIs at EB=1e-10.
    PaSTRi achieves 16.8× by exploiting this domain structure.
    """
    # Simplified: identify the most similar reference block and store only the ratio
    ref_block  = eri_block[0]               # use first row as reference
    scales     = eri_block / (ref_block + 1e-20)   # scaling factors (mostly constant)
    residuals  = eri_block - (scales * ref_block)
    print(f"Scale factor variance : {scales.var():.2e}  (low → high compressibility)")
    print(f"Residual range        : {residuals.min():.2e} to {residuals.max():.2e}")
    return scales, residuals

# simulate a small ERI sub-block
block = np.outer(np.exp(-np.arange(10)*0.1), np.exp(-np.arange(10)*0.1))
pastri_pattern_scaling_predict(block)
```

---

### 4.3 Quantum Circuit State Vectors

```python
import numpy as np

def quantum_sv_compress(state_vector: np.ndarray, eb: float):
    """
    Compress a quantum state vector (complex amplitudes).
    Strategy from Wu et al.: separate real/imaginary, XOR leading-zero reduction,
    then SZ2.1-style compression.
    
    For a 61-qubit simulation: reduces from 32 exabytes to 768 terabytes.
    """
    # Separate real and imaginary
    real_part = state_vector.real.astype(np.float32)
    imag_part = state_vector.imag.astype(np.float32)

    # Identify near-zero amplitudes (common in quantum states)
    magnitudes = np.abs(state_vector)
    near_zero  = magnitudes < eb
    print(f"Near-zero amplitudes (<{eb}): {near_zero.mean()*100:.1f}%")

    # Compress real and imaginary separately with SZ
    qcodes_r, recon_r = lorenzo_1d_compress(real_part, eb)
    qcodes_i, recon_i = lorenzo_1d_compress(imag_part, eb)
    
    return qcodes_r, qcodes_i, recon_r + 1j*recon_i

# 10-qubit state vector (2^10 = 1024 amplitudes)
n_qubits = 10
sv = np.random.randn(2**n_qubits) + 1j*np.random.randn(2**n_qubits)
sv /= np.linalg.norm(sv)  # normalize

qr, qi, recon_sv = quantum_sv_compress(sv, eb=1e-4)
print(f"Max error on |amplitude|: {np.max(np.abs(np.abs(recon_sv) - np.abs(sv))):.2e}")
```

---

### 4.4 Climate / Cosmology: Adaptive Per-Region Error Bounds

```python
import numpy as np

def adaptive_eb_cosmology(field: np.ndarray, base_eb: float,
                           density_threshold: float):
    """
    Jin et al. approach: high-density regions (halos) need tighter error bounds
    because cosmologists focus their analysis there.
    Low-density void regions can tolerate larger errors.
    
    This adaptive strategy improves compression ratio by up to 73%
    with the same post-analysis quality loss.
    """
    density = np.abs(field)  # proxy: use field magnitude as density

    # tight bound for halos, relaxed bound for voids
    eb_map = np.where(density > density_threshold,
                      base_eb,            # tight bound: halos
                      base_eb * 10.0)     # relaxed bound: voids

    print(f"Halo regions (tight EB={base_eb:.1e})  : {(density > density_threshold).mean()*100:.1f}%")
    print(f"Void regions (loose EB={base_eb*10:.1e}): {(density <= density_threshold).mean()*100:.1f}%")
    print("Adaptive EB → higher overall CR with same post-analysis accuracy.")
    return eb_map

field = np.random.exponential(scale=1.0, size=(128, 128))  # log-normal-like cosmological density
eb_map = adaptive_eb_cosmology(field, base_eb=1e-3, density_threshold=2.0)
```

---

### 4.5 MPI Communication Compression (CCOLL / HZCCL)

```python
import numpy as np

def compress_gradient_for_allreduce(gradient: np.ndarray, eb: float):
    """
    CCOLL/HZCCL approach: compress gradients before MPI_Allreduce.
    After allreduce on compressed data, decompress on each rank.
    
    Trade-off: compression error < eb is acceptable because:
    1. Gradient noise already exists in SGD.
    2. Error averages out across many ranks.
    
    Huang et al. showed limited impact on final model accuracy.
    Speedup: up to 4.5× over NCCL, 28.7× over Cray MPI.
    """
    import zlib
    qcodes, recon = lorenzo_1d_compress(gradient, eb)
    raw_bytes     = qcodes.astype(np.int16).tobytes()
    compressed    = zlib.compress(raw_bytes, level=1)  # level 1: fast

    cr = gradient.nbytes / len(compressed)
    print(f"Gradient shape : {gradient.shape}")
    print(f"Bandwidth saved: {(1 - 1/cr)*100:.1f}%  (CR={cr:.1f}×)")
    print(f"Max error on gradient: {np.max(np.abs(recon - gradient)):.2e}  ≤ EB={eb:.1e}")
    return compressed, recon

gradient = np.random.randn(10000).astype(np.float32) * 0.01  # typical DNN gradient scale
compress_gradient_for_allreduce(gradient, eb=1e-5)
```

---

## 5. How to Choose a Compressor

```
Use case                            Recommended compressor
─────────────────────────────────────────────────────────────────────────
Fast CPU, general scientific data   SZ3, QoZ
GPU (throughput priority)           cuSZp  (200–400 GB/s on A100)
GPU (ratio priority)                cuSZ, cuSZ-I
Maximum compression ratio           FAZ, SPERR, TTHRESH
Unstructured / non-uniform grids    MGARD
Derived-quantity error control      MGARD-Lambda
Relative error bound                SZ3 (log-mode), QoZ
Molecular dynamics                  MDZ
Quantum chemistry (ERI)             PaSTRi
Quantum circuit state vectors       SZ2.1 + XOR tricks (Wu et al.)
Climate data                        CliZ, SZ3
Seismic RTM                         HyZ (BR + SZx)
X-ray crystallography               ROIBIN-SZ
MPI collective communication        CCOLL / HZCCL
Federated learning gradients        FedSZ
```

### Error bound selection guide

```python
import numpy as np

def suggest_error_bound(data: np.ndarray, desired_cr: float = None,
                         acceptable_psnr_db: float = 60.0):
    """
    Rough heuristic: estimate what error bound achieves a target PSNR.
    PSNR = 20*log10(data_range / RMSE)
    For many simulation datasets, abs_eb ≈ 1e-4 * data_range is a good start.
    """
    data_range = data.max() - data.min()
    target_rmse = data_range / (10 ** (acceptable_psnr_db / 20))
    suggested_eb = target_rmse  # very rough: abs_eb ≈ target RMSE

    print(f"Data range     : {data_range:.4f}")
    print(f"Target PSNR    : {acceptable_psnr_db} dB")
    print(f"Suggested EB   : {suggested_eb:.2e}")
    print(f"Rule of thumb  : EB = 1e-4 × range = {data_range*1e-4:.2e}")
    print("Always validate post-analysis QoI (SSIM, CRPS, etc.) after choosing EB.")

data = np.sin(np.linspace(0, 2*np.pi, 10000))
suggest_error_bound(data, acceptable_psnr_db=80.0)
```

---

## 6. Dependencies

```bash
# Core numerical
pip install numpy scipy

# Wavelet transforms
pip install PyWavelets

# Tensor decomposition
pip install tensorly

# Deep learning
pip install torch

# Clustering (for VQ)
pip install scikit-learn

# ZFP Python bindings
pip install zfpy

# SZ3, MGARD, SPERR via libpressio (recommended for benchmarking)
pip install libpressio
# or via conda:
conda install -c conda-forge libpressio-tools

# SZ3 CLI
# https://github.com/szcompressor/SZ3

# ZFP CLI
# https://github.com/LLNL/zfp

# MGARD
# https://github.com/CODARcode/MGARD

# SPERR
# https://github.com/NCAR/SPERR

# cuSZp (GPU)
# https://github.com/szcompressor/cuSZp
```

---

## Quick Reference: Compression Pipeline

```
Input floating-point data (scientific field)
         │
         ▼
[Optional: Domain Transform]
   • log-transform  → enables relative EB
   • space-filling curve (zMesh) → improves smoothness for AMR
         │
         ▼
[Decorrelation — choose one or combine]
   • Lorenzo predictor (1D/2D/3D) ──── SZ family
   • Wavelet transform (DWT/OWT)  ──── ZFP, SPERR, FAZ
   • Tucker / HOSVD               ──── TTHRESH
   • DNN autoencoder              ──── HAE, AE-SZ
         │
         ▼  (produces near-zero residuals)
[Quantization]
   • Linear-scale  → SZ (fixed error per bin)
   • Log-scale     → NUMARCK
   • Vector (k-means) → MDZ
         │
         ▼  (produces sparse integer codes)
[Filtering — optional]
   • Data folding  → represent constant blocks with 1 value
   • Data extraction → handle outliers separately
         │
         ▼
[Lossless Encoding]
   • Huffman   → SZ, FAZ
   • Arithmetic → TTHRESH, MGARD
   • Zlib/Zstd → SZ, Bit Grooming
   • Run-length → cuSZp, TTHRESH
   • Embedded (bit-plane) → ZFP
         │
         ▼
Compressed bitstream  (10–1000× smaller, |error| ≤ ε guaranteed)
```
