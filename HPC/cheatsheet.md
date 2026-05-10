# Error-bounded Lossy Compression — Cheatsheet

## The Conceptual Distinction

- **6 Models** = the *overall strategy* (the blueprint for the compressor)
- **10 Techniques** = the *building blocks* that implement that strategy

A model is named after its **central decorrelation technique**, but every real
compressor assembles 3–5 techniques into a pipeline:

```
Input data
    │
    ▼
[Preprocessing]     ← DT, DS, BPC (early stage)
    │
    ▼
[Decorrelation]     ← THIS STEP names the model: PDP / OWT / HOSVD / DNN / DS / bit-manip
    │
    ▼  (residuals — sparse, near-zero)
[Quantization]      ← QT
    │
    ▼
[Filtering]         ← FTR (optional)
    │
    ▼
[Encoding]          ← LE
    │
    ▼
Compressed bitstream
```

The model is determined by which technique sits in the **decorrelation slot**.
Everything else (QT, LE, FTR, DT) is shared infrastructure that appears across models.

---

## Many-to-Many Mapping

```
                     ┌──── 10 TECHNIQUES ───────────────────────────────────────────┐
                     │ PDP  OWT  HOSVD  DNN   DT   DS   BPC  QT   FTR   LE         │
6 MODELS    ─────────┼──────────────────────────────────────────────────────────────┤
1. Decimation        │  ·    ·     ·     ·    ·   [DS]  ·    ·     ·    ●           │
2. Bit-manip         │  ·    ·     ·     ·    ·    ·   [BPC] ·    ○    ●           │
3. Transform         │  ·   [OWT]  ·     ·    ○    ·    ●    ○    ○    ●           │
4. Prediction        │ [PDP]  ·    ·     ·    ○    ·    ·    ●    ○    ●           │
5. HOSVD             │  ·    ·   [HOSVD] ·    ·    ·    ●    ·     ·    ●           │
6. Deep Learning     │  ○    ·     ·    [DNN] ○    ·    ·    ●    ○    ●           │
                     └──────────────────────────────────────────────────────────────┘
[X] = defining technique    ● = always used    ○ = sometimes used    · = not used
```

- **QT and LE appear in every model** — universal infrastructure
- **PDP, OWT, HOSVD, DNN** each define exactly one model — the differentiating core
- **DT, BPC, FTR** are cross-cutting — they augment multiple models

---

## The 6 Models

| # | Model | Defining technique | Pro | Con |
|---|-------|--------------------|-----|-----|
| 1 | Decimation | DS | Fastest compression | Expensive decompression (interpolation) |
| 2 | Bit-manipulation | BPC | Very fast, simple | Low ratio (ignores correlations) |
| 3 | Transformation | OWT/DWT | High quality, GPU-friendly | Hard to bound error tightly |
| 4 | Prediction | PDP | High ratio, tunable error | Slower; harder to parallelize |
| 5 | HOSVD | HOSVD | Extremely high ratio | O(n⁴) cost; no pointwise error bound |
| 6 | Deep Learning | DNN | Can learn complex patterns | Slow training; hard to bound error |

Speed/Quality spectrum:
```
Faster / Lower Quality  ←─────────────────────────→  Slower / Higher Quality
  Decimation  Bit-manip  Transform  Prediction  HOSVD  Deep-Learning
```

---

## The 10 Techniques

| # | Abbreviation | Full name | Role in pipeline | Used by |
|---|--------------|-----------|-----------------|---------|
| 1 | PDP | Point-wise Data Prediction | Decorrelation | SZ family, FPZIP |
| 2 | QT  | Quantization | Encode residuals into integers | SZ, cuSZ, most compressors |
| 3 | OWT | Orthogonal/Wavelet Transform | Decorrelation | ZFP, SPERR, FAZ |
| 4 | DT  | Domain Transform | Preprocessing (log, space-filling curve) | SZ (log mode), zMesh |
| 5 | BPC | Bit-Plane Coding | Decorrelation or encoding | ZFP, SZx, cuSZp |
| 6 | HOSVD | Higher-Order SVD / Tucker | Decorrelation | TTHRESH, TuckerMPI |
| 7 | DS  | Decimation / Sampling | Preprocessing | AMR codes, compressed sensing |
| 8 | FTR | Filtering | Remove insignificant values | SZx, SPERR, cuSZx |
| 9 | LE  | Lossless Encoding | Final compression | Every compressor |
| 10 | DNN | Deep Neural Network | Decorrelation or prediction | HAE, AE-SZ, SRNN-SZ |

### LE sub-types
| Encoder | Type | Used by |
|---------|------|---------|
| Huffman | Entropy | SZ1/2/3, FAZ |
| Arithmetic | Entropy | TTHRESH, MGARD |
| Zlib/Zstd | Dictionary | SZ, Bit Grooming |
| Run-length | Repeat reduction | cuSZp, TTHRESH |
| Embedded (bit-plane) | Progressive | ZFP |
| Fixed-length | Fast on GPU | cuSZp |

---

## Concrete Compressor Pipelines

| Compressor | Model | Pipeline |
|------------|-------|----------|
| SZ1 / SZ2 | Prediction | PDP → QT → LE (Huffman+Zstd) |
| SZ3 | Prediction | PDP → QT → LE + DT (log mode) |
| QoZ | Prediction | PDP → QT → LE (auto-tuned) |
| FAZ | Prediction + Transform | PDP + OWT → QT → LE |
| ZFP | Transform | DT → OWT → BPC |
| SPERR | Transform | OWT → QT + FTR → LE (zstd) |
| SZx | Bit-manip | FTR → BPC |
| cuSZp | Bit-manip | FTR + BPC → LE (fixed-length) |
| TTHRESH | HOSVD | HOSVD → BPC → LE (RLE+AC) |
| MGARD | Transform | DWT (multilevel) → QT → LE |
| FPZIP | Prediction | PDP (Lorenzo) → BPC |
| HAE | Deep Learning | DT → DNN → QT → FTR |
| AE-SZ | Deep Learning | DNN (predictor) → QT → LE |

---

## Key Takeaway

> **A model is a strategy. A technique is a tool.**
> No single technique is sufficient — every high-performance compressor
> combines 3–5 techniques. QT + LE are the universal glue.
> The decorrelation technique (PDP, OWT, HOSVD, DNN) is what
> differentiates compressors and dominates the compression ratio.

---

## Application → Compressor Quick Reference

| Domain | Challenge | Recommended |
|--------|-----------|-------------|
| General scientific (CPU) | Balanced | SZ3, QoZ |
| GPU (throughput) | 200–400 GB/s | cuSZp |
| GPU (ratio) | High CR on GPU | cuSZ, cuSZ-I |
| Maximum ratio | Slow is OK | FAZ, SPERR, TTHRESH |
| Unstructured grids | Non-uniform mesh | MGARD |
| Relative error bound | Multi-scale data | SZ3 (log-mode), QoZ |
| Molecular dynamics | Trajectory data | MDZ |
| Quantum chemistry | ERI tensors | PaSTRi |
| Quantum circuits | State vectors | SZ2.1 + XOR |
| Climate | Spatial structure | CliZ |
| Seismic RTM | Wavefield snapshots | HyZ |
| MPI collectives | Gradient bandwidth | CCOLL / HZCCL |

---

## Error Bound Quick Guide

```
Absolute EB:   |reconstructed - original| ≤ ε
Relative EB:   |reconstructed - original| / |original| ≤ εᵣ
               → convert via log-transform: compress log(data) with abs EB = log(1+εᵣ)

Choosing ε:
  Start with ε = 1e-4 × data_range
  Validate post-analysis QoI (SSIM, CRPS, PSNR, domain-specific metrics)
  Tighten if QoI degrades; loosen to improve ratio
```

---

## Speed vs. Ratio Tradeoff (from Table 6)

```
Ratio   ★★★★★  FAZ, SPERR, HH-NN, TTHRESH
        ★★★★   SZ3, QoZ, MGARD, ZFP
        ★★★    SZ1/2, cuSZ
        ★★     SZx, FPZIP

Speed   ★★★★★  cuSZp  (200–400 GB/s on A100)
        ★★★★   ZFP, SZx
        ★★★    SZ3, cuSZ
        ★★     SPERR, MGARD
        ★      TTHRESH, FAZ (slow compress, high ratio)
```
