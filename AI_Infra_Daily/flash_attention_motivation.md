# Flash Attention Motivation: The IO Bottleneck

## GPU Structure Overview

```
┌─────────────────────────────────────────────────────────┐
│                         GPU                             │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Streaming Multiprocessors (SMs)       │  │
│  │                                                    │  │
│  │   ┌─────────┐  ┌─────────┐       ┌─────────┐     │  │
│  │   │  SM 0   │  │  SM 1   │  ...  │  SM N   │     │  │
│  │   │┌───────┐│  │┌───────┐│       │┌───────┐│     │  │
│  │   ││ CUDA  ││  ││ CUDA  ││       ││ CUDA  ││     │  │
│  │   ││ Cores ││  ││ Cores ││       ││ Cores ││     │  │
│  │   │└───────┘│  │└───────┘│       │└───────┘│     │  │
│  │   │┌───────┐│  │┌───────┐│       │┌───────┐│     │  │
│  │   ││ SRAM  ││  ││ SRAM  ││       ││ SRAM  ││     │  │
│  │   ││(~192KB)│  ││(~192KB)│       ││(~192KB)│     │  │
│  │   │└───────┘│  │└───────┘│       │└───────┘│     │  │
│  │   └─────────┘  └─────────┘       └─────────┘     │  │
│  │                                                    │  │
│  │   Total SRAM across all SMs: ~20MB                 │  │
│  │   Bandwidth: ~19 TB/s                              │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                               │
│                          │  ← data bus (bottleneck)      │
│                          │                               │
│  ┌───────────────────────▼───────────────────────────┐  │
│  │              HBM (Off-chip DRAM)                   │  │
│  │                                                    │  │
│  │   Capacity: 40-80 GB                              │  │
│  │   Bandwidth: ~2 TB/s                              │  │
│  │                                                    │  │
│  │   Stores: Q, K, V, scores, attn_weights, output   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Components

- **CUDA Cores**: Thousands of small processors that execute math (matmul, add, exp) in parallel
- **SRAM (Shared Memory / Registers)**: Each SM has its own fast on-chip memory. Data must be here for CUDA cores to operate on it
- **HBM (High Bandwidth Memory)**: Large off-chip memory where tensors are stored. "High bandwidth" compared to CPU RAM, but still ~10x slower than SRAM

### The Fundamental Problem

```
SRAM:  ~19 TB/s bandwidth,  ~20 MB capacity
HBM:    ~2 TB/s bandwidth,  ~80 GB capacity
                 ↑
          10x speed gap
```

CUDA cores can crunch numbers far faster than HBM can feed them data. When an operation is **memory-bound** (more time spent loading/storing data than computing), the cores sit idle waiting for data. This is exactly what happens in naive attention.

---

## GPU Memory Hierarchy (Simplified)

```
┌─────────────────────────────┐
│   HBM (High Bandwidth Mem)  │  Large (40-80GB), slow (~2 TB/s)
│   - Q, K, V, Output live    │
└──────────────┬──────────────┘
               │  ← bottleneck here
┌──────────────▼──────────────┐
│        SRAM (On-chip)       │  Small (~20MB), fast (~19 TB/s)
│   - Actual computation      │
└─────────────────────────────┘
```

SRAM is ~10x faster than HBM, but ~1000x smaller. The GPU can compute far faster than it can move data to/from HBM.

---

## What Naive Attention Does (The Problem)

Each step reads from HBM, computes in SRAM, writes back to HBM:

```
Step 1:  HBM → SRAM: load Q, K
         SRAM:       compute scores = Q @ K^T
         SRAM → HBM: write scores (N×N)       ← huge matrix materialized

Step 2:  HBM → SRAM: load scores (N×N)
         SRAM:       compute softmax
         SRAM → HBM: write attn_weights (N×N)  ← again full N×N

Step 3:  HBM → SRAM: load attn_weights, V
         SRAM:       compute output = attn_weights @ V
         SRAM → HBM: write output
```

**6 HBM round-trips.** The N×N scores matrix is written and re-read — O(N²) memory traffic for data used only once.

---

## Arithmetic Intensity: Compute vs IO

| Operation | FLOPs | HBM Reads/Writes (bytes) |
|-----------|-------|--------------------------|
| Q @ K^T   | O(N²d) | Read Q,K: O(Nd) + Write scores: O(N²) |
| Softmax   | O(N²) | Read scores: O(N²) + Write weights: O(N²) |
| Weights @ V | O(N²d) | Read weights,V: O(N²+Nd) + Write output: O(Nd) |

The actual compute (FLOPs) is cheap. The bottleneck is **moving the N×N matrices back and forth through HBM**.

---

## Benchmark Evidence

From our naive attention benchmark:

| seq_len | latency | memory |
|---------|---------|--------|
| 128     | 0.10ms  | 10.12MB |
| 256     | 0.11ms  | 14.12MB |
| 512     | 0.12ms  | 28.12MB |
| 1024    | 0.20ms  | 80.12MB |
| 4096    | 5.10ms  | 1064.12MB |

Memory grows as O(N²) — dominated by the scores matrix `(B, H, N, N)`.

---

## Flash Attention Fix: Tiling

Process Q, K, V in small blocks that fit entirely in SRAM:

```
For each Q_block (tile of Q):
    Initialize running output, running softmax stats in SRAM
    For each K_block, V_block (tiles of K, V):
        Load Q_block, K_block, V_block into SRAM
        Compute partial scores = Q_block @ K_block^T     (in SRAM)
        Update running softmax statistics                  (in SRAM)
        Accumulate partial output = partial_weights @ V_block (in SRAM)
    Write final output block to HBM
```

### Key Differences

**Scores Matrix:**
- Naive: Fully materialized in HBM (entire N×N written and re-read)
- Flash: Never exists in HBM (computed and consumed per tile in SRAM)

**HBM Memory Usage:**
- Naive: O(N²) — stores Q, K, V, scores, attn_weights, output
- Flash: O(N) — only stores Q, K, V, output (no intermediate N×N matrices)

**HBM IO Passes:**
- Naive: 6 passes — each step (matmul, softmax, matmul) reads and writes through HBM
- Flash: 1 pass — load tiles in, write final output out, nothing in between

**Where Compute Happens:**
- Naive: SRAM does the math, but intermediate results bounce back to HBM between steps
- Flash: All three steps (QK^T, softmax, ×V) happen back-to-back in SRAM per tile

### Why Tiling Works

- Each tile is small enough to fit in SRAM (~20MB)
- Scores are computed, used for softmax, multiplied with V, and discarded — all within SRAM
- The challenge: softmax needs the full row to normalize, solved with **online softmax** (tracking running max and sum)

### IO Complexity

```
Naive:    O(N²)  HBM reads/writes
Flash:    O(N²d / M)  where M = SRAM size
```

Since M is a constant and d is typically small (64-128), Flash Attention reduces IO significantly while performing the exact same computation.
