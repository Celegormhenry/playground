# Day 15: FlashAttention Tiling Strategy Design

## Goal

Design the tiling strategy for a Triton FlashAttention kernel: compute `O = softmax(Q @ K^T / sqrt(d)) @ V` without materializing the N×N attention matrix in HBM.

---

## Dimensions and Notation

```
B  = batch size
H  = number of heads
N  = sequence length
d  = head dimension (typically 64 or 128)

Q, K, V: (B, H, N, d)
Output:   (B, H, N, d)
```

For tiling, we focus on a single head (single `(N, d)` slice). The kernel processes all `B*H` heads via the grid.

---

## Tiling Strategy: Outer Q, Inner KV

### Why This Order?

Each output row `O[i, :]` depends on the **full** row of attention weights — i.e., query `i` must attend to **all** keys. So for each Q-tile, we must loop over **all** K/V tiles.

```
For each Q_tile (BLOCK_M rows of Q):
    For each KV_tile (BLOCK_N rows of K, V):
        Compute partial scores, update running softmax, accumulate output
    Finalize and write output tile
```

The alternative (outer K, inner Q) would require writing partial results for **all** Q rows at each step, consuming more HBM bandwidth. Outer-Q keeps the accumulator in SRAM for the entire inner loop.

### Tile Sizes

```
BLOCK_M  = rows of Q per tile     (e.g., 64 or 128)
BLOCK_N  = rows of K/V per tile   (e.g., 64 or 128)
d        = full head dimension     (not tiled — loaded in full)
```

We do **not** tile along `d` because `d` is small (64–128) and the full row must be in SRAM for the dot product and output accumulation.

### SRAM Budget Check

Each program needs to hold simultaneously:

| Tensor | Shape | Elements | Bytes (fp32) |
|--------|-------|----------|-------------|
| Q tile | (BLOCK_M, d) | 64×128 = 8K | 32 KB |
| K tile | (BLOCK_N, d) | 64×128 = 8K | 32 KB |
| V tile | (BLOCK_N, d) | 64×128 = 8K | 32 KB |
| Scores tile | (BLOCK_M, BLOCK_N) | 64×64 = 4K | 16 KB |
| Output acc | (BLOCK_M, d) | 64×128 = 8K | 32 KB |
| Softmax stats (m, l) | (BLOCK_M,) × 2 | 128 | 0.5 KB |
| **Total** | | | **~145 KB** |

With BLOCK_M=BLOCK_N=64, d=128: ~145 KB per program. Fits in SM SRAM (~192 KB on A100). Smaller blocks (32) give more headroom; larger blocks (128) need careful budgeting.

---

## Grid Layout

```python
grid = (triton.cdiv(N, BLOCK_M), B * H)
```

- **Axis 0**: which Q-tile (along sequence length)
- **Axis 1**: which batch×head

Each program computes BLOCK_M rows of output for one head.

```python
pid_q = tl.program_id(0)   # which Q tile
pid_bh = tl.program_id(1)  # which batch*head

# Offset into the (B, H, N, d) tensor
batch = pid_bh // H
head = pid_bh % H
qkv_offset = batch * H * N * d + head * N * d
```

---

## Per-Program Algorithm

### Step 0: Load Q Tile (Once)

```
Q_tile = load Q[pid_q * BLOCK_M : (pid_q+1) * BLOCK_M, :]   # (BLOCK_M, d)
```

Q_tile stays in SRAM for the entire inner loop — loaded from HBM exactly once.

### Step 1: Initialize Accumulators

```
m = [-inf] * BLOCK_M          # running max per query row
l = [0.0] * BLOCK_M           # running sum of exp per query row
O = zeros(BLOCK_M, d)         # running output accumulator
```

### Step 2: Inner Loop Over KV Tiles

For `j = 0, 1, ..., ceil(N / BLOCK_N) - 1`:

```
K_tile = load K[j * BLOCK_N : (j+1) * BLOCK_N, :]   # (BLOCK_N, d)
V_tile = load V[j * BLOCK_N : (j+1) * BLOCK_N, :]   # (BLOCK_N, d)

# --- Compute scores ---
S = Q_tile @ K_tile^T / sqrt(d)         # (BLOCK_M, BLOCK_N)

# --- Online softmax update ---
m_block = rowmax(S)                      # (BLOCK_M,) — max of each row in this tile
m_new = max(m, m_block)                  # (BLOCK_M,) — new running max

# Correction factor for old statistics
correction = exp(m - m_new)              # (BLOCK_M,)

# Exponentiate scores with new max
P = exp(S - m_new[:, None])              # (BLOCK_M, BLOCK_N)

# Update running sum
l = l * correction + rowsum(P)           # (BLOCK_M,)

# Update output accumulator (rescale old output + add new contribution)
O = O * correction[:, None] + P @ V_tile   # (BLOCK_M, d)

m = m_new
```

### Finalize (end of online softmax)

```
# After all KV tiles processed, l is now the true denominator
O = O / l[:, None]                       # deferred normalization
store O → output[pid_q * BLOCK_M : (pid_q+1) * BLOCK_M, :]
```

---

## Data Flow Diagram

```
                        HBM
  ┌──────────────────────────────────────────────────────────┐
  │  Q (N, d)    K (N, d)    V (N, d)         O (N, d)      │
  └─────┬────────────┬───────────┬──────────────▲────────────┘
        │            │           │              │
   load once    load each    load each     write once
        │        iteration   iteration         │
        ▼            ▼           ▼              │
  ┌─────────────────────────────────────────────┼────────────┐
  │  SRAM (per program)                         │            │
  │                                             │            │
  │  Q_tile (BLOCK_M, d)  ← stays entire loop  │            │
  │  O_acc  (BLOCK_M, d)  ← stays entire loop ─┘            │
  │  m      (BLOCK_M,)    ← stays entire loop               │
  │  l      (BLOCK_M,)    ← stays entire loop               │
  │                                                          │
  │  K_tile (BLOCK_N, d)  ← replaced each iteration         │
  │  V_tile (BLOCK_N, d)  ← replaced each iteration         │
  │  S      (BLOCK_M, BLOCK_N)  ← temporary, never in HBM   │
  │  P      (BLOCK_M, BLOCK_N)  ← temporary, never in HBM   │
  │                                                          │
  │  Each iteration:                                         │
  │    S = Q_tile @ K_tileᵀ / √d                            │
  │    m_new = max(m, rowmax(S))                             │
  │    P = exp(S - m_new)                                    │
  │    l = l * exp(m - m_new) + rowsum(P)                    │
  │    O_acc = O_acc * exp(m - m_new) + P @ V_tile           │
  │    m = m_new                                             │
  │                                                          │
  │  After all iterations:                                   │
  │    O_acc = O_acc / l  → write to HBM                     │
  └──────────────────────────────────────────────────────────┘
```

---

## HBM Access Pattern

| What | Reads | Writes |
|------|-------|--------|
| Q | N×d (each tile loaded once) | — |
| K | N×d × (N/BLOCK_M) times | — |
| V | N×d × (N/BLOCK_M) times | — |
| O | — | N×d |
| Scores (N×N) | **never in HBM** | **never in HBM** |

Total HBM IO: `O(N²d / BLOCK_M)` — reduced from naive `O(N²)` by a factor related to SRAM tile size.

K and V are each read `ceil(N/BLOCK_M)` times total (once per Q-tile). Larger BLOCK_M = fewer passes over K/V = less HBM traffic.

---

## Triton Implementation Skeleton

```python
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N, d,
    stride_qn, stride_qd,   # Q strides
    stride_kn, stride_kd,   # K strides
    stride_vn, stride_vd,   # V strides
    stride_on, stride_od,   # O strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_q = tl.program_id(0)    # which Q tile
    pid_bh = tl.program_id(1)   # which batch*head

    # -- offset into this head's Q/K/V/O --
    # (computed from pid_bh, B, H, N, d)

    # -- load Q tile (BLOCK_M, d) --
    q_offs_n = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    q_offs_d = tl.arange(0, d)  # d is small, load full width
    q_ptrs = Q_ptr + q_offs_n[:, None] * stride_qn + q_offs_d[None, :] * stride_qd
    q_mask = q_offs_n[:, None] < N
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # -- init accumulators --
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    # -- inner loop over KV tiles --
    for j in range(0, N, BLOCK_N):
        kv_offs = j + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        # load K tile (BLOCK_N, d) and V tile (BLOCK_N, d)
        k_ptrs = K_ptr + kv_offs[:, None] * stride_kn + q_offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + kv_offs[:, None] * stride_vn + q_offs_d[None, :] * stride_vd
        K_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        V_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # scores: (BLOCK_M, BLOCK_N)
        S = tl.dot(Q_tile, tl.trans(K_tile)) * (1.0 / sqrt_d)
        # mask out-of-bounds keys
        S = tl.where(kv_mask[None, :], S, float('-inf'))

        # online softmax
        m_block = tl.max(S, axis=1)              # (BLOCK_M,)
        m_new = tl.maximum(m, m_block)
        correction = tl.exp(m - m_new)
        P = tl.exp(S - m_new[:, None])           # (BLOCK_M, BLOCK_N)
        l = l * correction + tl.sum(P, axis=1)
        O_acc = O_acc * correction[:, None] + tl.dot(P, V_tile)
        m = m_new

    # -- finalize --
    O_acc = O_acc / l[:, None]

    # -- store output --
    o_ptrs = O_ptr + q_offs_n[:, None] * stride_on + q_offs_d[None, :] * stride_od
    tl.store(o_ptrs, O_acc, mask=q_mask)
```

---

## Causal Masking Extension

For autoregressive models, query `i` should only attend to keys `j ≤ i`. In the inner loop:

```python
# After computing S, before softmax:
causal_mask = q_offs_n[:, None] >= kv_offs[None, :]  # (BLOCK_M, BLOCK_N)
S = tl.where(causal_mask, S, float('-inf'))
```

Optimization: if the entire KV tile is above the causal boundary (all `j > max(q_offs_n)`), skip it entirely:

```python
if j > (pid_q + 1) * BLOCK_M - 1:
    break  # all remaining KV tiles are fully masked
```

This cuts the inner loop roughly in half for causal attention.

---

## Block Size Trade-offs

| | Smaller (32) | Larger (128) |
|--|-------------|-------------|
| SRAM per program | ~40 KB | ~580 KB |
| Programs per SM | More (better occupancy) | Fewer |
| K/V reloads | More (N/32 passes) | Fewer (N/128 passes) |
| tl.dot efficiency | Lower (small tiles) | Higher (better tensor core util) |

**Recommended starting point**: `BLOCK_M=64, BLOCK_N=64, d=128` (~145 KB). This balances SRAM usage and occupancy on most GPUs. Tune with `@triton.autotune` after correctness is verified.

---

## Implementation Plan (Days 16–19)

| Day | Task | Key Challenge |
|-----|------|--------------|
| 16 | Implement Q @ K^T tile computation | 2D pointer arithmetic, scaling by 1/√d |
| 17 | Add online softmax to inner loop | Correction factor `exp(m_old - m_new)`, rescaling O_acc |
| 18 | Add P @ V aggregation | Accumulator update with rescaling, final normalization |
| 19 | Validate correctness | Compare against `F.scaled_dot_product_attention` for various (B, H, N, d) |

---

## Numerical Considerations

1. **Scale before softmax**: multiply S by `1/sqrt(d)` immediately, not after
2. **Float32 accumulators**: even if Q/K/V are fp16, keep `m`, `l`, `O_acc` in fp32
3. **-inf for masked positions**: ensures `exp(-inf) = 0`, doesn't corrupt max or sum
4. **Final division safety**: `l` can't be zero if at least one key is unmasked (exp(0) ≥ 1)
