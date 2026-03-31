# Day 11: Row Softmax Kernel in Triton

## What is Row Softmax?

Given a 2D matrix `x` of shape `(M, N)`, row softmax computes for each row independently:

```
softmax(x_i) = exp(x_i - max(row)) / sum(exp(row - max(row)))
```

The `max(row)` subtraction is critical for **numerical stability** — without it, `exp(large_number)` overflows to `inf`.

## How the Triton Kernel Works

### Grid: One Program Per Row

Unlike matmul (which uses a 2D grid of tiles), softmax uses a **1D grid of M programs** — one per row:

```python
grid = (M,)  # M programs, one per row
```

Each program is responsible for computing softmax over its entire row.

### Step-by-Step Inside Each Program

#### Step 1: Identify Which Row

```python
row = tl.program_id(axis=0)
```

#### Step 2: Compute Column Offsets and Mask

```python
col_offsets = tl.arange(0, BLOCK_SIZE)  # [0, 1, 2, ..., BLOCK_SIZE-1]
mask = col_offsets < N                   # guard out-of-bounds columns
```

`BLOCK_SIZE` is the next power of 2 >= N. If N=1000, BLOCK_SIZE=1024, so the last 24 lanes are masked off.

#### Step 3: Load the Row

```python
x_ptrs = x_ptr + row * stride_m + col_offsets
x = tl.load(x_ptrs, mask=mask, other=float('-inf'))
```

Masked-off positions get `-inf`, which becomes `exp(-inf) = 0` and doesn't affect the sum.

#### Step 4: Pass 1 — Find Row Max

```python
row_max = tl.max(x, axis=0)  # scalar: max across all BLOCK_SIZE elements
```

This is a **reduction** — Triton compiles it to an efficient warp-level reduce.

#### Step 5: Pass 2 — Exponentiate, Sum, Normalize

```python
numerator = tl.exp(x - row_max)          # shift and exp (all elements)
denominator = tl.sum(numerator, axis=0)   # another reduction
result = numerator / denominator          # element-wise divide
```

#### Step 6: Store Result

```python
out_ptrs = out_ptr + row * out_stride_m + col_offsets
tl.store(out_ptrs, result, mask=mask)
```

## Why This is Efficient

| Aspect | Naive PyTorch | Triton Kernel |
|--------|--------------|---------------|
| Memory passes | 3 separate passes (max, exp, sum) each reading from DRAM | 1 load from DRAM, all compute in SRAM |
| Kernel launches | 3+ separate CUDA kernels | 1 fused kernel |
| Data movement | Row loaded 3 times from global memory | Row loaded once, stays in registers |

The key insight: **fusing the max, exp, sum, and divide into a single kernel** avoids redundant global memory reads. Each row is loaded from DRAM exactly once.

## BLOCK_SIZE Choice

```python
BLOCK_SIZE = triton.next_power_of_2(N)
```

- Must be a power of 2 (Triton requirement for `tl.arange`)
- Should be >= N so the entire row fits in one load
- If N is very large (e.g. 65536), you'd need a loop over column chunks — our implementation handles rows that fit in a single block

## Connection to Online Softmax (Day 5)

| | Standard Softmax | Online Softmax | Triton Softmax |
|--|-----------------|----------------|----------------|
| Where | CPU/PyTorch | CPU/PyTorch | GPU kernel |
| Passes over data | 2 (max, then exp+sum) | 1 (streaming blocks) | 1 DRAM read (2 reductions in SRAM) |
| Use case | Reference implementation | FlashAttention building block | Standalone GPU softmax |

Online softmax from Day 5 processes data in blocks with running statistics — that pattern becomes critical in Day 17 when we add it to the attention kernel. Today's Triton softmax is simpler: load the whole row into SRAM, then do standard max-subtract-exp-sum there.

## Running the Code

```bash
python triton_softmax.py
```

This runs:
1. **Correctness tests** — compares against `F.softmax` for various shapes
2. **Numerical stability test** — verifies no NaN/Inf with large values
3. **Benchmark** — measures GB/s for Triton vs PyTorch across different row widths
