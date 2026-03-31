# Day 14: Triton Concepts — Week 2 Reference

A consolidated reference of everything learned in Week 2 (Days 8–13).

---

## 1. What Triton Is

Triton is a Python-based GPU kernel language. It sits between PyTorch (no kernel control) and CUDA (manual thread management). You write in **blocks**, not threads — Triton handles thread-level parallelism, shared memory, and memory coalescing automatically.

```
CUDA:   grid → blocks → threads       (you manage all 3)
Triton: grid → programs                (each program = one block of work)
```

---

## 2. Core Programming Model

### Program = One Tile of Work

Every Triton kernel is a **program** identified by `tl.program_id(axis)`. Each program processes a contiguous block of elements. The grid tells the GPU how many programs to launch.

```python
@triton.jit
def my_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                        # which program am I?
    offs = pid * BLOCK + tl.arange(0, BLOCK)       # my elements
    mask = offs < n                                 # guard OOB
    x = tl.load(x_ptr + offs, mask=mask)           # load
    tl.store(x_ptr + offs, x * 2, mask=mask)       # store
```

Launch: `my_kernel[(triton.cdiv(n, BLOCK),)](x, n, BLOCK=BLOCK)`

### Key Primitives

| Triton | Purpose |
|--------|---------|
| `@triton.jit` | JIT-compile to GPU code |
| `tl.program_id(axis)` | Which program instance (like `blockIdx`) |
| `tl.arange(0, N)` | Offset range within a block |
| `tl.load(ptr, mask)` | Masked load from GPU memory |
| `tl.store(ptr, val, mask)` | Masked store to GPU memory |
| `tl.constexpr` | Compile-time constant (block sizes) |
| `tl.dot(a, b)` | Tile-level matrix multiply |
| `tl.max / tl.sum` | Reductions (compiled to warp-level ops) |
| `tl.exp / tl.math.exp2` | Element-wise math |
| `tl.zeros((M, N), dtype)` | Initialize an accumulator |
| `tl.atomic_add(ptr, val)` | Atomic accumulate (for split-K, etc.) |
| `triton.cdiv(a, b)` | Ceiling division |
| `triton.next_power_of_2(n)` | Next power of 2 ≥ n |

---

## 3. Pointer Arithmetic

Triton uses C-style pointer arithmetic. Tensors are accessed via `base_ptr + offsets`:

```python
# 1D: load elements [pid*BLOCK .. pid*BLOCK + BLOCK)
offs = pid * BLOCK + tl.arange(0, BLOCK)
x = tl.load(x_ptr + offs, mask=offs < n)

# 2D: load a tile [rows, cols] using stride
row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # shape: (BLOCK_M,)
col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # shape: (BLOCK_N,)
ptrs = base_ptr + row_offs[:, None] * stride_row + col_offs[None, :] * stride_col
# ptrs shape: (BLOCK_M, BLOCK_N) — a 2D block of pointers
```

The `[:, None]` and `[None, :]` broadcasting creates a 2D grid of pointers from 1D offset vectors. This is the fundamental pattern for all 2D kernels (matmul, attention, etc.).

---

## 4. Masking

The last block often extends past the array boundary. Without masking → undefined behavior.

```
Array:  [0 1 2 3 4 5 6 7 8 9]    n=10, BLOCK=4
Block 2: indices [8 9 10 11]
Mask:            [T T  F  F]      ← prevents OOB access
```

Rules:
- **Loads**: `other=0.0` or `other=float('-inf')` for masked positions
- **Stores**: masked positions are skipped (no garbage written)
- **Reductions**: choose `other` carefully — `-inf` for max, `0.0` for sum

---

## 5. Kernel Patterns Learned

### Pattern 1: Element-wise (Vector Add — Day 8-9)

- **Grid**: 1D, one program per block of elements
- **Structure**: load → compute → store
- **Insight**: Trivially memory-bound; PyTorch's built-in is already optimal. Good for learning, not a performance showcase.

### Pattern 2: Tiled Reduction (Matmul — Day 10)

- **Grid**: 1D (flattened 2D), one program per output tile
- **Structure**: 2D tile decomposition with inner loop over K
- **Key idea**: Accumulator stays in registers (SRAM). Only the final result writes to HBM.

```python
# Each program computes one BLOCK_M × BLOCK_N tile of C
pid_m = pid // num_n_tiles
pid_n = pid % num_n_tiles
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

for k in range(0, K, BLOCK_K):
    a_tile = tl.load(a_ptrs, mask=..., other=0.0)
    b_tile = tl.load(b_ptrs, mask=..., other=0.0)
    acc += tl.dot(a_tile, b_tile)
    # advance pointers along K
```

### Pattern 3: Row-wise Reduction (Softmax — Day 11)

- **Grid**: 1D, one program per row
- **Structure**: load entire row → reduce (max, sum) → normalize → store
- **Key idea**: Fusing max + exp + sum + divide into one kernel avoids 3 separate HBM round-trips.

### Pattern 4: Online/Streaming Reduction (Online Softmax — Day 11)

- **Grid**: 1D, one program per row
- **Structure**: loop over row in chunks, maintain running `(m, l)` statistics
- **Key idea**: Handles rows wider than BLOCK_SIZE. This is the pattern used in FlashAttention.

```python
m = float('-inf')  # running max
l = 0.0            # running exp sum
for chunk in row_chunks:
    m_new = max(m, max(chunk))
    l = l * exp(m - m_new) + sum(exp(chunk - m_new))
    m = m_new
```

---

## 6. Optimization Techniques (Day 12-13)

### L2 Cache Swizzling (GROUP_SIZE)

Reorders which programs run on nearby SMs so they share B tiles in L2 cache:

```python
# Without swizzling: pid 0→(0,0), pid 1→(0,1), pid 2→(0,2), ...
# With swizzling:    pid 0→(0,0), pid 1→(1,0), pid 2→(0,1), pid 3→(1,1), ...
# Adjacent pids now share the same B columns → L2 cache reuse
```

### Autotuning (`@triton.autotune`)

Triton tries multiple configurations and picks the fastest:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],  # re-tune when these change
)
```

- `num_warps`: parallelism within a program (more warps = more threads)
- `num_stages`: software pipelining depth (overlap load and compute)

### Split-K

When M, N are small but K is large, there aren't enough output tiles to saturate the GPU. Split-K launches multiple programs per output tile, each handling a slice of K, then `tl.atomic_add` to accumulate.

```
Basic:    grid = (M_tiles × N_tiles,)             — 1 program per tile
Split-K:  grid = (M_tiles × N_tiles, SPLIT_K)     — SPLIT_K programs per tile
```

### exp2 Optimization

`exp2(x * log2(e))` maps to a single GPU hardware instruction, vs `exp(x)` which is a software approximation. Identical results, slightly faster.

---

## 7. Performance Mental Model

### Memory-Bound vs Compute-Bound

| | Memory-bound | Compute-bound |
|--|-------------|---------------|
| Bottleneck | HBM bandwidth | FLOPS |
| Example | Vector add, softmax | Large matmul |
| Metric | GB/s | TFLOPS |
| Optimization | Reduce HBM traffic (fuse ops) | Increase arithmetic intensity |

### Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / bytes moved
```

- Vector add: ~0.17 FLOP/byte → always memory-bound
- Matmul (N=4096): ~4096 FLOP/byte → compute-bound
- Softmax: ~5 FLOP/byte → memory-bound (optimization = fewer HBM passes)

### Benchmarking

```python
# Triton's built-in benchmarking
ms, min_ms, max_ms = triton.testing.do_bench(lambda: kernel_call(), quantiles=[0.5, 0.2, 0.8])

# Compute throughput
gbps = bytes_moved / ms * 1e-6          # for memory-bound ops
tflops = 2 * M * N * K / ms * 1e-9      # for matmul
```

---

## 8. Common Pitfalls

1. **Forgetting masks** → OOB access, silent wrong results
2. **Non-power-of-2 BLOCK_SIZE** → Triton compilation error with `tl.arange`
3. **Wrong `other` value** → `other=0.0` in a max reduction gives wrong answer; use `float('-inf')`
4. **Integer overflow in pointer math** → use `tl.int64` offsets for large tensors
5. **Not advancing pointers in K-loop** → reads same tile every iteration
6. **Missing `allow_tf32=False`** → `tl.dot` uses TF32 by default, which is lossy for fp32 validation

---

## 9. Triton vs CUDA Summary

| Aspect | CUDA | Triton |
|--------|------|--------|
| Unit of work | Thread | Program (block of elements) |
| Shared memory | Manual `__shared__` allocation | Automatic |
| Thread sync | `__syncthreads()` | Not needed |
| Memory coalescing | Manual alignment | Automatic |
| Warp-level ops | `__shfl_*`, `__reduce_*` | `tl.max`, `tl.sum` |
| Launch syntax | `<<<grid, block>>>` | `kernel[grid](args)` |
| Tuning | Manual | `@triton.autotune` |
| Language | C++ | Python |

Triton trades some low-level control for dramatically simpler code. For most attention/matmul workloads, Triton reaches 80-95% of hand-tuned CUDA performance.

---

## 10. Files Reference

| File | Day | What It Implements |
|------|-----|--------------------|
| `triton_vector_add.py` | 8-9 | Element-wise vector add + benchmark |
| `triton_matmul.py` | 10 | Tiled matmul kernel |
| `triton_matmul_pipelined.py` | 12-13 | Pipelined + autotuned + split-K matmul |
| `triton_softmax.py` | 11 | Row softmax (standard, exp2, online) + benchmark |
| `triton_basics_notes.md` | 8 | Vector add walkthrough |
| `triton_softmax_notes.md` | 11 | Softmax kernel walkthrough |
