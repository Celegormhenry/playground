# Day 8: Triton Basics — Vector Add

## What is Triton?

Triton is a **Python-based language for writing GPU kernels**. It sits between PyTorch (too high-level, no kernel control) and CUDA (too low-level, manual thread management).

In Triton, you think in **blocks of data**, not individual threads.

## Mental Model: Triton vs CUDA

```
CUDA:   grid → blocks → threads   (you manage all 3 levels)
Triton: grid → programs            (each program handles a BLOCK of elements)
```

A "program" is one instance of your kernel, identified by `tl.program_id(0)`. Each program processes `BLOCK_SIZE` elements.

Triton handles the thread-level parallelism for you — you just say "load this block, compute, store."

---

## Key Triton Primitives

| Triton | CUDA Equivalent | What It Does |
|--------|----------------|--------------|
| `@triton.jit` | `__global__` | JIT-compiles function into a GPU kernel |
| `tl.program_id(0)` | `blockIdx.x` | Which block am I? |
| `tl.arange(0, N)` | `threadIdx.x` | Range of offsets within a block |
| `tl.load(ptr, mask)` | `ptr[idx]` | Load from GPU memory (with bounds check) |
| `tl.store(ptr, val, mask)` | `ptr[idx] = val` | Write to GPU memory |
| `tl.constexpr` | template param | Compile-time constant (for block size) |
| `triton.cdiv(a, b)` | `(a + b - 1) / b` | Ceiling division |

---

## Vector Add: Line-by-Line Walkthrough

### 1. Kernel Definition

```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
```

- `@triton.jit` — tells Triton to JIT-compile this into GPU machine code
- `x_ptr`, `y_ptr`, `out_ptr` — **pointers** to GPU memory (like C pointers, not Python objects)
- `n_elements` — total size of the vectors
- `BLOCK_SIZE: tl.constexpr` — known at compile time, so Triton can unroll loops and optimize

### 2. Identify Which Block We Are

```python
pid = tl.program_id(axis=0)
```

- Each launched program gets a unique `pid` (0, 1, 2, ...)
- `axis=0` means the first (and here, only) grid dimension

### 3. Compute Offsets This Block Owns

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

- `tl.arange(0, BLOCK_SIZE)` generates `[0, 1, 2, ..., BLOCK_SIZE-1]`
- With `pid * BLOCK_SIZE`, each block covers a non-overlapping slice:
  - Block 0 → `[0, 1, ..., 1023]`
  - Block 1 → `[1024, 1025, ..., 2047]`
  - Block 2 → `[2048, 2049, ..., 3071]`
  - ...

### 4. Mask Out-of-Bounds

```python
mask = offsets < n_elements
```

- The last block may extend past the array end
- `mask` is a boolean vector — `True` for valid indices, `False` for out-of-bounds
- Example: if `n_elements = 2500` and block 2 covers `[2048..3071]`, indices 2500–3071 are masked out

### 5. Load, Compute, Store

```python
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
result = x + y
tl.store(out_ptr + offsets, result, mask=mask)
```

- **Pointer arithmetic**: `x_ptr + offsets` = "load elements at these positions"
- Masked loads return 0 for masked-out positions (safe, no segfault)
- Masked stores skip masked-out positions (no garbage written)
- `x + y` is element-wise over the entire block — Triton parallelizes this across threads automatically

### 6. Launching the Kernel

```python
BLOCK_SIZE = 1024
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
```

- `grid` = number of programs to launch = `ceil(n / BLOCK_SIZE)`
- `kernel[grid](args)` is the launch syntax (analogous to `<<<grid, block>>>` in CUDA)
- `BLOCK_SIZE` is passed as a keyword because it's a `constexpr`

---

## Why Masking Matters

Without masking, the last block would read/write past the end of the array → **undefined behavior** (garbage reads, memory corruption).

```
Array:  [0  1  2  3  4  5  6  7  8  9]     (n=10, BLOCK_SIZE=4)
Block 0: [0  1  2  3]     mask=[T T T T]
Block 1: [4  5  6  7]     mask=[T T T T]
Block 2: [8  9  ?  ?]     mask=[T T F F]   ← mask prevents OOB access
```

---

## Triton vs CUDA: Same Kernel Comparison

### CUDA Version
```c
__global__ void vector_add(float *x, float *y, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}
// Launch: vector_add<<<cdiv(n, 256), 256>>>(x, y, out, n);
```

### Triton Version
```python
@triton.jit
def vector_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)
```

Key difference: In CUDA, one thread handles one element. In Triton, one program handles `BLOCK_SIZE` elements — Triton decides how to map that to threads internally.

---

## Key Takeaways

1. **Triton abstracts threads** — you think in blocks, not individual threads
2. **Pointer arithmetic** — `ptr + offsets` is how you address memory (like C)
3. **Always mask** — the last block can go out of bounds
4. **`constexpr`** — block sizes must be compile-time constants for Triton to optimize
5. **`kernel[grid](args)`** — launch syntax; grid = number of programs
6. **Triton auto-tunes** — it handles shared memory, coalescing, and thread mapping for you

---

## Exercises

- [ ] Run `triton_vector_add.py` — verify correctness across sizes 1 to 1M
- [ ] Read the benchmark output — is Triton faster or same speed as PyTorch for vector add?
- [ ] Try changing `BLOCK_SIZE` to 512 or 2048 — does it affect performance?
- [ ] Think about: why is vector add a poor showcase for Triton's advantage? (Hint: it's memory-bound and trivially simple — PyTorch's built-in op is already optimal)
