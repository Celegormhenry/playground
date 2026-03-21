# Naive Scaled Dot-Product Attention — Tensor Transforms

## Input Tensors

Assume: **B** = batch, **H** = num_heads, **N** = seq_len, **d** = head_dim

| Tensor | Shape |
|--------|-------|
| Q (Query) | `(B, H, N, d)` |
| K (Key) | `(B, H, N, d)` |
| V (Value) | `(B, H, N, d)` |

---

## Step-by-Step Tensor Transforms

### Step 1: Transpose K

```
K^T = K.transpose(-2, -1)
```

| | Shape |
|---|---|
| K | `(B, H, N, d)` |
| K^T | `(B, H, d, N)` |

#### How `transpose(-2, -1)` works

`torch.Tensor.transpose(dim0, dim1)` swaps two dimensions of a tensor. Negative indices count from the end: `-1` is the last dim, `-2` is second-to-last.

For K with shape `(B, H, N, d)`, the dims are indexed as:

```
dim:    0    1    2    3
       (B)  (H)  (N)  (d)
               dim -2 ↑  ↑ dim -1
```

`transpose(-2, -1)` swaps dim 2 and dim 3:

```
Before: (B, H, N, d)    — each head holds N row-vectors of length d
After:  (B, H, d, N)    — each head holds d row-vectors of length N
```

**Concrete example** — a single head, N=3, d=2:

```
K = [[k00, k01],       shape: (3, 2)
     [k10, k11],       3 tokens, each a 2-dim vector
     [k20, k21]]

K^T = [[k00, k10, k20],  shape: (2, 3)
       [k01, k11, k21]]  rows and cols swapped
```

This is needed because `Q @ K^T` computes dot products between all pairs of query and key vectors:

```
Q @ K^T  →  (N, d) @ (d, N) = (N, N)
```

Without the transpose, the inner dimensions wouldn't align for matmul.

**Note:** `transpose` returns a **view** — no data is copied. It just changes how strides index into the same memory.

---

### Step 2: Compute Raw Scores (QK^T)

```
scores = Q @ K^T
```

| | Shape |
|---|---|
| Q | `(B, H, N, d)` |
| K^T | `(B, H, d, N)` |
| **scores** | **`(B, H, N, N)`** |

Each `(N, N)` matrix holds the dot-product similarity between every query position and every key position. Element `[i, j]` = how much query `i` attends to key `j`.

---

### Step 3: Scale

```
scores = scores / sqrt(d)
```

| | Shape |
|---|---|
| scores | `(B, H, N, N)` — unchanged |

Division by `sqrt(d)` (a scalar) keeps values from growing too large before softmax, which would push gradients toward zero.

---

### Step 4: Softmax

```
attn_weights = softmax(scores, dim=-1)
```

| | Shape |
|---|---|
| attn_weights | `(B, H, N, N)` — unchanged |

Softmax is applied along the **last dimension** (the key dimension). Each row now sums to 1 — it's a probability distribution over keys for each query.

---

### Step 5: Weighted Sum of Values

```
output = attn_weights @ V
```

| | Shape |
|---|---|
| attn_weights | `(B, H, N, N)` |
| V | `(B, H, N, d)` |
| **output** | **`(B, H, N, d)`** |

Each query position's output is a weighted combination of all value vectors, where the weights come from the softmax attention distribution.

---

## Summary Diagram

```
Q (B,H,N,d)   K (B,H,N,d)
     |              |
     |         transpose(-2,-1)
     |              |
     |         K^T (B,H,d,N)
     |              |
     +--- matmul ---+
            |
      scores (B,H,N,N)
            |
       / sqrt(d)
            |
      scores (B,H,N,N)
            |
       softmax(dim=-1)
            |
      attn_weights (B,H,N,N)       V (B,H,N,d)
            |                           |
            +-------- matmul -----------+
                        |
                  output (B,H,N,d)
```

## Memory Complexity

The `scores` matrix is `(B, H, N, N)` — **O(N^2)** in sequence length. This is the bottleneck that Flash Attention eliminates by tiling the computation and never materializing the full `N x N` matrix.
