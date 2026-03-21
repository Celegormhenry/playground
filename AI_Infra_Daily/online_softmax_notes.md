# Online Softmax (Running Max/Sum)

## Why Online Softmax?

In naive attention, we compute softmax over the **entire** row of scores at once:

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

This requires **two passes** over the data:
1. Pass 1: Find `max(x)` and compute `sum(exp(x_j - max(x)))`
2. Pass 2: Divide each `exp(x_i - max(x))` by the sum

For FlashAttention, we process Q*K^T in **tiles/blocks** — we never have all scores in memory at once. So we need a way to **incrementally update** the softmax as we see new blocks of scores.

---

## The Problem

Say we've computed softmax over the first block of scores `[x_1, ..., x_B]` and get:
- `m_1 = max(x_1, ..., x_B)` (running max)
- `l_1 = sum(exp(x_j - m_1))` for j in 1..B (running sum of exponentials)

Now we see a new block `[x_{B+1}, ..., x_{2B}]`. The global max might change! If it does, all our previous `exp()` values are wrong because they used the old max.

---

## The Solution: Online Softmax (Milakov & Gimelshein, 2018)

We maintain two running statistics:
- **m** = running max seen so far
- **l** = running sum of `exp(x_j - m)` seen so far

When a new block of scores arrives with local max `m_new`:

```
m_combined = max(m_old, m_new)

# Correction factor: rescale old sum to new max
l_combined = l_old * exp(m_old - m_combined) + l_new * exp(m_new - m_combined)
```

The key insight: **`exp(m_old - m_combined)`** is the correction factor. It rescales all the old exponentials from the old max to the new max, without revisiting them individually.

---

## Step-by-Step Example

Scores: `[2, 4, 1, 5, 3]`, processed in blocks of size 2.

### Block 1: [2, 4]
```
m = 4
l = exp(2-4) + exp(4-4) = exp(-2) + 1 = 0.135 + 1.0 = 1.135
```

### Block 2: [1, 5]
```
m_new = 5
m_combined = max(4, 5) = 5
l = 1.135 * exp(4-5) + exp(1-5) + exp(5-5)
  = 1.135 * 0.368 + 0.018 + 1.0
  = 0.418 + 0.018 + 1.0
  = 1.436
```

### Block 3: [3]
```
m_new = 3
m_combined = max(5, 3) = 5  (max doesn't change)
l = 1.436 * exp(5-5) + exp(3-5)
  = 1.436 * 1.0 + 0.135
  = 1.571
```

### Final softmax values:
```
softmax(x_i) = exp(x_i - 5) / 1.571
```

This matches what you'd get computing softmax over the full vector at once!

---

## How This Connects to FlashAttention

In FlashAttention, for each block of K (and V):
1. Compute a tile of scores: `S_block = Q_block @ K_block^T`
2. Find local max of this tile
3. Update running max `m` and running sum `l` using the online softmax formulas
4. **Rescale the running output accumulator**: `O = O * exp(m_old - m_new) + softmax_block @ V_block`

The output accumulator `O` also needs rescaling whenever the max changes, using the same correction factor.

```
# Pseudocode for one row of output
m = -inf
l = 0
O = 0

for each block j:
    S_j = q @ K_j^T / sqrt(d)       # score tile
    m_new = max(m, max(S_j))         # update running max

    # Rescale old accumulator and sum
    correction = exp(m - m_new)
    l = l * correction + sum(exp(S_j - m_new))
    O = O * correction + exp(S_j - m_new) @ V_j

    m = m_new

O = O / l  # final normalization
```

---

## Why This Matters for Memory

- **Naive**: Materializes full N×N attention matrix → O(N²) memory
- **Online/Tiled**: Only holds one tile of scores at a time → O(N) memory
- This is the core trick that makes FlashAttention possible — correct softmax without storing all scores

---

## Key Takeaways

1. Online softmax maintains `(m, l)` — running max and running sum
2. When max changes, multiply old sum by `exp(m_old - m_new)` to correct
3. The output accumulator needs the same correction
4. Result is **numerically identical** to standard softmax
5. This enables **tiled/blocked** attention with O(N) memory instead of O(N²)
