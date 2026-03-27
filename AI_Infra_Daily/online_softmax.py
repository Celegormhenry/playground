"""
Day 5: Online Softmax — running max/sum approach.

Demonstrates that processing scores in blocks gives identical results
to standard full-vector softmax.
"""

import torch
import torch.nn.functional as F


def standard_softmax(x: torch.Tensor) -> torch.Tensor:
    """Standard 2-pass softmax over last dimension.

    Example: x = [2.0, 4.0, 1.0]

    Pass 1 — find max (for numerical stability):
        m = max(2.0, 4.0, 1.0) = 4.0

    Pass 2 — subtract max, exponentiate, normalize:
        x - m    = [2-4, 4-4, 1-4]       = [-2.0, 0.0, -3.0]
        e        = exp([-2, 0, -3])       = [0.1353, 1.0, 0.0498]
        sum(e)   = 0.1353 + 1.0 + 0.0498 = 1.1851
        e/sum(e) = [0.1142, 0.8438, 0.0420]  <- final softmax
    """
    m = x.max(dim=-1, keepdim=True).values          # Pass 1: find max
    e = torch.exp(x - m)                             # Pass 2: shift & exp
    return e / e.sum(dim=-1, keepdim=True)           # normalize


def online_softmax(x: torch.Tensor, block_size: int = 2) -> torch.Tensor:
    """
    Online softmax: process x in blocks, maintaining running max and sum.

    For each block:
        1. Compute local max
        2. Update global running max
        3. Correct old running sum with exp(m_old - m_new)
        4. Add new block's contribution to running sum

    Args:
        x: (..., N) tensor
        block_size: number of elements per block
    Returns:
        softmax result, same shape as x
    """
    N = x.shape[-1]
    shape = x.shape[:-1]

    # TODO: Initialize running statistics
    #   m = running max, init to -inf, shape (*shape, 1)
    #   l = running sum, init to 0,    shape (*shape, 1)
    m = torch.full((*shape, 1), float('-inf'),device=x.device) # (*shape, 1)
    l = torch.zeros((*shape,1),device=x.device) # (*shape, 1)

    for start in range(0, N, block_size):
        end = min(start+block_size, N)
        x_block = x[..., start:end] # (*shape, block_size)
        m_block = x_block.max(dim = -1, keepdim = True).values # (*shape, 1)
        m_new = torch.maximum(m, m_block) # (*shape, 1)
        correction = torch.exp(m - m_new) # (*shape, 1)
        l = l * correction + torch.exp(x_block - m_new).sum(dim=-1, keepdim=True) # (*shape, 1)
        m = m_new # (*shape, 1)

    return torch.exp(x - m) / l # (..., N)


def online_softmax_with_output_accumulation(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 2
) -> torch.Tensor:
    """
    Tiled attention using online softmax — the FlashAttention pattern.

    For each row of Q, processes K/V in blocks while maintaining:
        - m: running max of scores
        - l: running sum of exp(scores - m)
        - O: running weighted sum of V (the output accumulator)

    Args:
        Q: (B, H, N, d)
        K: (B, H, N, d)
        V: (B, H, N, d)
        block_size: number of K/V rows per block
    Returns:
        output: (B, H, N, d)
    """
    B, H, N, d = Q.shape
    scale = d ** -0.5

    # TODO: Initialize accumulators
    #   m = running max,    shape (B, H, N, 1), init to -inf
    #   l = running sum,    shape (B, H, N, 1), init to 0
    #   O = output accumulator, shape (B, H, N, d), init to 0
    m = torch.full((B, H, N, 1), float('-inf'), device=Q.device)
    l = torch.zeros((B, H, N, 1), device=Q.device)
    O = torch.zeros((B, H, N, d), device=Q.device)

    # TODO: Loop over K, V blocks along the sequence dimension
    # Pseudocode for one row of output
    # m = -inf
    # l = 0
    # O = 0

    # for each block j:
    #     S_j = q @ K_j^T / sqrt(d)       # score tile
    #     m_new = max(m, max(S_j))         # update running max

    #     # Rescale old accumulator and sum
    #     correction = exp(m - m_new)
    #     l = l * correction + sum(exp(S_j - m_new))
    #     O = O * correction + exp(S_j - m_new) @ V_j

    #     m = m_new

    # O = O / l  # final normalization
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        K_block = K[:, :, start:end, :] #B, H, block_size, d
        V_block = V[:, :, start:end, :] #B, H, block_size, d
        S_block = torch.matmul(Q, K_block.transpose(-2, -1)) #B, H, N, block_size
        S_block = S_block * scale #B, H, N, block_size
        m_block = S_block.max(dim = -1, keepdim=True).values #B, H, N, 1
        m_new = torch.maximum(m, m_block) #B, H, N, 1
        correction_ratio = torch.exp(m-m_new) # B, H, N, 1
        P_block = torch.exp(S_block-m_new) # B, H, N, block_size
        l = l*correction_ratio + P_block.sum(dim=-1, keepdim=True) #B, H, N, 1
        O = O*correction_ratio + torch.matmul(P_block, V_block)  #B, H, N, d
        m = m_new

    # Final normalization
    O = O / l
    return O


# ──────────────────────────── Tests ────────────────────────────

def test_standard_softmax():
    """Standard softmax should match PyTorch's F.softmax and sum to 1."""
    torch.manual_seed(42)
    x = torch.randn(3, 10)

    out = standard_softmax(x)
    ref = F.softmax(x, dim=-1)

    assert torch.allclose(out, ref, atol=1e-6), "Should match F.softmax"
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        "Rows should sum to 1"
    assert (out >= 0).all(), "All values should be non-negative"

    # Large values — check numerical stability
    x_large = torch.tensor([[1000.0, 1001.0, 1002.0]])
    out_large = standard_softmax(x_large)
    assert not torch.any(torch.isnan(out_large)), "No NaNs with large values"
    assert not torch.any(torch.isinf(out_large)), "No Infs with large values"
    print("[PASS] standard_softmax matches F.softmax, sums to 1, numerically stable")


def test_online_softmax_matches_standard():
    """Online softmax should match standard softmax exactly."""
    torch.manual_seed(42)
    x = torch.randn(3, 10)

    ref = standard_softmax(x)
    for bs in [1, 2, 3, 5, 10]:
        out = online_softmax(x, block_size=bs)
        assert torch.allclose(out, ref, atol=1e-6), \
            f"Mismatch with block_size={bs}"
    print("[PASS] online_softmax matches standard softmax")


def test_online_softmax_sums_to_one():
    """Each row of online softmax should sum to 1."""
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    out = online_softmax(x, block_size=3)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        "Rows should sum to 1"
    print("[PASS] online_softmax rows sum to 1")


def test_tiled_attention_matches_naive():
    """Tiled attention with online softmax should match naive attention."""
    torch.manual_seed(42)
    B, H, N, d = 2, 4, 16, 32
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    # Reference: naive attention
    ref = F.scaled_dot_product_attention(Q, K, V)

    for bs in [1, 2, 4, 7, 16]:
        out = online_softmax_with_output_accumulation(Q, K, V, block_size=bs)
        assert torch.allclose(out, ref, atol=1e-5), \
            f"Tiled attention mismatch with block_size={bs}"
    print("[PASS] tiled attention matches PyTorch SDPA for all block sizes")


def test_online_softmax_numerical_stability():
    """Online softmax should handle large values without overflow."""
    x = torch.tensor([[1000.0, 1001.0, 1002.0]])
    out = online_softmax(x, block_size=1)
    ref = standard_softmax(x)
    assert torch.allclose(out, ref, atol=1e-6), "Should handle large values"
    assert not torch.any(torch.isnan(out)), "No NaNs"
    assert not torch.any(torch.isinf(out)), "No Infs"
    print("[PASS] numerically stable with large values")


def demo_running_stats():
    """Print step-by-step running max/sum to build intuition."""
    x = torch.tensor([2.0, 4.0, 1.0, 5.0, 3.0])
    block_size = 2

    print("\n=== Online Softmax Step-by-Step ===")
    print(f"Input: {x.tolist()}, block_size={block_size}\n")

    m = torch.tensor(float('-inf'))
    l = torch.tensor(0.0)

    for i, start in enumerate(range(0, len(x), block_size)):
        end = min(start + block_size, len(x))
        block = x[start:end]

        m_block = block.max()
        m_new = torch.maximum(m, m_block)
        correction = torch.exp(m - m_new)

        l_old = l.item()
        l = l * correction + torch.exp(block - m_new).sum()

        print(f"Block {i+1}: {block.tolist()}")
        print(f"  local max = {m_block.item():.1f}")
        print(f"  m: {m.item():.1f} -> {m_new.item():.1f}")
        print(f"  correction = exp({m.item():.1f} - {m_new.item():.1f}) = {correction.item():.4f}")
        print(f"  l: {l_old:.4f} * {correction.item():.4f} + sum(exp(block - {m_new.item():.1f})) = {l.item():.4f}")
        print()

        m = m_new

    result = torch.exp(x - m) / l
    ref = standard_softmax(x.unsqueeze(0)).squeeze(0)

    print(f"Final m = {m.item():.1f}, l = {l.item():.4f}")
    print(f"Online result:   {result.tolist()}")
    print(f"Standard result: {ref.tolist()}")
    print(f"Match: {torch.allclose(result, ref, atol=1e-6)}")


def bench_softmax_speed():
    """Compare speed of standard vs online softmax."""
    import time

    sizes = [64, 256, 1024, 4096]
    block_sizes = [16, 64, 256]
    warmup, repeats = 5, 50

    print("\n=== Softmax Speed Comparison ===")
    print(f"{'N':>6}  {'standard (ms)':>14}  ", end="")
    for bs in block_sizes:
        print(f"{'online bs=' + str(bs) + ' (ms)':>20}", end="  ")
    print()
    print("-" * (24 + 22 * len(block_sizes)))

    for N in sizes:
        x = torch.randn(32, N)

        # Warmup & bench standard
        for _ in range(warmup):
            standard_softmax(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            standard_softmax(x)
        t_std = (time.perf_counter() - t0) / repeats * 1000

        print(f"{N:>6}  {t_std:>14.4f}  ", end="")

        for bs in block_sizes:
            for _ in range(warmup):
                online_softmax(x, block_size=bs)
            t0 = time.perf_counter()
            for _ in range(repeats):
                online_softmax(x, block_size=bs)
            t_online = (time.perf_counter() - t0) / repeats * 1000
            print(f"{t_online:>20.4f}", end="  ")
        print()


def bench_attention_speed():
    """Compare speed of naive attention vs tiled (online softmax) attention."""
    import time

    configs = [(2, 4, 64, 32), (2, 4, 256, 64), (2, 4, 1024, 64)]
    block_sizes = [16, 64, 256]
    warmup, repeats = 3, 20

    print("\n=== Attention Speed Comparison ===")
    print(f"{'(B,H,N,d)':>16}  {'naive (ms)':>12}  ", end="")
    for bs in block_sizes:
        print(f"{'tiled bs=' + str(bs) + ' (ms)':>20}", end="  ")
    print()
    print("-" * (32 + 22 * len(block_sizes)))

    for B, H, N, d in configs:
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        # Naive attention
        for _ in range(warmup):
            F.scaled_dot_product_attention(Q, K, V)
        t0 = time.perf_counter()
        for _ in range(repeats):
            F.scaled_dot_product_attention(Q, K, V)
        t_naive = (time.perf_counter() - t0) / repeats * 1000

        print(f"{'(' + ','.join(map(str,[B,H,N,d])) + ')':>16}  {t_naive:>12.4f}  ", end="")

        for bs in block_sizes:
            for _ in range(warmup):
                online_softmax_with_output_accumulation(Q, K, V, block_size=bs)
            t0 = time.perf_counter()
            for _ in range(repeats):
                online_softmax_with_output_accumulation(Q, K, V, block_size=bs)
            t_tiled = (time.perf_counter() - t0) / repeats * 1000
            print(f"{t_tiled:>20.4f}", end="  ")
        print()

    print("\nNote: Python-level tiling is slower than naive PyTorch/C++ ops.")
    print("The real win comes from CUDA kernels that keep tiles in SRAM (FlashAttention).")


if __name__ == "__main__":
    test_standard_softmax()
    test_online_softmax_matches_standard()
    test_online_softmax_sums_to_one()
    test_tiled_attention_matches_naive()
    test_online_softmax_numerical_stability()
    demo_running_stats()
    print("\nAll tests passed!")
    bench_softmax_speed()
    bench_attention_speed()
