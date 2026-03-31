"""
Day 11: Triton Row Softmax Kernel

Implements row-wise softmax in Triton: for each row, compute
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))

Key concepts:
- One program per row — each program handles an entire row
- Two-pass approach within each program:
    Pass 1: find row max (for numerical stability)
    Pass 2: compute exp(x - max) and sum, then normalize
- For rows wider than BLOCK_SIZE, loop over columns in chunks
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,          # pointer to input  (M, N)
    out_ptr,        # pointer to output (M, N)
    M,              # number of rows
    N,              # number of columns
    stride_m,       # elements to skip to move one row in x
    out_stride_m,   # elements to skip to move one row in out
    BLOCK_SIZE: tl.constexpr,  # tile width (must be power of 2, >= N ideally)
):
    """Each program computes softmax for one row."""

    # TODO 1: Get the row index from program_id
    #   Use: tl.program_id(axis)
    row_id = tl.program_id(axis=0)

    # TODO 2: Compute column offsets [0, 1, ..., BLOCK_SIZE-1] and mask OOB columns
    #   Use: tl.arange(0, BLOCK_SIZE) for offsets
    #   Compare offsets against N for the mask
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # TODO 3: Compute pointers for this row and load the row
    #   Pointers: x_ptr + row * stride_m + col_offsets
    #   Use: tl.load(ptrs, mask=mask, other=float('-inf'))
    #   Note: masked positions get -inf so exp(-inf)=0, won't affect sum
    x_ptrs = x_ptr + row_id * stride_m + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=float('-inf'))


    # TODO 4: Pass 1 — find the row max for numerical stability
    #   Use: tl.max(x, axis=0) to reduce across the row
    row_max = tl.max(x, axis=0)

    # TODO 5: Pass 2 — subtract max, exponentiate, sum, normalize
    #   numerator = tl.exp(x - row_max)
    #   denominator = tl.sum(numerator, axis=0)
    #   result = numerator / denominator
    numerator = tl.exp(x - row_max)                                                                                                                                
    denominator = tl.sum(numerator, axis=0)                                                                                                                        
    result = numerator / denominator

    # TODO 6: Store the result
    #   Pointers: out_ptr + row * out_stride_m + col_offsets
    #   Use: tl.store(ptrs, result, mask=mask)
    out_ptrs = out_ptr + row_id * out_stride_m + col_offsets
    tl.store(out_ptrs, result, mask=mask)


@triton.jit
def softmax_exp2_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    out_stride_m,
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax using exp2 — single hardware instruction on NVIDIA GPUs."""
    row_id = tl.program_id(axis=0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x_ptrs = x_ptr + row_id * stride_m + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=float('-inf'))

    row_max = tl.max(x, axis=0)

    # exp2 optimization: exp(x) = exp2(x * log2(e))
    LOG2E: tl.constexpr = 1.4426950408889634  # log2(e)
    numerator = tl.math.exp2((x - row_max) * LOG2E)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    out_ptrs = out_ptr + row_id * out_stride_m + col_offsets
    tl.store(out_ptrs, result, mask=mask)


@triton.jit
def softmax_online_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    out_stride_m,
    BLOCK_SIZE: tl.constexpr,
):
    """Online softmax — processes the row in BLOCK_SIZE chunks.

    Works for any row width, even when N >> BLOCK_SIZE.
    Pass 1: loop over chunks to compute running max and sum
    Pass 2: loop again to compute exp(x - max) / sum and store
    """
    row_id = tl.program_id(axis=0)
    row_start = x_ptr + row_id * stride_m

    # ── Pass 1: compute row max and sum using online algorithm ──
    m = float('-inf')  # running max
    l = 0.0            # running sum of exp

    for start in range(0, N, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x_block = tl.load(row_start + col_offsets, mask=mask, other=float('-inf'))

        m_block = tl.max(x_block, axis=0)
        m_new = tl.maximum(m, m_block)
        # Correct old sum: exp(m_old - m_new) rescales previous terms
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x_block - m_new), axis=0)
        m = m_new

    # ── Pass 2: normalize and store ──
    out_row_start = out_ptr + row_id * out_stride_m

    for start in range(0, N, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x_block = tl.load(row_start + col_offsets, mask=mask, other=float('-inf'))

        result = tl.exp(x_block - m) / l
        tl.store(out_row_start + col_offsets, result, mask=mask)


def softmax_online(x: torch.Tensor, block_size: int = 4096) -> torch.Tensor:
    """Wrapper for online softmax kernel. Handles arbitrarily wide rows."""
    assert x.is_cuda
    assert x.ndim == 2

    M, N = x.shape
    out = torch.empty_like(x)
    # Use fixed block size — no need for next_power_of_2(N)
    BLOCK_SIZE = triton.next_power_of_2(min(block_size, N))
    grid = (M,)
    softmax_online_kernel[grid](x, out, M, N, x.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out


def softmax_exp2(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for the exp2-optimized softmax kernel."""
    assert x.is_cuda
    assert x.ndim == 2

    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_exp2_kernel[grid](x, out, M, N, x.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Wrapper that launches the Triton softmax kernel on a 2D tensor."""
    assert x.is_cuda
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"

    M, N = x.shape
    out = torch.empty_like(x)

    # TODO 7: Choose BLOCK_SIZE — must be a power of 2 >= N
    #   Use: triton.next_power_of_2(N)
    BLOCK_SIZE = triton.next_power_of_2(N)

    # TODO 8: Compute the grid — one program per row
    #   grid = (M,)
    grid = (M,)

    # TODO 9: Launch the kernel
    #   softmax_kernel[grid](x, out, M, N, x.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    softmax_kernel[grid](x, out, M, N, x.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)

    return out


# ── Correctness tests ────────────────────────────────────────────

def test_correctness():
    """Triton softmax should match PyTorch F.softmax for various shapes."""
    configs = [
        (1, 3),
        (4, 16),
        (32, 128),
        (64, 1000),     # non-power-of-2 columns
        (128, 1024),
        (256, 4096),
    ]
    for M, N in configs:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        triton_out = softmax(x)
        torch_out = F.softmax(x, dim=-1)
        assert torch.allclose(triton_out, torch_out, atol=1e-6), \
            f"Mismatch at ({M}, {N}): max diff = {(triton_out - torch_out).abs().max().item()}"

        # Check rows sum to 1
        row_sums = triton_out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"Rows don't sum to 1 at ({M}, {N})"

        print(f"  ({M:>5d}, {N:>5d})  max diff = {(triton_out - torch_out).abs().max().item():.2e}  ✓")
    print("All correctness tests passed!\n")


def test_numerical_stability():
    """Should handle large values without overflow."""
    x = torch.tensor([[1000.0, 1001.0, 1002.0]], device="cuda")
    out = softmax(x)
    ref = F.softmax(x, dim=-1)
    assert torch.allclose(out, ref, atol=1e-6), "Large value mismatch"
    assert not torch.any(torch.isnan(out)), "NaN detected"
    assert not torch.any(torch.isinf(out)), "Inf detected"
    print("  Numerical stability test passed! ✓\n")


# ── Benchmark ────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        line_arg="provider",
        line_vals=["triton", "triton_exp2", "triton_online", "torch"],
        line_names=["Triton", "Triton-exp2", "Triton-online", "PyTorch"],
        styles=[("blue", "-"), ("red", "-"), ("orange", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    elif provider == "triton_exp2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_exp2(x), quantiles=quantiles)
    elif provider == "triton_online":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_online(x), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.softmax(x, dim=-1), quantiles=quantiles)

    # 2 vectors worth of memory traffic: 1 read + 1 write, each M*N float32
    gbps = lambda ms: 2 * M * N * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    print("=== Correctness Test ===")
    test_correctness()

    print("=== Correctness Test (online) ===")
    for M, N in [(1, 3), (4, 16), (64, 1000), (128, 1024), (256, 4096), (64, 32768), (32, 65536)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = softmax_online(x)
        ref = F.softmax(x, dim=-1)
        assert torch.allclose(out, ref, atol=1e-5), \
            f"online mismatch at ({M}, {N}): max diff = {(out - ref).abs().max().item()}"
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
            f"online rows don't sum to 1 at ({M}, {N})"
        print(f"  ({M:>5d}, {N:>5d})  max diff = {(out - ref).abs().max().item():.2e}  ✓")
    print("All online correctness tests passed!\n")

    print("=== Correctness Test (exp2) ===")
    for M, N in [(1, 3), (64, 1000), (256, 4096)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        out = softmax_exp2(x)
        ref = F.softmax(x, dim=-1)
        assert torch.allclose(out, ref, atol=1e-6), \
            f"exp2 mismatch at ({M}, {N}): max diff = {(out - ref).abs().max().item()}"
        print(f"  ({M:>5d}, {N:>5d})  max diff = {(out - ref).abs().max().item():.2e}  ✓")
    print("All exp2 correctness tests passed!\n")

    print("=== Numerical Stability Test ===")
    test_numerical_stability()

    print("=== Benchmark ===")
    benchmark.run(print_data=True, save_path="/home/xfan/projects/playground/AI_Infra_Daily/bench_results")
