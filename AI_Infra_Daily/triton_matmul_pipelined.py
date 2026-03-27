"""
Triton Matrix Multiplication: Basic vs Pipelined vs PyTorch

Compares three implementations:
1. Basic kernel (from triton_matmul.py)
2. Pipelined kernel with software prefetching + L2 swizzling
3. PyTorch (cuBLAS)
"""

import torch
import triton
import triton.language as tl


# ── Basic kernel (same as triton_matmul.py) ──────────────────────────────────
@triton.jit
def matmul_basic(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)

    a_ptr = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptr = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a_mask = (offset_m[:, None] < M) & (offset_k[None, :] + k_start < K)
        b_mask = (offset_k[:, None] + k_start < K) & (offset_n[None, :] < N)
        a_tile = tl.load(a_ptr, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptr, mask=b_mask, other=0.0)
        acc += tl.dot(a_tile, b_tile, allow_tf32=False)
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ── Pipelined kernel with prefetch + L2 swizzling ───────────────────────────
@triton.jit
def matmul_pipelined(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)

    # L2 cache swizzling: group tiles so nearby pids share B tiles in L2
    num_tiles_in_group = GROUP_SIZE * num_n_tiles
    group_id = pid // num_tiles_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + ((pid % num_tiles_in_group) % group_size_m)
    pid_n = (pid % num_tiles_in_group) // group_size_m

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        a_mask = (offset_m[:, None] < M) & (offset_k[None, :] + k_start < K)
        b_mask = (offset_k[:, None] + k_start < K) & (offset_n[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a_tile, b_tile, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ── Autotuned kernel (basic structure, no swizzling) ─────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotuned(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)

    a_ptr = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptr = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a_mask = (offset_m[:, None] < M) & (offset_k[None, :] + k_start < K)
        b_mask = (offset_k[:, None] + k_start < K) & (offset_n[None, :] < N)
        a_tile = tl.load(a_ptr, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptr, mask=b_mask, other=0.0)
        acc += tl.dot(a_tile, b_tile, allow_tf32=False)
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ── Split-K kernel (autotuned) ───────────────────────────────────────────────
# Instead of 1 block doing ALL of K, split K across SPLIT_K blocks.
# Each block computes a partial sum, then atomic-adds into the output.
#
# Basic:    grid = (M_tiles * N_tiles,)          — 1 block per output tile
# Split-K:  grid = (M_tiles * N_tiles, SPLIT_K)  — SPLIT_K blocks per output tile
#
# Helps when M,N are small (few output tiles) but K is large.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 16}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr'],
)
@triton.jit
def matmul_splitk_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # axis=0: which output tile, axis=1: which K-split
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    # This block handles K range: [k_start_range, k_end_range)
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start_range = pid_k * k_per_split
    k_end_range = min(k_start_range + k_per_split, K)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)

    # Start A/B pointers at this split's K offset
    a_ptrs = a_ptr + offset_m[:, None] * stride_am + (k_start_range + offset_k[None, :]) * stride_ak
    b_ptrs = b_ptr + (k_start_range + offset_k[:, None]) * stride_bk + offset_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(k_start_range, k_end_range, BLOCK_K):
        a_mask = (offset_m[:, None] < M) & (offset_k[None, :] + k_start < k_end_range)
        b_mask = (offset_k[:, None] + k_start < k_end_range) & (offset_n[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a_tile, b_tile, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Atomic add partial results into C (since multiple blocks write to same tile)
    c_ptrs = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.atomic_add(c_ptrs, acc, mask=c_mask)


# ── Wrappers ─────────────────────────────────────────────────────────────────
def matmul_v1(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    matmul_basic[grid](a, b, c, M, N, K,
                       a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1),
                       BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return c


def matmul_v2(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    matmul_pipelined[grid](a, b, c, M, N, K,
                           a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1),
                           BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_SIZE=8)
    return c


def matmul_v3(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_autotuned[grid](a, b, c, M, N, K,
                           a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1))
    return c


def matmul_v4(a, b):
    M, K = a.shape
    K, N = b.shape
    # Must zero-init because we use atomic_add
    # Use pre_hook to re-zero c before each autotune trial
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K'],
    )
    matmul_splitk_kernel[grid](a, b, c, M, N, K,
                               a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                               c.stride(0), c.stride(1))
    return c


# ── Correctness ──────────────────────────────────────────────────────────────
def test_correctness():
    configs = [(64, 64, 64), (128, 256, 512), (333, 444, 555), (1024, 1024, 1024)]
    for M, N, K in configs:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = a @ b
        for name, fn in [("basic", matmul_v1), ("pipelined", matmul_v2), ("autotuned", matmul_v3), ("split-k", matmul_v4)]:
            out = fn(a, b)
            diff = (out - ref).abs().max().item()
            assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), \
                f"{name} mismatch at ({M},{N},{K}): {diff}"
            print(f"  {name:>10s}  ({M:>5d},{N:>5d},{K:>5d})  max diff = {diff:.2e}  ok")
    print("All correctness tests passed!\n")


# ── Benchmark ────────────────────────────────────────────────────────────────
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[(2**i, 2**i, 2**i) for i in range(7, 13)],  # 128 to 4096
        line_arg="provider",
        line_vals=["basic", "autotuned", "split-k", "torch"],
        line_names=["Basic", "Autotuned", "Split-K", "PyTorch"],
        styles=[("blue", "-"), ("orange", "-"), ("purple", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-square",
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "basic":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_v1(a, b), quantiles=quantiles)
    elif provider == "autotuned":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_v3(a, b), quantiles=quantiles)
    elif provider == "split-k":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_v4(a, b), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: a @ b, quantiles=quantiles)

    flops = 2.0 * M * N * K
    tflops = lambda ms: flops / ms * 1e-9
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ── Tall-K Benchmark (fixed M=N=64, sweep K) ────────────────────────────────
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["K"],
        x_vals=[2**i for i in range(8, 16)],  # 256 to 32768
        line_arg="provider",
        line_vals=["basic", "autotuned", "split-k", "torch"],
        line_names=["Basic", "Autotuned", "Split-K", "PyTorch"],
        styles=[("blue", "-"), ("orange", "-"), ("purple", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-tallK-M64-N64",
        args={"M": 64, "N": 64},
    )
)
def benchmark_tallk(M, N, K, provider):
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "basic":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_v1(a, b), quantiles=quantiles)
    elif provider == "autotuned":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_v3(a, b), quantiles=quantiles)
    elif provider == "split-k":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_v4(a, b), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: a @ b, quantiles=quantiles)

    flops = 2.0 * M * N * K
    tflops = lambda ms: flops / ms * 1e-9
    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    print("=== Correctness Test ===")
    test_correctness()

    print("=== Benchmark (mixed shapes) ===")
    benchmark.run(print_data=True, save_path="/home/xfan/projects/playground/AI_Infra_Daily/bench_results")

    print("\n=== Benchmark (tall-K: M=64, N=64, sweep K) ===")
    benchmark_tallk.run(print_data=True, save_path="/home/xfan/projects/playground/AI_Infra_Daily/bench_results")
