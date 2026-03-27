"""
Day 10: Triton Matrix Multiplication Kernel

Implements a tiled matmul kernel in Triton.
C = A @ B  where A is (M, K) and B is (K, N).

Key concepts:
- 2D grid: each program computes one BLOCK_M x BLOCK_N tile of C
- Inner loop iterates over K dimension in BLOCK_K chunks
- Accumulator stays in registers (SRAM) -- only final result written to DRAM
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (number of elements to skip to move one row/col)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes (compile-time constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Each program instance computes a BLOCK_M x BLOCK_N tile of C."""

    # TODO 1: Get the 1D program ID, then derive pid_m and pid_n from it.
    #   Use: tl.program_id(axis=0) to get the flat pid
    #   Use: tl.cdiv(N, BLOCK_N) to get the number of column tiles
    #   Then: pid_m = pid // num_n_tiles, pid_n = pid % num_n_tiles
    pid = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    # TODO 2: Compute row and column offsets for this tile.
    #   Use: tl.arange(0, BLOCK_M) and tl.arange(0, BLOCK_N)
    #   Shift by pid_m * BLOCK_M and pid_n * BLOCK_N respectively
    offset_m =  tl.arange(0, BLOCK_M)
    offset_n =  tl.arange(0, BLOCK_N)
    offset_m = pid_m * BLOCK_M + offset_m
    offset_n = pid_n * BLOCK_N + offset_n


    # TODO 3: Initialize K-dimension offsets and build 2D pointer blocks for A and B.
    #   Use: tl.arange(0, BLOCK_K) for offs_k
    #   A pointers [BLOCK_M, BLOCK_K]: a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    #   B pointers [BLOCK_K, BLOCK_N]: b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    offset_k = tl.arange(0, BLOCK_K) 
    a_ptr = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptr = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn


    # TODO 4: Create a float32 accumulator of shape (BLOCK_M, BLOCK_N), initialized to zero.
    #   Use: tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # TODO 5: Loop over K in steps of BLOCK_K. In each iteration:
    #   a) Build masks for A [BLOCK_M, BLOCK_K] and B [BLOCK_K, BLOCK_N] to guard OOB
    #   b) Load A and B tiles with tl.load(..., mask=..., other=0.0)
    #   c) Accumulate with: acc += tl.dot(a_tile, b_tile)
    #   d) Advance a_ptrs by BLOCK_K * stride_ak, b_ptrs by BLOCK_K * stride_bk
                                                                                                                                                                 
    for k_start in range(0, K, BLOCK_K):                                                                                                                           
        a_mask = (offset_m[:, None] < M) & (offset_k[None, :] + k_start < K)                                                                                           
        b_mask = (offset_k[:, None] + k_start < K) & (offset_n[None, :] < N)                                                                                           
                                                                                                                                                                    
        a_tile = tl.load(a_ptr, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptr, mask=b_mask, other=0.0)
        acc += tl.dot(a_tile, b_tile, allow_tf32=False)                                                                                                                              
    
        a_ptr += BLOCK_K * stride_ak                                                                                                                              
        b_ptr += BLOCK_K * stride_bk


    # TODO 6: Write the accumulator back to C.
    #   Build c_ptrs [BLOCK_M, BLOCK_N]: c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    #   Build a mask for OOB, then use tl.store(c_ptrs, acc, mask=...)
    # Build C pointers [BLOCK_M, BLOCK_N]                                                                                                                        
    c_ptrs = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn                                                                                     
                                                                                                                                                                    
    # Mask for OOB                                                                                                                                                 
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)                                                                                                         
                                                                                                                                                                    
    # Store                                                                                                                                                        
    tl.store(c_ptrs, acc, mask=c_mask)
    


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper that launches the Triton matmul kernel."""
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0], f"Incompatible shapes: {a.shape} @ {b.shape}"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # TODO 7: Choose tile sizes (powers of 2). Suggested: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    BLOCK_M=64 
    BLOCK_N=64
    BLOCK_K=32

    # TODO 8: Compute the grid size -- total number of output tiles.
    #   Use: triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    #   Grid is a 1-tuple: (num_tiles,)
    grid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (grid,)

    # TODO 9: Launch the kernel.
    #   matmul_kernel[grid](a, b, c, M, N, K,
    #       a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    #       BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    return c


# -- Correctness test --------------------------------------------------------
def test_correctness():
    configs = [
        (1, 1, 1),
        (16, 16, 16),
        (64, 64, 64),
        (128, 256, 512),
        (333, 444, 555),    # non-power-of-2 to test masking
        (1024, 1024, 1024),
    ]
    for M, N, K in configs:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        triton_out = matmul(a, b)
        torch_out = a @ b
        assert torch.allclose(triton_out, torch_out, atol=1e-3, rtol=1e-3), \
            f"Mismatch at ({M}, {N}, {K}): max diff = {(triton_out - torch_out).abs().max().item()}"
        print(f"  ({M:>5d}, {N:>5d}, {K:>5d})  max diff = {(triton_out - torch_out).abs().max().item():.2e}  ok")
    print("All correctness tests passed!\n")


# -- Benchmark ---------------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[(2**i, 2**i, 2**i) for i in range(7, 13)],  # 128 to 4096
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: a @ b, quantiles=quantiles)

    # FLOPS for matmul: 2 * M * N * K (multiply + add per output element)
    flops = 2.0 * M * N * K
    tflops = lambda ms: flops / ms * 1e-9  # ms -> s is 1e-3, TFLOPS is 1e-12, net 1e-9
    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    print("=== Correctness Test ===")
    test_correctness()

    print("=== Benchmark ===")
    benchmark.run(print_data=True, save_path="/home/xfan/projects/playground/AI_Infra_Daily/bench_results")
