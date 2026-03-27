"""
Day 8: Setup Triton & Vector Add Kernel

Your task: implement the vector add kernel and wrapper function.
Follow the TODOs below. Refer to triton_basics_notes.md if you get stuck.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,      # pointer to first input vector
    y_ptr,      # pointer to second input vector
    out_ptr,    # pointer to output vector
    n_elements, # total number of elements
    BLOCK_SIZE: tl.constexpr,  # number of elements per block (compile-time constant)
):
    # TODO 1: Get the program ID for axis 0
    #   Use: tl.program_id(axis)
    pid = tl.program_id(axis=0)

    # TODO 2: Compute the offsets this program handles
    #   Use: tl.arange(start, end) to generate [0, 1, ..., BLOCK_SIZE-1]
    #   Then shift by pid * BLOCK_SIZE to get global indices
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)                 

    # TODO 3: Create a mask for out-of-bounds indices
    #   Compare offsets against n_elements
    mask = offsets < n_elements

    # TODO 4: Load x and y from global memory
    #   Use: tl.load(pointer + offsets, mask=mask)
    x = tl.load(x_ptr+offsets, mask = mask)
    y = tl.load(y_ptr+offsets, mask = mask)

    # TODO 5: Compute the result and store it
    #   Use: tl.store(pointer + offsets, value, mask=mask)
    result = x + y
    tl.store(out_ptr + offsets, result, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper that launches the Triton kernel."""
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    out = torch.empty_like(x)
    n_elements = x.numel()

    # TODO 6: Choose a BLOCK_SIZE (power of 2, e.g. 1024)
    BLOCK_SIZE = 512

    # TODO 7: Compute the grid size (number of programs to launch)
    #   Use: triton.cdiv(a, b) for ceiling division
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # TODO 8: Launch the kernel
    #   Use: kernel_name[grid](args..., BLOCK_SIZE=BLOCK_SIZE)
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


# ── Correctness test (no changes needed) ─────────────────────────
def test_correctness():
    for n in [1, 127, 1024, 100_000, 1_048_576]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        triton_out = vector_add(x, y)
        torch_out = x + y
        assert torch.allclose(triton_out, torch_out), f"Mismatch at n={n}"
        print(f"  n={n:>10d}  ✓")
    print("All correctness tests passed!\n")


# ── Benchmark (no changes needed) ────────────────────────────────
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 25)],  # 4K to 16M
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-throughput",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add(x, y), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)

    # 3 vectors (2 reads + 1 write) × 4 bytes each
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    print("=== Correctness Test ===")
    test_correctness()

    print("=== Benchmark ===")
    benchmark.run(print_data=True, save_path="/home/xfan/projects/playground/AI_Infra_Daily/bench_results")
