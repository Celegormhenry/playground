"""
Flash Attention CUDA — correctness test & benchmark.

Compares:
  1. PyTorch SDPA (cuDNN/FA2 under the hood)
  2. Python tiled attention (online_softmax_with_output_accumulation)
  3. CUDA v1: tile K/V only (one block per query row)
  4. CUDA v2: tile both Q and K/V (one block per Q tile, 256 threads)
  5. CUDA v3: same as v2 but uses Tensor Cores (wmma) for matmuls
"""

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from online_softmax import online_softmax_with_output_accumulation

# ── JIT compile CUDA kernels ──
print("Compiling CUDA kernels (first run takes ~30s)...")
flash_cuda = load(
    name="flash_attention_cuda",
    sources=["flash_attention_cuda.cu"],
    verbose=False,
)
print("Done.\n")


# ───────────────────── Correctness ─────────────────────

def test_correctness():
    print("=== Correctness Tests ===")
    torch.manual_seed(42)

    for D in [32, 64, 128]:
        B, H, N = 2, 4, 128
        Q = torch.randn(B, H, N, D, device="cuda")
        K = torch.randn(B, H, N, D, device="cuda")
        V = torch.randn(B, H, N, D, device="cuda")

        ref = F.scaled_dot_product_attention(Q, K, V)

        # v1
        for bs in [16, 32, 64]:
            smem = 2 * bs * D * 4 + 20
            if smem > 48 * 1024:
                continue
            out = flash_cuda.flash_attn_v1_fwd(Q, K, V, bs)
            err = (out - ref).abs().max().item()
            assert err < 1e-3, f"v1 D={D} bs={bs}: err {err}"
        print(f"  v1 D={D:>3d}  [PASS]")

        # v2
        out = flash_cuda.flash_attn_v2_fwd(Q, K, V)
        err = (out - ref).abs().max().item()
        assert err < 1e-3, f"v2 D={D}: err {err}"
        print(f"  v2 D={D:>3d}  max_error={err:.2e}  [PASS]")

        # v3 (Tensor Core)
        out = flash_cuda.flash_attn_v3_fwd(Q, K, V)
        err = (out - ref).abs().max().item()
        assert err < 1e-2, f"v3 D={D}: err {err}"  # fp16 matmul → slightly higher tolerance
        print(f"  v3 D={D:>3d}  max_error={err:.2e}  [PASS]")

    print()


# ───────────────────── Benchmark ─────────────────────

def bench(fn, warmup=10, repeats=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def bench_attention():
    configs = [
        (2, 8, 256, 64),
        (2, 8, 512, 64),
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (4, 8, 1024, 64),
        (4, 8, 2048, 64),
    ]

    header = f"{'(B,H,N,d)':>18}  {'SDPA':>10}  {'py tiled':>10}  {'CUDA v1':>10}  {'CUDA v2':>10}  {'CUDA v3':>10}  {'v3/SDPA':>8}  {'v2->v3':>8}"
    print("=== Attention Speed Comparison (ms) ===")
    print(header)
    print("-" * len(header))

    for B, H, N, d in configs:
        Q = torch.randn(B, H, N, d, device="cuda")
        K = torch.randn(B, H, N, d, device="cuda")
        V = torch.randn(B, H, N, d, device="cuda")

        label = f"({B},{H},{N},{d})"

        t_sdpa = bench(lambda: F.scaled_dot_product_attention(Q, K, V))

        t_py = bench(
            lambda: online_softmax_with_output_accumulation(Q, K, V, block_size=32),
            warmup=3, repeats=10,
        )

        t_v1 = bench(lambda: flash_cuda.flash_attn_v1_fwd(Q, K, V, 64))

        t_v2 = bench(lambda: flash_cuda.flash_attn_v2_fwd(Q, K, V))

        t_v3 = bench(lambda: flash_cuda.flash_attn_v3_fwd(Q, K, V))

        v3_vs_sdpa = t_v3 / t_sdpa
        v2_to_v3 = t_v2 / t_v3

        print(f"{label:>18}  {t_sdpa:>10.4f}  {t_py:>10.4f}  {t_v1:>10.4f}  {t_v2:>10.4f}  {t_v3:>10.4f}  {v3_vs_sdpa:>7.1f}x  {v2_to_v3:>7.1f}x")

    print()
    print("v3/SDPA = how many times slower v3 is vs PyTorch (lower = better)")
    print("v2->v3  = speedup from Tensor Cores (higher = better)")


if __name__ == "__main__":
    test_correctness()
    bench_attention()
