"""
Days 16–17: FlashAttention Triton Kernel

Day 16 — Tiled QK score computation:
    Kernel that computes S = Q @ K^T * (1/sqrt(d)) using tiles.
    Grid: (num_q_tiles * num_k_tiles, B*H)
    Outputs the full N×N score matrix for validation.

Day 17 — Online softmax + output accumulation:
    Full FlashAttention forward: each program computes BLOCK_M rows of
    output for one head, looping over all K/V tiles with online softmax.
    The N×N attention matrix never touches HBM.
    Grid: (ceil(N/BLOCK_M), B*H)

Builds on:
    - Day 10: triton_matmul.py (tiled dot products, 2D pointer arithmetic)
    - Day 11: triton_softmax.py (online softmax kernel)
    - Day 15: attention_tiling_strategy.md (algorithm & SRAM budget)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════
# Day 16: Tiled QK Score Computation
# ═══════════════════════════════════════════════════════════════════
#
# Goal: verify that loading Q and K in tiles and computing
#       S_tile = Q_tile @ K_tile^T * scale
# gives the same result as the naive  S = Q @ K^T * scale.
#
# This kernel materializes the full N×N score matrix — FlashAttention
# avoids this, but we need it here to test the tiling arithmetic.
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def qk_tile_kernel(
    Q_ptr, K_ptr, S_ptr,
    N,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_sbh, stride_sn, stride_sk,
    scale,
    BLOCK_M: tl.constexpr,   # rows of Q per tile
    BLOCK_N: tl.constexpr,   # rows of K per tile (= cols of S per tile)
    D: tl.constexpr,         # head dimension (loaded in full, not tiled)
):
    """
    Compute one (BLOCK_M, BLOCK_N) tile of the score matrix:
        S[q_start:q_end, k_start:k_end] = Q_tile @ K_tile^T * scale

    Grid: (num_q_tiles * num_k_tiles, B * H)
      axis 0 → flat tile index (row-major over Q-tiles and K-tiles)
      axis 1 → which batch*head

    Hints:
      1. Decode program IDs:
         - pid_tile = tl.program_id(0), pid_bh = tl.program_id(1)
         - From pid_tile, compute pid_q (which Q tile) and pid_k (which K tile)
           using num_k_tiles = tl.cdiv(N, BLOCK_N)
      2. Compute row/col offsets:
         - q_offs = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
         - k_offs = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
         - d_offs = tl.arange(0, D)
      3. Build pointer arrays using strides (2D indexing):
         - q_ptrs = Q_ptr + pid_bh * stride_qbh + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd
         - Same pattern for K and S
      4. Load Q tile (BLOCK_M, D) and K tile (BLOCK_N, D) with masks for boundary
      5. Compute S_tile = tl.dot(Q_tile, tl.trans(K_tile), allow_tf32=False) * scale
      6. Store S_tile with combined mask (q_mask[:, None] & k_mask[None, :])
    """
    pid_tile = tl.program_id(0)
    pid_bh = tl.program_id(1)
    num_k_tiles = tl.cdiv(N, BLOCK_N)
    pid_q = pid_tile // num_k_tiles
    pid_k = pid_tile  % num_k_tiles

    q_offs = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)

    q_ptrs = Q_ptr + pid_bh*stride_qbh + q_offs[:, None]*stride_qn + d_offs[None, :] * stride_qd
    k_ptrs = K_ptr + pid_bh*stride_kbh + k_offs[:, None]*stride_kn + d_offs[None, :] * stride_kd
    s_ptrs = S_ptr + pid_bh*stride_sbh + q_offs[:, None]*stride_sn + k_offs[None, :] * stride_sk

    q_mask = q_offs < N   # 1D (BLOCK_M,)
    k_mask = k_offs < N   # 1D (BLOCK_N,)

    Q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)   # (BLOCK_M, D)
    K_tile = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)   # (BLOCK_N, D)

    S_tile = tl.dot(Q_tile, tl.trans(K_tile), allow_tf32=False) * scale

    s_mask = q_mask[:, None] & k_mask[None, :]   # (BLOCK_M, BLOCK_N)
    tl.store(s_ptrs, S_tile, mask=s_mask)










def qk_tile_scores(
    Q: torch.Tensor, K: torch.Tensor,
    BLOCK_M: int = 64, BLOCK_N: int = 64,
) -> torch.Tensor:
    """
    Wrapper: compute Q @ K^T * (1/sqrt(d)) via tiled Triton kernel.

    Args:
        Q, K: (B, H, N, d) on CUDA, float32
    Returns:
        S: (B, H, N, N) scaled score matrix

    Hints:
      1. Extract B, H, N, d from Q.shape; compute scale = d ** -0.5
      2. Flatten to (B*H, N, d) with .reshape().contiguous()
      3. Allocate output S: (B*H, N, N)
      4. Grid = (num_q_tiles * num_k_tiles, B*H)
      5. Call qk_tile_kernel[grid](...) passing all strides via .stride()
      6. Reshape output back to (B, H, N, N)
    """
    dtype = torch.float32
    B, H, N, d = Q.shape
    scale = d ** -0.5
    Q_flat = Q.reshape(B*H,N,d).contiguous()
    K_flat = K.reshape(B*H,N,d).contiguous()
    S_flat = torch.empty(B*H,N,N, device=Q.device, dtype = dtype)

    grid = (triton.cdiv(N,BLOCK_M)*triton.cdiv(N,BLOCK_N), B*H)

    qk_tile_kernel[grid](Q_flat, K_flat, S_flat,
    N,
    Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
    K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
    S_flat.stride(0), S_flat.stride(1), S_flat.stride(2),
    scale = scale,
    BLOCK_M = BLOCK_M,   # rows of Q per tile
    BLOCK_N = BLOCK_N,   # rows of K per tile (= cols of S per tile)
    D=d,         # head dimension (loaded in full, not tiled)
    )

    return S_flat.reshape(B, H, N, N)


# ═══════════════════════════════════════════════════════════════════
# Day 17: FlashAttention Forward — Online Softmax + V Accumulation
# ═══════════════════════════════════════════════════════════════════
#
# Core idea:  for each Q tile, loop over ALL K/V tiles while
# maintaining running softmax statistics (m, l) in registers.
#
# Per-iteration update (for each KV block j):
#     S_j      = Q_tile @ K_j^T * scale          (BLOCK_M, BLOCK_N)
#     m_new    = max(m, rowmax(S_j))              (BLOCK_M,)
#     corr     = exp(m_old - m_new)               (BLOCK_M,)
#     P_j      = exp(S_j - m_new)                 (BLOCK_M, BLOCK_N)
#     l        = l * corr + rowsum(P_j)           (BLOCK_M,)
#     O_acc    = O_acc * corr + P_j @ V_j         (BLOCK_M, D)
#     m        = m_new
#
# After all KV tiles:
#     O = O_acc / l                               (deferred normalization)
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr = False,
):
    """
    FlashAttention forward: compute O = softmax(Q @ K^T / sqrt(d)) @ V
    without materializing the N×N attention matrix.

    Grid: (ceil(N / BLOCK_M), B * H)
      axis 0 → which Q tile (BLOCK_M rows of output)
      axis 1 → which batch*head

    Hints:
      1. Decode pid_q and pid_bh from program IDs
      2. Load Q tile (BLOCK_M, D) — stays in SRAM the entire inner loop
         - q_offs = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
         - d_offs = tl.arange(0, D)
         - Build q_ptrs with strides, load with mask (q_offs < N)
      3. Initialize accumulators:
         - m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)  # running max
         - l = tl.zeros((BLOCK_M,), dtype=tl.float32)                # running sum
         - O_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)          # output
      4. Inner loop: for kv_start in range(0, N, BLOCK_N):
         a. Load K_tile (BLOCK_N, D) and V_tile (BLOCK_N, D)
         b. S = tl.dot(Q_tile, tl.trans(K_tile)) * scale    # (BLOCK_M, BLOCK_N)
         c. Mask out-of-bounds keys: S = tl.where(kv_mask[None, :], S, float('-inf'))
         d. Online softmax update:
            - m_block = tl.max(S, axis=1)
            - m_new = tl.maximum(m, m_block)
            - correction = tl.exp(m - m_new)
            - P = tl.exp(S - m_new[:, None])
            - l = l * correction + tl.sum(P, axis=1)
            - O_acc = O_acc * correction[:, None] + tl.dot(P.to(V_tile.dtype), V_tile)
            - m = m_new
      5. After loop: O_acc = O_acc / l[:, None]
      6. Store O_acc to output with mask
    """
    # TODO: implement the kernel body
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_offs = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)
    q_ptrs = Q_ptr + pid_bh*stride_qbh + q_offs[:, None]*stride_qn + d_offs[None, :] * stride_qd
    mask_q = q_offs < N
    Q_tile = tl.load(q_ptrs, mask=mask_q[:,None], other=0.0)

    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)  # running max
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)                # running sum
    O_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)          # output
    for kv_start in range(0, N, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)   # (BLOCK_N,)
        kv_mask = kv_offs < N

        # Load K tile: (BLOCK_N, D)
        k_ptrs = K_ptr + pid_bh * stride_kbh + kv_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        # Load V tile: (BLOCK_N, D)
        v_ptrs = V_ptr + pid_bh * stride_vbh + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        S = tl.dot(Q_tile, tl.trans(K_tile), allow_tf32=ALLOW_TF32) * scale
        S = tl.where(kv_mask[None, :], S, float('-inf'))

        #online softmax
        m_block = tl.max(S, axis=1)
        m_new = tl.maximum(m, m_block)
        correction = tl.exp(m - m_new)
        P = tl.exp(S - m_new[:, None])
        l = l * correction + tl.sum(P, axis=1)
        O_acc = O_acc * correction[:, None] + tl.dot(P.to(V_tile.dtype), V_tile, allow_tf32=ALLOW_TF32)
        m = m_new

    O_acc = O_acc / l[:, None]

    # Store output tile
    o_ptrs = O_ptr + pid_bh * stride_obh + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od
    tl.store(o_ptrs, O_acc, mask=mask_q[:, None])


def flash_attention_fwd(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    BLOCK_M: int = 64, BLOCK_N: int = 64,
    allow_tf32: bool = False,
) -> torch.Tensor:
    """
    FlashAttention forward pass using Triton.

    Args:
        Q, K, V: (B, H, N, d) float32 tensors on CUDA
        BLOCK_M:  rows of Q per tile
        BLOCK_N:  rows of K/V per tile
    Returns:
        O: (B, H, N, d) attention output

    Hints:
      1. Extract B, H, N, d; compute scale = d ** -0.5
      2. Flatten Q, K, V to (B*H, N, d) with .reshape().contiguous()
      3. Allocate O = torch.empty_like(Q_flat)
      4. Grid = (triton.cdiv(N, BLOCK_M), B * H)
      5. Call flash_attention_fwd_kernel[grid](...) with all strides
         - Don't forget num_stages=1
      6. Reshape O back to (B, H, N, d)
    """
    # TODO: implement the wrapper
    dtype = torch.float32
    B, H, N, d = Q.shape
    scale = d ** -0.5
    Q_flat = Q.reshape(B*H, N, d).contiguous()
    K_flat = K.reshape(B*H, N, d).contiguous()
    V_flat = V.reshape(B*H, N, d).contiguous()
    O_flat = torch.empty_like(Q_flat)

    grid = (triton.cdiv(N, BLOCK_M), B * H)

    flash_attention_fwd_kernel[grid](
        Q_flat, K_flat, V_flat, O_flat,
        N,
        Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
        K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
        V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
        O_flat.stride(0), O_flat.stride(1), O_flat.stride(2),
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=d,
        ALLOW_TF32=allow_tf32,
        num_stages=1,
    )

    return O_flat.reshape(B, H, N, d)


# ═══════════════════════════════════════════════════════════════════
# Autotuned FlashAttention — searches over BLOCK_M, BLOCK_N,
# num_warps, num_stages to find the fastest config per (N, d).
# ═══════════════════════════════════════════════════════════════════

def _fa_autotune_configs():
    configs = []
    for BLOCK_M in [32, 64, 128]:
        for BLOCK_N in [32, 64, 128]:
            for num_warps in [2, 4, 8]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config(
                            {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N},
                            num_warps=num_warps, num_stages=num_stages,
                        )
                    )
    return configs


@triton.autotune(configs=_fa_autotune_configs(), key=['N', 'D'])
@triton.jit
def flash_attention_fwd_kernel_tuned(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr = False,
):
    """Same algorithm as flash_attention_fwd_kernel, but with autotune."""
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_offs = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)
    q_ptrs = Q_ptr + pid_bh * stride_qbh + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd
    mask_q = q_offs < N
    Q_tile = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for kv_start in range(0, N, BLOCK_N):
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < N

        k_ptrs = K_ptr + pid_bh * stride_kbh + kv_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        v_ptrs = V_ptr + pid_bh * stride_vbh + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        S = tl.dot(Q_tile, tl.trans(K_tile), allow_tf32=ALLOW_TF32) * scale
        S = tl.where(kv_mask[None, :], S, float('-inf'))

        m_block = tl.max(S, axis=1)
        m_new = tl.maximum(m, m_block)
        correction = tl.exp(m - m_new)
        P = tl.exp(S - m_new[:, None])
        l = l * correction + tl.sum(P, axis=1)
        O_acc = O_acc * correction[:, None] + tl.dot(P.to(V_tile.dtype), V_tile, allow_tf32=ALLOW_TF32)
        m = m_new

    O_acc = O_acc / l[:, None]

    o_ptrs = O_ptr + pid_bh * stride_obh + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od
    tl.store(o_ptrs, O_acc, mask=mask_q[:, None])


def flash_attention_fwd_tuned(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    allow_tf32: bool = False,
) -> torch.Tensor:
    """Autotuned FlashAttention forward — Triton picks the best block sizes."""
    B, H, N, d = Q.shape
    scale = d ** -0.5
    Q_flat = Q.reshape(B * H, N, d).contiguous()
    K_flat = K.reshape(B * H, N, d).contiguous()
    V_flat = V.reshape(B * H, N, d).contiguous()
    O_flat = torch.empty_like(Q_flat)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), B * H)

    flash_attention_fwd_kernel_tuned[grid](
        Q_flat, K_flat, V_flat, O_flat,
        N,
        Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
        K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
        V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
        O_flat.stride(0), O_flat.stride(1), O_flat.stride(2),
        scale,
        D=d,
        ALLOW_TF32=allow_tf32,
    )

    return O_flat.reshape(B, H, N, d)


# ═══════════════════════════════════════════════════════════════════
# Tests — DO NOT MODIFY (use these to validate your implementation)
# ═══════════════════════════════════════════════════════════════════

def test_qk_tile_scores():
    """Day 16: tiled Q @ K^T should match naive Q @ K^T * scale."""
    print("=== Day 16: QK Tile Computation ===")
    torch.manual_seed(42)

    configs = [
        # (B, H, N, d, block_sizes_to_test)
        (1, 1, 64, 64,   [(32, 32), (64, 64), (64, 32)]),
        (2, 4, 128, 64,  [(32, 32), (64, 64), (64, 32)]),
        (2, 4, 256, 128, [(32, 32), (32, 64)]),           # smaller blocks for d=128 (SRAM limit)
        (1, 1, 100, 64,  [(32, 32), (64, 64)]),           # non-power-of-2 N (tests masking)
    ]

    for B, H, N, d, block_sizes in configs:
        Q = torch.randn(B, H, N, d, device="cuda")
        K = torch.randn(B, H, N, d, device="cuda")

        # Reference: naive Q @ K^T * scale
        scale = d ** -0.5
        ref = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Triton tiled version
        for bm, bn in block_sizes:
            out = qk_tile_scores(Q, K, BLOCK_M=bm, BLOCK_N=bn)
            err = (out - ref).abs().max().item()
            assert err < 1e-3, \
                f"QK mismatch (B={B},H={H},N={N},d={d}) bm={bm},bn={bn}: err={err}"

        print(f"  (B={B}, H={H}, N={N:>4d}, d={d:>3d})  [PASS]")

    print()


def test_flash_attention_correctness():
    """Day 17: FlashAttention should match PyTorch SDPA."""
    print("=== Day 17: Flash Attention — Online Softmax ===")
    torch.manual_seed(42)

    configs = [
        (1, 1, 64, 64,   [(32, 32), (64, 64), (64, 32)]),
        (2, 4, 128, 64,  [(32, 32), (64, 64), (64, 32)]),
        (2, 8, 256, 64,  [(32, 32), (64, 64)]),
        (2, 4, 256, 128, [(32, 32), (32, 64)]),             # smaller blocks for d=128
        (1, 1, 100, 64,  [(32, 32), (64, 64)]),             # non-power-of-2 N
    ]

    for B, H, N, d, block_sizes in configs:
        Q = torch.randn(B, H, N, d, device="cuda")
        K = torch.randn(B, H, N, d, device="cuda")
        V = torch.randn(B, H, N, d, device="cuda")

        ref = F.scaled_dot_product_attention(Q, K, V)

        for bm, bn in block_sizes:
            out = flash_attention_fwd(Q, K, V, BLOCK_M=bm, BLOCK_N=bn)
            err = (out - ref).abs().max().item()
            assert err < 1e-3, \
                f"Attn mismatch (B={B},H={H},N={N},d={d}) bm={bm},bn={bn}: err={err}"

        print(f"  (B={B}, H={H}, N={N:>4d}, d={d:>3d})  max_err={err:.2e}  [PASS]")

    print()


def test_flash_attention_numerical_stability():
    """Online softmax should handle large logits without overflow."""
    print("=== Numerical Stability ===")
    torch.manual_seed(0)
    B, H, N, d = 1, 1, 64, 64

    # Large Q values → large scores before softmax
    Q = torch.randn(B, H, N, d, device="cuda") * 10.0
    K = torch.randn(B, H, N, d, device="cuda")
    V = torch.randn(B, H, N, d, device="cuda")

    out = flash_attention_fwd(Q, K, V)
    ref = F.scaled_dot_product_attention(Q, K, V)

    assert not torch.any(torch.isnan(out)), "NaN in output"
    assert not torch.any(torch.isinf(out)), "Inf in output"
    err = (out - ref).abs().max().item()
    assert err < 1e-2, f"Large-logit error: {err}"
    print(f"  max_err={err:.2e}  [PASS]\n")


def bench_flash_attention():
    """Compare Triton FlashAttention vs PyTorch SDPA."""
    print("=== Benchmark: Triton FlashAttn vs PyTorch SDPA (ms) ===")
    header = f"{'(B,H,N,d)':>18}  {'SDPA':>10}  {'Triton FA':>10}  {'ratio':>8}"
    print(header)
    print("-" * len(header))

    configs = [
        (2, 8, 256, 64),
        (2, 8, 512, 64),
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (4, 8, 1024, 64),
    ]

    for B, H, N, d in configs:
        Q = torch.randn(B, H, N, d, device="cuda")
        K = torch.randn(B, H, N, d, device="cuda")
        V = torch.randn(B, H, N, d, device="cuda")

        t_sdpa = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V)
        )
        t_triton = triton.testing.do_bench(
            lambda: flash_attention_fwd(Q, K, V)
        )
        ratio = t_triton / t_sdpa

        label = f"({B},{H},{N},{d})"
        print(f"{label:>18}  {t_sdpa:>10.4f}  {t_triton:>10.4f}  {ratio:>7.1f}x")

    print()
    print("ratio = Triton / SDPA  (lower = better, <1x means Triton is faster)")


def bench_precision_modes():
    """Compare Triton FlashAttention across fp32, tf32, fp16 vs PyTorch SDPA."""
    print("=== Benchmark: fp32 vs tf32 vs fp16 vs SDPA (ms) ===")
    header = f"{'config':>22}  {'SDPA fp32':>10}  {'SDPA fp16':>10}  {'Tri fp32':>10}  {'Tri tf32':>10}  {'Tri fp16':>10}"
    print(header)
    print("-" * len(header))

    configs = [
        # GPT-2 style: H=12, d=64
        (1, 12, 512, 64),
        (1, 12, 1024, 64),
        (1, 12, 2048, 64),
        # LLaMA-7B style: H=32, d=128
        (1, 32, 512, 128),
        (1, 32, 1024, 128),
        (1, 32, 2048, 128),
        (1, 32, 4096, 128),
    ]

    for B, H, N, d in configs:
        Q_f32 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
        K_f32 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
        V_f32 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
        Q_f16 = Q_f32.half()
        K_f16 = K_f32.half()
        V_f16 = V_f32.half()

        # SDPA fp32
        t_sdpa_f32 = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q_f32, K_f32, V_f32))
        # SDPA fp16
        t_sdpa_f16 = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q_f16, K_f16, V_f16))
        # Triton fp32 (allow_tf32=False)
        t_tri_f32 = triton.testing.do_bench(
            lambda: flash_attention_fwd(Q_f32, K_f32, V_f32, allow_tf32=False))
        # Triton tf32 (allow_tf32=True)
        t_tri_tf32 = triton.testing.do_bench(
            lambda: flash_attention_fwd(Q_f32, K_f32, V_f32, allow_tf32=True))
        # Triton fp16
        t_tri_f16 = triton.testing.do_bench(
            lambda: flash_attention_fwd(Q_f16, K_f16, V_f16, allow_tf32=False))

        label = f"({B},{H},{N},{d})"
        print(f"{label:>22}  {t_sdpa_f32:>10.4f}  {t_sdpa_f16:>10.4f}  {t_tri_f32:>10.4f}  {t_tri_tf32:>10.4f}  {t_tri_f16:>10.4f}")

    print()
    print("fp32  = exact float32 matmul")
    print("tf32  = TF32 Tensor Core (fp32 input, 10-bit mantissa matmul)")
    print("fp16  = half-precision input & matmul")


def bench_autotune():
    """Compare autotuned Triton vs fixed block size vs SDPA."""
    print("=== Benchmark: Autotune vs Fixed vs SDPA — fp16 (ms) ===")
    print("(First run includes autotune search time, subsequent runs use cached best config)\n")
    header = f"{'config':>22}  {'SDPA fp16':>10}  {'Tri fixed':>10}  {'Tri tuned':>10}  {'tuned/SDPA':>10}"
    print(header)
    print("-" * len(header))

    configs = [
        # GPT-2 style
        (1, 12, 512, 64),
        (1, 12, 1024, 64),
        (1, 12, 2048, 64),
        # LLaMA style
        (1, 32, 512, 128),
        (1, 32, 1024, 128),
        (1, 32, 2048, 128),
        (1, 32, 4096, 128),
    ]

    for B, H, N, d in configs:
        Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)

        t_sdpa = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V))
        t_fixed = triton.testing.do_bench(
            lambda: flash_attention_fwd(Q, K, V, allow_tf32=False))
        # Warm up autotune (searches on first call)
        flash_attention_fwd_tuned(Q, K, V, allow_tf32=False)
        t_tuned = triton.testing.do_bench(
            lambda: flash_attention_fwd_tuned(Q, K, V, allow_tf32=False))

        ratio = t_tuned / t_sdpa
        label = f"({B},{H},{N},{d})"
        print(f"{label:>22}  {t_sdpa:>10.4f}  {t_fixed:>10.4f}  {t_tuned:>10.4f}  {ratio:>9.2f}x")

    print()


if __name__ == "__main__":
    test_qk_tile_scores()
    test_flash_attention_correctness()
    test_flash_attention_numerical_stability()
    bench_flash_attention()
