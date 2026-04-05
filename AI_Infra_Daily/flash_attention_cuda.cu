/*
 * flash_attention_cuda.cu
 *
 * Three FlashAttention kernels:
 *   v1: tiles K/V only (one block per query row, D threads)
 *   v2: tiles BOTH Q and K/V (one block per Q tile, 256 threads, matmuls in smem)
 *   v3: same as v2 but uses Tensor Cores (wmma) for matmuls (requires sm_70+)
 *
 * All matrices: (B, H, N, D)
 *   B = batch size, H = heads, N = seq length, D = head dim
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

// ════════════════════════════════════════════════════════════
//  V1 KERNEL — tile K/V only (one block per query row)
// ════════════════════════════════════════════════════════════

template <int BLOCK_KV, int D>
__global__ void flash_attn_v1_kernel(
    const float* __restrict__ Q,   // (B, H, N, D)
    const float* __restrict__ K,   // (B, H, N, D)
    const float* __restrict__ V,   // (B, H, N, D)
    float* __restrict__ O,         // (B, H, N, D)
    int N, float scale)
{
    // grid = (B, H, N), block = (D_padded,)
    // Each block: one query row, D threads
    int b = blockIdx.x, h = blockIdx.y, i = blockIdx.z;
    int H = gridDim.y;
    int tid = threadIdx.x;
    if (tid >= D) return;

    int bh = (b * H + h) * N * D;
    float q_val = Q[bh + i * D + tid];        // Q[b,h,i,tid] — (1,) register

    float o_acc = 0.0f, m = -FLT_MAX, l = 0.0f;

    extern __shared__ float smem[];
    float* K_tile = smem;                      // (BLOCK_KV, D) in smem
    float* V_tile = smem + BLOCK_KV * D;       // (BLOCK_KV, D) in smem

    for (int j = 0; j < N; j += BLOCK_KV) {
        int tile_len = min(BLOCK_KV, N - j);

        for (int r = 0; r < tile_len; r++) {
            K_tile[r * D + tid] = K[bh + (j + r) * D + tid];
            V_tile[r * D + tid] = V[bh + (j + r) * D + tid];
        }
        __syncthreads();

        for (int r = 0; r < tile_len; r++) {
            float dot = q_val * K_tile[r * D + tid];        // (1,) partial
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, off);

            __shared__ float warp_sums[4];
            int warp_id = tid / 32, lane = tid % 32;
            if (lane == 0) warp_sums[warp_id] = dot;
            __syncthreads();

            float s = 0.0f;
            if (tid < 4) {
                s = (tid < (D + 31) / 32) ? warp_sums[tid] : 0.0f;
                for (int off = 2; off > 0; off >>= 1)
                    s += __shfl_down_sync(0xf, s, off);
            }
            __shared__ float score;
            if (tid == 0) score = s * scale;
            __syncthreads();

            float s_val = score;
            float m_new = fmaxf(m, s_val);
            float corr = expf(m - m_new);
            float p = expf(s_val - m_new);
            l = l * corr + p;
            o_acc = o_acc * corr + p * V_tile[r * D + tid];
            m = m_new;
        }
        __syncthreads();
    }

    O[bh + i * D + tid] = o_acc / l;
}

// ════════════════════════════════════════════════════════════
//  V2 KERNEL — tile BOTH Q and K/V (the FlashAttention-2 pattern)
// ════════════════════════════════════════════════════════════
//
// Key differences from v1:
//   - One block handles BLOCK_Q query rows (not just 1)
//   - Q tile loaded once into smem, reused across all KV tiles
//   - S = Q @ K^T computed as full matmul (no warp reduction per element)
//   - 256 threads cooperatively compute S and O matmuls
//
// Shared memory layout for one block:
//   Q_smem:   (BLOCK_Q,  D)        — query tile,  loaded once
//   K_smem:   (BLOCK_KV, D)        — key tile,    streamed
//   V_smem:   (BLOCK_KV, D)        — value tile,  streamed
//   S_smem:   (BLOCK_Q,  BLOCK_KV) — score tile,  overwritten with P = softmax weights
//   O_smem:   (BLOCK_Q,  D)        — output accumulator
//   m_smem:   (BLOCK_Q,)           — running max per Q row
//   l_smem:   (BLOCK_Q,)           — running sum per Q row
//   mn_smem:  (BLOCK_Q,)           — new max (scratch)

template <int BLOCK_Q, int BLOCK_KV, int D, int NUM_THREADS>
__global__ void flash_attn_v2_kernel(
    const float* __restrict__ Q,   // (B, H, N, D)
    const float* __restrict__ K,   // (B, H, N, D)
    const float* __restrict__ V,   // (B, H, N, D)
    float* __restrict__ O,         // (B, H, N, D)
    int N, float scale)
{
    int tid = threadIdx.x;                         // 0..NUM_THREADS-1 (e.g. 0..255)

    // ── Grid mapping ──
    // grid = (ceil(N/BLOCK_Q), H, B)
    //   blockIdx.x = Q tile index  (each tile = BLOCK_Q rows)
    //   blockIdx.y = head
    //   blockIdx.z = batch
    int q_start = blockIdx.x * BLOCK_Q;            // first Q row for this block
    int h = blockIdx.y;
    int b = blockIdx.z;
    int H = gridDim.y;
    int bh = (b * H + h) * N * D;                  // flat offset for (b,h) slice

    // ── Shared memory ──
    extern __shared__ float smem[];
    float* Q_smem  = smem;                                    // (BLOCK_Q, D)
    float* K_smem  = Q_smem  + BLOCK_Q * D;                   // (BLOCK_KV, D)
    float* V_smem  = K_smem  + BLOCK_KV * D;                  // (BLOCK_KV, D)
    float* S_smem  = V_smem  + BLOCK_KV * D;                  // (BLOCK_Q, BLOCK_KV)
    float* O_smem  = S_smem  + BLOCK_Q * BLOCK_KV;            // (BLOCK_Q, D)
    float* m_smem  = O_smem  + BLOCK_Q * D;                   // (BLOCK_Q,)
    float* l_smem  = m_smem  + BLOCK_Q;                       // (BLOCK_Q,)
    float* mn_smem = l_smem  + BLOCK_Q;                       // (BLOCK_Q,)  scratch for m_new

    // ── Initialize accumulators ──
    // O_smem: (BLOCK_Q, D) = 0
    for (int i = tid; i < BLOCK_Q * D; i += NUM_THREADS)
        O_smem[i] = 0.0f;
    // m_smem: (BLOCK_Q,) = -inf,  l_smem: (BLOCK_Q,) = 0
    for (int i = tid; i < BLOCK_Q; i += NUM_THREADS) {
        m_smem[i] = -FLT_MAX;
        l_smem[i] = 0.0f;
    }

    // ── Load Q tile: (BLOCK_Q, D) from HBM -> smem (once, reused) ──
    for (int i = tid; i < BLOCK_Q * D; i += NUM_THREADS) {
        int qi = i / D, dx = i % D;                // qi: row in tile, dx: d-element
        int g_row = q_start + qi;                   // global Q row
        Q_smem[qi * D + dx] = (g_row < N) ? Q[bh + g_row * D + dx] : 0.0f;
    }
    __syncthreads();

    // ═══════════ Main loop: stream K/V tiles ═══════════
    // Iterations: ceil(N / BLOCK_KV)
    for (int kv_start = 0; kv_start < N; kv_start += BLOCK_KV) {
        int kv_len = min(BLOCK_KV, N - kv_start);  // last tile may be shorter

        // ── Load K tile: (kv_len, D) and V tile: (kv_len, D) HBM -> smem ──
        for (int i = tid; i < kv_len * D; i += NUM_THREADS) {
            int r = i / D, c = i % D;
            K_smem[r * D + c] = K[bh + (kv_start + r) * D + c];
            V_smem[r * D + c] = V[bh + (kv_start + r) * D + c];
        }
        __syncthreads();

        // ── Compute S = Q_tile @ K_tile^T  ──
        // S_smem[qi][kj] = (sum_{d} Q_smem[qi][d] * K_smem[kj][d]) * scale
        // Shape: (BLOCK_Q, kv_len)
        // Work: BLOCK_Q * kv_len elements, each a dot product of length D
        //       256 threads share the work — each computes ~2 dot products
        {
            int S_total = BLOCK_Q * kv_len;         // e.g. 16 * 32 = 512
            for (int i = tid; i < S_total; i += NUM_THREADS) {
                int qi = i / kv_len;                // Q row in tile      (0..BLOCK_Q-1)
                int kj = i % kv_len;                // K row in tile      (0..kv_len-1)
                float dot = 0.0f;
                for (int d = 0; d < D; d++)         // loop over head dim
                    dot += Q_smem[qi * D + d] * K_smem[kj * D + d];
                S_smem[qi * BLOCK_KV + kj] = dot * scale;   // (1,) -> smem
            }
        }
        __syncthreads();

        // ── Online softmax step 1: find new max per Q row ──
        // mn_smem[qi] = max(m_smem[qi], max_{kj} S_smem[qi][kj])
        for (int qi = tid; qi < BLOCK_Q; qi += NUM_THREADS) {
            float tile_max = -FLT_MAX;
            for (int kj = 0; kj < kv_len; kj++)
                tile_max = fmaxf(tile_max, S_smem[qi * BLOCK_KV + kj]);
            mn_smem[qi] = fmaxf(m_smem[qi], tile_max);
        }
        __syncthreads();

        // ── Online softmax step 2: rescale old accumulators ──
        // correction[qi] = exp(m_old[qi] - m_new[qi])        — (1,) per Q row
        // O_smem[qi][dx] *= correction[qi]                    — (BLOCK_Q, D) rescaled
        // l_smem[qi]     *= correction[qi]                    — (BLOCK_Q,) rescaled
        for (int i = tid; i < BLOCK_Q * D; i += NUM_THREADS) {
            int qi = i / D;
            O_smem[i] *= expf(m_smem[qi] - mn_smem[qi]);
        }
        for (int qi = tid; qi < BLOCK_Q; qi += NUM_THREADS)
            l_smem[qi] *= expf(m_smem[qi] - mn_smem[qi]);
        __syncthreads();

        // ── Online softmax step 3: convert S -> P = exp(S - m_new) ──
        // P_smem[qi][kj] = exp(S_smem[qi][kj] - mn_smem[qi])
        // Shape: (BLOCK_Q, kv_len), overwrites S_smem
        {
            int S_total = BLOCK_Q * kv_len;
            for (int i = tid; i < S_total; i += NUM_THREADS) {
                int qi = i / kv_len, kj = i % kv_len;
                S_smem[qi * BLOCK_KV + kj] = expf(S_smem[qi * BLOCK_KV + kj] - mn_smem[qi]);
            }
        }
        __syncthreads();

        // ── Update l: l[qi] += sum_{kj} P[qi][kj] ──
        for (int qi = tid; qi < BLOCK_Q; qi += NUM_THREADS) {
            float l_add = 0.0f;
            for (int kj = 0; kj < kv_len; kj++)
                l_add += S_smem[qi * BLOCK_KV + kj];
            l_smem[qi] += l_add;
        }

        // ── Accumulate O += P @ V_tile ──
        // O_smem[qi][dx] += sum_{kj} P_smem[qi][kj] * V_smem[kj][dx]
        // Shape: (BLOCK_Q, D) += (BLOCK_Q, kv_len) @ (kv_len, D)
        // Work: BLOCK_Q * D elements, each a dot product of length kv_len
        //       256 threads share the work — each computes ~4 elements
        for (int i = tid; i < BLOCK_Q * D; i += NUM_THREADS) {
            int qi = i / D;                         // Q row in tile      (0..BLOCK_Q-1)
            int dx = i % D;                         // d-element          (0..D-1)
            float o_add = 0.0f;
            for (int kj = 0; kj < kv_len; kj++)    // loop over KV tile
                o_add += S_smem[qi * BLOCK_KV + kj] * V_smem[kj * D + dx];
            O_smem[i] += o_add;                     // (1,) accumulate
        }

        // ── Update m ──
        for (int qi = tid; qi < BLOCK_Q; qi += NUM_THREADS)
            m_smem[qi] = mn_smem[qi];
        __syncthreads();
    }

    // ── Final: O = O / l, write smem -> HBM ──
    // O[b,h,q_start+qi,dx] = O_smem[qi][dx] / l_smem[qi]
    for (int i = tid; i < BLOCK_Q * D; i += NUM_THREADS) {
        int qi = i / D, dx = i % D;
        int g_row = q_start + qi;
        if (g_row < N)
            O[bh + g_row * D + dx] = O_smem[i] / l_smem[qi];
    }
}


// ════════════════════════════════════════════════════════════
//  V3 KERNEL — Tensor Core (wmma) acceleration
// ════════════════════════════════════════════════════════════
//
// Same algorithm as v2 (tile Q + KV, online softmax) but replaces
// scalar dot-product loops with wmma 16×16×16 matmuls:
//   - Q, K, V converted to fp16 in shared memory for Tensor Cores
//   - S = Q @ K^T  via wmma (fp16 input, fp32 accumulate)
//   - O += P @ V   via wmma (fp16 input, fp32 accumulate)
//   - Softmax stays fp32 for numerical stability
//
// Requires compute capability >= 7.0 (Volta or newer)

using namespace nvcuda;

template <int BLOCK_Q, int BLOCK_KV, int D, int NUM_THREADS>
__global__ void flash_attn_v3_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, float scale)
{
    static_assert(BLOCK_Q % 16 == 0 && BLOCK_KV % 16 == 0 && D % 16 == 0,
                  "wmma requires dimensions to be multiples of 16");

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int num_warps = NUM_THREADS / 32;

    int q_start = blockIdx.x * BLOCK_Q;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int H = gridDim.y;
    int bh = (b * H + h) * N * D;

    // ── Shared memory layout ──
    // Half-precision buffers (for wmma):
    //   Q_half:   (BLOCK_Q, D)           — loaded once
    //   K_T_half: (D, BLOCK_KV)          — K transposed, reloaded each iter
    //   V_half:   (BLOCK_KV, D)          — reloaded each iter
    //   P_half:   (BLOCK_Q, BLOCK_KV)    — softmax weights for wmma
    // Float buffers (for softmax + accumulation):
    //   S_smem:   (BLOCK_Q, BLOCK_KV)    — raw scores
    //   O_smem:   (BLOCK_Q, D)           — output accumulator
    //   m_smem:   (BLOCK_Q,)             — running max
    //   l_smem:   (BLOCK_Q,)             — running sum
    //   mn_smem:  (BLOCK_Q,)             — new max scratch
    extern __shared__ char smem_raw[];
    half*  Q_half   = (half*)smem_raw;
    half*  K_T_half = Q_half   + BLOCK_Q * D;
    half*  V_half   = K_T_half + D * BLOCK_KV;
    half*  P_half   = V_half   + BLOCK_KV * D;
    float* S_smem   = (float*)(P_half + BLOCK_Q * BLOCK_KV);
    float* O_smem   = S_smem   + BLOCK_Q * BLOCK_KV;
    float* m_smem   = O_smem   + BLOCK_Q * D;
    float* l_smem   = m_smem   + BLOCK_Q;
    float* mn_smem  = l_smem   + BLOCK_Q;

    // ── Initialize accumulators ──
    for (int i = tid; i < BLOCK_Q * D; i += NUM_THREADS)
        O_smem[i] = 0.0f;
    for (int i = tid; i < BLOCK_Q; i += NUM_THREADS) {
        m_smem[i] = -FLT_MAX;
        l_smem[i] = 0.0f;
    }

    // ── Load Q tile as fp16 (loaded once, reused across all KV tiles) ──
    for (int i = tid; i < BLOCK_Q * D; i += NUM_THREADS) {
        int qi = i / D, dx = i % D;
        int g_row = q_start + qi;
        Q_half[qi * D + dx] = __float2half(
            (g_row < N) ? Q[bh + g_row * D + dx] : 0.0f);
    }
    __syncthreads();

    // ═══════════ Main loop: stream K/V tiles ═══════════
    for (int kv_start = 0; kv_start < N; kv_start += BLOCK_KV) {
        int kv_len = min(BLOCK_KV, N - kv_start);

        // ── Load K (transposed) and V as fp16 ──
        // K(r, c) stored as K^T(c, r) for wmma: Q @ K^T = Q_half @ K_T_half
        for (int i = tid; i < BLOCK_KV * D; i += NUM_THREADS) {
            int r = i / D, c = i % D;
            float k_val = (r < kv_len) ? K[bh + (kv_start + r) * D + c] : 0.0f;
            float v_val = (r < kv_len) ? V[bh + (kv_start + r) * D + c] : 0.0f;
            K_T_half[c * BLOCK_KV + r] = __float2half(k_val);
            V_half[r * D + c] = __float2half(v_val);
        }
        __syncthreads();

        // ── S = Q_half @ K_T_half via wmma ──
        // (BLOCK_Q, D) @ (D, BLOCK_KV) → (BLOCK_Q, BLOCK_KV)
        // Each warp computes one 16×16 output tile
        {
            constexpr int nm = BLOCK_Q / 16;
            constexpr int nn = BLOCK_KV / 16;
            constexpr int total_tiles = nm * nn;
            for (int t = warp_id; t < total_tiles; t += num_warps) {
                int tm = t / nn, tn = t % nn;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                for (int k = 0; k < D; k += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
                    wmma::load_matrix_sync(a, &Q_half[tm * 16 * D + k], D);
                    wmma::load_matrix_sync(b, &K_T_half[k * BLOCK_KV + tn * 16], BLOCK_KV);
                    wmma::mma_sync(acc, a, b, acc);
                }
                wmma::store_matrix_sync(
                    &S_smem[tm * 16 * BLOCK_KV + tn * 16], acc, BLOCK_KV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ── Apply scale + mask padding columns ──
        for (int i = tid; i < BLOCK_Q * BLOCK_KV; i += NUM_THREADS) {
            int kj = i % BLOCK_KV;
            S_smem[i] = (kj < kv_len) ? S_smem[i] * scale : -FLT_MAX;
        }
        __syncthreads();

        // ── Online softmax step 1: find new row max ──
        for (int qi = tid; qi < BLOCK_Q; qi += NUM_THREADS) {
            float tile_max = -FLT_MAX;
            for (int kj = 0; kj < BLOCK_KV; kj++)
                tile_max = fmaxf(tile_max, S_smem[qi * BLOCK_KV + kj]);
            mn_smem[qi] = fmaxf(m_smem[qi], tile_max);
        }
        __syncthreads();

        // ── Online softmax step 2: rescale old O and l ──
        for (int i = tid; i < BLOCK_Q * D; i += NUM_THREADS) {
            int qi = i / D;
            O_smem[i] *= expf(m_smem[qi] - mn_smem[qi]);
        }
        for (int qi = tid; qi < BLOCK_Q; qi += NUM_THREADS)
            l_smem[qi] *= expf(m_smem[qi] - mn_smem[qi]);
        __syncthreads();

        // ── Online softmax step 3: P = exp(S - m_new), convert to fp16 ──
        for (int i = tid; i < BLOCK_Q * BLOCK_KV; i += NUM_THREADS) {
            int qi = i / BLOCK_KV;
            float p = expf(S_smem[i] - mn_smem[qi]);
            S_smem[i] = p;                       // keep fp32 for l update
            P_half[i] = __float2half(p);          // fp16 copy for wmma
        }
        __syncthreads();

        // ── Update l += rowsum(P) ──
        for (int qi = tid; qi < BLOCK_Q; qi += NUM_THREADS) {
            float l_add = 0.0f;
            for (int kj = 0; kj < BLOCK_KV; kj++)
                l_add += S_smem[qi * BLOCK_KV + kj];
            l_smem[qi] += l_add;
        }

        // ── O += P_half @ V_half via wmma ──
        // (BLOCK_Q, BLOCK_KV) @ (BLOCK_KV, D) → (BLOCK_Q, D)
        // Load existing O into accumulator, add P@V, store back
        {
            constexpr int nm = BLOCK_Q / 16;
            constexpr int nn = D / 16;
            constexpr int total_tiles = nm * nn;
            for (int t = warp_id; t < total_tiles; t += num_warps) {
                int tm = t / nn, tn = t % nn;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::load_matrix_sync(
                    acc, &O_smem[tm * 16 * D + tn * 16], D, wmma::mem_row_major);
                for (int k = 0; k < BLOCK_KV; k += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
                    wmma::load_matrix_sync(a, &P_half[tm * 16 * BLOCK_KV + k], BLOCK_KV);
                    wmma::load_matrix_sync(b, &V_half[k * D + tn * 16], D);
                    wmma::mma_sync(acc, a, b, acc);
                }
                wmma::store_matrix_sync(
                    &O_smem[tm * 16 * D + tn * 16], acc, D, wmma::mem_row_major);
            }
        }

        // ── Update m ──
        for (int qi = tid; qi < BLOCK_Q; qi += NUM_THREADS)
            m_smem[qi] = mn_smem[qi];
        __syncthreads();
    }

    // ── Final: O = O / l, write smem → HBM ──
    for (int i = tid; i < BLOCK_Q * D; i += NUM_THREADS) {
        int qi = i / D, dx = i % D;
        int g_row = q_start + qi;
        if (g_row < N)
            O[bh + g_row * D + dx] = O_smem[i] / l_smem[qi];
    }
}


// ════════════════════════════════════════════════════════════
//  HOST LAUNCHERS
// ════════════════════════════════════════════════════════════

torch::Tensor flash_attn_v1_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int block_size) {
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kFloat32 && Q.dim() == 4);
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);

    auto O = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);
    int threads = ((D + 31) / 32) * 32;
    dim3 grid(B, H, N);
    size_t smem = 2 * block_size * D * sizeof(float) + 5 * sizeof(float);

    TORCH_CHECK(smem <= 48 * 1024, "v1: smem overflow, use smaller block_size");

    #define V1_LAUNCH(BS, DD) \
        flash_attn_v1_kernel<BS, DD><<<grid, threads, smem>>>( \
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), \
            O.data_ptr<float>(), N, scale);
    #define V1_D(BS) \
        if (D==32) { V1_LAUNCH(BS,32) } else if (D==64) { V1_LAUNCH(BS,64) } \
        else if (D==128) { V1_LAUNCH(BS,128) } else { TORCH_CHECK(false, "D must be 32/64/128"); }

    if      (block_size==16)  { V1_D(16)  }
    else if (block_size==32)  { V1_D(32)  }
    else if (block_size==64)  { V1_D(64)  }
    else { TORCH_CHECK(false, "v1 block_size must be 16/32/64"); }

    #undef V1_LAUNCH
    #undef V1_D
    return O;
}


torch::Tensor flash_attn_v2_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kFloat32 && Q.dim() == 4);
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);

    auto O = torch::empty_like(Q);                          // (B, H, N, D)
    float scale = 1.0f / sqrtf((float)D);

    // Choose tile sizes based on D to fit in 48KB smem
    // smem = (2*BQ*D + 2*BKV*D + BQ*BKV + 3*BQ) * 4 bytes
    //
    //   D=32:  BQ=16, BKV=64 → 14.2 KB
    //   D=64:  BQ=16, BKV=32 → 26.2 KB
    //   D=128: BQ=16, BKV=16 → 33.2 KB

    constexpr int NUM_THREADS = 256;

    #define V2_LAUNCH(BQ, BKV, DD) { \
        constexpr int smem_floats = 2*(BQ)*(DD) + 2*(BKV)*(DD) + (BQ)*(BKV) + 3*(BQ); \
        size_t smem = smem_floats * sizeof(float); \
        dim3 grid(((N) + (BQ) - 1) / (BQ), H, B); \
        flash_attn_v2_kernel<BQ, BKV, DD, NUM_THREADS><<<grid, NUM_THREADS, smem>>>( \
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), \
            O.data_ptr<float>(), N, scale); \
    }

    if      (D == 32)  { V2_LAUNCH(16, 64, 32)  }
    else if (D == 64)  { V2_LAUNCH(16, 32, 64)  }
    else if (D == 128) { V2_LAUNCH(16, 16, 128) }
    else { TORCH_CHECK(false, "D must be 32, 64, or 128"); }

    #undef V2_LAUNCH
    return O;
}


torch::Tensor flash_attn_v3_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kFloat32 && Q.dim() == 4);
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);

    auto O = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);

    constexpr int NUM_THREADS = 128;  // 4 warps
    constexpr int BLOCK_Q = 16;

    // smem = half_elems * 2 + float_elems * 4 bytes
    // half:  Q(BQ*D) + K^T(D*BKV) + V(BKV*D) + P(BQ*BKV)
    // float: S(BQ*BKV) + O(BQ*D) + m(BQ) + l(BQ) + mn(BQ)
    #define V3_LAUNCH(BKV, DD) { \
        constexpr int half_elems = BLOCK_Q*(DD) + (DD)*(BKV) + (BKV)*(DD) + BLOCK_Q*(BKV); \
        constexpr int float_elems = BLOCK_Q*(BKV) + BLOCK_Q*(DD) + 3*BLOCK_Q; \
        size_t smem = half_elems * sizeof(half) + float_elems * sizeof(float); \
        dim3 grid(((N) + BLOCK_Q - 1) / BLOCK_Q, H, B); \
        flash_attn_v3_kernel<BLOCK_Q, BKV, DD, NUM_THREADS><<<grid, NUM_THREADS, smem>>>( \
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), \
            O.data_ptr<float>(), N, scale); \
    }

    if      (D == 32)  { V3_LAUNCH(64, 32)  }
    else if (D == 64)  { V3_LAUNCH(64, 64)  }
    else if (D == 128) { V3_LAUNCH(32, 128) }
    else { TORCH_CHECK(false, "v3: D must be 32, 64, or 128"); }

    #undef V3_LAUNCH
    return O;
}


// ════════════════════════════════════════════════════════════
//  PYBIND
// ════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_v1_fwd", &flash_attn_v1_fwd, "FlashAttention v1 (tile KV only)");
    m.def("flash_attn_v2_fwd", &flash_attn_v2_fwd, "FlashAttention v2 (tile Q + KV)");
    m.def("flash_attn_v3_fwd", &flash_attn_v3_fwd, "FlashAttention v3 (Tensor Core / wmma)");
}
