"""
Method 5: HOSVD / Tucker Decomposition
────────────────────────────────────────
Model  : HOSVD-based
Idea   : Higher-Order SVD decomposes a tensor T into a small core G and factor
         matrices {U1, U2, U3}. Compressing G gives high ratio by exploiting
         global correlations across all dimensions simultaneously.
Pro    : Extremely high compression ratio, especially for smooth multi-D fields.
Con    : O(n^4) cost for 3D data; only L2 error control (no pointwise bound).
Used by: TTHRESH, TuckerMPI, ATC.

T ≈ G ×₁ U1 ×₂ U2 ×₃ U3    (Tucker decomposition)
  G  : compressed core tensor  (rank r1×r2×r3)
  Uₖ : factor matrix for mode k  (nₖ × rₖ)
"""

import numpy as np
from utils import header, subheader

header("5. HOSVD / TUCKER DECOMPOSITION")

def tucker_compress(tensor: np.ndarray, rank: tuple):
    """Sequential HOSVD: apply truncated SVD mode by mode."""
    G  = tensor.copy().astype(np.float64)
    Us = []
    for mode, r in enumerate(rank):
        n_mode = G.shape[mode]
        unf    = np.reshape(np.moveaxis(G, mode, 0), (n_mode, -1))
        U, _, _ = np.linalg.svd(unf, full_matrices=False)
        U_r    = U[:, :r]
        G      = np.tensordot(U_r.T, G, axes=([1], [mode]))
        G      = np.moveaxis(G, 0, mode)
        Us.append(U_r)
    return G, Us

def tucker_decompress(G: np.ndarray, Us: list):
    T = G.copy()
    for mode, U in enumerate(Us):
        T = np.tensordot(U, T, axes=([1], [mode]))
        T = np.moveaxis(T, 0, mode)
    return T

# ── Input: 3D smooth field (sin × cos × exp) ─────────────────────────────────
x, y, z = np.meshgrid(np.linspace(0, np.pi, 8),
                       np.linspace(0, np.pi, 8),
                       np.linspace(0, np.pi, 8))
T = np.sin(x) * np.cos(y) * np.exp(-z / 4)

print(f"\nINPUT tensor: shape={T.shape}  total={T.size} values  {T.nbytes} bytes")
print(f"  Value range : [{T.min():.4f}, {T.max():.4f}]")
print(f"  Sample slice T[0,:,0] = {T[0,:,0].round(4)}")

# ── Compression at different ranks ───────────────────────────────────────────
subheader("Rank sweep — compression ratio vs. error")

print(f"\n  {'rank':>12}  {'orig':>6}  {'stored':>8}  {'ratio':>7}  {'max_err':>10}  {'RMSE':>12}")
for rank in [(8,8,8), (6,6,6), (4,4,4), (3,3,3), (2,2,2), (1,1,1)]:
    G, Us   = tucker_compress(T, rank)
    Tr      = tucker_decompress(G, Us)
    stored  = G.size + sum(U.size for U in Us)
    ratio   = T.size / stored
    max_err = np.abs(Tr - T).max()
    rmse    = np.sqrt(np.mean((Tr - T)**2))
    print(f"  {str(rank):>12}  {T.size:>6}  {stored:>8}  {ratio:>6.1f}×  {max_err:>10.4f}  {rmse:>12.6f}")

# ── Inspect the core tensor ───────────────────────────────────────────────────
subheader("Core tensor structure (rank 3,3,3)")

G33, Us33 = tucker_compress(T, (3, 3, 3))

print(f"\nCore tensor G shape: {G33.shape}  ({G33.size} values)")
print(f"\nG[:,:,0] (first slice):")
print(G33[:, :, 0].round(4))
print(f"\nG[:,:,1] (second slice):")
print(G33[:, :, 1].round(4))
print(f"\nObservation: energy concentrates in G[0,0,0] — most other entries near zero.")
print(f"  G[0,0,0] = {G33[0,0,0]:.4f}  (dominant)")
print(f"  Mean |G| = {np.abs(G33).mean():.4f}")
print(f"  Max |G|  = {np.abs(G33).max():.4f}")

# ── Factor matrices ────────────────────────────────────────────────────────────
subheader("Factor matrices (orthonormal bases per dimension)")

for k, U in enumerate(Us33):
    print(f"\n  U{k+1} shape: {U.shape}  (8 data points → 3 basis vectors)")
    print(f"  U{k+1}[:,0] = {U[:,0].round(4)}  (dominant basis vector)")
    print(f"  Orthonormality check: U{k+1}.T @ U{k+1} ≈\n{(U.T @ U).round(4)}")
