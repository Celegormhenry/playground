"""
Method 7: Domain Transform (DT)
─────────────────────────────────
Role   : Preprocessing module (applied before the main compression pipeline)
Idea   : Transform the data domain to make error control easier or to improve
         data smoothness (hence prediction accuracy).
Types  : Log-transform (for relative error bound), space-filling curve (AMR).
Key theorem (from the paper):
  Enforcing pointwise relative error bound εᵣ on data d is equivalent to
  enforcing absolute error bound log(1+εᵣ) on log(d).
  → compress log(data) with standard absolute-EB compressor, then exp() on decompress.
Used by: SZ3 (log mode), QoZ, zMesh (space-filling curve for AMR).
"""

import numpy as np
from utils import header, subheader, lorenzo_1d

header("7. DOMAIN TRANSFORM (DT)")

# ════════════════════════════════════════════════════════════════════════════
# 7a. Why we need the log transform
# ════════════════════════════════════════════════════════════════════════════

subheader("7a. Problem — absolute EB fails on multi-scale data")

data = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
abs_eb = 0.001   # EB tuned for the small values

print(f"\nINPUT  : {data}")
print(f"Abs EB : {abs_eb}  (tuned to protect small values like 0.001)")

# simulate what happens with abs EB on the large values
print(f"\nRelative error if we use abs EB={abs_eb} uniformly:")
print(f"  {'value':>8}  {'abs_err':>8}  {'rel_err':>10}  {'acceptable?':>12}")
for v in data:
    rel = abs_eb / abs(v) * 100
    ok  = "✓" if rel <= 1.0 else "✗  too loose!"
    print(f"  {v:>8.3f}  {abs_eb:>8.4f}  {rel:>9.2f}%  {ok}")

print(f"\n→ abs EB={abs_eb} protects small values but is far too loose for large ones.")
print(f"  To protect 1000.0 at 0.1% rel accuracy, you'd need abs EB=1.0 — which")
print(f"  would completely destroy the information in 0.001.")

# ════════════════════════════════════════════════════════════════════════════
# 7b. Solution — log transform
# ════════════════════════════════════════════════════════════════════════════

subheader("7b. Solution — log-transform for uniform relative error bound")

er     = 0.01   # 1% relative error bound
abs_eb_log = np.log(1.0 + er)

print(f"\nRelative EB  : {er*100}%")
print(f"Abs EB in log domain = log(1 + {er}) = {abs_eb_log:.6f}")
print(f"\nProof sketch:")
print(f"  Want: |x̂ - x| / |x| ≤ εᵣ")
print(f"  Let  ŷ = log(x̂),  y = log(x)")
print(f"  |ŷ - y| = |log(x̂/x)| = |log(1 + (x̂-x)/x)| ≈ |(x̂-x)/x| for small error")
print(f"  So controlling |ŷ - y| ≤ log(1+εᵣ) controls the relative error.")

# ── Step 1: Log-transform ─────────────────────────────────────────────────────
log_data = np.log(data)
print(f"\nSTEP 1 — log-transform:")
print(f"  original : {data}")
print(f"  log(data): {log_data.round(4)}")
print(f"  The log data is now smooth and uniformly spaced → Lorenzo predicts well.")

# ── Step 2: Compress log domain with absolute EB ──────────────────────────────
qcodes, recon_log = lorenzo_1d(log_data, eb=abs_eb_log)
print(f"\nSTEP 2 — compress log_data with abs EB={abs_eb_log:.4f}:")
print(f"  qcodes        : {qcodes}")
print(f"  recon_log     : {recon_log.round(4)}")
print(f"  max log error : {np.abs(recon_log - log_data).max():.6f}  ≤ {abs_eb_log:.6f}")

# ── Step 3: Exponentiate back ─────────────────────────────────────────────────
recon = np.exp(recon_log)
rel_errs = np.abs(recon - data) / data * 100

print(f"\nSTEP 3 — exponentiate back (decompression):")
print(f"  {'original':>10}  {'recon':>10}  {'rel_err':>10}  {'≤ {er*100:.0f}%?':>8}")
for v, r, e in zip(data, recon, rel_errs):
    ok = "✓" if e <= er*100 else "✗"
    print(f"  {v:>10.4f}  {r:>10.4f}  {e:>9.3f}%  {ok}")

print(f"\nMax relative error: {rel_errs.max():.3f}%  ≤ {er*100:.0f}%  ✓")
print(f"Every value protected uniformly, regardless of magnitude.")

# ════════════════════════════════════════════════════════════════════════════
# 7c. Comparison: abs EB vs log-domain EB
# ════════════════════════════════════════════════════════════════════════════

subheader("7c. Head-to-head comparison")

print(f"\n  Method                  qcodes                          max_rel_err")

# Naive abs EB (set to protect 0.001 at 1%)
naive_eb = 0.001 * 0.01    # = 1e-5, to get 1% on 0.001
qcodes_naive, recon_naive = lorenzo_1d(data, eb=naive_eb)
rel_naive = (np.abs(recon_naive - data) / data * 100).max()
print(f"  Abs EB={naive_eb:.0e} (naive)  {qcodes_naive}  {rel_naive:.3f}%")

qcodes_log, recon_log2 = lorenzo_1d(np.log(data), eb=abs_eb_log)
recon_log_orig = np.exp(recon_log2)
rel_log = (np.abs(recon_log_orig - data) / data * 100).max()
print(f"  Log-domain EB={abs_eb_log:.4f}  {qcodes_log}  {rel_log:.3f}%")

print(f"\n  Naive approach uses {len(np.unique(qcodes_naive))} unique codes (large range).")
print(f"  Log-domain uses     {len(np.unique(qcodes_log))} unique codes (smaller range → better Huffman).")
