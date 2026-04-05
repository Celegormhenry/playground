"""
Flash Attention Benchmark Script

Compares Triton FlashAttention (fixed / autotuned / tf32 / fp16)
against PyTorch SDPA across realistic model configs.

Usage:
    python flash_attn_benchmark.py
"""

import torch
import torch.nn.functional as F
import triton
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from flash_attention_triton import (
    flash_attention_fwd,
    flash_attention_fwd_tuned,
)

# ═══════════════════════════════════════════════════════════════════
# Configs
# ═══════════════════════════════════════════════════════════════════

CONFIGS_GPT2 = [
    # (B, H, N, d)  — GPT-2 style: H=12, d=64
    (1, 12, 512, 64),
    (1, 12, 1024, 64),
    (1, 12, 2048, 64),
]

CONFIGS_LLAMA = [
    # (B, H, N, d)  — LLaMA-7B style: H=32, d=128
    (1, 32, 512, 128),
    (1, 32, 1024, 128),
    (1, 32, 2048, 128),
    (1, 32, 4096, 128),
]


# ═══════════════════════════════════════════════════════════════════
# 1. Precision Benchmark: fp32 vs tf32 vs fp16
# ═══════════════════════════════════════════════════════════════════

def bench_precision():
    print("=" * 80)
    print("Benchmark 1: Precision Modes (fp32 / tf32 / fp16) vs SDPA")
    print("=" * 80)
    header = f"{'config':>22}  {'SDPA fp32':>10}  {'SDPA fp16':>10}  {'Tri fp32':>10}  {'Tri tf32':>10}  {'Tri fp16':>10}"
    print(header)
    print("-" * len(header))

    all_configs = CONFIGS_GPT2 + CONFIGS_LLAMA
    results = []

    for B, H, N, d in all_configs:
        Q_f32 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
        K_f32 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
        V_f32 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
        Q_f16 = Q_f32.half()
        K_f16 = K_f32.half()
        V_f16 = V_f32.half()

        t_sdpa_f32 = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q_f32, K_f32, V_f32))
        t_sdpa_f16 = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q_f16, K_f16, V_f16))
        t_tri_f32 = triton.testing.do_bench(
            lambda: flash_attention_fwd(Q_f32, K_f32, V_f32, allow_tf32=False))
        t_tri_tf32 = triton.testing.do_bench(
            lambda: flash_attention_fwd(Q_f32, K_f32, V_f32, allow_tf32=True))
        t_tri_f16 = triton.testing.do_bench(
            lambda: flash_attention_fwd(Q_f16, K_f16, V_f16, allow_tf32=False))

        label = f"({B},{H},{N},{d})"
        print(f"{label:>22}  {t_sdpa_f32:>10.4f}  {t_sdpa_f16:>10.4f}  {t_tri_f32:>10.4f}  {t_tri_tf32:>10.4f}  {t_tri_f16:>10.4f}")

        results.append({
            'B': B, 'H': H, 'N': N, 'd': d,
            'sdpa_f32': t_sdpa_f32, 'sdpa_f16': t_sdpa_f16,
            'tri_f32': t_tri_f32, 'tri_tf32': t_tri_tf32, 'tri_f16': t_tri_f16,
        })

    print()
    return results


# ═══════════════════════════════════════════════════════════════════
# 2. Autotune Benchmark: fixed vs tuned vs SDPA (fp16)
# ═══════════════════════════════════════════════════════════════════

def bench_autotune():
    print("=" * 80)
    print("Benchmark 2: Autotune vs Fixed Block Size vs SDPA (fp16)")
    print("=" * 80)
    header = f"{'config':>22}  {'SDPA fp16':>10}  {'Tri fixed':>10}  {'Tri tuned':>10}  {'tuned/SDPA':>10}"
    print(header)
    print("-" * len(header))

    all_configs = CONFIGS_GPT2 + CONFIGS_LLAMA
    results = []

    for B, H, N, d in all_configs:
        Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)

        t_sdpa = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V))
        t_fixed = triton.testing.do_bench(
            lambda: flash_attention_fwd(Q, K, V, allow_tf32=False))
        # Warm up autotune
        flash_attention_fwd_tuned(Q, K, V, allow_tf32=False)
        t_tuned = triton.testing.do_bench(
            lambda: flash_attention_fwd_tuned(Q, K, V, allow_tf32=False))

        ratio = t_tuned / t_sdpa
        label = f"({B},{H},{N},{d})"
        print(f"{label:>22}  {t_sdpa:>10.4f}  {t_fixed:>10.4f}  {t_tuned:>10.4f}  {ratio:>9.2f}x")

        results.append({
            'B': B, 'H': H, 'N': N, 'd': d,
            'sdpa': t_sdpa, 'fixed': t_fixed, 'tuned': t_tuned,
        })

    print()
    return results


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_precision(results):
    """Plot precision benchmark: latency lines for GPT-2 and LLaMA configs."""
    gpt2 = [r for r in results if r['d'] == 64]
    llama = [r for r in results if r['d'] == 128]

    colors = {
        'SDPA fp32': '#4C72B0', 'SDPA fp16': '#55A868',
        'Triton fp32': '#C44E52', 'Triton tf32': '#8172B2', 'Triton fp16': '#CCB974',
    }
    markers = {
        'SDPA fp32': 's', 'SDPA fp16': 's',
        'Triton fp32': 'o', 'Triton tf32': 'o', 'Triton fp16': 'o',
    }
    linestyles = {
        'SDPA fp32': '--', 'SDPA fp16': '--',
        'Triton fp32': '-', 'Triton tf32': '-', 'Triton fp16': '-',
    }
    keys = [
        ('sdpa_f32', 'SDPA fp32'), ('sdpa_f16', 'SDPA fp16'),
        ('tri_f32', 'Triton fp32'), ('tri_tf32', 'Triton tf32'), ('tri_f16', 'Triton fp16'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # GPT-2
    seq = [r['N'] for r in gpt2]
    for key, name in keys:
        vals = [r[key] for r in gpt2]
        ax1.plot(seq, vals, marker=markers[name], color=colors[name],
                 linestyle=linestyles[name], linewidth=2, markersize=7, label=name)
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('GPT-2 Style (H=12, d=64)', fontsize=13, fontweight='bold')
    ax1.set_xticks(seq)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')

    # LLaMA (skip Triton fp32 — too slow)
    seq = [r['N'] for r in llama]
    keys_llama = [k for k in keys if k[0] != 'tri_f32']
    for key, name in keys_llama:
        vals = [r[key] for r in llama]
        ax2.plot(seq, vals, marker=markers[name], color=colors[name],
                 linestyle=linestyles[name], linewidth=2, markersize=7, label=name)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('LLaMA-7B Style (H=32, d=128)\n(Triton fp32 omitted)', fontsize=13, fontweight='bold')
    ax2.set_xticks(seq)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    path = 'bench_results/flash_attn_precision_bench.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_autotune(results):
    """Plot autotune benchmark: grouped bar chart."""
    gpt2 = [r for r in results if r['d'] == 64]
    llama = [r for r in results if r['d'] == 128]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = ['#55A868', '#C44E52', '#8172B2']
    labels = ['SDPA fp16', 'Triton fixed', 'Triton tuned']
    data_keys = ['sdpa', 'fixed', 'tuned']

    for ax, data, title in [
        (ax1, gpt2, 'GPT-2 (H=12, d=64)'),
        (ax2, llama, 'LLaMA-7B (H=32, d=128)'),
    ]:
        seq = [str(r['N']) for r in data]
        x = np.arange(len(seq))
        width = 0.25

        for i, (key, label, color) in enumerate(zip(data_keys, labels, colors)):
            vals = [r[key] for r in data]
            ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, edgecolor='white')

        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seq)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = 'bench_results/flash_attn_autotune_bench.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    prec_results = bench_precision()
    plot_precision(prec_results)
    print()

    tune_results = bench_autotune()
    plot_autotune(tune_results)
