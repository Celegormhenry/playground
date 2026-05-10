"""Shared utilities used across all examples."""

import numpy as np
import heapq
from collections import Counter

SEP  = "─" * 70
SEP2 = "═" * 70

np.set_printoptions(precision=4, suppress=True, linewidth=100)

def header(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

def subheader(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def lorenzo_1d(data: np.ndarray, eb: float):
    """1D Lorenzo predictor with reconstructed-data-driven policy."""
    n      = len(data)
    qcodes = np.zeros(n, dtype=np.int32)
    recon  = np.zeros(n, dtype=np.float64)
    delta  = 2.0 * eb
    for i in range(n):
        pred      = recon[i-1] if i > 0 else 0.0
        qcodes[i] = int(np.round((data[i] - pred) / delta))
        recon[i]  = pred + qcodes[i] * delta
    return qcodes, recon

def huffman_build(symbols: list) -> dict:
    """Build a Huffman code table from a list of symbols."""
    freq = Counter(symbols)
    heap = [[w, [sym, ""]] for sym, w in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][1][0]: "0"}
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for p in lo[1:]: p[1] = "0" + p[1]
        for p in hi[1:]: p[1] = "1" + p[1]
        heapq.heappush(heap, [lo[0]+hi[0]] + lo[1:] + hi[1:])
    return {sym: code for sym, code in heap[0][1:]}
