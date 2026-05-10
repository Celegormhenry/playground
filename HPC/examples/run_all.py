"""
Run all I/O examples in sequence.
Usage: python3 run_all.py [example_number]
       python3 run_all.py        # runs all
       python3 run_all.py 4      # runs only example 04
"""

import sys
import runpy

EXAMPLES = [
    ("01", "01_decimation.py",        "Decimation / Sampling"),
    ("02", "02_bit_manipulation.py",  "Bit Manipulation"),
    ("03", "03_wavelet_transform.py", "Wavelet Transform"),
    ("04", "04_prediction_lorenzo.py","Prediction (Lorenzo)"),
    ("05", "05_hosvd_tucker.py",      "HOSVD / Tucker"),
    ("06", "06_quantization.py",      "Quantization Types"),
    ("07", "07_domain_transform.py",  "Domain Transform"),
    ("08", "08_bit_plane_coding.py",  "Bit-Plane Coding"),
    ("09", "09_data_folding.py",      "Data Folding (Filtering)"),
    ("10", "10_lossless_encoding.py", "Lossless Encoding"),
    ("11", "11_full_pipeline.py",     "Full Pipeline (SZ3-style)"),
]

SEP = "═" * 70

def run(num, filename, title):
    print(f"\n{SEP}")
    print(f"  Running example {num}: {title}")
    print(f"  File: {filename}")
    print(SEP)
    runpy.run_path(filename, run_name="__main__")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        for num, filename, title in EXAMPLES:
            run(num, filename, title)
    else:
        target = sys.argv[1].zfill(2)
        match  = [(n, f, t) for n, f, t in EXAMPLES if n == target]
        if not match:
            print(f"Unknown example '{sys.argv[1]}'. Available: {[n for n,_,_ in EXAMPLES]}")
            sys.exit(1)
        run(*match[0])
