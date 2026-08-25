#!/usr/bin/env python3
"""bench/axes/02_ozaki_gemm_reduce.py — reduce raw Ozaki-II/AMX GEMM
throughput+accuracy runs into results.json fragment + markdown fragment.

Invoked by bench/axes/02_ozaki_gemm.sh:
    02_ozaki_gemm_reduce.py --workdir DIR --amx-ok 0|1 --ozaki-ok 0|1
        --ozaki-fast-ok 0|1 --json-out PATH --md-out PATH
"""
import argparse
import json
import os
import re
import statistics
import sys

BENCH_RE = re.compile(r"^BENCH n=(?P<n>\d+) ns_samples=\[(?P<samples>[\d,]*)\]$")
SAMPLE_RE = re.compile(
    r"^SAMPLE r=(?P<r>\d+) c=(?P<c>\d+) approx=(?P<approx>\S+) "
    r"exact=(?P<exact>\S+) relerr=(?P<relerr>\S+)$"
)


def parse_throughput(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            m = BENCH_RE.match(line.strip())
            if not m:
                continue
            n = int(m.group("n"))
            samples = [int(x) for x in m.group("samples").split(",") if x]
            gflops = [(2.0 * n ** 3) / (s / 1e9) / 1e9 for s in samples if s > 0]
            rows.append({
                "n": n,
                "raw_ns_samples": samples,
                "gflops_samples": gflops,
                "gflops_median": statistics.median(gflops) if gflops else None,
            })
    return rows


def parse_accuracy(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            m = SAMPLE_RE.match(line.strip())
            if not m:
                continue
            rows.append({
                "r": int(m.group("r")), "c": int(m.group("c")),
                "approx": float(m.group("approx")), "exact": float(m.group("exact")),
                "relerr": float(m.group("relerr")),
            })
    return rows


def accuracy_summary(rows):
    if not rows:
        return None
    rel = [r["relerr"] for r in rows]
    return {"max_relerr": max(rel), "mean_relerr": statistics.mean(rel), "n_samples": len(rel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--amx-ok", required=True)
    ap.add_argument("--ozaki-ok", required=True)
    ap.add_argument("--ozaki-fast-ok", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--md-out", required=True)
    args = ap.parse_args()

    tags_ok = {
        "amx": args.amx_ok == "1",
        "ozaki": args.ozaki_ok == "1",
        "ozaki-fast": args.ozaki_fast_ok == "1",
    }

    throughput = {}
    accuracy = {}
    for tag, ok in tags_ok.items():
        if not ok:
            throughput[tag] = None
            accuracy[tag] = None
            continue
        throughput[tag] = parse_throughput(os.path.join(args.workdir, f"throughput.{tag}.out"))
        accuracy[tag] = accuracy_summary(parse_accuracy(os.path.join(args.workdir, f"accuracy.{tag}.out")))

    result = {
        "axis": "ozaki_ii_gemm",
        "claims_tested": [
            "Ozaki-II CRT GEMM computes the exact product and rounds once, "
            "where BLAS rounds every accumulation — measured here as error "
            "against a TRUE exact-rational reference, not just a "
            "differently-ordered float accumulation",
            "the fast tier can beat vendor BLAS throughput at large N "
            "while trading exactness for ~1e-8 accuracy",
        ],
        "accuracy_honesty_note": (
            "The correctness gate this axis's methodology follows "
            "(tests/gpu/ozaki_correctness_gate.sh) verifies Ozaki-II exact "
            "against a TOL=1e-9 threshold, not literal bit-exactness — its "
            "own reference is a naive f64 accumulation, accurate to "
            "~K*epsilon. At the modest K this axis samples (N=64), vendor "
            "BLAS's own few accumulation steps can measure MORE accurate "
            "against a true exact-rational reference than Ozaki-II's fixed "
            "~1e-13 CRT-reconstruction floor — that is not a regression, it "
            "is what a fixed reconstruction-precision floor vs a "
            "K-dependent accumulation error look like at small K. Ozaki-II's "
            "accuracy advantage over vendor BLAS is expected to widen as K "
            "grows and/or input dynamic range widens, since BLAS's per-step "
            "rounding error grows with K while Ozaki-II's stays governed by "
            "its fixed moduli budget. Report the numbers as measured; do "
            "not round this nuance away."
        ),
        "note_on_cuda_numbers": (
            "CHANGELOG.md's CUDA INT8-Ozaki numbers (RTX 3090: 4.74 TFLOP/s-eq, "
            "8.8x cublasDgemm; RTX PRO 6000 Blackwell: ~30 TFLOP/s, 20x) were "
            "measured on that other hardware, not this machine, and are NOT "
            "reproduced by this axis — this host has no NVIDIA GPU. Cited as "
            "prior published measurements only."
        ),
        "throughput_gflops_by_n": throughput,
        "accuracy_vs_exact_rational_reference": accuracy,
        "availability": tags_ok,
    }

    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)

    lines = []
    lines.append("### Axis 2: Ozaki-II CRT exact f64 GEMM vs vendor BLAS\n")
    lines.append("**Throughput (GF/s, median of repeated samples)**\n")
    all_ns = sorted({row["n"] for tag in throughput.values() if tag for row in tag})
    lines.append("| N | AMX (vendor BLAS) | Ozaki-II exact | Ozaki-II fast |")
    lines.append("|---:|---:|---:|---:|")
    for n in all_ns:
        def cell(tag):
            rows = throughput.get(tag)
            if not rows:
                return "unavailable"
            for row in rows:
                if row["n"] == n and row["gflops_median"] is not None:
                    return f"{row['gflops_median']:.1f}"
            return "n/a"
        lines.append(f"| {n} | {cell('amx')} | {cell('ozaki')} | {cell('ozaki-fast')} |")

    lines.append("\n**Accuracy vs an exact-rational reference (max/mean relative error over sampled entries)**\n")
    lines.append("| Kernel | max relerr | mean relerr | samples |")
    lines.append("|---|---:|---:|---:|")
    for tag in ("amx", "ozaki", "ozaki-fast"):
        summ = accuracy.get(tag)
        if not summ:
            lines.append(f"| {tag} | unavailable | unavailable | - |")
        else:
            lines.append(f"| {tag} | {summ['max_relerr']:.3e} | {summ['mean_relerr']:.3e} | {summ['n_samples']} |")
    lines.append(f"\n{result['accuracy_honesty_note']}\n")

    with open(args.md_out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
