#!/usr/bin/env python3
"""bench/axes/03_flat_rss_reduce.py — reduce native+VM flat-RSS sweep rows
into results.json fragment + markdown fragment.

Invoked by bench/axes/03_flat_rss.sh:
    03_flat_rss_reduce.py --native-rows PATH --vm-rows PATH --vm-available 0|1
        --json-out PATH --md-out PATH
"""
import argparse
import json
import sys


def read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def flatness_verdict(rows, x_key, rss_key="peak_rss_mb", allowance=1.5):
    ok_rows = [r for r in rows if r.get("ok")]
    if len(ok_rows) < 2:
        return None
    ok_rows.sort(key=lambda r: r[x_key])
    first = ok_rows[0][rss_key]
    last = ok_rows[-1][rss_key]
    ratio_x = ok_rows[-1][x_key] / ok_rows[0][x_key] if ok_rows[0][x_key] else None
    ratio_rss = last / first if first else None
    flat = (ratio_rss is not None and ratio_rss <= allowance)
    return {
        "first_x": ok_rows[0][x_key], "first_rss_mb": first,
        "last_x": ok_rows[-1][x_key], "last_rss_mb": last,
        "x_ratio": ratio_x, "rss_ratio": ratio_rss,
        "flat_within_allowance": flat, "allowance": allowance,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-rows", required=True)
    ap.add_argument("--vm-rows", required=True)
    ap.add_argument("--vm-available", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--md-out", required=True)
    args = ap.parse_args()

    native_rows = read_jsonl(args.native_rows)
    vm_rows = read_jsonl(args.vm_rows)
    vm_available = args.vm_available == "1"

    native_flat = flatness_verdict(native_rows, "ticks")
    vm_flat = flatness_verdict(vm_rows, "n") if vm_available else None

    vm_evac_comparison = None
    for r in vm_rows:
        if r.get("evac_off_peak_rss_mb") is not None:
            vm_evac_comparison = {
                "n": r["n"], "evac_on_peak_rss_mb": r["peak_rss_mb"],
                "evac_off_peak_rss_mb": r["evac_off_peak_rss_mb"],
            }

    result = {
        "axis": "flat_rss_under_resident_load",
        "claims_tested": [
            "a 100k-tick resident daemon loop (native, ESH-0214e shape) holds "
            "flat peak RSS as tick count grows",
            "the bytecode-VM with-region evacuator (PR #461 / SW-14) holds "
            "flat peak RSS as with-region iteration count grows, and "
            "materially less RSS than the evacuator disabled",
        ],
        "methodology_note": (
            "Each sweep point is peak RSS of one fresh process at that tick "
            "count (/usr/bin/time -l|-v) — Eshkol has no in-process RSS "
            "sampler, so an intra-run curve is not available; the sweep "
            "across tick counts is the curve. A leak shows up as RSS "
            "growing with ticks; flat allocation does not, regardless of "
            "per-point noise."
        ),
        "native": {"sweep": native_rows, "flatness": native_flat},
        "vm": {
            "available": vm_available,
            "sweep": vm_rows if vm_available else None,
            "flatness": vm_flat,
            "evacuator_on_vs_off_at_largest_point": vm_evac_comparison,
        },
    }

    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)

    lines = []
    lines.append("### Axis 3: flat-RSS under resident load\n")
    lines.append("**Native (AOT) — ESH-0214e-shaped resident daemon loop**\n")
    lines.append("| ticks | peak RSS (MB) | ok |")
    lines.append("|---:|---:|:---:|")
    for r in sorted(native_rows, key=lambda r: r["ticks"]):
        lines.append(f"| {r['ticks']} | {r['peak_rss_mb']} | {'yes' if r['ok'] else 'NO'} |")
    if native_flat:
        lines.append(f"\n{native_flat['first_x']} -> {native_flat['last_x']} ticks "
                      f"({native_flat['x_ratio']:.1f}x more work): "
                      f"{native_flat['first_rss_mb']}MB -> {native_flat['last_rss_mb']}MB peak RSS "
                      f"({native_flat['rss_ratio']:.2f}x). "
                      f"{'FLAT' if native_flat['flat_within_allowance'] else 'NOT FLAT'} "
                      f"(allowance {native_flat['allowance']}x).\n")

    lines.append("\n**VM (bytecode, eshkol-vm-standalone-test) — with-region sweep**\n")
    if vm_available:
        lines.append("| iterations | peak RSS (MB) | ok |")
        lines.append("|---:|---:|:---:|")
        for r in sorted(vm_rows, key=lambda r: r["n"]):
            lines.append(f"| {r['n']} | {r['peak_rss_mb']} | {'yes' if r['ok'] else 'NO'} |")
        if vm_flat:
            lines.append(f"\n{vm_flat['first_x']} -> {vm_flat['last_x']} iterations "
                          f"({vm_flat['x_ratio']:.1f}x more work): "
                          f"{vm_flat['first_rss_mb']}MB -> {vm_flat['last_rss_mb']}MB peak RSS "
                          f"({vm_flat['rss_ratio']:.2f}x). "
                          f"{'FLAT' if vm_flat['flat_within_allowance'] else 'NOT FLAT'} "
                          f"(allowance {vm_flat['allowance']}x).\n")
        if vm_evac_comparison:
            lines.append(f"\nEvacuator on vs off at n={vm_evac_comparison['n']}: "
                          f"{vm_evac_comparison['evac_on_peak_rss_mb']}MB (on) vs "
                          f"{vm_evac_comparison['evac_off_peak_rss_mb']}MB (off).\n")
    else:
        lines.append("unavailable — eshkol-vm-standalone-test was not built "
                      "(configure with -DESHKOL_BUILD_TESTS=ON)\n")

    with open(args.md_out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
