#!/usr/bin/env python3
"""bench/combine_results.py — merge the fingerprint + four per-axis JSON
fragments into one results.json, and the four per-axis markdown fragments
into one results.md.

Invoked by bench/run_public_benchmarks.sh:
    combine_results.py --fingerprint PATH --axis1 PATH --axis2 PATH
        --axis3 PATH --axis4 PATH --axis1-md PATH ... --json-out PATH
        --md-out PATH --smoke 0|1 --started-at ISO8601 --finished-at ISO8601
"""
import argparse
import json


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_text(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint", required=True)
    ap.add_argument("--axis1")
    ap.add_argument("--axis2")
    ap.add_argument("--axis3")
    ap.add_argument("--axis4")
    ap.add_argument("--axis1-md")
    ap.add_argument("--axis2-md")
    ap.add_argument("--axis3-md")
    ap.add_argument("--axis4-md")
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--md-out", required=True)
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--started-at", required=True)
    ap.add_argument("--finished-at", required=True)
    args = ap.parse_args()

    fingerprint = load_json(args.fingerprint)

    result = {
        "schema": "eshkol-public-benchmarks-v1",
        "smoke_mode": args.smoke == "1",
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "environment": fingerprint,
        "axes": {
            "1_exact_ad_cost_curves": load_json(args.axis1) if args.axis1 else None,
            "2_ozaki_ii_gemm": load_json(args.axis2) if args.axis2 else None,
            "3_flat_rss_under_resident_load": load_json(args.axis3) if args.axis3 else None,
            "4_differentiable_quantum_kernels": load_json(args.axis4) if args.axis4 else None,
        },
    }

    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)

    lines = []
    lines.append("# Eshkol public benchmark results\n")
    mode = "SMOKE (fast harness check — not a measurement run)" if result["smoke_mode"] else "FULL measurement run"
    lines.append(f"Mode: {mode}  \nStarted: {args.started_at}  \nFinished: {args.finished_at}\n")
    if fingerprint:
        cpu = fingerprint.get("cpu", {})
        os_info = fingerprint.get("os", {})
        git = fingerprint.get("git", {})
        lines.append("## Environment\n")
        lines.append(f"- OS: {os_info.get('name')} {os_info.get('version')} (kernel {os_info.get('kernel')})")
        lines.append(f"- CPU: {cpu.get('model')} ({cpu.get('physical_cores')} physical / {cpu.get('logical_cores')} logical cores)")
        lines.append(f"- GPU: {fingerprint.get('gpu_model')}")
        lines.append(f"- Memory: {fingerprint.get('memory_bytes', 0) / (1024**3):.0f} GiB")
        lines.append(f"- Compiler: {fingerprint.get('compiler', {}).get('cc')}")
        lines.append(f"- LLVM: {fingerprint.get('compiler', {}).get('llvm')}")
        lines.append(f"- BLAS: {fingerprint.get('blas')}")
        lines.append(f"- git: {git.get('sha')} ({git.get('branch')}), dirty={git.get('dirty')}")
        lines.append(f"- eshkol version: {fingerprint.get('eshkol_version')}")
        build = fingerprint.get("build", {})
        lines.append(f"- build: {build.get('type')}, quantum_enabled={build.get('quantum_enabled')}, gpu_enabled={build.get('gpu_enabled')}")
        lines.append(f"- load average at capture: {fingerprint.get('load_average_at_capture')}\n")

    for md_path in (args.axis1_md, args.axis2_md, args.axis3_md, args.axis4_md):
        if not md_path:
            continue
        text = load_text(md_path)
        if text:
            lines.append(text)

    with open(args.md_out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
