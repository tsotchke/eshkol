#!/usr/bin/env python3
"""bench/axes/04_quantum_kernels_reduce.py — reduce H2-Hessian + VQE runs
into results.json fragment + markdown fragment.
"""
import argparse
import json
import re
import statistics


def parse_kv_lines(path, patterns):
    """patterns: dict of key -> (regex, caster). Returns dict of key -> value
    (first match wins), or {} if the file is missing."""
    out = {}
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return out
    for key, (rx, caster) in patterns.items():
        m = re.search(rx, text)
        if m:
            try:
                out[key] = caster(m.group(1))
            except ValueError:
                out[key] = m.group(1)
    return out


def ns_stats(ns_list_str):
    vals = [int(x) for x in ns_list_str.split() if x]
    if not vals:
        return None
    return {
        "raw_ns_samples": vals,
        "median_ns": statistics.median(vals),
        "min_ns": min(vals),
        "mean_ns": statistics.mean(vals),
    }


def fmt_ns(ns):
    if ns is None:
        return "n/a"
    if ns >= 1e9:
        return f"{ns/1e9:.3f} s"
    if ns >= 1e6:
        return f"{ns/1e6:.1f} ms"
    return f"{ns:.0f} ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2-ok", required=True)
    ap.add_argument("--h2-out", required=True)
    ap.add_argument("--h2-wall-ns", required=True)
    ap.add_argument("--quantum-enabled", required=True)
    ap.add_argument("--vqe-ok", required=True)
    ap.add_argument("--vqe-out", required=True)
    ap.add_argument("--vqe-wall-ns", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--md-out", required=True)
    args = ap.parse_args()

    h2_ok = args.h2_ok == "1"
    h2_values = parse_kv_lines(args.h2_out, {
        "r_star_bohr": (r"equilibrium R\*\s*=\s*([0-9.eE+-]+)", float),
        "force_constant_ha_per_bohr2": (r"force constant d2E/dR2 =\s*([0-9.eE+-]+)", float),
        "frequency_cm1": (r"vibrational frequency =\s*([0-9.eE+-]+)", float),
        "energy_ha": (r"E\(R\*\)\s*=\s*([0-9.eE+-]+)", float),
    }) if h2_ok else {}
    h2_timing = ns_stats(args.h2_wall_ns)

    quantum_enabled = args.quantum_enabled == "1"
    vqe_ok = args.vqe_ok == "1"
    vqe_values = parse_kv_lines(args.vqe_out, {
        "exact_ground_energy_ha": (r"H2 exact ground energy \(Ha\):\s*([0-9.eE+-]+)", float),
        "vqe_optimized_energy_ha": (r"H2 VQE optimized energy \(Ha\):\s*([0-9.eE+-]+)", float),
        "abs_vqe_minus_exact_ha": (r"H2 \|VQE - exact\| \(Ha\):\s*([0-9.eE+-]+)", float),
        "gradient_entries": (r"H2 VQE gradient entries:\s*(\d+)", int),
    }) if vqe_ok else {}
    vqe_timing = ns_stats(args.vqe_wall_ns)

    result = {
        "axis": "differentiable_quantum_kernels",
        "claims_tested": [
            "H2 vibrational frequency from an EXACT 2nd-order derivative "
            "(derivative-n) of a hand-written STO-3G Born-Oppenheimer "
            "energy surface, no finite differences",
            "VQE H2 ground-state energy + native adjoint gradient recovers "
            "Moonlab's exact ground-energy oracle within tolerance",
        ],
        "h2_vibrational_hessian": {
            "source": "examples/h2_vibrational.esk",
            "requires_quantum_build": False,
            "ok": h2_ok,
            "values": h2_values,
            "wall_clock": h2_timing,
            "wall_clock_includes_compile": False,
            "note": "wall_clock is the compiled AOT binary's run time; "
                    "compile time is excluded (see h2.compile.log per run in the work dir)",
        },
        "vqe_h2_energy_and_gradient": {
            "source": "tests/quantum/vqe_test.esk",
            "requires_quantum_build": True,
            "quantum_build_available": quantum_enabled,
            "ok": vqe_ok,
            "values": vqe_values,
            "wall_clock": vqe_timing,
            "wall_clock_includes_compile": False,
        },
    }

    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)

    lines = []
    lines.append("### Axis 4: differentiable quantum kernels\n")
    lines.append("**H2 vibrational frequency (exact 2nd-order AD Hessian, no quantum build required)**\n")
    if h2_ok:
        lines.append(f"- equilibrium R* = {h2_values.get('r_star_bohr')} bohr")
        lines.append(f"- force constant d2E/dR2 = {h2_values.get('force_constant_ha_per_bohr2')} Ha/bohr^2")
        lines.append(f"- vibrational frequency = {h2_values.get('frequency_cm1')} cm^-1")
        if h2_timing:
            lines.append(f"- wall clock (median of {len(h2_timing['raw_ns_samples'])} runs): {fmt_ns(h2_timing['median_ns'])}")
    else:
        lines.append("- unavailable (compile or run failed — see work dir logs)")

    lines.append("\n**VQE H2 energy + native adjoint gradient (requires -DESHKOL_QUANTUM_ENABLED=ON)**\n")
    if not quantum_enabled:
        lines.append("- unavailable — this build is not quantum-enabled\n")
    elif vqe_ok:
        lines.append(f"- H2 exact ground energy = {vqe_values.get('exact_ground_energy_ha')} Ha")
        lines.append(f"- H2 VQE optimized energy = {vqe_values.get('vqe_optimized_energy_ha')} Ha")
        lines.append(f"- |VQE - exact| = {vqe_values.get('abs_vqe_minus_exact_ha')} Ha")
        lines.append(f"- gradient entries = {vqe_values.get('gradient_entries')}")
        if vqe_timing:
            lines.append(f"- wall clock (median of {len(vqe_timing['raw_ns_samples'])} runs): {fmt_ns(vqe_timing['median_ns'])}")
    else:
        lines.append("- quantum build available but the run did not pass — see work dir logs\n")

    with open(args.md_out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
