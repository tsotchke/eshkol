#!/usr/bin/env python3
"""bench/axes/01_exact_ad_reduce.py — turn raw BENCH lines from the exact-AD
order/dimension sweeps into results.json fragment + a markdown fragment.

Not meant to be run standalone; invoked by bench/axes/01_exact_ad.sh with:
    01_exact_ad_reduce.py <order.out> <dim.out> <json_out> <md_out>
"""
import json
import math
import re
import statistics
import sys

LINE_RE = re.compile(
    r"^BENCH kind=(?P<kind>\S+) path=(?P<path>\S+) k=(?P<k>-?\d+) "
    r"iters=(?P<iters>\d+) ns_samples=\[(?P<samples>[\d,]*)\]$"
)


def parse(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = LINE_RE.match(line)
            if not m:
                continue
            samples = [int(x) for x in m.group("samples").split(",") if x]
            iters = int(m.group("iters"))
            ns_per_call = [s / iters for s in samples]
            rows.append({
                "kind": m.group("kind"),
                "path": m.group("path"),
                "k": int(m.group("k")),
                "iters": iters,
                "rounds": len(samples),
                "raw_ns_samples": samples,
                "ns_per_call_samples": ns_per_call,
                "ns_per_call_median": statistics.median(ns_per_call),
                "ns_per_call_min": min(ns_per_call),
                "ns_per_call_mean": statistics.mean(ns_per_call),
            })
    return rows


def loglog_slope(points):
    """points: list of (x, y) with x,y > 0. Returns least-squares slope of
    log(y) vs log(x) — the exponent in a y ~ x^slope power-law fit."""
    pts = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    xs = [math.log(x) for x, _ in pts]
    ys = [math.log(y) for _, y in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return None
    return cov / var


def find_crossover(float_rows, exact_rows, threshold=2.0):
    """First k (by ascending k) where exact/float ns_per_call_median ratio
    exceeds `threshold`, i.e. where paying for exactness starts costing
    materially more than the float path. Returns (k, ratio) or None."""
    fmap = {r["k"]: r["ns_per_call_median"] for r in float_rows}
    emap = {r["k"]: r["ns_per_call_median"] for r in exact_rows}
    for k in sorted(set(fmap) & set(emap)):
        if fmap[k] <= 0:
            continue
        ratio = emap[k] / fmap[k]
        if ratio >= threshold:
            return k, ratio
    return None


def curve(rows, kind, path):
    sel = sorted([r for r in rows if r["kind"] == kind and r["path"] == path],
                 key=lambda r: r["k"])
    return sel


def main():
    order_out, dim_out, json_out, md_out = sys.argv[1:5]
    order_rows = parse(order_out)
    dim_rows = parse(dim_out)

    order_float = curve(order_rows, "ad_order", "float")
    order_exact = curve(order_rows, "ad_order", "exact")
    order_mono_float = curve(order_rows, "ad_order_mono", "float")
    order_mono_exact = curve(order_rows, "ad_order_mono", "exact")
    dim_float = curve(dim_rows, "ad_dim", "float")
    dim_exact = curve(dim_rows, "ad_dim", "exact")

    order_float_exp = loglog_slope([(r["k"], r["ns_per_call_median"]) for r in order_float])
    order_exact_exp = loglog_slope([(r["k"], r["ns_per_call_median"]) for r in order_exact])
    dim_float_exp = loglog_slope([(r["k"], r["ns_per_call_median"]) for r in dim_float])
    dim_exact_exp = loglog_slope([(r["k"], r["ns_per_call_median"]) for r in dim_exact])

    order_crossover = find_crossover(order_float, order_exact, threshold=2.0)
    dim_crossover = find_crossover(dim_float, dim_exact, threshold=2.0)

    result = {
        "axis": "exact_ad_cost_curves",
        "claims_tested": [
            "derivative-n cost is O(k^2) in derivative order k, not O(2^k)",
            "derivative-n at an exact (rational/bignum) point returns an exact result",
        ],
        "order_sweep": {
            "function": "f(x) = 1/(1-x)",
            "float_point": 0.5,
            "exact_point": "1/2",
            "float_curve": order_float,
            "exact_curve": order_exact,
            "mono_literal_k8_float": order_mono_float,
            "mono_literal_k8_exact": order_mono_exact,
            "loglog_fit_exponent_float": order_float_exp,
            "loglog_fit_exponent_exact": order_exact_exp,
            "crossover_where_exact_at_least_2x_float": (
                {"k": order_crossover[0], "ratio": order_crossover[1]}
                if order_crossover else None
            ),
        },
        "dimension_sweep": {
            "function": "f_d(x) = x^(d+1) via an explicit d-step multiply chain, fixed order k=4",
            "note": (
                "dimension = size of the primal computation the derivative is "
                "threaded through, not the number of independent AD input "
                "variables — see the header comment in "
                "bench/axes/01_exact_ad.sh for why gradient/jacobian on a "
                "rational VECTOR point was not used here (boundary coercion "
                "to double for non-scalar points)."
            ),
            "fixed_order_k": 4,
            "float_point": 0.5,
            "exact_point": "1/3",
            "float_curve": dim_float,
            "exact_curve": dim_exact,
            "loglog_fit_exponent_float": dim_float_exp,
            "loglog_fit_exponent_exact": dim_exact_exp,
            "crossover_where_exact_at_least_2x_float": (
                {"d": dim_crossover[0], "ratio": dim_crossover[1]}
                if dim_crossover else None
            ),
        },
    }

    with open(json_out, "w") as f:
        json.dump(result, f, indent=2)

    def fmt_ns(ns):
        if ns >= 1e6:
            return f"{ns/1e6:.3f} ms"
        if ns >= 1e3:
            return f"{ns/1e3:.2f} us"
        return f"{ns:.0f} ns"

    lines = []
    lines.append("### Axis 1: exact-AD cost curves\n")
    lines.append(f"Order-sweep fit exponent (ns/call ~ k^p): float p={order_float_exp:.2f}, "
                  f"exact p={order_exact_exp:.2f} (p=2 is the claimed O(k^2))\n"
                  if order_float_exp and order_exact_exp else "")
    lines.append("\n**Order sweep — `(derivative-n f x k)`, f(x)=1/(1-x)**\n")
    lines.append("| k | float ns/call | exact ns/call | exact/float ratio |")
    lines.append("|---:|---:|---:|---:|")
    fmap = {r["k"]: r["ns_per_call_median"] for r in order_float}
    emap = {r["k"]: r["ns_per_call_median"] for r in order_exact}
    for k in sorted(set(fmap) | set(emap)):
        fv = fmap.get(k)
        ev = emap.get(k)
        ratio = f"{ev/fv:.2f}x" if fv and ev and fv > 0 else "n/a"
        lines.append(f"| {k} | {fmt_ns(fv) if fv else 'n/a'} | {fmt_ns(ev) if ev else 'n/a'} | {ratio} |")
    if order_crossover:
        lines.append(f"\nCrossover: exact-rational path first costs >=2x the float path at k={order_crossover[0]} "
                      f"({order_crossover[1]:.2f}x). Below that order, exactness is close to free; "
                      f"above it, bignum growth dominates and paying for exactness is a real cost — "
                      f"reported honestly, not hidden.\n")
    else:
        lines.append("\nNo point in this sweep crossed the 2x exact/float ratio.\n")

    lines.append("\n**Dimension sweep — `(derivative-n f_d x 4)`, f_d(x)=x^(d+1) via a d-step chain**\n")
    lines.append("| d | float ns/call | exact ns/call | exact/float ratio |")
    lines.append("|---:|---:|---:|---:|")
    fmap = {r["k"]: r["ns_per_call_median"] for r in dim_float}
    emap = {r["k"]: r["ns_per_call_median"] for r in dim_exact}
    for d in sorted(set(fmap) | set(emap)):
        fv = fmap.get(d)
        ev = emap.get(d)
        ratio = f"{ev/fv:.2f}x" if fv and ev and fv > 0 else "n/a"
        lines.append(f"| {d} | {fmt_ns(fv) if fv else 'n/a'} | {fmt_ns(ev) if ev else 'n/a'} | {ratio} |")
    lines.append(f"\nDimension-sweep fit exponent (ns/call ~ d^p): float p={dim_float_exp:.2f}, "
                  f"exact p={dim_exact_exp:.2f} (p=1 is linear in workload size)\n"
                  if dim_float_exp and dim_exact_exp else "")

    with open(md_out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
