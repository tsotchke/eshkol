#!/usr/bin/env python3
"""check_wasm_size_budget.py — enforce ADR-0064 WASM binary size ceiling.

Default ceiling is +5% over the recorded v1.4.0-connection baseline for
site/static/eshkol-vm.wasm so architectural VM parity work never introduces
uncontrolled WASM bloat without dead-code elimination or lazy-loading.
"""

import argparse
import os
import sys

BASELINE_VM_WASM_BYTES = 1762451  # v1.4.0-connection baseline (site/static/eshkol-vm.wasm)
MAX_GROWTH_RATIO = 1.05           # hard +5% ceiling per ADR-0064


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--wasm",
        default="site/static/eshkol-vm.wasm",
        help="Path to eshkol-vm.wasm to verify",
    )
    ap.add_argument(
        "--baseline",
        type=int,
        default=BASELINE_VM_WASM_BYTES,
        help="Baseline byte size (default: v1.4.0-connection baseline)",
    )
    ap.add_argument(
        "--max-growth-pct",
        type=float,
        default=5.0,
        help="Maximum allowed growth percentage over baseline (default: 5.0)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.wasm):
        print(f"FAIL: wasm binary not found: {args.wasm}", file=sys.stderr)
        return 2

    actual = os.path.getsize(args.wasm)
    ceiling = int(args.baseline * (1.0 + args.max_growth_pct / 100.0))
    delta = actual - args.baseline
    pct = (delta / args.baseline) * 100.0 if args.baseline else 0.0

    status = "PASS" if actual <= ceiling else "FAIL"
    print(
        f"wasm_size_budget: {status} "
        f"(actual={actual} B, baseline={args.baseline} B, "
        f"delta={delta:+d} B [{pct:+.2f}%], ceiling={ceiling} B [+{args.max_growth_pct:.1f}%])"
    )
    return 0 if actual <= ceiling else 1


if __name__ == "__main__":
    raise SystemExit(main())
