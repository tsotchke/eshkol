#!/usr/bin/env python3
"""Bounded TCP/UDP loopback verification through the public core.net module."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path,
                        default=root / os.environ.get("BUILD_DIR", "build") / "eshkol-run")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error("timeout must be positive and at most 600 seconds")
    try:
        result = subprocess.run(
            [str(args.runner.resolve()), "-r", str(root / "tests/v1_2_edge_cases/net_roundtrip_test.esk")],
            cwd=root, env=dict(os.environ, ESHKOL_PATH=str(root / "lib"), ESHKOL_JIT_CACHE="0"),
            capture_output=True, text=True, timeout=args.timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: network gate: {error}", file=sys.stderr)
        return 1
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode or not {"PASS: TCP roundtrip", "PASS: UDP roundtrip"}.issubset(result.stdout.splitlines()):
        print(f"FAIL: network gate (runner exit {result.returncode})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
