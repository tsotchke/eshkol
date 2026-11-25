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
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error("timeout must be positive and at most 600 seconds")
    # The original round trip plus the lease suite: slow peers, reset,
    # deadlines, cancellation, guaranteed close and descriptor cycles.
    suites = [
        ("tests/v1_2_edge_cases/net_roundtrip_test.esk", {"PASS: TCP roundtrip", "PASS: UDP roundtrip"}),
        ("tests/network/net_core_test.esk", {"PASS: TCP leases", "PASS: UDP leases", "RESULT: PASS"}),
    ]
    for source, required in suites:
        try:
            result = subprocess.run(
                ["bash", "-c", 'source "$1"; shift; eshkol_test_timeout "$@"',
                 "eshkol-net-gate", str(root / "scripts/lib/test_isolation.sh"), str(args.timeout),
                 str(args.runner.resolve()), "-r", str(root / source),
                 "-L" + str(args.runner.resolve().parent)],
                cwd=root, env=dict(os.environ, ESHKOL_PATH=str(root / "lib"), ESHKOL_JIT_CACHE="0",
                                  ESHKOL_LIB_DIR=str(args.runner.resolve().parent),
                                  ESHKOL_TEST_TIMEOUT_FORCE_PYTHON="1"),
                capture_output=True, text=True)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"FAIL: network gate: {error}", file=sys.stderr)
            return 1
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        lines = result.stdout.splitlines()
        if result.returncode or not required.issubset(lines) or any(l.startswith("FAIL:") for l in lines):
            print(f"FAIL: network gate {source} (runner exit {result.returncode})", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
