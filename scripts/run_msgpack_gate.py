#!/usr/bin/env python3
"""Require MessagePack wire golden tests and strict UTF-8 refusal tests."""
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
    args = parser.parse_args()
    for test, marker in [("msgpack_test.esk", "RESULT: OK"),
                         ("msgpack_utf8_test.esk", "PASS: MessagePack UTF-8 70 checks")]:
        try:
            result = subprocess.run(
                [str(args.runner.resolve()), "-r", str(root / "tests/v1_2_edge_cases" / test)],
                cwd=root, env=dict(os.environ, ESHKOL_PATH=str(root / "lib"), ESHKOL_JIT_CACHE="0"),
                capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"FAIL: {test}: {error}", file=sys.stderr)
            return 1
        print(result.stdout, end="")
        if result.returncode or marker not in [line.strip() for line in result.stdout.splitlines()]:
            print(result.stderr, end="", file=sys.stderr)
            print(f"FAIL: {test} (runner exit {result.returncode})", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
