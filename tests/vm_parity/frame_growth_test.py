#!/usr/bin/env python3
"""Exercise call-frame growth on source execution."""
import os
import subprocess
import sys


def run(vm, source):
    result = subprocess.run([vm, source], capture_output=True, text=True,
                            check=False, timeout=120,
                            env={**os.environ, "ESHKOL_VM_NO_DISASM": "1"})
    return result


def main():
    if len(sys.argv) != 3:
        print("usage: frame_growth_test.py VM SOURCE", file=sys.stderr)
        return 2
    result = run(sys.argv[1], sys.argv[2])
    if (result.returncode or result.stdout.strip() != "1200" or
            "FRAME OVERFLOW" in result.stderr or "ERROR" in result.stderr):
        print(f"FAIL: frame growth: rc={result.returncode} "
              f"stdout={result.stdout!r} stderr={result.stderr!r}", file=sys.stderr)
        return 1
    print("PASS: non-tail recursion crossed initial frame capacity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
