#!/usr/bin/env python3
"""LE-07: a VM frame overflow must be observable as a failing process."""

import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: frame_overflow_exit_status_test.py VM SOURCE", file=sys.stderr)
        return 2
    result = subprocess.run(
        [sys.argv[1], sys.argv[2]],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        print("FAIL: frame overflow returned zero", file=sys.stderr)
        return 1
    if "FRAME OVERFLOW" not in result.stderr:
        print("FAIL: frame overflow diagnostic missing", file=sys.stderr)
        return 1
    print("PASS: frame overflow exits nonzero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
