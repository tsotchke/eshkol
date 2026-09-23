#!/usr/bin/env python3
"""A configured frame ceiling must fail cleanly when exceeded."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: frame_ceiling_exit_status_test.py VM CEILING", file=sys.stderr)
        return 2
    vm, raw_ceiling = sys.argv[1:]
    ceiling = int(raw_ceiling)
    scratch = Path(__file__).resolve().parents[2] / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frame-ceiling-", dir=scratch) as directory:
        source = Path(directory) / "ceiling.esk"
        source.write_text(
            "(define (d n) (if (= n 0) 0 (+ 1 (d (- n 1)))))\n"
            f"(display (d {ceiling + 1}))\n"
        )
        result = subprocess.run(
            [vm, str(source)], capture_output=True, text=True,
            check=False, timeout=120,
            env={**os.environ, "ESHKOL_VM_NO_DISASM": "1"},
        )
    if result.returncode == 0 or "FRAME OVERFLOW" not in result.stderr or result.stdout:
        print(f"FAIL: ceiling {ceiling}: rc={result.returncode} "
              f"stdout={result.stdout!r} stderr={result.stderr!r}", file=sys.stderr)
        return 1
    print(f"PASS: frame ceiling {ceiling} exits nonzero without output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
