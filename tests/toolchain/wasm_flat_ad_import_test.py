#!/usr/bin/env python3
"""Focused test for generated flat-AD import glue.

The generator's self-test needs only Python; the two runtime checks
instantiate real WASM modules and need Node.js. Pass the interpreter with
--node (CMake passes the one it found). Without Node the self-test still runs
and the test exits 77, which CTest reports as skipped rather than passed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIPPED = 77


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="", help="Node.js interpreter for the runtime checks")
    args = parser.parse_args()
    generator = ROOT / "scripts" / "generate_wasm_import_glue.py"
    commands = [[sys.executable, str(generator), "--selftest"]]
    if args.node:
        commands += [
            [args.node, str(Path(__file__).with_suffix(".js"))],
            [args.node, str(ROOT / "tests/toolchain/wasm_exact_runtime_test.js")],
        ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    if not args.node:
        print("SKIP: Node.js not found; the WASM runtime import checks did not run")
        return SKIPPED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
