#!/usr/bin/env python3
"""Execute compiler-emitted closure calls in both production WASM hosts."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--node", default="")
    args = parser.parse_args()
    if not args.node:
        print("SKIP: Node.js is required for actual WASM execution")
        return 77
    env = dict(os.environ, ESHKOL_PATH=str(ROOT / "lib"), ESHKOL_JIT_CACHE="0")
    # These are ordinary browser artifacts. Native coverage imports have no
    # browser implementation; the separate native/VM gates measure execution.
    env.pop("ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR", None)
    with tempfile.TemporaryDirectory(prefix="wasm-closure-") as directory:
        wasm = Path(directory) / "split.wasm"
        commands = [[str(args.compiler.resolve()), "--wasm",
                     str(ROOT / "tests/toolchain/wasm_closure_browser.esk"), "-o", str(wasm)],
                    [args.node, str(ROOT / "tests/toolchain/wasm_closure_runtime_test.js"), str(wasm)]]
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, env=env, check=False, timeout=90)
            if result.returncode:
                return result.returncode
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
