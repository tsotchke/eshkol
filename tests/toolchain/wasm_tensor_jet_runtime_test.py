#!/usr/bin/env python3
"""Execute compiler-emitted tensor derivatives and an independent wasm32 ABI caller."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--node", default="")
    parser.add_argument("--clang", default="")
    args = parser.parse_args()
    if not args.node or not args.clang:
        print("SKIP: Node.js and Clang with wasm32 support are required")
        return 77
    env = dict(os.environ, ESHKOL_PATH=str(ROOT / "lib"), ESHKOL_JIT_CACHE="0")
    with tempfile.TemporaryDirectory(prefix="wasm-tensor-jets-") as directory:
        out = Path(directory)
        program, abi = out / "program.wasm", out / "abi.wasm"
        commands = [
            [str(args.compiler.resolve()), "--wasm", str(ROOT / "tests/toolchain/wasm_tensor_jet_browser.esk"), "-o", str(program)],
            # Like Eshkol's WASM output, this is a raw object with exported
            # functions and imported memory. No external linker is involved.
            [args.clang, "--target=wasm32-unknown-unknown", "-ffreestanding", "-O2", "-c",
             str(ROOT / "tests/toolchain/wasm_tensor_jet_abi.c"), "-o", str(abi)],
            [args.node, str(ROOT / "tests/toolchain/wasm_tensor_jet_runtime_test.js"), str(program), str(abi)],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
