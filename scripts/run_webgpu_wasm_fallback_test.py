#!/usr/bin/env python3
"""Compile the real WebGPU C backend and exercise no-device fallback in Node.

Requires Emscripten and Node. No browser adapter is requested; this proves
WASM integration, not GPU execution or precision.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emxx", default=os.environ.get("EMXX", "em++"))
    parser.add_argument("--node", default=os.environ.get("NODE", "node"))
    parser.add_argument("--work-dir", type=Path, default=root / ".scratch/webgpu-wasm-fallback")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(work / "tmp"), EMCC_CORES=os.environ.get("EMCC_CORES", "1"))
    output = work / "fallback.cjs"
    commands = [
        [args.emxx, "-std=c++17", "-I" + str(root / "inc"),
         str(root / "lib/backend/gpu/gpu_memory_webgpu.cpp"),
         str(root / "tests/webgpu/webgpu_wasm_fallback_test.cpp"),
         "-sENVIRONMENT=node", "-sEXIT_RUNTIME=1", "-o", str(output)],
        [args.node, str(output)],
    ]
    for label, command, timeout in zip(("compile", "execute"), commands, (180, 30)):
        try:
            result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                    text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"FAIL WebGPU WASM {label}: {error}", file=sys.stderr)
            return 1
        (work / (label + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            print(result.stdout + result.stderr, file=sys.stderr)
            return 1
        if label == "execute" and "PASS WebGPU WASM no-device CPU fallback" not in result.stdout.splitlines():
            print("FAIL WebGPU WASM: missing assertion marker", file=sys.stderr)
            return 1
    print("PASS WebGPU WASM no-device CPU fallback")
    return 0


if __name__ == "__main__":
    sys.exit(main())
