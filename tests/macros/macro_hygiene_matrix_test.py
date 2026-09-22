#!/usr/bin/env python3
"""R7RS syntax-rules hygiene matrix on native JIT, native AOT and the VM.

Runs tests/vm_parity/corpus/93_macro_hygiene_matrix.esk on every engine and
requires every hand-computed check to pass on each (ADR-0026, SW-192).
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--native", required=True)
parser.add_argument("--vm", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
source = root / "tests/vm_parity/corpus/93_macro_hygiene_matrix.esk"
expected = len(re.findall(r'^\s*\(hyg "', source.read_text(), re.MULTILINE))
scratch = root / ".scratch"
scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="macro-hygiene-", dir=scratch) as work:
    env = dict(os.environ, ESHKOL_JIT_CACHE_DIR=work + "/jit",
               ESHKOL_AOT_MODULE_CACHE_DIR=work + "/aot-cache",
               ESHKOL_VM_NO_DISASM="1")

    def run(label, command, check=True):
        result = subprocess.run(command, cwd=root, env=env, text=True,
                                capture_output=True, timeout=300)
        output = result.stdout + result.stderr
        if result.returncode:
            raise SystemExit(f"FAIL [{label}] exit {result.returncode}: {command}\n{output}")
        if not check:
            return
        passed = [l for l in result.stdout.splitlines() if l.startswith("PASS hygiene:")]
        failed = [l for l in output.splitlines() if l.startswith("FAIL hygiene:")]
        if failed or len(passed) != expected:
            raise SystemExit(f"FAIL [{label}] {len(passed)}/{expected} checks passed\n"
                             + "\n".join(failed) + "\n" + output[-2000:])

    run("native JIT", [args.native, "-r", str(source)])
    executable = work + "/hygiene-matrix"
    run("native AOT compile", [args.native, str(source), "-o", executable], check=False)
    run("native AOT", [executable])
    run("VM", [args.vm, str(source)])
print(f"PASS: {expected} hygiene checks agree on native JIT, AOT, and VM")
