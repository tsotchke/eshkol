#!/usr/bin/env python3
"""Independent expected-value checks for native JIT/AOT and VM macro scope."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--native", required=True)
parser.add_argument("--vm", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
source = root / "tests/vm_parity/corpus/92_macro_definition_scope.esk"
scratch = root / ".scratch"
scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="macro-scope-", dir=scratch) as work:
    env = dict(os.environ, ESHKOL_JIT_CACHE_DIR=work + "/jit",
               ESHKOL_AOT_MODULE_CACHE_DIR=work + "/aot-cache",
               ESHKOL_VM_NO_DISASM="1")

    def run(command, check_values=True, expected=None):
        result = subprocess.run(command, cwd=root, env=env, text=True,
                                capture_output=True, timeout=120)
        output = result.stdout + result.stderr
        if result.returncode or "FAIL macro:" in output:
            raise SystemExit(f"FAIL: {command}\n{output}")
        if expected is not None and result.stdout.splitlines() != expected:
            raise SystemExit(f"FAIL: expected {expected}: {command}\n{output}")
        if check_values and expected is None and sum(line.startswith("PASS macro:")
                                for line in result.stdout.splitlines()) != 14:
            raise SystemExit(f"FAIL: expected 14 value checks: {command}\n{output}")

    run([args.native, "-r", str(source)])
    run([args.vm, str(source)])
    executable = work + "/macro-scope"
    run([args.native, str(source), "-o", executable], check_values=False)
    run([executable])
    bindings = root / "tests/macros/native_binding_scope_test.esk"
    expected = ["3", "4", "7", "6", "2", "3", "5", "(x 1)"]
    run([args.native, "-r", str(bindings)], expected=expected)
    run([args.native, str(bindings), "-o", executable], check_values=False)
    run([executable], expected=expected)
    traversal = root / "tests/macros/native_traversal_test.esk"
    expected = ["10", "2", "10", "15", "15", "12", "9"]
    run([args.native, "-r", str(traversal)], expected=expected)
    run([args.native, str(traversal), "-o", executable], check_values=False)
    run([executable], expected=expected)
print("PASS: 14 macro scope values agree on native JIT, AOT, and VM")
