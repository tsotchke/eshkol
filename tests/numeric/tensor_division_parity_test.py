#!/usr/bin/env python3
"""Native JIT/VM parity for IEEE tensor division and exact-zero errors."""
import os
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[2]
build = pathlib.Path(sys.argv[1]).resolve()


def run(engine, program):
    source = root / "tests/numeric" / program
    env = dict(os.environ, ESHKOL_VM_NO_DISASM="1")
    command = ([str(build / "eshkol-run"), "-r", str(source), "-L", str(build)]
               if engine == "native" else
               [str(build / "eshkol-vm-standalone-test"), str(source)])
    return subprocess.run(command, cwd=root, env=env, text=True,
                          capture_output=True, timeout=120)


outputs = {}
for engine in ("native", "vm"):
    result = run(engine, "tensor_division_ieee_test.esk")
    if result.returncode or "ERROR:" in result.stderr:
        raise AssertionError(f"{engine} inexact division failed: {result.stderr[-500:]}")
    outputs[engine] = result.stdout

assert outputs["native"] == outputs["vm"], "native/VM tensor division differs"
lines = outputs["native"].splitlines()
assert len(lines) == 8, f"expected 8 result lines, got {len(lines)}"
assert lines[:3] == ["+inf.0", "-inf.0", "+nan.0"]
assert lines[3:5] == ["#((+inf.0))", "#((+nan.0))"]
assert lines[5] == "#(+inf.0 +inf.0 +nan.0)"
assert lines[6] == "#((+inf.0 -inf.0) (+inf.0 -inf.0))"
assert lines[7].count("-inf.0") == 64

for engine in ("native", "vm"):
    result = run(engine, "exact_rational_zero_test.esk")
    assert result.returncode != 0 and any(
        phrase in result.stderr.lower() for phrase in ("division by zero", "divide by zero")), (
        f"{engine} exact rational division did not raise: {result}")
    assert result.stdout == "", f"{engine} printed a value after exact division by zero"

print("PASS native/VM IEEE tensor division: 8 identical lines; exact zero raises")
