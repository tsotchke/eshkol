#!/usr/bin/env python3
"""Real native JIT/AOT results for self-calls inside tail-analysis boundaries."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

runner, lib_dir = sys.argv[1:]
cases = {
    "non-tail-match": ("(let loop ((i 3)) (match i (0 42) (_ (+ 1 (loop (- i 1))))))", "45"),
    "tail-match": ("(let loop ((i 10)) (match i (0 42) (_ (loop (- i 1)))))", "42"),
    "non-tail-values": ("(let loop ((i 3)) (if (= i 0) 42 (let-values (((x) (values (loop (- i 1))))) (+ x 1))))", "45"),
    "non-tail-star-values": ("(let loop ((i 3)) (if (= i 0) 42 (let*-values (((x) (values (loop (- i 1))))) (+ x 1))))", "45"),
    "non-tail-do": ("(let loop ((i 3)) (if (= i 0) 42 (do ((j 0 (+ j 1))) ((= j 1) (+ 1 (loop (- i 1)))))))", "45"),
    "non-tail-callcc": ("(let loop ((i 3)) (if (= i 0) 42 (call/cc (lambda (k) (+ 1 (loop (- i 1)))))))", "45"),
    "non-tail-wind": ("(let loop ((i 3)) (if (= i 0) 42 (dynamic-wind (lambda () #t) (lambda () (+ 1 (loop (- i 1)))) (lambda () #t))))", "45"),
    # Preserve completion of a large ordinary tail loop while fixing the
    # skipped forms. This checks operand evaluation and actual stack behavior.
    "ordinary-tail-control": ("(let loop ((i 100000) (sum 0)) (if (= i 0) sum (loop (- i 1) (+ sum 1))))", "100000"),
}
env = os.environ.copy()
env["ESHKOL_JIT_CACHE"] = "0"

def run(argv, expected=None):
    result = subprocess.run(argv, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, timeout=60)
    if result.returncode != 0 or (expected is not None and result.stdout.strip() != expected):
        raise SystemExit(f"FAIL: {argv!r} exit={result.returncode} expected={expected!r}\n"
                         f"stdout={result.stdout}\nstderr={result.stderr}")

with tempfile.TemporaryDirectory(prefix="eshkol-tco-control-flow-") as tmp:
    root = Path(tmp)
    for name, (expr, expected) in cases.items():
        source = root / (name + ".esk")
        source.write_text("(display " + expr + ")\n(newline)\n", encoding="utf-8")
        run([runner, "-n", "-r", str(source)], expected)
        binary = root / name
        run([runner, "-n", str(source), "-o", str(binary), "-L", lib_dir])
        run([str(binary)], expected)
        print(f"PASS: {name} jit+aot={expected}")
print(f"PASS: named-let control flow {len(cases)} cases across JIT and AOT")
