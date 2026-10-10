#!/usr/bin/env python3
"""Expansion renames every binder at any nesting depth, JIT and AOT.

The expander gives each binder a fresh name and rewrites its references. It
must reach every level of the program, so a reference nested deeper than any
set bound still finds its binder; and only nesting that expansion itself
introduces is limited, so a macro that grows without end still stops with a
diagnostic. Depths straddle 1000, the former traversal bound. With --vm, the
bytecode VM must either produce the same value for a deep program or refuse
it with a nonzero exit; it may never run it to a different answer.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--native", required=True)
parser.add_argument("--vm")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]


def nested_lets(depth):
    opens = "".join(f"(let ((x{i} {i})) " for i in range(1, depth + 1))
    return f"(display {opens}(+ x1 x{depth}){')' * depth})\n(newline)\n"


def mixed_binders(depth):
    """let, let*, letrec, lambda application and let-values, in rotation."""
    text = f"(+ x1 x{depth})"
    for i in range(depth, 0, -1):
        kind = i % 5
        if kind == 0:
            text = f"(let ((x{i} {i})) {text})"
        elif kind == 1:
            text = f"(let* ((x{i} {i})) {text})"
        elif kind == 2:
            text = f"(letrec ((x{i} {i})) {text})"
        elif kind == 3:
            text = f"((lambda (x{i}) {text}) {i})"
        else:
            text = f"(let-values (((x{i}) (values {i}))) {text})"
    return f"(display {text})\n(newline)\n"


def deep_source_macro_uses(depth, uses):
    """Macro uses written in the source, nested inside `depth` calls."""
    return ("(define-syntax inc (syntax-rules () ((_ e) (+ 1 e))))\n"
            f"(display {'(+ 1 ' * depth}{'(inc ' * uses}0{')' * (depth + uses)})\n(newline)\n")


RUNAWAY = ("(define-syntax grow (syntax-rules () ((_ e) (list (grow e)))))\n"
           "(display (grow 1))\n(newline)\n")

scratch = root / ".scratch"
scratch.mkdir(exist_ok=True)
checks = 0
with tempfile.TemporaryDirectory(prefix="macro-depth-", dir=scratch) as work:
    work = Path(work)
    env = dict(os.environ, ESHKOL_JIT_CACHE="0", ESHKOL_VM_NO_DISASM="1",
               ESHKOL_JIT_CACHE_DIR=str(work / "jit"),
               ESHKOL_AOT_MODULE_CACHE_DIR=str(work / "aot-cache"))

    def run(label, command):
        result = subprocess.run(command, cwd=root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=300)
        return result.returncode, result.stdout

    def expect_value(label, source, value, aot=False):
        global checks
        path = work / f"{label}.esk"
        path.write_text(source)
        rc, out = run(label, [args.native, "-r", str(path)])
        if rc or str(value) not in out.splitlines():
            raise SystemExit(f"FAIL [{label} JIT] exit {rc}, expected {value}\n{out[-3000:]}")
        checks += 1
        if aot:
            binary = work / label
            rc, out = run(label, [args.native, str(path), "-o", str(binary)])
            if rc:
                raise SystemExit(f"FAIL [{label} AOT compile] exit {rc}\n{out[-3000:]}")
            rc, out = run(label, [str(binary)])
            if rc or str(value) not in out.splitlines():
                raise SystemExit(f"FAIL [{label} AOT] exit {rc}, expected {value}\n{out[-3000:]}")
            checks += 1
        print(f"PASS: {label} = {value}", flush=True)

    for depth in (997, 998, 1500):
        expect_value(f"nested_lets_{depth}", nested_lets(depth), depth + 1, aot=depth == 1500)
    expect_value("mixed_binders_1200", mixed_binders(1200), 1201, aot=True)
    expect_value("deep_source_macro_uses_1200", deep_source_macro_uses(1200, 20), 1220, aot=True)

    if args.vm:
        path = work / "nested_lets_1500.esk"
        rc, out = run("vm", [args.vm, str(path)])
        if rc == 0 and "1501" not in out.splitlines():
            raise SystemExit(f"FAIL [VM nested_lets_1500] exit 0 without 1501\n{out[-3000:]}")
        checks += 1
        print(f"PASS: VM nested_lets_1500 {'= 1501' if rc == 0 else f'refused (exit {rc})'}", flush=True)

    path = work / "runaway.esk"
    path.write_text(RUNAWAY)
    rc, out = run("runaway", [args.native, "-r", str(path)])
    if rc == 0 or "macro expansion depth limit exceeded" not in out:
        raise SystemExit(f"FAIL [runaway] exit {rc}, expected the depth diagnostic\n{out[-3000:]}")
    checks += 1
    print("PASS: a self-nesting macro stops with the depth diagnostic", flush=True)

print(f"PASS: {checks} macro expansion depth checks")
