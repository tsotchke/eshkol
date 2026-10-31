#!/usr/bin/env python3
"""Algebraic effects gate (ADR-0045).

1. Semantics: every fixture in tests/effects/*.esk runs on the native JIT
   (-r), as a native AOT binary, and on the bytecode VM; each transcript must
   equal tests/effects/expected/<name>.txt. Transcripts are normalised the way
   scripts/run_continuation_tests.sh normalises them (compiler noise dropped,
   all newlines removed, because the VM ends every `display` with a newline).
2. Effect typing: every tests/effects/typing/*.esk is compiled with
   --strict-types and ESHKOL_DUMP_EFFECTS=1.
     pos_*  must compile, and each `;; expect-row: name <row>` must be the row
            the checker printed for that procedure;
     neg_*  must be rejected, with every `;; expect-error: text` in the
            diagnostics.

Prints PASSED/FAILED per check and `PASS: effects gate` or `FAIL: effects gate`.
With --trace-dir it writes one runtime event per criterion to
<trace-dir>/v19_effects.jsonl (kind v1_9_types; names v19_algebraic_effects and
v19_effect_types).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOISE = re.compile(r"^(WARN|INFO:|DEBUG|\[ESKB\]|\[GPU\]|\s*\[compiled:|=== Eshkol VM|"
                   r"=== Execution complete ===|remark:|warning: <unknown>|\[REPL\]|"
                   r"eshkol-vm: note:)")


def normalise(text):
    return "".join(line for line in text.splitlines() if not NOISE.match(line))


def run(cmd, timeout, env=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           errors="replace", env=env)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT after %ds" % timeout


class Gate:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def check(self, label, ok, detail=""):
        if ok:
            self.passed += 1
            print("PASSED " + label)
        else:
            self.failed += 1
            self.failures.append(label)
            print("FAILED " + label)
            if detail:
                for line in detail.rstrip().splitlines()[:12]:
                    print("  " + line)


def semantics(gate, args, env):
    fixtures = sorted((REPO / "tests/effects").glob("*.esk"))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    for f in fixtures:
        name = f.stem
        exp = REPO / "tests/effects/expected" / (name + ".txt")
        if not exp.exists():
            gate.check(f"{name}::expected-file-present", False, f"missing {exp}")
            continue
        want = normalise(exp.read_text())

        rc, out, err = run([args.runner, "-r", str(f)], args.timeout, env)
        got = normalise(out)
        gate.check(f"{name}::native-jit", rc == 0 and got == want,
                   f"exit {rc}\nexpected: {want}\nactual  : {got}\n{err[-800:]}")

        exe = work / name
        rc, out, err = run([args.runner, "-o", str(exe), str(f)], args.timeout, env)
        if rc != 0 or not exe.exists():
            gate.check(f"{name}::native-aot", False, f"AOT compile exit {rc}\n{err[-800:]}")
        else:
            rc, out, err = run([str(exe)], args.timeout, env)
            got = normalise(out)
            gate.check(f"{name}::native-aot", rc == 0 and got == want,
                       f"exit {rc}\nexpected: {want}\nactual  : {got}\n{err[-800:]}")

        if args.vm:
            venv = dict(env, ESHKOL_VM_NO_DISASM="1", ESHKOL_VM_REGION_QUIET="1")
            rc, out, err = run([args.vm, str(f)], args.timeout, venv)
            got = normalise(out)
            gate.check(f"{name}::vm", rc == 0 and got == want,
                       f"exit {rc}\nexpected: {want}\nactual  : {got}\n{err[-800:]}")


def typing(gate, args, env):
    work = Path(args.work)
    tenv = dict(env, ESHKOL_DUMP_EFFECTS="1")
    for f in sorted((REPO / "tests/effects/typing").glob("*.esk")):
        name = f.stem
        text = f.read_text()
        rows = re.findall(r"^;; expect-row: (\S+) (<.*>)\s*$", text, re.M)
        errors = re.findall(r"^;; expect-error: (.*?)\s*$", text, re.M)
        exe = work / ("typing_" + name)
        rc, out, err = run([args.runner, "--strict-types", "-o", str(exe), str(f)], args.timeout, tenv)
        printed = dict(re.findall(r"^effects: (\S+) : (<.*>)$", err, re.M))
        if name.startswith("pos_"):
            gate.check(f"typing/{name}::accepted", rc == 0, f"exit {rc}\n{err[-1200:]}")
            for proc, row in rows:
                gate.check(f"typing/{name}::row::{proc}", printed.get(proc) == row,
                           f"expected {row}, inferred {printed.get(proc)}")
        else:
            gate.check(f"typing/{name}::rejected", rc != 0, f"exit {rc}\n{err[-1200:]}")
            for e in errors:
                gate.check(f"typing/{name}::diagnostic", e in err,
                           f"missing diagnostic: {e}\n{err[-1200:]}")


def emit(trace_dir, results):
    if not trace_dir:
        return
    d = Path(trace_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "v19_effects.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for name, (value, snippet) in results.items():
            fh.write(json.dumps({"kind": "v1_9_types", "name": name, "value": value,
                                 "snippet": snippet[:2000], "confidence": 0.95}) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default=str(REPO / "build/eshkol-run"))
    ap.add_argument("--vm", default=str(REPO / "build/eshkol-vm-standalone-test"))
    ap.add_argument("--work", default=str(REPO / "build/test-work/effects-gate"))
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    if not os.access(args.runner, os.X_OK):
        print(f"FAIL: effects gate (no runner at {args.runner})")
        return 2
    if args.vm and not os.access(args.vm, os.X_OK):
        print(f"FAIL: effects gate (no VM at {args.vm})")
        return 2
    env = dict(os.environ)
    env["ESHKOL_PATH"] = str(REPO / "lib") + (":" + env["ESHKOL_PATH"] if env.get("ESHKOL_PATH") else "")

    sem = Gate()
    semantics(sem, args, env)
    typ = Gate()
    typing(typ, args, env)

    results = {
        "v19_algebraic_effects": ("PASS" if sem.failed == 0 and sem.passed > 0 else "FAIL",
                                  f"semantics {sem.passed} passed, {sem.failed} failed "
                                  f"(JIT, AOT, VM); failures: {sem.failures[:8]}"),
        "v19_effect_types": ("PASS" if typ.failed == 0 and typ.passed > 0 else "FAIL",
                             f"typing {typ.passed} passed, {typ.failed} failed; "
                             f"failures: {typ.failures[:8]}"),
    }
    emit(args.trace_dir, results)
    total_failed = sem.failed + typ.failed
    print(f"\neffects semantics: {sem.passed} passed, {sem.failed} failed")
    print(f"effects typing: {typ.passed} passed, {typ.failed} failed")
    print("PASS: effects gate" if total_failed == 0 else "FAIL: effects gate")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
