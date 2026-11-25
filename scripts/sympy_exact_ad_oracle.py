#!/usr/bin/env python3
"""SymPy differential oracle for Eshkol's exact-AD surface.

Runs a fixed corpus (tests/sympy_oracle/exact_ad_corpus.json) through
Eshkol's built-in AD operators -- derivative-n, taylor, derivative,
gradient, hessian, jacobian -- under JIT (`eshkol-run -r`) and AOT
(`eshkol-run -o` then the binary), computes the same quantities with SymPy
on exact rationals, and compares them case by case:

  expect_exact = true   the Eshkol result must be exact and equal SymPy's
                        rational value
  expect_exact = false  an inexact Eshkol result must agree to f64 rounding;
                        an exact one must equal SymPy exactly

Each case and engine emits one ICC runtime_event (kind `sympy_oracle`,
name `sympy_exact_ad_<id>_<engine>`), and the run emits the summary event
`sympy_exact_ad_oracle_pilot` (PASS only when every case agrees on both
engines). SymPy runs in a separate interpreter: ESHKOL_SYMPY_PYTHON when
set, else this interpreter. When no interpreter can import SymPy the run
records SKIP with the reason and exits 77 (CTest SKIP_RETURN_CODE).

Exit codes: 0 PASS, 1 FAIL, 2 infrastructure error, 77 SKIP.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

SUMMARY_NAME = "sympy_exact_ad_oracle_pilot"
KIND = "sympy_oracle"
REL_TOL = 1e-11
ABS_TOL = 1e-11

SYMPY_REFERENCE = r'''
import json, sys
import sympy as sp
corpus = json.load(open(sys.argv[1]))
out = {"sympy_version": sp.__version__, "cases": {}}
def rat(s):
    return sp.Rational(s)
def ser(v):
    v = sp.simplify(v)
    if v.is_Rational:
        return {"exact": str(v)}
    return {"exact": None, "approx": str(sp.N(v, 40))}
for case in corpus["cases"]:
    names = case["vars"]
    syms = sp.symbols(names)
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    env = {n: s for n, s in zip(names, syms)}
    env.update({"Rational": sp.Rational, "Subs": sp.Subs, "Derivative": sp.Derivative,
                "y": sp.Symbol("y"), "exp": sp.exp, "sin": sp.sin, "log": sp.log})
    point = {s: rat(p) for s, p in zip(syms, case["point"])}
    def parse(e):
        return sp.sympify(e, locals=env)
    op = case["op"]
    if op == "jacobian":
        exprs = [parse(e) for e in case["sympy"]]
        value = [[ser(sp.diff(e, s).subs(point)) for s in syms] for e in exprs]
    else:
        e = parse(case["sympy"])
        x = syms[0]
        if op == "derivative-n":
            value = [ser(sp.diff(e, x, n).subs(point) if n > 0 else e.subs(point)) for n in case["orders"]]
        elif op == "taylor":
            value = [ser((sp.diff(e, x, k).subs(point) if k > 0 else e.subs(point)) / sp.factorial(k))
                     for k in range(case["order"] + 1)]
        elif op == "derivative" or (op == "gradient" and case.get("scalar_point")):
            value = ser(sp.diff(e, x).subs(point))
        elif op == "hessian" and case.get("scalar_point"):
            value = ser(sp.diff(e, x, 2).subs(point))
        elif op == "gradient":
            value = [ser(sp.diff(e, s).subs(point)) for s in syms]
        elif op == "hessian":
            value = [[ser(sp.diff(e, a, b).subs(point)) for b in syms] for a in syms]
        else:
            raise SystemExit("unknown op " + op)
    out["cases"][case["id"]] = value
json.dump(out, sys.stdout)
'''

ESHKOL_PRELUDE = r'''
(define (oracle-flatten v)
  (cond ((number? v) (list v))
        ((null? v) '())
        ((pair? v) (append (oracle-flatten (car v)) (oracle-flatten (cdr v))))
        ((vector? v) (oracle-flatten (vector->list v)))
        (else (list v))))
(define (oracle-all-exact? v)
  (let loop ((l (oracle-flatten v)))
    (or (null? l) (and (number? (car l)) (exact? (car l)) (loop (cdr l))))))
(define (oracle-emit id v)
  (display "ORACLE ") (display id) (display " ")
  (display (if (oracle-all-exact? v) "exact" "inexact")) (display " ")
  (write v) (newline))
'''


def esk_point(case):
    pts = case["point"]
    if len(case["vars"]) == 1 and case["op"] not in ("gradient", "hessian", "jacobian"):
        return pts[0]
    if case.get("scalar_point"):
        return pts[0]
    return "(vector " + " ".join(pts) + ")"


def esk_expr(case):
    f, op, p = case["eshkol"], case["op"], esk_point(case)
    if op == "derivative-n":
        return "(list " + " ".join(f"(derivative-n {f} {p} {n})" for n in case["orders"]) + ")"
    if op == "taylor":
        return f"(taylor {f} {p} {case['order']})"
    return f"({op} {f} {p})"


def build_program(corpus):
    lines = [ESHKOL_PRELUDE]
    for case in corpus["cases"]:
        lines.append(f"(oracle-emit \"{case['id']}\" {esk_expr(case)})")
    lines.append('(display "ORACLE-END") (newline)')
    return "\n".join(lines) + "\n"


def tokenize(text):
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            out.append(c)
            i += 1
        elif c == "#" and i + 1 < len(text) and text[i + 1] == "(":
            out.append("(")
            i += 2
        else:
            j = i
            while j < len(text) and not text[j].isspace() and text[j] not in "()":
                j += 1
            out.append(text[i:j])
            i = j
    return out


def parse_datum(tokens, pos=0):
    tok = tokens[pos]
    if tok == "(":
        items, pos = [], pos + 1
        while tokens[pos] != ")":
            item, pos = parse_datum(tokens, pos)
            items.append(item)
        return items, pos + 1
    return parse_number(tok), pos + 1


def parse_number(tok):
    if "." in tok or "e" in tok.lower() or "inf" in tok or "nan" in tok:
        return float(tok.replace("+inf.0", "inf").replace("-inf.0", "-inf").replace("+nan.0", "nan"))
    return Fraction(tok)


def parse_engine_output(text):
    results = {}
    for line in text.splitlines():
        if not line.startswith("ORACLE "):
            continue
        _, cid, exactness, datum = line.split(" ", 3)
        value, _ = parse_datum(tokenize(datum))
        results[cid] = (exactness == "exact", value)
    return results, "ORACLE-END" in text


def ref_value(r):
    if isinstance(r, list):
        return [ref_value(x) for x in r]
    if r["exact"] is not None:
        return Fraction(r["exact"])
    return float(r["approx"])


def compare(got, ref, expect_exact, path="value"):
    if isinstance(ref, list):
        if not isinstance(got, list) or len(got) != len(ref):
            return f"{path}: shape mismatch (got {got!r}, SymPy {ref!r})"
        for i, (g, r) in enumerate(zip(got, ref)):
            err = compare(g, r, expect_exact, f"{path}[{i}]")
            if err:
                return err
        return None
    if isinstance(got, list):
        return f"{path}: got a collection where SymPy has a scalar"
    if isinstance(got, Fraction):
        if isinstance(ref, Fraction):
            return None if got == ref else f"{path}: exact {got} != SymPy {ref}"
        return f"{path}: exact {got} where SymPy is irrational {ref}"
    if expect_exact:
        return f"{path}: inexact {got} where the contract promises an exact result"
    target = float(ref)
    if abs(got - target) <= max(ABS_TOL, REL_TOL * abs(target)):
        return None
    return f"{path}: {got!r} differs from SymPy {target!r}"


class Trace:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def emit(self, name, value, snippet):
        event = {"kind": KIND, "name": name, "value": value, "snippet": snippet,
                 "confidence": 1.0, "recorded_epoch": int(time.time())}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def find_sympy_python():
    candidates = []
    if os.environ.get("ESHKOL_SYMPY_PYTHON"):
        candidates.append(os.environ["ESHKOL_SYMPY_PYTHON"])
    candidates.append(sys.executable)
    for cand in candidates:
        try:
            probe = subprocess.run([cand, "-I", "-c", "import sympy; print(sympy.__version__)"],
                                   capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return cand, probe.stdout.strip()
    return None, None


def run_engine(engine, eshkol_run, lib_dir, program, work, env):
    if engine == "jit":
        cmd = [eshkol_run, "-r", str(program), f"-L{lib_dir}"]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        return proc.returncode, proc.stdout + proc.stderr
    binary = work / "sympy_exact_ad_oracle_aot"
    comp = subprocess.run([eshkol_run, "-O", "1", "-o", str(binary), str(program), f"-L{lib_dir}"],
                          capture_output=True, text=True, env=env, timeout=1800)
    if comp.returncode != 0:
        return comp.returncode, comp.stdout + comp.stderr
    proc = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eshkol-run", required=True)
    ap.add_argument("--lib-dir", required=True, help="build directory holding the stdlib objects")
    ap.add_argument("--corpus", default=str(repo / "tests/sympy_oracle/exact_ad_corpus.json"))
    ap.add_argument("--trace-dir", default=os.environ.get("TRACE_DIR", str(repo / "scripts/icc_traces")))
    ap.add_argument("--work-dir", default=None, help="where the generated program and AOT binary go")
    ap.add_argument("--engines", default="jit,aot")
    args = ap.parse_args()

    trace = Trace(Path(args.trace_dir) / "sympy_exact_ad_oracle.jsonl")
    corpus = json.loads(Path(args.corpus).read_text())
    engines = [e for e in args.engines.split(",") if e]

    sympy_python, sympy_version = find_sympy_python()
    if not sympy_python:
        reason = ("SymPy is not importable by ESHKOL_SYMPY_PYTHON or this interpreter; "
                  "set ESHKOL_SYMPY_PYTHON to a Python with SymPy to run the oracle")
        trace.emit(SUMMARY_NAME, "SKIP", reason)
        print(f"SKIP: {reason}")
        return 77

    work = Path(args.work_dir) if args.work_dir else Path(args.trace_dir) / "sympy-oracle-work"
    work.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus).resolve()
    ref_proc = subprocess.run([sympy_python, "-I", "-c", SYMPY_REFERENCE, str(corpus_path)],
                              capture_output=True, text=True, timeout=1800)
    if ref_proc.returncode != 0:
        msg = "SymPy reference computation did not complete: " + ref_proc.stderr.strip()[-400:]
        trace.emit(SUMMARY_NAME, "INFRA", msg)
        print("INFRA: " + msg)
        return 2
    reference = json.loads(ref_proc.stdout)

    program = work / "sympy_exact_ad_oracle.esk"
    program.write_text(build_program(corpus))
    env = dict(os.environ)
    env.setdefault("ESHKOL_PATH", str(repo / "lib"))
    env.setdefault("ESHKOL_LIB_DIR", args.lib_dir)
    env["ESHKOL_JIT_CACHE"] = "0"

    total = passed = 0
    failures = []
    for engine in engines:
        rc, output = run_engine(engine, args.eshkol_run, args.lib_dir, program, work, env)
        results, complete = parse_engine_output(output)
        if rc != 0 or not complete:
            failures.append(f"{engine}: program exit {rc}, complete={complete}")
        for case in corpus["cases"]:
            cid = case["id"]
            total += 1
            name = f"sympy_exact_ad_{cid}_{engine}"
            if cid not in results:
                err = "no result line"
            else:
                exact, got = results[cid]
                ref = ref_value(reference["cases"][cid])
                err = compare(got, ref, case["expect_exact"])
                if not err and case["expect_exact"] and not exact:
                    err = "result is not exact"
            if err:
                failures.append(f"{engine}/{cid}: {err}")
                trace.emit(name, "FAIL", f"{case['op']} {cid} [{engine}]: {err}")
                print(f"FAIL: {engine} {cid}: {err}")
            else:
                passed += 1
                trace.emit(name, "PASS", f"{case['op']} {cid} [{engine}] agrees with SymPy {sympy_version}")
                print(f"PASS: {engine} {cid}")

    ok = not failures and passed == total and total > 0
    summary = (f"{passed}/{total} case-engine pairs agree with SymPy {sympy_version} "
               f"({len(corpus['cases'])} cases x {','.join(engines)})")
    if failures:
        summary += "; first: " + failures[0]
    trace.emit(SUMMARY_NAME, "PASS" if ok else "FAIL", summary)
    print(("PASS: " if ok else "FAIL: ") + SUMMARY_NAME + " " + summary)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
