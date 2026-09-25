#!/usr/bin/env python3
"""Route guard for numeric builtins on derivative carriers (SW-210, SW-211).

A numeric builtin that meets a Taylor carrier without a recurrence answers
with its primal: the derivative comes back 0 and nothing fails. SW-210 was
eleven builtins in that state. This gate fails when the class can recur:

  1. Every builtin the native code generator lowers through its math
     dispatch (`codegenMathFunction(op, "name")`) has a unary row or a
     TAYLOR_ALIAS in lib/core/taylor_recurrences.def, and every builtin it
     lowers through `codegenBinaryMathFunction` has a binary row.
  2. Every row has a `case ESH_TAYLOR_UOP_<name>` / `ESH_TAYLOR_OP_<name>`
     rule in lib/core/runtime_taylor.c, or belongs to the integral family
     predicate `tr_is_ext_uop`.
  3. Every procedure with a plain complex kernel (complexRuntimeSuffix in
     lib/backend/complex_codegen.cpp) has a carrier formula in
     complexCarrierMath (lib/backend/llvm_codegen.cpp), so a complex value
     carrying a derivative never reaches a kernel that reads its primal.

`selftest` proves each check goes red on the shape it exists to catch.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

UN_RE = re.compile(r'^TAYLOR_UN\(\s*(\w+)\s*,\s*\d+\s*,\s*"([^"]+)"', re.MULTILINE)
BIN_RE = re.compile(r'^TAYLOR_BIN\(\s*(\w+)\s*,\s*\d+\s*,\s*"([^"]+)"', re.MULTILINE)
ALIAS_RE = re.compile(r'^TAYLOR_ALIAS\(\s*"([^"]+)"\s*,\s*(\w+)\s*\)', re.MULTILINE)
MATH_RE = re.compile(r'codegenMathFunction\(op,\s*"([^"]+)"\)')
BINMATH_RE = re.compile(r'codegenBinaryMathFunction\(op,\s*"([^"]+)"\)')
NAME_EQ_RE = re.compile(r'func_name\s*==\s*"([^"]+)"')


def function_body(text: str, signature: str) -> str:
    """The brace-balanced body of the first function whose header contains `signature`."""
    start = text.find(signature)
    if start < 0:
        return ""
    brace = text.find("{", start)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace:i + 1]
    return ""


def check(defs: str, runtime: str, codegen: str, complex_cg: str) -> list[str]:
    errors: list[str] = []
    unary = dict((sexpr, name) for name, sexpr in UN_RE.findall(defs))
    binary = dict((sexpr, name) for name, sexpr in BIN_RE.findall(defs))
    aliases = dict(ALIAS_RE.findall(defs))
    if not unary or not binary:
        return ["taylor_recurrences.def: no TAYLOR_UN/TAYLOR_BIN rows parsed"]

    def unary_row(spelling: str) -> str | None:
        if spelling in aliases:
            return aliases[spelling]
        return unary.get(spelling)

    math_names = sorted(set(MATH_RE.findall(codegen)))
    if not math_names:
        errors.append("llvm_codegen.cpp: no codegenMathFunction(op, \"...\") call sites parsed")
    for name in math_names:
        if unary_row(name) is None:
            errors.append(f"unary builtin '{name}' is lowered through codegenMathFunction "
                          f"but has no TAYLOR_UN row or TAYLOR_ALIAS: a Taylor carrier "
                          f"would get its primal back")
    for name in sorted(set(BINMATH_RE.findall(codegen))):
        if name not in binary:
            errors.append(f"binary builtin '{name}' is lowered through "
                          f"codegenBinaryMathFunction but has no TAYLOR_BIN row")

    ext_body = function_body(runtime, "static int tr_is_ext_uop(")
    for sexpr, name in sorted(unary.items()):
        rule = f"ESH_TAYLOR_UOP_{name}"
        if f"case {rule}" not in runtime and rule not in ext_body:
            errors.append(f"unary row '{name}' has no rule in runtime_taylor.c")
    for sexpr, name in sorted(binary.items()):
        if f"case ESH_TAYLOR_OP_{name}" not in runtime and name not in ("add", "sub", "mul", "div"):
            errors.append(f"binary row '{name}' has no rule in runtime_taylor.c")

    suffix_body = function_body(complex_cg, "ComplexCodegen::complexRuntimeSuffix(")
    carrier_body = function_body(codegen, "Value* complexCarrierMath(")
    suffix_names = set(NAME_EQ_RE.findall(suffix_body))
    carrier_names = set(NAME_EQ_RE.findall(carrier_body))
    if not suffix_names:
        errors.append("complex_codegen.cpp: complexRuntimeSuffix names not parsed")
    for name in sorted(suffix_names - carrier_names):
        errors.append(f"complex procedure '{name}' has a plain complex kernel but no carrier "
                      f"formula in complexCarrierMath: a complex value carrying a derivative "
                      f"would lose it")
    return errors


def read(repo: Path) -> tuple[str, str, str, str]:
    return (
        (repo / "lib/core/taylor_recurrences.def").read_text(encoding="utf-8"),
        (repo / "lib/core/runtime_taylor.c").read_text(encoding="utf-8"),
        (repo / "lib/backend/llvm_codegen.cpp").read_text(encoding="utf-8"),
        (repo / "lib/backend/complex_codegen.cpp").read_text(encoding="utf-8"),
    )


def selftest(repo: Path) -> int:
    defs, runtime, codegen, complex_cg = read(repo)
    base = check(defs, runtime, codegen, complex_cg)
    if base:
        print("selftest: the real tree must be clean first:\n  " + "\n  ".join(base))
        return 1
    cases = {
        "unrouted unary builtin": (defs, runtime,
            codegen + '\ncodegenMathFunction(op, "erf");\n', complex_cg),
        "dropped alias": (defs.replace('TAYLOR_ALIAS("ceil",  ceiling)', ''), runtime,
            codegen, complex_cg),
        "row without a runtime rule": (
            defs.replace('TAYLOR_UN(sigmoid, 12,',
                         'TAYLOR_UN(erfc, 99, "erfc", erfc, 0.5)\nTAYLOR_UN(sigmoid, 12,'),
            runtime, codegen, complex_cg),
        "unrouted binary builtin": (defs, runtime,
            codegen + '\ncodegenBinaryMathFunction(op, "hypot");\n', complex_cg),
        "complex kernel without a carrier formula": (defs, runtime,
            codegen.replace('if (func_name == "asinh")', 'if (func_name == "asinh_removed")'),
            complex_cg),
    }
    bad = 0
    for label, args in cases.items():
        errs = check(*args)
        status = "red" if errs else "GREEN (wrong)"
        print(f"selftest {label}: {status}")
        if not errs:
            bad += 1
    if bad:
        print(f"FAIL: {bad} selftest case(s) did not go red")
        return 1
    print("PASS: taylor unary route guard selftest")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="check", choices=["check", "selftest"])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    repo = Path(args.repo)
    if args.mode == "selftest":
        return selftest(repo)
    errors = check(*read(repo))
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    defs = read(repo)[0]
    print(f"PASS: taylor unary route guard ({len(UN_RE.findall(defs))} unary rows, "
          f"{len(ALIAS_RE.findall(defs))} aliases, {len(BIN_RE.findall(defs))} binary rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
