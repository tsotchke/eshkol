#!/usr/bin/env python3
"""
Task #110: Python FFI regression test.

Verifies:
1. Structured return values (ints, floats, bools, lists, strings) are
   converted to native Python types — not "<eshkol object>" strings.
2. Gradient/derivative work without SIGSEGV (previously crashed).
3. Runtime errors in the Eshkol code raise Python exceptions instead of
   killing the whole host process via exit(1).
4. The evaluation context stays usable after an error.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
import eshkol

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS: {name}")
    else:
        FAIL += 1
        print(f"FAIL: {name}")


def main():
    ctx = eshkol.Context()

    # ---- Structured returns ----
    check("int return type", ctx.eval("(+ 1 2)") == 3)
    check("float return type", abs(ctx.eval("(sqrt 16)") - 4.0) < 1e-9)
    check("bool true", ctx.eval("#t") is True)
    check("bool false", ctx.eval("#f") is False)
    check("null", ctx.eval("'()") is None)
    check("string", ctx.eval('"hello"') == "hello")
    check("flat list", ctx.eval("(list 1 2 3)") == [1, 2, 3])
    check("nested list", ctx.eval("(list 1 (list 2 3) 4)") == [1, [2, 3], 4])

    # ---- Derivatives / gradients (previously SIGSEGV'd) ----
    check("derivative sin 0.5",
          abs(ctx.derivative("sin", 0.5) - 0.8775825618903728) < 1e-9)
    check("AD gradient on tape",
          abs(ctx.eval("""
(let* ((tape (ad-tape-new))
       (x (ad-var tape 1.0))
       (y (ad-sin tape x)))
  (ad-backward tape y)
  (ad-gradient tape x))
""") - 0.5403023058681398) < 1e-9)

    # ---- Error recovery: eshkol raise must not exit(1) the Python process ----
    try:
        ctx.eval('(+ "hello" 3)')
        check("type-error raises Python exception", False)
    except RuntimeError as e:
        check("type-error raises Python exception", "Eshkol exception" in str(e))

    # ---- Context remains usable after error ----
    check("context usable after error", ctx.eval("(+ 10 20)") == 30)

    # ---- Multiple errors in sequence ----
    for i in range(3):
        try:
            ctx.eval('(+ "x" 1)')
        except RuntimeError:
            pass
    check("context usable after 3 errors", ctx.eval("(* 7 6)") == 42)

    print()
    print("=== Summary ===")
    print(f"Passed: {PASS}")
    print(f"Failed: {FAIL}")
    if FAIL == 0:
        print("RESULT: ALL PASS")
        return 0
    else:
        print("RESULT: FAILURES DETECTED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
