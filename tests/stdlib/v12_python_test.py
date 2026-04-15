#!/usr/bin/env python3
"""v1.2 Python bindings test suite."""

import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "build")
sys.path.insert(0, build_dir)

try:
    import eshkol
except ImportError:
    print("SKIP: eshkol Python module not built (cmake -DESHKOL_PYTHON_BINDINGS=ON)")
    sys.exit(0)

passed = 0
failed = 0

def test(name, expr, expected, tolerance=None):
    global passed, failed
    try:
        result = expr()
        if tolerance is not None:
            ok = abs(result - expected) < tolerance
        else:
            ok = result == expected
        if ok:
            passed += 1
        else:
            print(f"FAIL: {name}: got {result!r}, expected {expected!r}")
            failed += 1
    except Exception as e:
        print(f"FAIL: {name}: {e}")
        failed += 1

ctx = eshkol.Context()

# Arithmetic
test("1+2", lambda: ctx.eval("(+ 1 2)"), 3)
test("3*4", lambda: ctx.eval("(* 3 4)"), 12)
test("100-58", lambda: ctx.eval("(- 100 58)"), 42)

# Floating point
test("sin(0)", lambda: ctx.eval("(sin 0)"), 0.0, tolerance=1e-10)
test("exp(0)", lambda: ctx.eval("(exp 0)"), 1.0, tolerance=1e-10)

# Boolean
test("#t", lambda: ctx.eval("#t"), True)
test("#f", lambda: ctx.eval("#f"), False)

# String
test("hello", lambda: ctx.eval('"hello"'), "hello")

# Derivative
test("d/dx sin(0.5)", lambda: ctx.derivative("sin", 0.5), 0.8775825618903728, tolerance=1e-6)
test("d/dx x^2 at 3", lambda: ctx.derivative("(lambda (x) (* x x))", 3.0), 6.0, tolerance=1e-6)

# Summary
print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
if failed > 0:
    sys.exit(1)
print("ALL PASS")
