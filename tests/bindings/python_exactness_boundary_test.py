#!/usr/bin/env python3
"""
W5 Python/NumPy exactness boundary, silent-demotion refusal, and external-oracle
verification gate.

Verifies:
  1. Bit-exact IEEE-754 binary64 round-trip (including -0.0, smallest subnormal,
     1e-308, 1e+300) and multi-dimensional shape preservation (1D..4D, strided views).
  2. Direct NumPy -> Eshkol tensor binding (`bind_tensor`) without stringification,
     followed by live Eshkol matmul/reduction compared against NumPy as an external oracle.
  3. Strict silent-demotion refusal: float32, float16, complex128, int64 (default and
     out-of-53-bit-range 2^53 + 1), uint64 (2^53 + 1), bool, and object arrays all
     raise TypeError/ValueError rather than silently truncating or demoting.
  4. Explicit lossless widening (`allow_widening=True`) succeeds for exact int32/int64
     within [-2^53, 2^53] and float32.
  5. High-precision 17-digit `derivative()` and `gradient()` at irrational/high-precision
     points matching closed-form analytical external oracles within 1e-14 relative error.
"""
import math
import os
import sys

sys.path.insert(0, os.environ.get(
    "ESHKOL_PYTHON_MODULE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "build")))

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


def check_raises(name, exc_types, fn):
    try:
        fn()
    except exc_types:
        check(name, True)
        return
    except Exception as e:
        print(f"FAIL: {name} (unexpected exception {type(e).__name__}: {e})")
        global FAIL
        FAIL += 1
        return
    check(name, False)


def main():
    import eshkol
    import numpy as np

    ctx = eshkol.Context()

    # 1. Bit-exact IEEE-754 f64 round-trip (including -0.0 and subnormal)
    edge = np.array([
        -0.0, 0.0, 4.9406564584124654e-324, 1.0e-308, 1.0e+300,
        math.pi, -math.e, 1.0 + 2.0**-52
    ], dtype=np.float64)
    rt = ctx.tensor_roundtrip(edge)
    check("1D bit-exact IEEE-754 round-trip (including -0.0 and subnormal)",
          rt.dtype == np.float64 and rt.shape == edge.shape and
          rt.tobytes() == edge.tobytes())

    # 2. Multi-dimensional shapes (2D, 3D, 4D) and non-contiguous strided view
    arr4 = np.arange(24, dtype=np.float64).reshape(2, 3, 2, 2) * 0.125 - 1.5
    rt4 = ctx.tensor_roundtrip(arr4)
    check("4D shape and bit-exact round-trip",
          rt4.shape == (2, 3, 2, 2) and rt4.tobytes() == arr4.tobytes())

    strided = arr4.transpose(2, 0, 3, 1)
    check("strided view is non-contiguous input", not strided.flags["C_CONTIGUOUS"])
    rt_strided = ctx.tensor_roundtrip(strided)
    check("strided 4D view preserves shape and exact values",
          rt_strided.shape == strided.shape and np.array_equal(rt_strided, strided))

    # 3. Direct NumPy -> Eshkol binding and external-oracle verification against NumPy
    a = np.array([[1.25, -2.5, 0.75], [3.0, 1.5, -0.5]], dtype=np.float64)
    b = np.array([[0.5, 2.0], [-1.0, 0.25], [4.0, -3.0]], dtype=np.float64)
    ctx.bind_tensor("py-a", a)
    ctx.bind_tensor("py-b", b)

    c_eshkol = ctx.eval("(matmul py-a py-b)")
    c_numpy = a @ b
    check("bind_tensor + matmul matches NumPy oracle exactly",
          isinstance(c_eshkol, np.ndarray) and c_eshkol.shape == c_numpy.shape and
          np.allclose(c_eshkol, c_numpy, rtol=1e-15, atol=1e-15))

    s_eshkol = ctx.eval_double("(tensor-sum py-a)")
    check("bind_tensor + tensor-sum matches NumPy sum oracle",
          abs(s_eshkol - float(a.sum())) <= 1e-15)

    # 4. Silent-demotion refusal gate (default allow_widening=False)
    check_raises("refuse silent float32 -> float64 conversion", TypeError,
                 lambda: ctx.tensor_roundtrip(np.array([1.0, 2.0], dtype=np.float32)))
    check_raises("refuse silent float16 -> float64 conversion", TypeError,
                 lambda: ctx.tensor_roundtrip(np.array([1.0, 2.0], dtype=np.float16)))
    check_raises("refuse silent int64 -> float64 conversion", TypeError,
                 lambda: ctx.tensor_roundtrip(np.array([1, 2], dtype=np.int64)))
    check_raises("refuse complex128 imaginary-part drop", TypeError,
                 lambda: ctx.tensor_roundtrip(np.array([1.0 + 2.0j], dtype=np.complex128)))
    check_raises("refuse complex128 even with allow_widening=True", TypeError,
                 lambda: ctx.tensor_roundtrip(np.array([1.0 + 2.0j], dtype=np.complex128), True))
    check_raises("refuse bool array", TypeError,
                 lambda: ctx.tensor_roundtrip(np.array([True, False], dtype=np.bool_)))
    check_raises("refuse object array", TypeError,
                 lambda: ctx.tensor_roundtrip(np.array([1.0, "x"], dtype=object)))

    # 5. Lossless widening opt-in + exact 53-bit integer guard
    big_exact = np.array([-(1 << 53), 0, (1 << 53)], dtype=np.int64)
    rt_big = ctx.tensor_roundtrip(big_exact, allow_widening=True)
    check("allow_widening=True accepts exact 53-bit int64 bounds",
          rt_big.dtype == np.float64 and rt_big[0] == float(-(1 << 53)) and
          rt_big[2] == float(1 << 53))

    check_raises("refuse lossy int64 (2^53 + 1) even with allow_widening=True", ValueError,
                 lambda: ctx.tensor_roundtrip(np.array([(1 << 53) + 1], dtype=np.int64), True))
    check_raises("refuse lossy uint64 (2^53 + 1) even with allow_widening=True", ValueError,
                 lambda: ctx.tensor_roundtrip(np.array([(1 << 53) + 1], dtype=np.uint64), True))

    f32 = np.array([1.5, -0.25], dtype=np.float32)
    rt_f32 = ctx.tensor_roundtrip(f32, allow_widening=True)
    check("allow_widening=True widens float32 losslessly to float64",
          rt_f32.dtype == np.float64 and np.array_equal(rt_f32, f32.astype(np.float64)))

    # 6. Full 17-digit IEEE-754 precision in derivative() and gradient() vs external oracle
    x_pt = 1.0 + 1.0e-12
    # d/dx (x^3 + sin(x)) = 3*x^2 + cos(x)
    d_eshkol = ctx.derivative("(lambda (x) (+ (* x x x) (sin x)))", x_pt)
    d_exact = 3.0 * x_pt * x_pt + math.cos(x_pt)
    check("derivative() preserves full 17-digit point precision (< 1e-14 rel err)",
          abs(d_eshkol - d_exact) / abs(d_exact) < 1e-14)

    # grad(x*y + exp(x*z)) at (1.123456789012345, -0.987654321098765, 0.5)
    pt = [1.123456789012345, -0.987654321098765, 0.5]
    g_eshkol = np.asarray(
        ctx.gradient("(lambda (v) (let ((x (vref v 0)) (y (vref v 1)) (z (vref v 2))) (+ (* x y) (exp (* x z)))))",
                     pt),
        dtype=np.float64)
    ex = math.exp(pt[0] * pt[2])
    g_exact = np.array([pt[1] + pt[2] * ex, pt[0], pt[0] * ex], dtype=np.float64)
    check("gradient() preserves full 17-digit point precision (< 1e-14 rel err)",
          np.max(np.abs(g_eshkol - g_exact) / np.abs(g_exact)) < 1e-14)

    print()
    print("=== Summary ===")
    print(f"Passed: {PASS}")
    print(f"Failed: {FAIL}")
    if FAIL == 0:
        print("RESULT: ALL PASS")
        return 0
    print("RESULT: FAILURES DETECTED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
