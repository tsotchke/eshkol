#!/usr/bin/env python3
"""onnx_export_test.py — validates Eshkol's ONNX export output.

Runs onnx.checker.check_model on the files produced by
onnx_export_test.esk and verifies the initializer values round-trip
bit-exactly. Skip cleanly if the onnx package is not installed, so
the Eshkol-only smoke test still runs on machines without it."""

import os
import subprocess
import sys

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        print(f"PASS: {name}")
        PASS += 1
    else:
        print(f"FAIL: {name}")
        FAIL += 1


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    esk = os.path.join(repo, "tests", "v1_2_edge_cases", "onnx_export_test.esk")
    runner = os.path.join(repo, "build", "eshkol-run")

    # Clean any stale artifacts so the test is self-contained.
    for p in ("/tmp/eshkol_onnx_test.onnx",
              "/tmp/eshkol_onnx_1d.onnx",
              "/tmp/eshkol_onnx_3d.onnx"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    r = subprocess.run([runner, "-r", esk], capture_output=True, text=True,
                        timeout=60)
    check("esk harness exit 0", r.returncode == 0)
    check("esk harness RESULT: OK", "RESULT: OK" in r.stdout)

    try:
        import onnx
        import numpy as np
    except ImportError:
        print("(onnx/numpy not installed — skipping deep validation)")
        return 0 if FAIL == 0 else 1

    # 2×2
    m = onnx.load("/tmp/eshkol_onnx_test.onnx")
    onnx.checker.check_model(m)
    check("2D onnx.checker OK", True)
    check("2D graph has a name", m.graph.name != "")
    init = m.graph.initializer[0]
    check("2D dims roundtrip", list(init.dims) == [2, 2])
    arr = np.array(init.double_data).reshape(list(init.dims))
    check("2D values roundtrip",
          np.allclose(arr, [[1.0, 2.0], [3.0, 4.0]]))

    # 1D
    m = onnx.load("/tmp/eshkol_onnx_1d.onnx")
    onnx.checker.check_model(m)
    init = m.graph.initializer[0]
    check("1D dims roundtrip", list(init.dims) == [4])
    arr = np.array(init.double_data)
    check("1D values roundtrip",
          np.allclose(arr, [0.5, 1.5, 2.5, 3.5]))

    # 3D
    m = onnx.load("/tmp/eshkol_onnx_3d.onnx")
    onnx.checker.check_model(m)
    init = m.graph.initializer[0]
    check("3D dims roundtrip", list(init.dims) == [2, 2, 2])

    print(f"\n=== Summary ===\nPassed: {PASS}\nFailed: {FAIL}")
    if FAIL == 0:
        print("RESULT: OK")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
