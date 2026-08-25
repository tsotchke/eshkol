#!/usr/bin/env python3
"""
Audit H1 regression test: NumPy capsule lifetime for the Python bindings.

bindings/python/eshkol_module.cpp's tensor branch of ffi_value_to_python
hands back a zero-copy NumPy array whose data pointer aliases memory owned
by the Eshkol runtime (the FFI context's arena). Before the fix, the
array's `base` capsule had a no-op deleter and held no reference back to
the context — so deleting the Python `Context` object while the array was
still alive called eshkol_ffi_shutdown() immediately, with nothing to stop
it, and any later read of the array was a read of memory the context no
longer guaranteed was valid.

The fix (see EshkolCtxHandle in eshkol_module.cpp) makes every exported
array's capsule hold its own strong reference (std::shared_ptr) to the
context. eshkol_ffi_shutdown() — the shared_ptr's deleter — only actually
runs once every holder, including every live array's capsule, has let go.
Deleting the Context object no longer invalidates arrays it already
produced.

This test proves the tie is real without needing the runtime to already
have a hook that frees the arena on shutdown (it doesn't, today — the
arena is process-lifetime, which is exactly why this was a *latent*
hazard rather than a crash-on-every-run). It registers its OWN hook
through the same public, documented mechanism a real embedder would use
to free its own arena-backed state on shutdown — eshkol_register_shutdown_hook(),
declared in inc/eshkol/core/runtime.h and demonstrated in-tree by
tests/core/runtime_shutdown_teardown_race_test.cpp — and has that hook
overwrite the exact bytes backing a live NumPy array. That models
precisely the scenario the audit H1 comment describes ("if a caller
destroys the FFI context ... while numpy still references the array,
reads see freed memory"): whatever a shutdown hook does to arena-backed
state, it must not touch memory a live NumPy array still depends on.

  BEFORE the fix: deleting the Context runs eshkol_ffi_shutdown()
  synchronously, the hook fires immediately, and the array's data is
  silently corrupted — a textbook SILENT-WRONG defect (wrong values, no
  diagnostic, exit 0; see .icc/silent-wrong-ledger.yaml SW-44).

  AFTER the fix: the array's capsule keeps the context alive, shutdown
  (and therefore the hook) is deferred until the array itself is
  released, so the read always sees the original, correct values.

Wired into ctest as python_bindings_capsule_lifetime when
ESHKOL_PYTHON_BINDINGS is ON and pybind11 + numpy are available; skips
(exit 0) otherwise, matching the existing eshkol Python binding test
convention (tests/stdlib/v12_python_test.py).
"""
import ctypes
import gc
import os
import sys

# CMake sets ESHKOL_PYTHON_MODULE_DIR to the eshkol_py build output
# directory so this runs correctly regardless of the build directory name
# (build, build-asan, build-xla, ...). Fall back to the historical
# "<repo>/build" convention used by the other loose binding test scripts
# for a plain manual run.
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


def main():
    try:
        import eshkol
    except ImportError as e:
        print(f"SKIP: eshkol Python module not built ({e})")
        return 0

    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy not installed")
        return 0

    # eshkol_py force-loads / whole-archives libeshkol-static.a and links
    # eshkol-repl-lib (see CMakeLists.txt's ESHKOL_PYTHON_BINDINGS block),
    # so the extension's own dynamic symbol table carries the full public
    # C FFI, including eshkol_register_shutdown_hook. Re-dlopen()ing the
    # exact file Python already imported just bumps its refcount and
    # hands back the same loaded image — this is not a second copy of the
    # runtime, just a handle we can call C functions through.
    libeshkol = ctypes.CDLL(eshkol.__file__)

    HOOK_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libeshkol.eshkol_register_shutdown_hook.restype = ctypes.c_uint32
    libeshkol.eshkol_register_shutdown_hook.argtypes = [
        HOOK_FN, ctypes.c_void_p, ctypes.c_char_p,
    ]

    fired = {"count": 0}

    def make_poison_hook(addr, nbytes, pattern):
        def hook(_ctx, _reason):
            ctypes.memset(addr, pattern, nbytes)
            fired["count"] += 1
            return 0
        # Keep the ctypes callback trampoline alive for the whole test —
        # letting it get garbage-collected while still registered with the
        # runtime would itself be a (different, ctypes-level) UAF.
        return HOOK_FN(hook)

    # ---- Core scenario: delete the Context while a NumPy array it
    #      produced is still alive. The array must keep reading the
    #      correct, original values — never freed/poisoned memory, and
    #      never a crash. ----
    ctx = eshkol.Context()
    a = ctx.eval("#(1.0 2.0 3.0 4.0)")
    check("tensor eval produced a numpy array", isinstance(a, np.ndarray))
    expected = list(a)  # snapshot before anything can touch the buffer

    addr = a.ctypes.data
    nbytes = a.nbytes
    hook_ref = make_poison_hook(addr, nbytes, 0xEF)
    hook_id = libeshkol.eshkol_register_shutdown_hook(hook_ref, None, b"h1-poison-test")
    check("shutdown hook registered", hook_id != 0)

    # CPython refcounting deallocates `ctx` here synchronously (no cycles
    # are involved), so this is the deterministic point where, pre-fix,
    # eshkol_ffi_shutdown() ran and the hook fired. gc.collect() is
    # belt-and-suspenders for any non-refcounting Python implementation.
    del ctx
    gc.collect()

    # This is the discriminating check: on the unfixed binding, the hook
    # has already fired by this point and `a`'s bytes are 0xEF, not the
    # original values.
    check("shutdown hook did NOT fire while the array was still live",
          fired["count"] == 0)
    check("array values are intact after del(ctx)",
          list(a) == expected)

    # Releasing the array's own reference must finally let shutdown (and
    # therefore the hook) run — the fix defers teardown, it does not leak
    # it forever.
    del a
    gc.collect()
    check("shutdown hook fired once the last live reference was released",
          fired["count"] == 1)

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
