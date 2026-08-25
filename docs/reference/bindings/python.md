# Python Bindings

`bindings/python/eshkol_module.cpp` is a [pybind11](https://pybind11.readthedocs.io/)
extension module that embeds the Eshkol JIT runtime inside a Python process —
the opposite direction from [Agent FFI](../agent/ffi.md), which calls C
*from* Eshkol. Shipped since v1.2-scale; the capsule-lifetime fix described
below (audit H1) landed in the v1.3.5 line, PR #458.

## Building

```bash
cmake -S . -B build -DESHKOL_PYTHON_BINDINGS=ON
cmake --build build --target eshkol_py -j
```

Requires `pybind11` (`find_package(pybind11 QUIET)`) and NumPy on the
`PYTHONPATH` used at import time; the CMake block silently no-ops the
target (with a configure-time message, not a hard error) if `pybind11` is
not found. The build output is `eshkol.cpython-<tag>-<platform>.so` (the
target is named `eshkol_py`, but `OUTPUT_NAME` is set to `eshkol` so
`import eshkol` works directly against the build directory).

## API

`eshkol.Context()` constructs one JIT runtime instance. State (definitions,
`define`d variables) persists across `eval()` calls on the same `Context`;
create a new one for an isolated session.

| Method | Signature | Notes |
|---|---|---|
| `eval` | `eval(source: str) -> object` | Evaluates Eshkol source, returns a Python value: `int`, `float`, `bool`, `str`, `None`, `list` (Scheme pairs/lists), or a NumPy array (tensors — see below). |
| `eval_double` | `eval_double(source: str) -> float` | Convenience path that skips the general value conversion. |
| `eval_file` | `eval_file(path: str) -> None` | Loads and evaluates a file. Rejects an empty path, a path over 4 KiB, or one containing a NUL byte (#195 hardening) without touching the filesystem first. |
| `derivative` | `derivative(func: str, x: float) -> float` | `func` **must** be a complete `(lambda (x) ...)` (or `(λ (x) ...)`) source string — see the validation note below. |
| `gradient` | `gradient(func: str, point: list[float]) -> numpy.ndarray` | Same `(lambda ...)` requirement; `point` is the argument vector. |

Verified this cycle against a from-source build of commit `1b5729d0` (`#458`
merged onto `694c3179`), `PYTHONPATH=build2 python3`:

```python
import eshkol
ctx = eshkol.Context()
ctx.eval("(+ 1 2)")                                    # => 3
ctx.eval("#(1.0 2.0 3.0)")                              # => numpy.ndarray([1., 2., 3.])
ctx.derivative("(lambda (x) (sin x))", 0.5)             # => 0.8775825618903728  (== cos(0.5))
ctx.gradient("(lambda (x y) (* x y))", [2.0, 3.0])      # => numpy.ndarray([3., 2.])
```

**`func_source` validation note (finding, not part of #458):** the module's
own docstring example, `ctx.derivative('sin', 0.5)`, does not work against
this build — `derivative`/`gradient` validate that `func_source` is
syntactically a complete `(lambda ...)` form (balanced parens, no string
literals) as an injection guard added for #191, and a bare identifier like
`'sin'` fails that check with `ValueError: derivative: func_source must be
a lambda expression`. Use the `(lambda (x) ...)` form shown above. This is
independent of the H1 fix below and pre-dates it; flagged here rather than
silently repeated as a working example.

## Capsule lifetime guarantee (audit H1, PR #458)

Every tensor `eval()` returns is a **zero-copy** NumPy array backed
directly by memory in the Eshkol runtime's arena — there is no copy at the
FFI boundary. Before PR #458, that array's NumPy `base` capsule held no
reference back to the owning context: deleting the Python `Context` object
(or letting its last reference drop) called `eshkol_ffi_shutdown()`
immediately, and any live array — or a slice/view/reshape of one — was
left aliasing memory the runtime no longer guaranteed was valid. Exit 0, no
exception, no crash: a silent wrong-value defect, tracked as
`.icc/silent-wrong-ledger.yaml` SW-44 (bucket `SILENT-WRONG`).

The fix: `EshkolContext` now holds its context as
`std::shared_ptr<eshkol_ffi_context_t>` with `eshkol_ffi_shutdown` as the
deleter, and every tensor NumPy array's `base` capsule carries a second,
independent copy of that same `shared_ptr`. NumPy propagates `base`
through slicing, reshaping, and views, so as long as *any* array derived
from a context is alive, the shared_ptr's use-count stays above zero and
the real shutdown is deferred — regardless of whether the Python `Context`
wrapper itself has already been deleted. The guarantee in one sentence:
**an array `eval()` hands back stays valid for as long as you hold a
reference to it, independent of the `Context` object's own lifetime.**
`.copy()` on the Python side is still available for a caller that wants a
fully independent array (e.g. to stop pinning the runtime's resources),
but it is no longer required for correctness.

Re-verified for this documentation wave —
`tests/bindings/python_capsule_lifetime_test.py` (also wired into `ctest`
as `python_bindings_capsule_lifetime`), run directly against
`ESHKOL_PYTHON_MODULE_DIR=build2`:

```
PASS: tensor eval produced a numpy array
PASS: shutdown hook registered
PASS: shutdown hook did NOT fire while the array was still live
PASS: array values are intact after del(ctx)
PASS: shutdown hook fired once the last live reference was released
RESULT: ALL PASS (5/5)
```

**Scope note, carried over from the fix commit:** this closes the
*lifetime* hazard only. A separate, still-open issue — `double`-only
demotion and exactness across the Python/NumPy boundary (an Eshkol exact
rational or bignum crossing into a plain Python `float`) — is explicitly
out of scope here and is tracked as a v1.4.0-connection interop-wave item
(see [KNOWN_ISSUES.md](../../KNOWN_ISSUES.md) and ROADMAP.md's W5
workstream).

## See also

- [Agent FFI reference index](../agent/ffi.md) — the other direction:
  calling C from Eshkol via `extern`/`:real`.
- `tests/v1_2_edge_cases/python_ffi_test.py`,
  `tests/stdlib/v12_python_test.py` — the pre-existing loose Python-binding
  regression scripts (manual-run convention, predate the ctest wiring
  above).
- `inc/eshkol/eshkol_ffi.h` — the underlying C API the pybind11 module
  wraps (`eshkol_ffi_init`, `eshkol_ffi_eval`, tensor accessors,
  `eshkol_register_shutdown_hook`).
