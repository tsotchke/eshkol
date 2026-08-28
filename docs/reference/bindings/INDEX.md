# Language Bindings — Reference

Eshkol embedded inside another language's process, the opposite direction
from [Agent FFI](../agent/ffi.md) (which calls out to native C *from*
Eshkol). Each page covers one host language's binding: what it wraps, how
to build it, its API surface, and any lifetime/exactness caveats at the
language boundary.

## Pages

| Page | Contents |
|------|----------|
| [python.md](python.md) | `bindings/python/eshkol_module.cpp` — a pybind11 extension embedding the Eshkol JIT runtime in a Python process. `Context.eval`/`eval_double`/`eval_file`/`derivative`/`gradient`, build flags (`-DESHKOL_PYTHON_BINDINGS=ON`), NumPy zero-copy tensor interop, and the capsule-lifetime fix (audit H1, #458). |

## Cross-cutting notes

- **Direction matters**: a binding here embeds Eshkol inside a host
  language's process (Python calling in). [Agent FFI](../agent/ffi.md) is
  the other direction — Eshkol calling out to native C runtimes
  (`agent.*` modules). A feature request that sounds like "call X from
  Eshkol" belongs in the agent/FFI reference, not here.
- **Exactness at the boundary**: a host language's native numeric types
  (Python `float`, NumPy arrays) are not Eshkol's exact rational/bignum
  tower. Converting a value across a binding may silently demote it — see
  each page's own caveats before relying on a round-trip being exact.

## See also

- [Agent & FFI reference](../agent/INDEX.md) — the other embedding
  direction: `extern`/`:real` declarations and the `agent.*` modules that
  call native C from Eshkol.
- [Runtime reference](../runtime/INDEX.md) — CLI, environment variables,
  memory, parallelism, JIT.
