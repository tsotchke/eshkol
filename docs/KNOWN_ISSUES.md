# Known Issues — Eshkol v1.3.0-evolve

**Status**: Production release

---

## Resolved in v1.1 (Previously Listed as Planned)

✅ `eval` — Dynamic code evaluation via REPL JIT
✅ `call/cc` + `dynamic-wind` — First-class continuations
✅ Exact arithmetic — Bignums and rational numbers (35 codegen gaps fixed)
✅ Bytevectors — R7RS bytevector operations
✅ Package manager — `eshkol-pkg` with registry
✅ LSP server — `eshkol-lsp` for IDE integration
✅ GPU acceleration — Metal (Apple Silicon) + CUDA (NVIDIA), forward and backward
✅ Complex numbers — First-class type with AD support
✅ Parallel primitives — `parallel-map/fold/filter/execute`, `future`/`force`
✅ Signal processing — FFT/IFFT, window functions, FIR/IIR, Butterworth
✅ Optimization algorithms — Gradient descent, Adam, L-BFGS, conjugate gradient
✅ Records — R7RS `define-record-type`
✅ Backward pass dispatch — GPU → BLAS/AMX → scalar (mirrors forward hierarchy)
✅ Windows — Tier 1 native build via Visual Studio 2022 + LLVM 21

---

## Design Choices (Not Limitations)

**Arena memory (OALR) instead of garbage collection**
Deterministic O(1) allocation with zero GC pauses. Arena regions are lexically scoped and freed automatically on scope exit. This is a deliberate architectural choice for real-time, financial, and embedded workloads where latency predictability matters. Eshkol will never have a garbage collector.

**Gradual typing (warnings, not errors)**
Type annotations are optional and informational. This preserves Scheme's exploratory programming model. Programs compile and run regardless of type warnings. This is the intended behavior — Eshkol is a dynamically-typed language with optional static analysis, not a statically-typed language with escape hatches.

**Hybrid arena model (global + per-thread)**
Global arena for main thread, per-thread arenas (1 MB, lazily allocated) for parallel workers. Zero contention for parallel workloads. This is an implementation strength, not a trade-off.

---

## Hardware Constraints

**Metal SF64 software float64 emulation**
Apple Silicon lacks hardware float64 compute shaders. Eshkol uses SF64 software emulation (~200 GFLOPS) for GPU double-precision. The cost model automatically prefers CPU cBLAS/AMX (~1.2 TFLOPS) when faster — GPU is only selected for matrices exceeding cBLAS capacity (~31K×31K and larger).

**Conv2d backward is CPU-only**
The conv2d backward pass uses stride-based scatter/gather indexing that doesn't map to GEMM. This is inherent to the convolution transpose operation. LayerNorm/BatchNorm backward are reductions, which are inherently sequential.

**Windows has no GPU acceleration in the lite/native path**
Native Windows builds do not include Metal (macOS-only) or CUDA acceleration yet. GPU acceleration on Windows will require a dedicated Windows GPU backend path.

---

## Type System Scope

**Rank-1 polymorphism only**
`forall` quantification works at the outermost level. Higher-rank types (rank-2+) are not supported. This limits certain advanced functional programming patterns (e.g., ST monad encoding). Planned for a future release.

**Dependent types: tensor dimensions only**
The HoTT type system supports dependent types for tensor shape verification at compile time. Full dependent types (arbitrary value-level computation in types) are not implemented.

---

## Current Limitations (VM)

### Top-level mutual recursion grouping
Top-level mutual recursion requires consecutive function defines. Interleaved non-define expressions break groups, causing forward references to fail. Workaround: place all mutually recursive defines together without intervening expressions.

### Tensor nested syntax not supported in VM parser
`#((1 2) (3 4))` nested tensor syntax is not supported in the bytecode VM parser. Use flat tensors with `reshape` in the native compiler instead.

### Gradient limited to single-variable functions in VM
`gradient` in the VM only handles single-variable functions. Use `derivative` for scalar AD in the REPL.

---

## Tracked Open Issues (v1.3.0-evolve)

Edge-case findings surfaced by the adversarial-testing harnesses (see
[TESTING.md](TESTING.md)). Each has a minimal repro and a ledger entry under
`.swarm/tasks/ESH-*.json`. None block ordinary use; all are also listed in the
[CHANGELOG](../CHANGELOG.md) Known Issues section.

**Automatic differentiation**
- Vector gradient-of-gradient silently returns zeros — use nested scalar
  `derivative` for exact higher-order results (ESH-0096).
- `hessian`/`laplacian` SIGSEGV when the evaluation point is a tensor literal
  `#(...)` / `(tensor ...)`; a `(vector ...)` point works (ESH-0095).
- Vector-param AD op combined with a captured local parameter fails LLVM
  verification (`PtrToInt source must be pointer`) (ESH-0072, ESH-0097).

**Recursion depth**
- Deep non-tail recursion (~270k frames) dies with SIGILL and no diagnostic;
  stdlib `sort`/`length`/`filter` are non-tail-recursive and fail on very large
  inputs; mutual tail calls are not TCO'd (ESH-0098, ESH-0101, ESH-0102, ESH-0108).

**Language edges**
- A closure created inside a named-let loop that `set!`s a global loses the
  mutation (ESH-0094).
- Exact rational arithmetic degrades to double once a bignum is involved
  (ESH-0105).
- Long-form `(quasiquote x)`/`(unquote x)` and nested quasiquote (level >= 2)
  are not fully wired (ESH-0104, ESH-0107).
- JIT compile of a ~10k-deep nested expression uses excessive RSS/time; AOT is
  unaffected (ESH-0103).

**VM parity**
- 27 bytecode-VM behavioral divergences and 351 parity gaps are documented and
  tracked in `tests/vm_parity/PARITY.tsv` (see [VM_PARITY.md](VM_PARITY.md)).

---

## Roadmap (Future Releases)

These are planned features, not deficiencies in the current release:

| Feature | Target | Current Alternative |
|---------|--------|-------------------|
| Full R7RS library export filtering semantics | v1.3 | `define-library` and R7RS `import` forms, including `only`/`except`/`rename`/`prefix`, lower through the existing `require`/`provide` module system |
| Visual debugger UI | v1.3 | GDB/LLDB on the DWARF data already emitted by `-g`; `--dump-ir` for IR-level inspection |
| Full C callbacks from foreign threads | v1.3 | `extern` C function calls (in-thread) work; native HTTP, SQLite, subprocess, fs-watch FFI surfaces shipped in v1.2 |
| Extended Python bindings | v1.4 | Stable C FFI with pybind11 + NumPy zero-copy interop shipped in v1.2 |
| Distributed computing | v1.3 | Single-machine thread pool with `parallel-map`/`parallel-fold`/`future` |
| Multi-GPU dispatch | v1.3 | Single GPU (Metal or CUDA) chosen automatically by the cost model |
| Vulkan compute shaders | v1.3 | Metal (macOS) + CUDA (Linux/Windows-with-NVIDIA) |
| ONNX export | v1.3 | Eshkol's native `.eshkol-model` serialiser shipped in v1.2; manual file I/O for other formats |
| Profile-guided optimisation (PGO) | v1.3 | LLVM `-O3` + SIMD micro-kernels |
| Mobile / embedded targets | v1.4 + freestanding-platform programme | Desktop / server only |

---

## Reporting Issues

1. Check [Feature Matrix](FEATURE_MATRIX.md) for implementation status
2. Review this document for known constraints
3. File issue on GitHub: https://github.com/tsotchke/eshkol/issues
4. Provide: Eshkol version, platform, minimal reproduction

---

## See Also

- [Feature Matrix](FEATURE_MATRIX.md) — Implementation status
- [Roadmap](../ROADMAP.md) — Future development plans
- [API Reference](API_REFERENCE.md) — Complete function documentation
