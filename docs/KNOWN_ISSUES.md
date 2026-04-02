# Known Issues — Eshkol v1.1.11-accelerate

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
✅ Windows — Tier 1 native build via MSYS2/MinGW64

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

**Windows has no GPU acceleration**
Windows build (MSYS2/MinGW64) does not include Metal (macOS-only) or CUDA (requires NVIDIA toolkit). GPU acceleration on Windows is planned when Vulkan Compute support is added.

---

## Type System Scope

**Rank-1 polymorphism only**
`forall` quantification works at the outermost level. Higher-rank types (rank-2+) are not supported. This limits certain advanced functional programming patterns (e.g., ST monad encoding). Planned for a future release.

**Dependent types: tensor dimensions only**
The HoTT type system supports dependent types for tensor shape verification at compile time. Full dependent types (arbitrary value-level computation in types) are not implemented.

---

## Roadmap (Future Releases)

These are planned features, not deficiencies in the current release:

| Feature | Target | Current Alternative |
|---------|--------|-------------------|
| `define-library` / R7RS import with renaming | v1.3 | `require`/`provide` module system |
| Visual debugger | v1.2 | GDB/LLDB + `--dump-ir` |
| Full FFI with callbacks | v1.2 | `extern` C function calls work |
| Python bindings | v1.2 | File I/O or subprocess interop |
| Distributed computing | v1.2 | Single-machine thread pool |
| Multi-GPU dispatch | v1.2 | Single GPU (Metal or CUDA) |
| Vulkan Compute | v1.2 | Metal (macOS) + CUDA (Linux) |
| Model serialization / ONNX | v1.2 | Manual file I/O |
| Profile-Guided Optimization | v1.3 | LLVM -O3 + SIMD micro-kernels |
| Mobile/embedded targets | v1.4 | Desktop/server only |

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
