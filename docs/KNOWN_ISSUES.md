# Known Issues — Eshkol v1.3.4-evolve

**Status**: Production release

---

## Resolved in v1.3.4-evolve

- **Resident-loop retention with persistent mutation.** A tail-recursive loop
  that mutates persistent state (a knowledge base, workspace, or growing list)
  on every iteration used to get no automatic per-iteration reclamation and
  leaked one iteration's transient garbage forever. It is now lowered with a
  per-loop nursery region (ESH-0214e), so such a loop is flat at 34 MB —
  identical to its explicit `with-region` twin. `with-region` is no longer
  required to get flat RSS in a resident loop. See
  [memory-model](reference/runtime/memory-model.md#automatic-per-iteration-reclamation-in-resident-loops-esh-0214e).
- **`parallel-map` corrupted collection-valued results past the parallel
  threshold.** A closure whose body used per-iteration scope reclamation (an
  internal named-let loop, or a builtin such as `memv`) could return
  dangling/overlapping structure. Scope reclamation now degrades to commit-only
  on pool workers sharing the thread-safe arena, so results are identical to
  serial `map`.
- **`gradient` misdispatched when the callable was reached indirectly.**
  `(gradient f point)` through a function parameter/wrapper, and the curried
  `((gradient f) point)`, are now byte-identical to the direct call — the
  operator recovers the callable's arity from its closure metadata. There is no
  finite-difference fallback anywhere in the gradient path.
- **Floating-point printing was fixed-precision.** `display`, `write`, and
  `number->string` now emit the shortest decimal that reads back as the identical
  `double` (R7RS 6.2.6), byte-identical on the native and VM backends.
- **Hosted-VM tensor matmul parity was incomplete.** `arange` (1/2/3-arg),
  nested-literal tensor operands, and multi-dimensional `tensor-ref` /
  `tensor-set!` now match native codegen on the bytecode VM.

## Resolved in v1.1 (Previously Listed as Planned)

- `eval` — Dynamic code evaluation via REPL JIT
- `call/cc` + `dynamic-wind` — First-class continuations
- Exact arithmetic — Bignums and rational numbers (35 codegen gaps fixed)
- Bytevectors — R7RS bytevector operations
- Package manager — `eshkol-pkg` with registry
- LSP server — `eshkol-lsp` for IDE integration
- GPU acceleration — Metal (Apple Silicon) + CUDA (NVIDIA), forward and backward
- Complex numbers — First-class type with AD support
- Parallel primitives — `parallel-map/fold/filter/execute`, `future`/`force`
- Signal processing — FFT/IFFT, window functions, FIR/IIR, Butterworth
- Optimization algorithms — Gradient descent, Adam, L-BFGS, conjugate gradient
- Records — R7RS `define-record-type`
- Backward pass dispatch — GPU to BLAS/AMX to scalar (mirrors forward hierarchy)
- Windows — Tier 1 native build via Visual Studio 2022 + LLVM 21

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

### Reverse-mode gradient on the VM — implementation in progress
The bytecode VM's AD surface is currently scalar `derivative` only; `gradient`
in the VM handles single-variable functions today. A VM reverse-mode `gradient`
implementation is in flight, with full native/VM parity targeted for this
release line. Use `derivative` for scalar AD in the REPL in the meantime.

---

## Tracked Open Issues

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
  inputs (ESH-0098, ESH-0101, ESH-0108). Mutual tail calls ARE now proper R7RS
  tail calls (emitted as LLVM `musttail`) and run in O(1) stack — ESH-0102
  resolved (2026-07-04). The remaining exception is a higher-order tail call that
  forwards a stack-allocated closure argument, which falls back to a bounded call.
- Plain named-let TCO loops used to overflow the native stack around
  n≈300k-500k even with zero `guard`/`call/cc`/dynamic-alloca in the loop body
  (e.g. `(let loop ((n 0)) (if (>= n N) n (loop (+ n 1))))`). **Status: fixed**
  (ESH-0223) — no longer reproduces on current master. Re-tested after the
  define-loop TCO + per-iteration arena reclamation (ESH-0214b/#192), the
  iterative reader (#191), and deep region-escape (#210) landed: the bare
  named-let loop now runs flat with O(1) stack (~28 MB RSS, <1 s at N=1e7 under
  a 512 KB stack ulimit), matching the top-level-define TCO guarantee.
  Regression test `tests/tco/named_let_long_loop_test.esk` (and Test 1 of
  `tests/tco/named_let_tail_positions_test.esk`, which runs the identical shape
  to 1e7).

**Language edges**
- A closure created inside a named-let loop that `set!`s a global loses the
  mutation (ESH-0094).
- **A lambda that closes over a TCO'd self-recursive function's OWN
  loop-carried parameter and is passed to `derivative`/`gradient` (and, once
  merged, `taylor`) reads a stale/corrupted value or segfaults** once the
  function actually recurses — e.g. a captured double reads back as
  unrelated garbage, a captured list corrupts and crashes downstream `cdr`
  calls. Root cause: `derivative()`/`gradient()`'s free-variable-capture
  reconstruction didn't recognize a TCO loop alloca's value-vs-pointer
  capture convention (the same class of bug `map`'s `isTcoLoopAlloca` check
  already guarded against). Independently flagged by three adversarial
  agents during the P8/P9/P11 campaign passes and confirmed to be **one**
  root cause, **one** ticket. **Status: fixed** 2026-07-06 in
  `lib/backend/autodiff_codegen.cpp` (ESH-0221); regression test
  `tests/closures/tco_loop_capture_test.esk`.
- `(apply f (list ...))` used as a loop's back-edge used to grow the native
  call stack by one frame per iteration (and, for a named-let, could not even
  resolve the local loop name — it warned "apply: Unknown function" and
  returned `'()`). **Status: fixed** (ESH-0227). A statically spelled
  `(apply f leading... (list ...))` whose target `f` names the enclosing
  function's active TCO loop and whose total argument count matches the loop's
  arity now lowers to the same O(1)-stack loop back-edge a direct
  `(f arg ...)` self-call gets. The whole-function tail analysis recognizes
  apply-self-calls, so a *non-tail* apply-self-call still correctly disables the
  transform (the apply stays a normal call), and a dynamically-shaped final
  list is left as an ordinary (non-tail) apply. Fixed in
  `lib/backend/tail_call_codegen.cpp` and `lib/backend/llvm_codegen.cpp`;
  regression test `tests/tco/apply_loop_tail_test.esk`.
- `sleep-ms` used not to type-check its argument — the AOT/JIT builtin cast the
  tagged value's raw `.data` field straight to `int64_t` with no tag check, so a
  non-numeric argument reinterpreted whatever bits were there (a pointer, for
  heap-tagged values) as a millisecond count instead of raising a type error;
  the bytecode VM path silently no-op'd non-numbers. **Status: fixed**
  (ESH-0228). Both paths now accept only fixnums/flonums (valid values keep
  their behavior: sleep n ms, or a no-op for `<= 0`) and raise a clean,
  catchable "Type error in sleep-ms: expected number" on any non-number. Fixed
  in `lib/core/system_builtins.c` and `lib/backend/vm_native.c`; regression
  test `tests/system/sleep_ms_test.esk`.
- Exact rational arithmetic degrades to double once a bignum is involved
  (ESH-0105).
- Long-form `(quasiquote x)`/`(unquote x)` and nested quasiquote (level >= 2)
  are not fully wired (ESH-0104, ESH-0107).
- JIT compile of a ~10k-deep nested expression uses excessive RSS/time; AOT is
  unaffected (ESH-0103).

**VM parity**
- 27 bytecode-VM behavioral divergences and 351 parity gaps are documented and
  tracked in `tests/vm_parity/PARITY.tsv` (see [VM_PARITY.md](VM_PARITY.md)).
- A prior campaign pass reported "5 pre-existing surface-audit failures" for
  `scripts/run_vm_parity.sh`. Re-verified 2026-07-08 against current master
  (post-v1.3.0-evolve tag) with a full rebuild: `scripts/run_vm_parity.sh`
  passes clean end to end (stage 1 surface audit: 920 codegen symbols, all
  VM-supported or waived; stages 2-3 corpus differential + OOS probes: 56/56).
  No reproducible surface-audit failure currently exists on this branch — the
  earlier report is presumed to have referred to a transient state before a
  since-landed fix, or to a CI lane not exercised by this verification (e.g.
  ASAN/XLA). Not filing a ticket for a failure that doesn't currently
  reproduce; re-open if a specific lane is found to still fail.

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
