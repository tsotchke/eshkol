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
- **`gradient` was native-only on the bytecode VM.** Forward/reverse-mode
  `gradient` — direct, through a callable parameter, and curried — now runs on
  the VM byte-identically to native codegen (#337). `op:GRADIENT` and
  `op:DERIVATIVE` are `vm-supported`; higher-order nesting (gradient-of-derivative
  / Taylor tower) remains native-only.
- **`hessian`/`laplacian` crashed at a tensor-literal or variable-bound point.**
  The differentiation point was classified from the AST node kind, so a variable
  bound to a vector (or a `#(...)` / `(tensor ...)` literal, or a `(the ...)`
  wrapper) could take the wrong path and SIGSEGV; a cons-routed `gradient` could
  also return a silently wrong value. Points are now classified by their runtime
  value, so every point form gives the same result (#343). `laplacian` shares the
  fixed `hessian` path (it is the trace of the Hessian). Resolves ESH-0095.
- **Arity-1 whole-point tensor-loss gradients silently zeroed or crashed.** A
  loss whose body applies elementwise arithmetic to its whole vector/tensor
  argument now backpropagates exactly for a scalar-valued output and raises a
  clean `jacobian` diagnostic for a vector-valued one, instead of dropping the
  tangent or dereferencing a non-AD value (#338).

### Resolved in the v1.3.4-evolve correctness wave

The second half of the cycle was a consumer-hardening correctness wave whose
organising principle is that a wrong answer must not be able to look like a
right one. The change that made the rest findable is listed first.

- **A diagnosed program built and ran anyway.** The compiler printed
  `ERROR: …` and then emitted, linked and executed a binary regardless, so a
  diagnosed program produced a wrong answer instead of a failed build. An
  emitted error diagnostic now prevents artifact emission and execution. This
  converted a family of silent wrong answers into build failures — which is how
  most of the entries below were found.
- **`derivative`, `gradient` and `hessian` returned garbage at exact points.**
  At a rational or bignum point the operators produced garbage, `#()`, or a
  SIGSEGV, and lost exactness where the Taylor tower keeps it. They now route
  through the same Taylor-tower pass as `derivative-n`, with the contract being
  identity in **value and exactness**: `(derivative f x)` equals
  `(derivative-n f x 1)` and `(hessian f x)` equals `(derivative-n f x 2)`. The
  tier demotes on the first transcendental, matching R7RS exactness contagion.
  The inexact path is proven unchanged. (#393)
- **`gradient` was silently wrong when a loop filled a vector with
  `vector-set!`.** Loops containing `vector-set!` get a per-iteration nursery
  reset, and the write barrier that must promote escapees excluded dual numbers
  (a headerless 16-byte payload in the nursery). The nursery recycled the dual;
  the same address was reallocated on the next iteration; the primal survived
  and the **tangent was silently corrupted**. One shared "carries an arena
  pointer" predicate now governs the barrier, the `with-region` escape and the
  nursery recycle, and tape-retained AD nodes allocate from the tape's owning
  arena. (#396)
- **Gradients of runtime closures with declared arity 17–32 crashed, returned
  all zeros, or raised a type error.** The declared ceiling was 32 but a
  `MAX_CALL_ARGS_LIMIT=16` clamp made everything above 16 undefined. The arity
  spread is now outlined into one shared per-module helper that pins the whole
  declared range. (#398)
- **The bytecode VM decided exactness from a result's *value* rather than its
  operands' tags,** so inexact arithmetic silently became exact. Exactness is
  now decided by operand tag on the VM's whole numeric surface. (#394)
- **Native flonum `modulo`, `remainder`, `quotient` and the floor-division
  family were wrong across the board.** The double path converted through
  `int64` and packed the result **exact**, so `(quotient 7.0 2.0)` reported
  `exact? #t`; a `1e20` clamp constant leaked out and division by `0.0`
  returned the clamp. Results now stay double, and a zero divisor raises
  uniformly across `quotient` / `remainder` / `modulo`, including the mixed
  bignum route.
- **`iota` silently ignored its `start` and `step` arguments.** The stdlib
  defined it as strictly 1-argument while callers were already writing
  `(iota 5 1)` and `(iota 5 0 2)`; codegen discarded the extra arguments, so
  both returned `(0 1 2 3 4)` with no error. Both SRFI-1 forms now work, and
  passing too many arguments is a fatal arity error rather than a silent drop
  (ESH-0362).
- **`define-library` could not see a library defined in the same file.** The
  form validated its library name and threw it away, so a same-unit `import`
  failed. Fixed at the root on **all three** engines. The VM lane was the real
  gap — it knew none of `define-library` / `import` / `export` — and three
  latent VM defects were fixed with it: `provide` emitted nothing (slot shift),
  the module loader desynchronised its POP, and a fail-open forward reference is
  now refused instead of silently mis-binding. (#402)
- **`--shared-lib` never linked a shared library, and exited 0 anyway.** It now
  produces a real, C-ABI-correct shared library in both directions, with the
  `[2 x i64]` register pair and the Windows sret convention handled, and refuses
  32-bit targets loudly. (#377)
- **`with-region` was mis-lowered on the bytecode VM, two independent ways, and
  returned an untagged value on native.** (#380)
- **A named-let loop procedure used as a first-class value SIGSEGVed.** Escaped
  named-let closures now get real capture cells; nested tensor literals coerce.
  Both VM routes now hold a loop procedure escaped through a global or a
  let-bound cell. (#381)
- **`(load …)` and module resolution diverged between the JIT, AOT and import
  paths.** Four private copies of the resolver are unified behind one
  `platform::resolve_module_source_path`, and the requiring-file is owned by a
  scoped source context so the facts cannot drift. The documented search order
  (file directory, then project root / CWD, then `-I` / `ESHKOL_PATH`, then the
  bundled `lib`) is now the code's only order. (#407)
- **Non-local exits leaked regions.** A `raise` / `guard` or a `call/cc` escape
  crossing an open region now closes it after deep-promoting the in-flight
  value, and restores the allocation-routing slot before any arena is freed.
  This also fixes `with-region`, which previously leaked its region on a `raise`
  out of the body. (#341)
- **The `stdlib_display` test division and two harness defects that produced
  false verdicts.** The toolchain-fingerprint guard tried BSD `stat -f` before
  GNU `stat -c`; on GNU coreutils `-f` means `--file-system`, so every green
  Linux run was declared `INVALID RUN`. And the stale-directory prune globbed an
  unmatched pattern into `du`, which under `set -euo pipefail` killed the
  calling suite silently — the cause of a false red on the language-coverage
  floor.

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

### Tensor nested syntax
Resolved in v1.3.4-evolve, and the direction of the remaining asymmetry is the
opposite of what this entry used to claim. A nested collection is now
classified **by value** rather than by how it was spelled, so every tensor
operation accepts a runtime-built nest of lists and/or vectors at any rank, and
`#(#(1.0 2.0) #(3.0 4.0))` remains the rank-2 tensor literal it always was.
A ragged nest is no longer a parse-time refusal of the whole translation unit;
it lowers to an ordinary nested vector and raises one catchable error at the
operation that demanded a tensor. `tensor` and `matmul` are `vm-supported` in
`tests/vm_parity/PARITY.tsv`. Where the two engines still differ on the
`(tensor <rectangular nested collection>)` constructor, the manifest is the
authority — consult the row rather than this page.

### Region handles and `with-region` on the VM
The surface is shared; the reclamation is not. `region-open?` is
`vm-supported`, and `region-open` / `region-close` / `op:WITH_REGION` are
`native-only-justified` because the name resolves on both substrates and the
handle protocol, its validation and every error message are byte-identical (one
shared C implementation). A VM program using regions computes the same answer
as native — it simply does not get the memory back, because the region arena,
the allocation-slot hijack and the escape promotion are native-arena constructs
and the VM heap has no escape evacuator.

### Reverse-mode gradient on the VM
`gradient` now runs on the bytecode VM at full parity with native codegen
(#337): forward/reverse-mode, arity-resolved (scalar / N-argument / arity-1
whole-vector) and including the curried `((gradient f) point)` form,
byte-identical across the VM's source and bytecode axes. The one remaining
native-only case is higher-order nesting (gradient-of-derivative / Taylor
tower); use native codegen (`eshkol-run`) for nested higher-order AD.

### WASM batch compilation: multiple `define-syntax` forms
Under the WASM batch compilation path, a program with two or more
`define-syntax` forms mis-evaluates the first macro's arithmetic. The VM-native
path and the browser REPL's incremental compilation path both evaluate the same
program correctly. Found by the WASM execute-and-diff lane and tracked as an
XFAIL gate in `tests/wasm_diff/EXCLUSIONS.tsv`; use the VM-native or incremental
REPL path in the meantime.

---

## Tracked Open Issues

Edge-case findings surfaced by the adversarial-testing harnesses (see
[TESTING.md](TESTING.md)). Each has a minimal repro; the older ones carry a
ledger entry under `.swarm/tasks/ESH-*.json`, and the items filed during the
v1.3.4-evolve correctness wave are tracked as build items for v1.3.5. None
block ordinary use.

**Found during the v1.3.4-evolve correctness wave (new, honest knowns)**

- **AD ignores function shadowing — silent wrong answer.** A parameter that
  shadows a global function name is differentiated as the *global*, not as the
  shadowing binding: a case whose correct derivative is `4` answers `6`. This
  affects inexact points as well as exact ones. It is why the new exact-input
  differentiation tier (#393) uses a whitelist — a body may mention only its own
  parameter — rather than accepting arbitrary bodies. Silent-wrong class; the
  highest-priority open AD defect.
- **The Taylor tower cannot nest.** `derivative-n` applied to the result of
  `derivative-n` silently returns `0` where the jet path gives the correct
  value. Every *well-defined* single-level route is exact; see the
  derivative-closure entry under **Automatic differentiation** below, which is
  the same root cause seen from the closure side. Closing it needs a
  "differentiate a tower" runtime step inside the emitted derivative wrapper.
- **`i128` has no branch in the generic arithmetic opcodes.** The dedicated
  `i128-add` / `-sub` / `-mul` / `-neg` / shift / comparison / division surface
  is complete and bit-identical on both engines. Generic `+` / `*` over `i128`
  values is not wired on **either** side: native is fatal and the VM answers a
  wrong value. Use the `i128-*` operators; `i128` deliberately lives off the
  numeric tower and never auto-promotes, so this is a missing opcode branch
  rather than a tower-contagion question.
- **The VM lane ignores a path-literal `(load "x.esk")`.** After the
  load-path unification (#407) the native, JIT and AOT paths share one resolver.
  The VM lane still resolves only the CWD `lib/<dotted>` form and silently
  ignores a path literal. Tracked for v1.3.5; use the dotted module form on the
  VM in the meantime.
- **The attention backward pass is open.** The embedding and Fréchet-mean
  backward passes landed this release with gradient checks; attention's backward
  pass is not yet closed out.
- **qLLM oracle exporters can crash under the in-process JIT on macOS.** In
  traced or coverage-instrumented contexts under load, the qLLM oracle's
  exporters can crash on the in-process JIT lane. This is an **advisory** lane:
  the AOT lane is unaffected, the exported values are unaffected, and the
  release gate is 10/10. Under investigation.

**Automatic differentiation**
- Vector gradient-of-gradient silently returns zeros — use nested scalar
  `derivative` for exact higher-order results (ESH-0096). The same shape reached
  through a **curried** gradient closure behaves the same way: with
  `(define g (gradient f))`, `(jacobian g point)` returns a zero matrix, and
  `(gradient g)` correctly refuses (the gradient of an ℝⁿ→ℝⁿ function is
  undefined) with a diagnostic naming `jacobian`. Use `(hessian f point)` — it is
  exact. The curried gradient itself, `(g point)` / `(g x y …)`, is exact and
  byte-identical to `(gradient f point)`. Curried *scalar* higher-order
  derivatives are exact to 3rd order (ESH-0369).
- **`derivative-n` / `taylor` applied to a derivative *closure* returns 0.** With
  `(define df (derivative f))`, `(derivative-n df x k)` yields `0` — the tower
  path seeds a heap Taylor tower, but the closure `(derivative f)` returns is
  jet-transparent, not tower-transparent, so it reads the tower-tagged argument
  as a jet and the result carries no tower for extraction to read. Every
  *well-defined* route to the same number is exact: `(derivative-n f x k)` on the
  base function, `(derivative df)` / `(derivative (derivative f))` (the curried
  spelling, ESH-0369), and `(derivative (lambda (x) (df x)) x0)`. Closing this
  needs a "differentiate a tower" runtime step (`c_k(f') = (k+1)·c_{k+1}(f)`)
  inside the emitted derivative wrapper — a build item, not a limit of the
  mathematics.
- Vector-param AD op combined with a captured local parameter fails LLVM
  verification (`PtrToInt source must be pointer`) (ESH-0072, ESH-0097).
- **Resident training loops accumulate RSS unless each step is scoped.** The
  automatic per-iteration nursery (ESH-0214e) reclaims *structural* mutation, not
  the reverse-mode AD tape, so a long-running gradient/training loop is excluded
  from automatic reclamation by design. Wrap each optimization step in an
  explicit `(with-region ...)` to get flat RSS — the tape's node-pointer array is
  now reclaimed with the region (#345), so a per-step `with-region` is fully flat.
  A lighter-weight tape mark/release API is planned so a bare training loop can
  reclaim without a per-step region.

**Recursion depth**
- The stdlib list operations that used to fail on very large inputs no longer
  do. `sort` is an iterative vector merge sort (ESH-0098 resolved): 1,000,000
  reversed elements sort in 176 MB peak RSS, 5,000,000 in 848 MB, on the JIT and
  AOT alike. `length` and `filter` complete on 1,000,000-element lists
  (ESH-0108 resolved). Deep non-tail user recursion reaches at least 270k frames
  without a diagnostic-free death. Mutual tail calls ARE now proper R7RS
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

**Knowledge base and consciousness engine**
- **Quoted pattern-variable queries never match — silent wrong answer.**
  `(kb-query kb '(parent alice ?child))` never returns a match, on any target
  (native, VM, or WASM). A quoted `?x` compiles to a plain symbol, but the
  pattern detector recognizes only string-spelled logic variables, and
  `unify` / `logic-var?` / `kb-query` each use a different, mutually
  incompatible representation of what counts as a logic variable. Affects
  every knowledge-base pattern query written with the documented
  `'(pattern ?var)` syntax. Fix (a single unified logic-variable
  representation) is scheduled for v1.3.5.
- **Factor-graph free energy is wrong in the browser build only.**
  `(free-energy fg evidence)` returns `-0.0` regardless of the supplied
  evidence when run under the browser's WASM VM; native `eshkol-run`
  computes the correct value. Root cause is under investigation and looks
  like a wasm32-specific divergence together with a related
  observation-clamping defect. Fix scheduled for v1.3.5.

**VM parity**
- The VM implements a documented subset of the language, tracked row-by-row in
  `tests/vm_parity/PARITY.tsv` (see [VM_PARITY.md](VM_PARITY.md)): 951 rows —
  578 `vm-supported`, 44 `native-only-justified`, 329 `gap`, of which 17 are
  verified behavioral divergences with reproducible programs under
  `tests/vm_parity/found/` and the rest acknowledged holes. `op:GRADIENT` and
  `op:DERIVATIVE` moved to `vm-supported` this release (#337), and
  `op:IMPORT` / `op:PROVIDE` / `op:REQUIRE` followed with the same-unit
  `define-library` fix (#402) — with no new waivers. The differential gate is
  140/140 on the release cut.
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
