# Eshkol v1.2.0-scale — Production Readiness

**Release Date**: May 1, 2026

Eshkol v1.2.0-scale is the *production-readiness* release. The v1.1
line proved the math (autodiff, tensors, the consciousness engine);
v1.2 makes it shippable: trained models save and load, error messages
point at the actual line, the Python FFI is stable and zero-copy,
deep recursion doesn't blow the stack on Darwin, and a long tail of
correctness/security bugs that surfaced under real workloads is now
fixed.

The headline addition isn't a feature — it's the **62-test edge-case
regression suite** that catches every fix in this release going
forward.

## What's New in v1.2.0-scale

### Production Deployment

- **Model serialization** — `.eshkol-model` is an ESKB-extended
  binary format that round-trips trained networks (architecture +
  weights + metadata) so you can save a model on the training box
  and load it on the inference box.
- **Stable C ABI + Python bindings** — `inc/eshkol/c_abi.h` is the
  versioned public header; `pip install eshkol` gives you pybind11
  bindings with NumPy zero-copy interop.  Gradient computation,
  structured returns, and error recovery all crossed the FFI cleanly
  after the v1.2 hardening pass.
- **Per-thread arenas** — concurrent code paths
  (parallel-map workers, thread-pool tasks) now use thread-local
  arena slots so allocation in one worker never stomps another's.
- **Image I/O** — PNG/JPEG/BMP read/write/resize for vision
  pipelines.  `deps/stb/` is the current backend; v1.3+ replaces it
  with native platform APIs (CoreGraphics on macOS, system
  libpng/libjpeg on Linux, GDI+ on Windows).
- **Plotting stdlib** — inline matplotlib-style charts via PNG output
  for notebook-style workflows.

### Compiler Diagnostics

- **Actionable error messages** — compile errors now report the
  exact source line and column with a caret underline:
  ```
  /path/to/file.esk:6:4: error: Unknown function: undefined-fn
      6 |   (undefined-fn 1 2 3))
        |    ^
  ```
  Previously every error pointed at line 1 because the parser's
  comment-stripping reader consumed newlines.  The fix preserves
  newlines from comments and threads a cumulative file-line counter
  across `eshkol_parse_next_ast_from_stream` calls.

### Stdlib + Language

- **JSON Schema validation** (Draft 7 subset) — `json-schema-valid?`
  and `json-schema-validate` for experiment-manifest /
  preregistration enforcement.  Supports type, properties, required,
  additionalProperties, items, min/max length/items, minimum/maximum
  (with exclusive variants), enum, const, pattern (substring),
  oneOf / anyOf / allOf / not.  Auto-loaded via stdlib.
- **R7RS-compliant scoping for stdlib redefines** — user `(define
  (foo …))` after `(require stdlib)` cleanly shadows stdlib's `foo`
  at link time (LinkOnceODR linkage on stdlib functions) and at
  call-site lowering (variadic-info hygiene clears stale entries
  on redefine).  Previously a user redefine of a variadic stdlib
  function with a fixed-arity signature compiled with an
  arity-mismatch warning and crashed at runtime.
- **AD scalar derivative on inline lambdas** — `(derivative
  (lambda (x) …) point)` inside a wrapper function now correctly
  flows through the runtime closure dispatch.  Previously it
  returned -inf / wrong values because the new-style derivative
  codegen path bailed out without calling the closure for
  function-parameter operands.
- **Reflection** — `procedure-arity`, `record-fields`, `describe`
  for runtime inspection of user-defined procedures and records.
- **Memoization / LRU cache stdlib** — `(memoize fn :lru 256)`.
- **PRNG seeding + deterministic replay** — `(seed-prng! …)` and
  per-stream isolation for reproducible experiments.
- **Lazy sequences / streams** (SRFI 41).
- **Time API** — ISO-8601 parse/format + duration types.
- **Regex capture groups** — `regex-group` / `match-groups`.
- **CLI argument parser** — `(parse-args)` for noesis CLI entry
  points.
- **call-with-values + URL/base64url encoding** finalised.

### Build, Link, and Platform

- **macOS deep recursion** — every binary now ships with
  `LC_MAIN.stacksize = 512 MB` (`-Wl,-stack_size,0x20000000`) on
  Darwin.  The flag had only been wired into one of the two link
  paths in `eshkol-run`; the common compile-and-link path silently
  inherited the 8 MB default and any non-tail-recursive Scheme code
  hit `eshkol_check_recursion_depth + 4` with a SIGSEGV on its own
  frame push.
- **`--wasm` is self-contained** — the WASM emit path no longer
  falls through to native clang++ link.  `eshkol-run file.esk
  --wasm -o foo.wasm` produces the .wasm via LLVM in-memory codegen
  and exits cleanly; no spurious "_main referenced from
  initial-undefines" link errors.
- **Stdlib functions are weak-linked** — user code can override a
  stdlib symbol with their own definition without a "duplicate
  symbol" link error.  The fix mirrors the Windows
  `WeakAnyLinkage` path onto macOS/Linux LinkOnceODR.
- **AD value-typed captures** — derivatives that close over
  function-parameter `tagged_value` Arguments (e.g. `loss-fn`
  capturing `input`/`target`/`b` in `compute-loss-gradient`) no
  longer fail LLVM IR verification with "PtrToInt source must be
  pointer".
- **CI**: new `linux-x64-asan-ubsan` lane runs the v1.2 edge-case
  suite under `-DESHKOL_ENABLE_ASAN=ON -DESHKOL_ENABLE_UBSAN=ON`.
  `ESHKOL_ENABLE_TSAN=ON` and `ESHKOL_ENABLE_MSAN=ON` are scaffolded;
  TSan/MSan-built libstdc++ on apt.llvm.org is a v1.3 prerequisite.
- **`stdlib.o` rebuild correctness** — `file(GLOB_RECURSE …
  CONFIGURE_DEPENDS)` now watches every `lib/{core,math,signal,
  random,web,tensor,quantum,ml}/*.esk` so editing a transitive
  required module triggers a stdlib rebuild.  Previously only edits
  to `lib/stdlib.esk` itself did.

### Hardening

- **CRITICAL**: shell-string injection in `agent_subprocess.c` —
  fixed by switching to `posix_spawn` with `argv` arrays; the
  `popen("sh -c …")` path is gone.
- **CRITICAL**: Python FFI derivative-method AST injection in
  `eshkol_module.cpp` — fixed by canonicalising via the parser
  rather than string-substituting into source.
- **HIGH** (3 items): integer-overflow guards on arena, KB-load,
  and image-IO size computations (`__builtin_mul_overflow`).
- **HIGH** (4 items): path-traversal defence with percent-decode +
  component-check, TOCTOU race fixes on `stat → open`, and a
  Windows-subprocess buffer-size off-by-one.
- **HIGH**: 36 silent-swallow sites across the runtime now either
  surface the error or are documented as intentional.
- **MEDIUM**: ReDoS-resistant regex engine (counted-quantifier
  backtracker with bounded-state ceiling), SQL-injection guards on
  the persistence path, URL validator that rejects scheme
  smuggling.

### Testing

- **62-test v1.2 edge-case suite** at `tests/v1_2_edge_cases/`
  covering symbol consistency under gensym, AD tape state across
  worker threads, parser line tracking, stdlib symbol resolution,
  the JSON Schema validator, every real bug fix in this release.
  Runs under `bash scripts/run_v1_2_edge_cases_tests.sh` (also
  invoked by `run_all_tests.sh`).  Includes 3 shell-style tests
  for compile-time diagnostics that don't fit the `.esk → run →
  check exit`  shape.
- **Master suite EXIT=0** end-to-end across 36 sub-suites:
  features, stdlib, list, memory, modules, types, typesystem,
  autodiff, ml, neural, json, system, complex, cpp_type, parser,
  control_flow, logic, bignum, rational, parallel, signal,
  optimization, examples, xla, gpu, error_handling, macros, repl,
  web, tco, io, benchmark, migration, codegen, numeric,
  v1_2_edge_cases.

## Carry-forward to v1.3

- **Native media stack** — replace `deps/stb/` with CoreGraphics /
  libpng+libjpeg / GDI+ on the three host platforms so we stop
  vendoring third-party media code.
- **AD `input2` plumbing for non-matmul tensor ops** — the
  backward kernels for conv2d / batchnorm / layernorm / attention
  / multi-head-attention exist; the forward implementations need
  to be rewritten to multi-channel / per-feature shape so the
  Wengert tape can consume them.  Matmul (the only AD-supported
  tensor op exercised by the suite today) is correctly wired.
- **TSan / MSan CI lanes** — pending TSan/MSan-built libstdc++.
- **Spec-doc generator (`eshkol-doc`)** — extract type signatures
  + docstrings from the indexed module graph.
- **True module-private internals** — `(provide …)` is currently
  informational under both AOT and JIT (Bug Z); v1.3 reintroduces
  a proper rename pass while keeping cross-file calls to provided
  symbols working.
- **AD-1 follow-up** — re-extract `codegenDerivativeMonolith` into
  the new code path; the v1.2 fix delegates from `derivative()`
  to the monolith as a stop-gap.

---

# Eshkol v1.1.13-accelerate — Windows ARM64 + Release Workflow + VM Closure Fixes

**Release Date**: April 9, 2026

Eshkol v1.1.13-accelerate adds native Windows ARM64 support, rewrites the release workflow into a 16-lane build matrix that produces lite/XLA/CUDA variants for every supported platform, fixes two critical bytecode-VM closure bugs that affected the browser REPL and gradient descent demos, hardens setjmp/longjmp on Windows for both x64 and ARM64, and overhauls the website for full mobile responsiveness.

## What's New in v1.1.13-accelerate

### Windows ARM64 Native Support

- Full build path for Windows ARM64 via VS 2022 + ClangCL + LLVM 21 aarch64 SDK
- CMake auto-detects `clang_rt.builtins-{x86_64|aarch64}.lib` based on `CMAKE_VS_PLATFORM_NAME`
- Multi-arch DIA SDK lookup: scans both `Program Files` and `Program Files (x86)` for both `amd64` and `arm64`
- REPL JIT links the architecture-appropriate LLVM target libraries (`LLVMAArch64*` on ARM64, `LLVMX86*` on x64)
- 16 release artifacts per tag: 6 Windows (x64/arm64 × lite/xla/cuda), 6 Linux, 4 macOS

### setjmp/longjmp Cross-Platform Hardening

- Windows ARM64: uses `Intrinsic::sponentry` as the hidden `_setjmpex` context (matches Clang lowering)
- Windows x64: switched from `Intrinsic::localaddress` to `Intrinsic::frameaddress(0)` for the hidden context — produces stable, correctly-aligned frames
- Removed all compile-time `#ifdef _WIN32` branches in favor of runtime `Triple::isOSWindows()` — proper cross-compilation
- Dynamic `jmp_buf` sizing via `eshkol_jmp_buf_size()` runtime helper (no more hard-coded 256-byte buffers)

### Runtime Symbol Renames

POSIX shim functions are now renamed with an `eshkol_` prefix to disambiguate from MSVC's deprecated POSIX shims:

`fopen → eshkol_fopen`, `access → eshkol_access`, `remove → eshkol_remove`, `rename → eshkol_rename`, `mkdir → eshkol_mkdir`, `rmdir → eshkol_rmdir`, `chdir → eshkol_chdir`, `stat → eshkol_stat`, `opendir → eshkol_opendir`

Generated programs call `eshkol_runtime_init()` at start of `main()` (non-REPL mode).

### Codegen Error Handling

- New `fatal_codegen_error_` flag — codegen now fails hard on undefined-function/undefined-variable/private-symbol errors instead of silently emitting `printf`/`exit` runtime stubs
- New `declared_functions_by_ast` map keyed by AST node identity — fixes function resolution when multiple `define`s share a name within the same module

### VM Closure Bug Fixes (browser REPL + bytecode VM)

Two critical closure-handling bugs in the bytecode VM that broke autodiff demos involving captured upvalues:

- **Named-let nested closure PC offset**: When a lambda is created inside a `(let loop ...)` body, the loop's bytecode is inlined into the parent function with PC adjustments — but the inner lambda's `OP_CLOSURE` constant (its `func_pc`) was *not* offset by the loop's start position. The inner closure ended up jumping to a stale location with the wrong upvalue count, manifesting as "UPVALUE INDEX OUT OF BOUNDS" plus gradient always equal to 1 in named-let gradient descent.
- **Native 252 upvalue relay**: When a lambda inside a function captures a variable via the parent's upvalue (`is_local=false`), native 252 was reading `vm->stack[vm->fp + slot]` — treating the upvalue index as a stack-frame offset, reading whichever local happened to be at that slot. Fix: read from `vm->stack[vm->fp - 1]` (the parent closure per the calling convention), then index into `parent_cl->closure.upvalues[slot]`.

Together these restore correct gradients for **every** autodiff demo on the website. The "Train a Neural Network" front-page card now converges to ~0.891 over 3 data points, and the named-let gradient descent in `/learn` chapter 5 converges to `y/x`.

### CI / Release Workflow Overhaul

- Release workflow rewritten as two matrices: `unix-release-matrix` (10 jobs) + `windows-release-matrix` (6 jobs)
- New `publish-release` job downloads all artifacts, generates `SHA256SUMS.txt`, and publishes the GitHub release
- Per-architecture LLVM SDK caching on Windows runners (cache key includes `${arch}` and SDK version)
- CI workflow updated: `windows-2022` → `windows-latest`, `max-parallel: 2` Windows throttling
- Removed Docker-based XLA/CUDA build paths in favor of native CMake builds

### Website — Mobile Responsiveness

- Hamburger nav menu collapses the 7 top-level nav links on screens ≤720px; opens as a full-width dropdown; auto-closes when a link is clicked
- `html, body { overflow-x: hidden }` plus `min-width: 0` on flex/grid children — no more horizontal page scroll on any viewport (verified across 5 viewport sizes × 6 routes)
- Code blocks (`runnable-code` wrappers) now scroll horizontally *inside* the block instead of pushing the page wider
- `.docs-layout` switched from `1fr` to `minmax(0, 1fr)` — fixes the docs page reporting 972px wide on a 375px viewport
- `.comparison-table` becomes scrollable on ≤720px so the comparison table on `/downloads` doesn't push the page

### Browser REPL Error Display

- REPL now captures stderr (compile warnings, parse errors) into `_vmStderr` and displays them as `error: undefined variable 'foo'` instead of silently re-prompting
- Suppresses the trailing `()` NIL fallback when a compile error fired
- Shows `error: could not parse expression` when nothing parses
- Same fix applied to runnable code blocks (Run ▶ buttons across the site)

### Test Results

- 35/35 test suites, 100% pass rate on macOS ARM64, Linux x64, Linux ARM64, Windows x64, Windows ARM64
- 32/32 runnable site examples verified end-to-end in headless Chromium across mobile/tablet/desktop viewports
- All 16 release-build lanes green on the v1.1.13-accelerate tag

---

# Eshkol v1.1.12-accelerate — Toolchain Unification + Platform Hardening

**Release Date**: April 7, 2026

Eshkol v1.1.12-accelerate unifies the toolchain on LLVM 21 across all platforms, adds a native Windows build path via Visual Studio 2022 + ClangCL, fixes ARM64 and Windows x64 ABI issues in the runtime, adds clean URL routing to the website, and expands CI/CD coverage.

## What's New in v1.1.12-accelerate

### LLVM 21 Toolchain Unification

- Standardized entire build on LLVM 21 across Linux, macOS, and Windows
- New `cmake/LLVMToolchain.cmake`: authoritative LLVM version discovery and enforcement at configure time
- New `scripts/lib/llvm21-env.sh`: platform-aware LLVM 21 activation for all shell scripts
- Hard version check: configure fails with a clear error if LLVM major version is not exactly 21
- Removed misleading `LLVM 18+` compatibility branches from backend codegen

### Native Windows Build (Visual Studio 2022)

- Full native build via Visual Studio 2022 + ClangCL + LLVM 21 SDK
- Configures with `Visual Studio 17 2022` generator and `-T ClangCL`
- `region_escape_tagged_value_into` ABI fix: passes `eshkol_tagged_value_t` by pointer to satisfy Windows x64 calling convention for 16-byte aggregates

### ARM64 ABI Fix

- Fixed `call_thunk_closure` in `arena_memory.cpp:3908`: ARM64 returns 16-byte structs in register pairs (x0:x1), not via hidden return buffer
- Resolves dynamic-wind + call/cc thunk invocation on Apple Silicon and Linux ARM64

### Mutual TCO Fix

- `llvm_codegen.cpp`: version-gated tail call kind — `TCK_MustTail` on LLVM < 18, `TCK_Tail` on LLVM ≥ 18
- Fixes "LLVM ERROR: cannot use musttail" on Linux with LLVM 21

### Website — Clean URL Routing

- Navigation now uses `/downloads`, `/learn`, `/docs` etc. instead of `/#/downloads`
- GitHub Pages 404-redirect SPA routing for direct URL access
- History API (`pushState`/`popstate`) replaces `hashchange`

### CI/CD Expansion

- New GitLab CI matrix: Linux x64/arm64 × lite/XLA/CUDA + macOS × lite/XLA + Windows
- GitHub CI updated to LLVM 21 baseline across all runners
- Docker parity images (`docker/debian/`, `docker/ubuntu/`) updated to LLVM 21

### Test Results

- 35/35 test suites, 438/438 tests, 100% pass rate (macOS ARM64, Linux x64)

---

# Eshkol v1.1.11-accelerate - Performance Acceleration Release

**Release Date**: March 27, 2026

Eshkol v1.1-accelerate builds on the v1.0-foundation with comprehensive performance acceleration. Every v1.1 roadmap item is now complete: XLA backend (5/5), SIMD vectorization (4/4), concurrency (5/5), extended math (5/5), bignum/rational (6/6), consciousness engine (4/4), R7RS extensions (6/6), dual backend (7/7), and Windows platform (5/5) -- totaling 47/47 items.

## What's New in v1.1-accelerate

### Web Platform

Eshkol compiles to WebAssembly and runs in the browser. The project website ([eshkol.ai](https://eshkol.ai)) is itself written in Eshkol — 1,500+ lines compiled to a 502KB WASM binary.

- **Browser REPL**: A 63-opcode bytecode interpreter compiled via Emscripten runs in the browser with 555+ built-in functions. Users can evaluate Eshkol expressions without installing anything.
- **Automatic Differentiation in Browser**: Forward-mode AD via dual numbers works through the bytecode VM. Arithmetic opcodes detect dual number operands and dispatch to dual arithmetic (product rule, quotient rule, chain rule). `(derivative (lambda (x) (* x x)) 3.0)` returns `6` in the browser.
- **Interactive Examples**: Every code example on the website has a Run button with inline output. Examples span AD, neural network training, ODE solving, knowledge base queries, and exact arithmetic.
- **59 DOM Bindings**: Create elements, manipulate styles, handle events, draw on canvas, manage routing, access local storage — all from Eshkol compiled to WASM.
- **8-Chapter Interactive Textbook**: Progressive tutorial from basics through AD, tensors, scientific computing, and the consciousness engine — every example runnable.

### Bytecode VM — Production Complete

The bytecode VM is a fully production-grade execution engine:

- **555+ built-in functions** including character operations, bitwise logic, type predicates, string processing, list utilities, math extensions, complex numbers, and port I/O
- **Automatic differentiation in the VM**: Forward-mode AD via dual number propagation through all arithmetic and transcendental operations
- **R7RS control flow**: `call/cc` with continuation capture/restore, `guard`/`raise`, `dynamic-wind`, `values`/`call-with-values`
- **Exact arithmetic**: Rational literals (`1/3`), bignums, complex numbers, `+nan.0`/`+inf.0`/`-inf.0`
- **Consciousness engine**: Knowledge base queries with pattern matching, factor graphs, global workspace
- **Mutual recursion**: Top-level function defines can reference each other
- **System integration**: `directory-entries`, `command-line`, thread pool
- **176/176 tests passing**

### XLA Backend (Dual-Mode Architecture)

Tensor operations now dispatch through a multi-tier acceleration hierarchy:
- **StableHLO/MLIR path**: When MLIR is available, emits StableHLO ops for HW-optimized execution
- **LLVM-direct path**: Default mode with hand-tuned LLVM IR generation
- **Threshold dispatch**: XLA (>=100K elements) -> cBLAS (>=64) -> SIMD (>=64) -> scalar
- 6 core operations fully wired: matmul, elementwise, reduce, transpose, broadcast, slice

### SIMD Vectorization

Tensor loops are now explicitly vectorized with LLVM loop metadata and 64-byte aligned allocation:
- CPU feature detection for SSE2, SSE4.1, AVX, AVX2, AVX-512, and NEON
- SIMD micro-kernels for all tensor arithmetic and activation functions
- Loop vectorization metadata attached to all tensor operation back-edges
- Platform-specific tuning via cache-blocked matrix multiplication

### Signal Processing Library

New `signal.filters` module with 13 DSP functions:
- **Window functions**: Hamming, Hann, Blackman, Kaiser (with inline Bessel I0)
- **Convolution**: Direct O(N*M) and FFT-based O(N log N)
- **Filters**: FIR filter application, IIR Direct Form I
- **Butterworth design**: Lowpass, highpass, bandpass via bilinear transform
- **Analysis**: Frequency response (magnitude + phase)

### Optimization Algorithms

New `ml.optimization` module with 4 gradient-based optimizers:
- **Gradient descent** with configurable learning rate and convergence tolerance
- **Adam** (Adaptive Moment Estimation) with bias correction
- **L-BFGS** with two-loop recursion and backtracking Armijo line search
- **Conjugate gradient** (Fletcher-Reeves) with automatic restarts

All optimizers use the builtin `gradient` function (forward-mode AD with dual numbers).

### Parallelism & Concurrency

- `parallel-map`, `parallel-fold`, `parallel-filter`, `parallel-for-each`
- `future`/`force` for asynchronous computation
- Work-stealing thread pool with hardware-aware sizing
- Thread-safe arena memory management

### Arbitrary-Precision Arithmetic

- Bignum integers with full R7RS compliance (35 codegen gaps fixed)
- Rational numbers (exact fractions)
- Automatic overflow promotion (int64 -> bignum) and demotion
- All arithmetic, comparison, and I/O operations for both types

### Consciousness Engine

Novel AI primitives integrated at the compiler level:
- Logic programming (unification, substitutions, knowledge bases)
- Active inference (factor graphs, belief propagation, free energy minimization)
- Global workspace theory (modules, softmax competition, content broadcasting)
- 22 builtin operations spanning logic, inference, and workspace

### Dual Backend Architecture

Eshkol now ships with a complete bytecode VM alongside the LLVM native compiler:
- **Bytecode VM**: 63 opcodes, 250+ native calls, ESKB binary format, invoked via `-B` flag
- **Weight Matrix Transformer**: 55/55 tests passing, 3-way verified, 307K parameters
- **qLLM Bridge**: Eshkol-to-qLLM tensor conversion for semiclassical inference

### Windows Platform Support

Native Windows builds are now supported:
- **MSYS2/MinGW64 native build** (PR #9 by mattneel)
- UTF-8-safe REPL with proper console code page handling
- Runtime DLL bundling for standalone distribution
- Path normalization for Windows-style backslash paths

### R7RS Compliance

- `call/cc` and `dynamic-wind`
- `guard`/`raise` exception handling
- Bytevectors, `let-syntax`/`syntax-rules`, symbol operations
- Tail call optimization validation
- `(load "path")` R7RS file loading support

### GPU Backends

- Metal backend for Apple Silicon (SF64 software float64 emulation)
- CUDA backend with cuBLAS integration
- 5 GPU operations: elementwise, matmul, reduce, softmax, transpose

## Test Results

35 test suites passing with 438 test files covering all subsystems.

---

# Eshkol v1.0.0-foundation - Production Release

**Release Date**: December 12, 2025

We are proud to announce the **production release** of Eshkol v1.0-foundation - a complete, production-ready programming language that delivers unprecedented integration of compiler-level automatic differentiation, deterministic arena memory management, and homoiconic native code execution.

## What is Eshkol?

Eshkol is a production-grade Scheme dialect built on LLVM infrastructure, designed for gradient-based optimization, neural network development, and scientific computing. It combines functional programming elegance with native performance while eliminating garbage collection entirely.

## v1.0-foundation Achievements

### Complete Production Compiler

Eshkol v1.0-foundation delivers a fully functional compiler with:

- **Modular LLVM backend** with 21 specialized code generation modules
- **HoTT-inspired gradual type system** with bidirectional type checking
- **Comprehensive parser** supporting S-expressions, type annotations, pattern matching, and macros
- **Ownership and escape analysis** for automatic allocation strategy optimization
- **Module system** with dependency resolution and circular dependency detection
- **Interactive REPL** with LLVM ORC JIT compilation
- **170+ test files** providing comprehensive verification

### Compiler-Integrated Automatic Differentiation

First-class AD system operating at compiler, runtime, and LLVM IR levels:

- **Forward-mode AD** using dual number arithmetic
- **Reverse-mode AD** with computational graph and tape stack
- **Symbolic AD** through AST transformation
- **Nested gradients** up to 32 levels deep
- **8 vector calculus operators**: derivative, gradient, jacobian, hessian, divergence, curl, laplacian, directional-derivative
- **Polymorphic implementation** supporting int64, double, dual numbers, AD nodes, and tensors

### Deterministic Memory Management (OALR)

Zero garbage collection with ownership-aware lexical regions:

- **Arena allocation** with O(1) bump-pointer allocation
- **Escape analysis** automatically determining stack/region/shared allocation
- **with-region syntax** for lexical memory scopes  
- **Ownership tracking** preventing use-after-move at compile time
- **Fully deterministic** - zero GC pauses for real-time applications

### Comprehensive Language Features

**300+ language elements including:**
- 39 special forms (define, lambda, let/let*/letrec, if/cond/case/match, etc.)
- 60+ list operations with full Scheme compatibility
- 30+ string utilities
- 25+ tensor operations
- 10 hash table operations
- Complete I/O system with ports and exception handling
- Hygienic macros (syntax-rules)
- Pattern matching with 7 pattern types
- Multiple return values (values, call-with-values, let-values)

### Rich Standard Library

Modular library organization with pure Eshkol implementations:

- **stdlib.esk** - Central module re-exporting core functionality
- **math.esk** - Linear algebra (det, inv, solve), numerical integration, root finding, statistics
- **core.functional** - compose, curry, flip combinators
- **core.list** - higher-order functions, transformations, queries, sorting
- **core.strings** - extended string manipulation
- **core.json** - JSON parsing and serialization
- **core.data** - CSV processing, Base64 encoding

### Production-Ready Infrastructure

- **Cross-platform**: macOS (Intel/Apple Silicon), Linux (x86_64/ARM64), Windows (MSYS2/MinGW64)
- **Docker containers**: Debian and Ubuntu images
- **CMake build system**: Modern, maintainable build infrastructure
- **Comprehensive documentation**: Language specification, user reference, API docs
- **Package generation**: Homebrew formula, Debian packages

## Installation

### Quick Start

```bash
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run a program
build/eshkol-run tests/neural/nn_working.esk

# Start interactive REPL
build/eshkol-repl
```

### System Requirements

- **LLVM** 10.0+ (14+ recommended)
- **CMake** 3.14+
- **C17/C++20 compiler** (GCC 8+, Clang 6+)
- **readline** (optional, for REPL enhancements)

## Example: Neural Network Training

```scheme
(require stdlib)

;; Sigmoid activation
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))

;; Mean squared error loss
(define (mse-loss pred target)
  (let ((diff (- pred target)))
    (* 0.5 (* diff diff))))

;; Forward pass
(define (forward weights bias input)
  (sigmoid (+ (tensor-dot weights input) bias)))

;; Compute loss gradient for backpropagation
(define (loss-gradient weights bias input target)
  (gradient 
    (lambda (params)
      (mse-loss 
        (forward (vref params 0) (vref params 1) input)
        target))
    (vector weights bias)))

;; Training works - automatic differentiation handles the calculus
```

## What Makes v1.0-foundation Special

### 1. Compiler-Integrated AD - Not a Library

Unlike JAX, PyTorch, or TensorFlow, Eshkol's automatic differentiation is built into the **compiler itself**, operating on AST, runtime values, and LLVM IR simultaneously. This enables differentiation of **any** Eshkol function without framework constraints or graph tracing overhead.

### 2. Homoiconic Native Code

Lambdas compile to LLVM-native code but retain their source S-expressions in closure structures, enabling both **runtime introspection** and **native performance** - a combination no other compiled language achieves.

### 3. Zero Garbage Collection

Arena-based memory management provides **fully deterministic** performance without GC pauses, making Eshkol suitable for real-time systems, trading algorithms, and control systems where predictable timing is critical.

### 4. Production-Quality Implementation

This isn't a research prototype - it's a complete compiler with comprehensive testing, thorough documentation, and a clear architectural foundation for future expansion.

## Documentation

- **[Language Specification](COMPLETE_LANGUAGE_SPECIFICATION.md)** - Complete technical specification
- **[Language Reference](ESHKOL_V1_LANGUAGE_REFERENCE.md)** - User-focused reference with examples
- **[Vision Documents](docs/vision/)** - Purpose, competitive analysis, roadmap
- **[Architecture Guide](docs/ESHKOL_V1_ARCHITECTURE.md)** - Technical architecture overview
- **[API Reference](docs/API_REFERENCE.md)** - Comprehensive function documentation
- **[Quickstart](docs/QUICKSTART.md)** - Hands-on tutorial

## Known Limitations

v1.1-accelerate builds on v1.0-foundation. Remaining planned features:

- **Distributed computing** - Planned v1.2 (Q2 2026)

See [ROADMAP.md](ROADMAP.md) and [docs/vision/FUTURE_ROADMAP.md](docs/vision/FUTURE_ROADMAP.md) for detailed development plans.

## Next Steps

### For Users

1. **Explore the REPL**: `build/eshkol-repl`
2. **Try the examples**: `build/eshkol-run tests/autodiff/*.esk`
3. **Read the docs**: Start with [ESHKOL_V1_LANGUAGE_REFERENCE.md](ESHKOL_V1_LANGUAGE_REFERENCE.md)
4. **Experiment with AD**: The automatic differentiation system is production-ready

### For Contributors

1. **Review architecture**: [docs/ESHKOL_V1_ARCHITECTURE.md](docs/ESHKOL_V1_ARCHITECTURE.md)
2. **Check the roadmap**: [ROADMAP.md](ROADMAP.md) for v1.1/v1.2 plans
3. **See contribution guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
4. **Join development**: See open issues on GitHub for contribution areas

### For Researchers

1. **Study the AD implementation**: [docs/vision/ADDENDUM_TECHNICAL_WHITE_PAPER_V1.md](docs/vision/ADDENDUM_TECHNICAL_WHITE_PAPER_V1.md)
2. **Examine memory architecture**: [docs/breakdown/MEMORY_MANAGEMENT.md](docs/breakdown/MEMORY_MANAGEMENT.md)
3. **Analyze type system**: [docs/breakdown/TYPE_SYSTEM.md](docs/breakdown/TYPE_SYSTEM.md)
4. **Explore homoiconic closures**: [docs/vision/AI_FOCUS.md](docs/vision/AI_FOCUS.md)

## Acknowledgments

Eshkol v1.0-foundation represents years of research and implementation, synthesizing ideas from:
- **Scheme** for elegant functional programming
- **LLVM** for world-class code generation
- **Homotopy Type Theory** for rigorous type foundations
- **Region-based memory** research for deterministic allocation

We thank early testers and contributors who provided valuable feedback during development.

## License

Eshkol is released under the **MIT License** - see [LICENSE](LICENSE) for details.

## Contact

- **GitHub Repository**: https://github.com/tsotchke/eshkol
- **Issues**: Bug reports and feature requests
- **Discussions**: Technical questions and community engagement

---

**Eshkol v1.0-foundation** establishes a new standard for programming languages combining automatic differentiation, deterministic memory, and homoiconic native code. This is not a preview - this is only the beginning. Eshkol has a production-grade compiler ready for gradient-based computing, neural network development, and scientific applications where mathematical correctness and performance are non-negotiable.

*Where mathematical elegance meets uncompromising performance.*
