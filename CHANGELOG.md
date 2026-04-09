# Changelog

All notable changes to Eshkol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.13-accelerate] - 2026-04-09

### Windows ARM64 + Release Workflow Overhaul + VM Closure Bug Fixes

#### Windows ARM64 Native Support
- Full build path for Windows ARM64 via VS 2022 + ClangCL + LLVM 21 aarch64 SDK
- New CMake auto-detection of `clang_rt.builtins-{x86_64|aarch64}.lib` based on `CMAKE_VS_PLATFORM_NAME`
- Multi-arch DIA SDK lookup (both `Program Files` and `Program Files (x86)` for both `amd64` and `arm64`)
- REPL JIT now links the architecture-appropriate LLVM target libraries (`LLVMAArch64*` on ARM64, `LLVMX86*` on x64)

#### setjmp/longjmp Cross-Platform Hardening
- Windows ARM64: uses `Intrinsic::sponentry` as the hidden `_setjmpex` context (matches Clang lowering)
- Windows x64: switched from `Intrinsic::localaddress` to `Intrinsic::frameaddress(0)` for the hidden `_setjmpex` context
- Removed compile-time `#ifdef _WIN32` branches in favor of runtime `Triple::isOSWindows()` checks — proper cross-compilation
- Dynamic `jmp_buf` sizing via `eshkol_jmp_buf_size()` runtime helper (no more hard-coded 256-byte buffers)

#### Runtime Symbol Renames (Windows POSIX shim disambiguation)
- `fopen` → `eshkol_fopen`, `access` → `eshkol_access`, `remove` → `eshkol_remove`, `rename` → `eshkol_rename`, `mkdir` → `eshkol_mkdir`, `rmdir` → `eshkol_rmdir`, `chdir` → `eshkol_chdir`, `stat` → `eshkol_stat`, `opendir` → `eshkol_opendir`
- Avoids MSVC's deprecated POSIX shim warnings on Windows
- Generated programs now call `eshkol_runtime_init()` at start of `main` (non-REPL mode)

#### Codegen Error Handling
- New `fatal_codegen_error_` flag — codegen now **fails hard** on undefined-function/undefined-variable/private-symbol errors instead of silently emitting `printf`/`exit` runtime stubs
- New `declared_functions_by_ast` map keyed by AST node identity — fixes function resolution when multiple defines share a name within the same module

#### VM Closure Bug Fixes (browser REPL + bytecode VM)
- **Named-let nested closure PC offset**: When a lambda is created inside a `let loop` body, the loop's bytecode is inlined into the parent function with PC adjustments — but the inner lambda's `OP_CLOSURE` constant (its `func_pc`) was *not* offset by the loop's start position, causing the inner closure to jump to a stale location with the wrong upvalue count. Symptom: "UPVALUE INDEX OUT OF BOUNDS" + gradient always equal to 1 in named-let gradient descent
- **Native 252 upvalue relay**: When a lambda inside a function captures a variable via the parent's upvalue (`is_local=false`), native 252 was reading `vm->stack[vm->fp + slot]` — treating the upvalue index as a stack-frame offset. Fix: read from `vm->stack[vm->fp - 1]` (the parent closure per the calling convention), then index into `parent_cl->closure.upvalues[slot]`. Together with the named-let fix, this restores correct gradients for all autodiff demos involving captured upvalues
- Both fixes verified end-to-end: gradient descent converges, train demo returns ~0.891, named-let gradient descent converges to y/x

#### CI / Release Workflow
- Release workflow rewritten as two matrices (`unix-release-matrix` × 10 + `windows-release-matrix` × 6) plus a `publish-release` job that downloads all artifacts, generates `SHA256SUMS.txt`, and publishes the GitHub release
- New release lanes: `windows-arm64-{lite,xla,cuda}`, `windows-x64-{lite,xla,cuda}`, `linux-{x64,arm64}-{lite,xla,cuda}`, `macos-{x64,arm64}-{lite,xla}` — 16 total per release
- Per-architecture LLVM SDK caching on Windows runners (cache key includes `${arch}` and SDK version)
- CI workflow updated: `windows-2022` → `windows-latest`, `max-parallel: 2` Windows throttling
- Removed Docker-based XLA/CUDA build paths in favor of native CMake builds

#### Website Mobile Responsiveness
- Hamburger nav menu collapses 7 nav links on screens ≤720px; opens as full-width dropdown; auto-closes when a link is clicked
- `html, body { overflow-x: hidden }` plus `min-width: 0` on flex/grid children — no more horizontal page scroll on any viewport
- Code blocks (`runnable-code` wrappers) now scroll horizontally *inside* the block instead of pushing the page wider
- `.docs-layout` switched from `1fr` to `minmax(0, 1fr)` — fixes the docs page being 972px wide on a 375px viewport
- `.comparison-table` becomes scrollable on ≤720px so the comparison table on `/downloads` doesn't push the page

#### Browser REPL Error Display
- REPL now captures stderr (compile warnings, parse errors) into `_vmStderr` and displays them as `error: undefined variable 'foo'` instead of silently re-prompting
- Suppresses the trailing `()` NIL fallback when a compile error fired
- Shows `error: could not parse expression` when nothing parses
- Same fix applied to runnable code blocks (Run ▶ buttons across the site)

#### Test Results
- 35/35 test suites, 100% pass rate (macOS ARM64, Linux x64, Windows x64, Windows ARM64)
- 32/32 runnable site examples verified in headless Chromium across mobile/tablet/desktop viewports

### Bytecode VM — Production Complete

The bytecode VM is now a fully production-grade execution engine with 555+ built-in functions, forward-mode automatic differentiation, R7RS control flow, exact arithmetic, and the consciousness engine.

- **Automatic differentiation**: Forward-mode AD via dual number propagation. Arithmetic and transcendental functions automatically track derivatives. `(derivative (lambda (x) (* x x)) 3.0)` → `6`
- **R7RS control flow**: `call/cc` with full continuation capture/restore and dynamic-wind unwinding, `guard`/`raise` exception handling, `values`/`call-with-values`
- **Exact arithmetic**: Rational literals (`1/3`), arbitrary-precision integers, complex numbers, R7RS special floats (`+nan.0`, `+inf.0`, `-inf.0`)
- **Consciousness engine**: Knowledge base queries with `?`-wildcard pattern matching, factor graphs with belief propagation, global workspace
- **555+ built-in functions**: Character operations, bitwise operations, type predicates, string processing (`split`, `join`, `trim`, `reverse`, `repeat`), list operations (`take`, `drop`, `any`, `every`, `find`), math extensions (`cosh`, `sinh`, `tanh`), complex numbers, port I/O
- **Mutual recursion**: Top-level function defines can reference each other without forward declarations
- **System integration**: `directory-entries` (POSIX readdir), `command-line` (argc/argv), thread pool
- **176/176 tests passing**

### Web Platform

- **eshkol.ai**: Complete website written in Eshkol (1,400 lines), compiled to WebAssembly
- **Browser REPL**: 63-opcode bytecode interpreter with 555+ builtins, running in WebAssembly via Emscripten
- **AD in the browser**: Automatic differentiation works through the REPL — gradient descent converges in the browser
- **Interactive learning**: 8-chapter textbook and 10-example gallery where every code example has a Run button
- **Live documentation**: Docs page loads markdown directly from GitHub with syntax highlighting
- **Downloads**: Platform-aware downloads page with GitHub Releases API integration
- **GitHub Pages deployment**: Automated via `.github/workflows/pages.yml`

---

## [1.1.12-accelerate] - 2026-04-07

### Toolchain Unification + Platform Hardening Release

#### LLVM 21 Toolchain Unification
- Standardized entire build on LLVM 21 across Linux, macOS, and Windows (previously mixed LLVM 17/18)
- New `cmake/LLVMToolchain.cmake`: authoritative LLVM version discovery and enforcement at configure time
- New `scripts/lib/llvm21-env.sh`: platform-aware LLVM 21 activation for all shell scripts
- All platform scripts now hand off LLVM policy to CMake instead of embedding independent logic
- Hard version check: configure fails with a clear error if LLVM major version is not exactly 21
- Removed misleading `LLVM 18+` compatibility branches from backend codegen

#### Native Windows Support
- Full build via Visual Studio 2022 + ClangCL + LLVM 21 SDK
- Configures with `Visual Studio 17 2022` generator and `-T ClangCL`
- `region_escape_tagged_value_into` ABI fix: now passes `eshkol_tagged_value_t` by pointer (`const eshkol_tagged_value_t*`) to satisfy Windows x64 calling convention for 16-byte aggregates

#### ARM64 ABI Fix
- Fixed `call_thunk_closure` in `arena_memory.cpp`: ARM64 returns 16-byte `eshkol_tagged_value_t` in registers (not via hidden return buffer as on x86/Windows)
- Added `#if defined(__aarch64__)` dispatch — direct return ABI on ARM64, hidden-buffer ABI on x86/Windows
- Resolves dynamic-wind + call/cc thunk invocation on Apple Silicon and Linux ARM64

#### Mutual TCO Fix
- `llvm_codegen.cpp`: version-gated tail call kind — `TCK_MustTail` on LLVM < 18, `TCK_Tail` on LLVM ≥ 18
- Fixes "LLVM ERROR: cannot use musttail" on Linux (LLVM 21 rejects musttail for aggregate-return functions)

#### Website
- Clean URL routing: navigation now uses `/downloads`, `/learn`, `/docs` etc. instead of `/#/downloads`
- GitHub Pages 404-redirect SPA routing for direct URL access
- Updated LLVM requirement strings: LLVM 17+ → LLVM 21+
- Updated WASM size stats to reflect current build sizes

#### CI/CD Expansion
- New GitLab CI matrix: Linux x64/arm64 × lite/XLA/CUDA + macOS × lite/XLA + Windows
- GitHub CI updated to LLVM 21 baseline across all runners
- Docker parity images (`docker/debian/`, `docker/ubuntu/`) updated to LLVM 21

#### Test Results
- 35/35 test suites, 438/438 tests, 100% pass rate (local, macOS ARM64)

---

## [1.1.11-accelerate] - 2026-03-27

### Performance Acceleration Release

Eshkol v1.1-accelerate delivers comprehensive performance acceleration through XLA integration, SIMD vectorization, parallelism primitives, and expanded math/ML libraries.

#### XLA Backend Integration
- Dual-mode architecture: StableHLO/MLIR path (when MLIR available) + LLVM-direct path (default)
- 6 core tensor operations wired through XLA: matmul, elementwise, reduce, transpose, broadcast, slice
- Threshold-based dispatch: XLA (>=100K elements) -> cBLAS (>=64) -> SIMD (>=64) -> scalar
- JIT compilation for dynamic shapes via LLVM ORC
- CPU/GPU code generation from single source with unified dispatch hierarchy

#### SIMD Vectorization
- CPU feature detection: SSE2, SSE4.1, AVX, AVX2, AVX-512, NEON (ARM64)
- Hand-written SIMD micro-kernels for tensor arithmetic (add, sub, mul, div)
- SIMD-accelerated activation functions: ReLU, sigmoid, GELU, LeakyReLU, SiLU
- SIMD dot product with horizontal sum reduction
- LLVM loop vectorization metadata on all tensor loop back-edges
- 64-byte AVX-512 aligned tensor memory allocation
- Platform-specific tuning via cache-blocked matrix kernels

#### Parallelism Primitives
- `parallel-map`, `parallel-fold`, `parallel-filter`, `parallel-for-each`
- `future`, `force`, `future-ready?` for asynchronous computation
- Work-stealing thread pool scheduler with hardware-aware sizing
- Thread-safe arena memory management

#### Extended Math Library
- **Complex numbers**: Full R7RS complex arithmetic with autodiff integration
- **FFT/IFFT**: Cooley-Tukey radix-2 implementation
- **Signal processing filters** (13 functions): Hamming/Hann/Blackman/Kaiser windows, direct and FFT-based convolution, FIR/IIR filter application, Butterworth filter design (lowpass/highpass/bandpass), frequency response analysis
- **Statistical distributions**: Normal, Poisson, Binomial, Exponential, Uniform, Geometric, Bernoulli (in stdlib)
- **Optimization algorithms** (7 functions): Gradient descent, Adam (adaptive moment estimation), L-BFGS (limited-memory BFGS with two-loop recursion), conjugate gradient (Fletcher-Reeves), backtracking Armijo line search

#### Arbitrary-Precision Arithmetic
- Bignum (arbitrary-precision integers) with full R7RS compliance
- Rational numbers (exact fractions) with all arithmetic operations
- Automatic int64 -> bignum overflow promotion and bignum -> int64 demotion
- Bitwise operations on bignums (two's complement semantics)
- 35 codegen gaps audited and fixed across arithmetic, comparison, conversion, and I/O

#### Consciousness Engine
- Logic programming primitives: unification, substitutions, knowledge base
- Active inference engine: factor graphs, belief propagation, free energy minimization
- Global workspace theory: modules, softmax competition, content broadcasting
- 22 builtin operations for logic, inference, and workspace manipulation
- CPT mutation with belief reconvergence for real-time learning

#### R7RS Compliance Extensions
- `call/cc` and `dynamic-wind` with proper continuation semantics
- `guard`/`raise` exception handling
- Bytevectors with full R7RS operations
- `let-syntax` / `syntax-rules` hygienic macros
- Tail call optimization validation
- Symbol operations (`symbol->string`, `string->symbol`)

#### GPU Backends
- Metal backend for Apple Silicon with SF64 software float64 emulation
- CUDA backend with cuBLAS integration and real compute kernels
- 5 GPU operations: elementwise, matmul, reduce, softmax, transpose

#### Production Hardening
- All 47/47 roadmap items completed (including GPU 5/5, Signal Processing 4/4, Web Platform 3/3)
- Tensor bounds checking with runtime validation
- Metal buffer leak fix (@autoreleasepool)
- REPL complex type handling
- Module visibility enforcement
- 35 test suites passing (438 test files)

#### Dual Backend Architecture (NEW)
- **Bytecode VM**: 63-opcode register+stack interpreter (eshkol_vm.c, 8457 lines) with 250+ native call IDs covering the full language
  - 15 runtime libraries: complex, rational, bignum, dual, autodiff, tensor, logic, inference, workspace, string, IO, hashtable, bytevector, multivalue, parameter
  - ESKB binary format with LEB128 encoding, CRC32 checksums, section-based layout
  - Bytecode emission via `-B` flag: `eshkol-run input.esk -B output.eskb`
  - VM linked into compiler build (ESHKOL_VM_LIBRARY_MODE)
- **Weight Matrix Transformer**: Programs as neural network weights (weight_matrices.c, 2299 lines)
  - d_model=36, 5 layers, FFN_DIM=512, 307K parameters
  - 3-way verification: reference interpreter = simulated transformer = matrix-based forward pass
  - 55/55 tests passing, exports QLMW binary format for qLLM loading
- **qLLM Bridge**: Eshkol-qLLM tensor conversion with AD integration (qllm_bridge.h)

#### Windows Platform Support (NEW)
- Native Windows build via MSYS2/MinGW64 (contributed by mattneel, PR #9)
- UTF-8-safe REPL console output
- Runtime DLL bundling in CI artifacts
- MSYS-style file path normalization
- Platform runtime abstraction layer (platform_runtime.cpp/h)

#### Production Hardening (continued)
- ARM64 parallel ABI fix: struct return -> output pointer for eshkol_parallel_execute/map/fold/filter
- REPL CodeGenOptLevel::None fix for ARM64 3+ arg stdlib struct passing
- Cons cell header fix: arena_allocate_cons_with_header for proper HEAP_PTR display
- LinkOnceODRLinkage for stdlib symbol override prevention (no more duplicate symbols)
- Precompiled module discovery: collect_all_submodules() for automatic stdlib sub-module detection
- Weight matrix stack overflow fix: double-buffer State cur/nxt replaces 1.15MB trace[8192]
- `(load "path/to/file.esk")`: R7RS-compatible file loading (alias for require with path conversion)
- Port type check fix: flag bit detection instead of exact HEAP_PTR equality
- Substring bounds overflow protection
- Tensor reshape OOM null check

---

## [1.0.0-foundation] - 2025-12-12

### Production Release

Eshkol v1.0-foundation represents a complete, production-ready compiler with unprecedented integration of automatic differentiation, deterministic memory management, and homoiconic native code execution.

#### Core Compiler Implementation
- Modular LLVM backend with 21 specialized codegen modules
- Recursive descent parser with HoTT type expression support
- Bidirectional type checker with gradual typing
- Ownership and escape analysis for memory optimization
- Module system with dependency resolution and cycle detection
- Hygienic macro system (define-syntax with syntax-rules)
- Exception handling (guard/raise with R7RS semantics)

#### Automatic Differentiation System
- **Forward-mode AD**: Dual number arithmetic for efficient first derivatives
- **Reverse-mode AD**: Computational graph with tape stack for gradient computation
- **Symbolic AD**: Compile-time AST transformation
- **Nested gradients**: Up to 32 levels deep via global tape stack
- **Vector calculus operators** (8 total):
  - `derivative` - First derivative (forward-mode)
  - `gradient` - Gradient vector (reverse-mode)
  - `jacobian` - Jacobian matrix for vector functions
  - `hessian` - Hessian matrix (second derivatives)
  - `divergence` - Vector field divergence (∇·F)
  - `curl` - Vector field curl (∇×F, 3D only)
  - `laplacian` - Laplacian operator (∇²f)
  - `directional-derivative` - Derivative in specified direction
- Polymorphic arithmetic supporting int64/double/dual/tensor/AD-node

#### Memory Management (OALR)
- **Arena allocation**: O(1) bump-pointer with deterministic cleanup
- **Ownership tracking**: Compile-time analysis (owned, moved, borrowed states)
- **Escape analysis**: Automatic stack/region/shared allocation decisions
- **with-region syntax**: Lexical memory scopes
- **Zero garbage collection**: Fully deterministic performance
- **Global arena**: 64KB default block size, expandable
- **Region stack**: 16-level nesting depth

#### Tagged Value System
- 16-byte runtime representation with 8-bit type tags
- Immediate types (0-7): NULL, INT64, DOUBLE, BOOL, CHAR, SYMBOL, DUAL_NUMBER
- Consolidated types (8-9): HEAP_PTR, CALLABLE with object header subtypes
- 8-byte object headers for heap objects (subtype, flags, ref_count, size)
- 32-byte cons cells with complete tagged values (car and cdr)
- Mixed-type lists with zero type erasure

#### Closure System
- Static capture analysis during parsing
- Environment encoding with packed info (captures | fixed_params | is_variadic)
- Homoiconic display via embedded S-expressions
- Lambda registry for function pointer → S-expression mapping
- Variadic function support (fixed + rest parameters, or all-args-as-list)

#### Data Structures
- N-dimensional tensors with autodiff integration
- Hash tables (FNV-1a hashing, open addressing, 0.75 load factor)
- Heterogeneous vectors (Scheme-compatible)
- Strings with UTF-8 support
- Proper and improper lists
- Exception objects with source locations

#### Language Features (300+ Total)
- **39 special forms**: define, lambda, let/let*/letrec, if/cond/case/match, quote/quasiquote, etc.
- **300+ built-in functions**: Complete Scheme R7RS subset
- **60+ list operations**: map, filter, fold, compound accessors (caar through cddddr), etc.
- **30+ string utilities**: join, split, trim, case conversion, search, replace
- **25+ tensor operations**: element-wise arithmetic, linear algebra, reductions, transformations
- **10 hash table operations**: ref, set!, has-key?, remove!, keys, values, count, clear!
- **8 autodiff operators**: Complete vector calculus support
- Scheme-compatible syntax (R7RS subset)
- Module system with `require`/`provide`
- Pattern matching with 7 pattern types
- First-class functions and closures
- Tail call optimization (self-recursion → loops)
- Hygienic macros (syntax-rules)

#### Standard Library (Modular)
- `stdlib.esk` - Re-exports core modules
- `math.esk` - Linear algebra (det, inv, solve), numerical integration, root finding, statistics
- `core.functional.*` - compose, curry, flip
- `core.list.*` - higher-order, transforms, queries, search, sort, convert, generate, compound accessors
- `core.strings.*` - Extended string manipulation
- `core.json.*` - JSON parsing and serialization
- `core.data.*` - CSV processing, Base64 encoding
- `core.control.*` - Trampoline for deep recursion

#### Development Tools
- **eshkol-run**: Standalone compiler with multiple output modes
- **eshkol-repl**: Interactive REPL with LLVM ORC JIT compilation
- **CMake build system**: Cross-platform with Docker support
- **Comprehensive test suite**: 170+ test files covering all features
- **stdlib.o**: Pre-compiled standard library

#### Platform Support
- macOS (Intel x86_64, Apple Silicon ARM64)
- Linux (x86_64, ARM64)
- Docker containers (Debian, Ubuntu)

#### Build Requirements
- LLVM 17
- CMake 3.14+
- C17 runtime, C++20 compiler
- readline (optional, for REPL features)
