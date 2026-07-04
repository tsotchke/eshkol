# Eshkol Development Roadmap

> **This is the canonical roadmap.** It is the single source of truth for
> Eshkol's release plan. Other roadmap-shaped documents are derived
> views — they exist to answer specific questions and should not
> contradict this file:
>
> - [`docs/COMPILER_ROADMAP.md`](docs/COMPILER_ROADMAP.md) — engineering
>   detail by release line (M0–M4 task tiers), including the per-release
>   work breakdown for compiler engineers.
> - [`docs/breakdown/ROADMAP.md`](docs/breakdown/ROADMAP.md) — short
>   summary for readers in the breakdown / per-subsystem documentation
>   set.
> - [`docs/vision/FUTURE_ROADMAP.md`](docs/vision/FUTURE_ROADMAP.md) —
>   forward-looking vision and long-horizon research items.
> - [`docs/NOESIS_TRAJECTORY.md`](docs/NOESIS_TRAJECTORY.md) — the
>   Noesis-readiness view, tracked separately because Noesis has a
>   distinct downstream cadence.
>
> When any of those docs disagrees with this one, this one is correct
> and the others should be updated.

This roadmap tracks Eshkol's evolution from the **completed v1.0-foundation release** through upcoming versions that will establish Eshkol as the definitive platform for gradient-based computing and integrated AI.

> **Parallel platform program**: The internal freestanding / kernel / embedded architecture work begins during `v1.2-scale` as a mergeable infrastructure program and converges publicly at `v1.8-platform`. See [docs/platform/README.md](docs/platform/README.md) and [docs/platform/ROADMAP_ALIGNMENT.md](docs/platform/ROADMAP_ALIGNMENT.md).

---

## v1.0-foundation (2025) - COMPLETED

**Production Release Delivered**

### Completed Core Implementation
- [x] LLVM-based modular backend with 21 specialized codegen modules
- [x] Recursive descent parser with HoTT type expression support
- [x] Bidirectional type checker with gradual typing
- [x] Ownership and escape analysis for memory optimization
- [x] Module system with dependency resolution and cycle detection
- [x] Hygienic macro system (syntax-rules)
- [x] R7RS Scheme compatibility (subset)

### Completed Automatic Differentiation
- [x] Forward-mode AD (dual numbers)
- [x] Reverse-mode AD (computational graphs)
- [x] Nested gradients (32-level tape stack)
- [x] Vector calculus operators (8 total): derivative, gradient, jacobian, hessian, divergence, curl, laplacian, directional-derivative
- [x] Polymorphic arithmetic (int64/double/dual/tensor/AD-node)

### Completed Memory Management
- [x] Arena allocation with OALR (Ownership-Aware Lexical Regions)
- [x] Escape analysis (stack/region/shared allocation decisions)
- [x] with-region syntax for lexical memory scopes
- [x] Ownership tracking (owned, moved, borrowed states)
- [x] Zero garbage collection - fully deterministic

### Completed Data Structures
- [x] 16-byte tagged values with consolidated types
- [x] 32-byte cons cells supporting mixed-type lists
- [x] N-dimensional tensors with autodiff integration
- [x] Hash tables (FNV-1a hashing, open addressing)
- [x] Heterogeneous vectors
- [x] Exception handling (guard/raise)

### Completed Standard Library
- [x] 60+ list operations
- [x] 30+ string utilities
- [x] Functional programming (compose, curry, flip)
- [x] JSON/CSV/Base64 support
- [x] Math library (linear algebra, numerical methods, statistics)

### Completed Development Tools
- [x] Interactive REPL with LLVM ORC JIT
- [x] Standalone compiler (eshkol-run)
- [x] Library compilation mode
- [x] Comprehensive test suite (430+ files)
- [x] CMake build system
- [x] Docker containers

---

## v1.1-accelerate (Q1 2026) - COMPLETED

**Focus:** Performance acceleration through XLA, SIMD, and parallelism

### XLA Backend Integration
- [x] XLA type system and codegen infrastructure
- [x] XLA fusion for tensor operation chains
- [x] Automatic kernel generation
- [x] CPU/GPU code generation from single source
- [x] JIT compilation for dynamic shapes

### SIMD Vectorization
- [x] SSE/AVX/NEON instruction generation
- [x] Loop vectorization for tensor operations
- [x] Memory alignment optimization
- [x] Platform-specific tuning

### Concurrency Primitives
- [x] `parallel-map` for data parallelism
- [x] `parallel-fold` for parallel reduction
- [x] `future` for asynchronous computation
- [x] Work-stealing thread pool scheduler
- [x] Thread-safe memory management

### Extended Math Library
- [x] Complex numbers with autodiff
- [x] FFT/IFFT operations
- [x] Signal processing filters
- [x] Statistical distributions
- [x] Optimization algorithms (L-BFGS, conjugate gradient)

### Arbitrary-Precision Arithmetic (Added)
- [x] Bignum (arbitrary-precision integers) — full R7RS compliance
- [x] Rational numbers (exact fractions)
- [x] Overflow detection and automatic promotion (int64 → bignum)
- [x] Bignum demotion (normalize back to int64 when possible)
- [x] Bitwise operations on bignums (two's complement semantics)
- [x] All arithmetic, comparison, predicate, equality ops for bignums

### Consciousness Engine (Added)
- [x] Logic programming primitives (unification, substitutions, knowledge base)
- [x] Active inference engine (factor graphs, belief propagation, free energy)
- [x] Global workspace theory implementation (modules, softmax competition)
- [x] 22 builtin operations for logic, inference, and workspace

### R7RS Compliance Extensions (Added)
- [x] call/cc and dynamic-wind
- [x] guard/raise exception handling
- [x] Bytevectors
- [x] let-syntax / syntax-rules hygienic macros
- [x] Tail call optimization validation
- [x] Symbol operations
- [x] `(load "path")` R7RS file loading support

### Dual Backend Architecture (Added)
- [x] Bytecode VM — 63-opcode register+stack interpreter with 250+ native call IDs
- [x] ESKB binary format — section-based bytecode container with CRC32
- [x] `-B` flag for bytecode emission from eshkol-run
- [x] VM compiler integration — eshkol_vm.c linked into compiler build
- [x] Weight matrix transformer — programs as neural network weights (126/126 inline, 123/123 traced, 3-way verified)
- [x] qLLM bridge — Eshkol↔qLLM tensor conversion with AD integration

### GPU Acceleration (Added)
- [x] Metal SF64/DF64/F32/FP24/FP53 precision tiers
- [x] Ozaki-II CRT-based exact matrix multiplication
- [x] CUDA backend with cuBLAS
- [x] Cost-model dispatch (SIMD → cBLAS → GPU)
- [x] Occupancy-aware kernel configuration

### Signal Processing (Added)
- [x] FFT/IFFT (Cooley-Tukey radix-2)
- [x] Window functions (Hamming, Hann, Blackman, Kaiser)
- [x] FIR/IIR filters
- [x] Butterworth filter design (lowpass, highpass, bandpass)

### Web Platform (Added)
- [x] WebAssembly compilation target
- [x] Browser-based REPL (web/)
- [x] JavaScript interop

### v1.1.12 Additions (April 2026)
- [x] Production bytecode VM (555+ builtins, 176/176 tests)
- [x] Forward-mode AD in bytecode VM (dual number propagation through all opcodes)
- [x] eshkol.ai website written in Eshkol, compiled to WASM (8 pages, browser REPL)
- [x] Interactive documentation with runnable code examples
- [x] GitHub Pages deployment workflow
- [x] R7RS control flow in VM: call/cc, guard/raise, dynamic-wind, values
- [x] Exact arithmetic in VM: rational literals, +nan.0/+inf.0/-inf.0

### v1.1.13 Additions (April 2026)
- [x] Native Windows ARM64 build path (VS 2022 + ClangCL + LLVM 21 aarch64 SDK)
- [x] 16-lane release matrix (linux/macos/windows × x64/arm64 × lite/xla/cuda)
- [x] Per-arch LLVM SDK caching on Windows runners
- [x] Two critical VM closure bug fixes (named-let nested closure PC offset, native 252 upvalue relay)
- [x] Windows setjmp hardening: x64 frameaddress, ARM64 sponentry, dynamic jmp_buf sizing
- [x] Runtime symbol renames (eshkol_fopen, eshkol_access, eshkol_remove, etc.) for MSVC POSIX shim disambiguation
- [x] Codegen fatal error flag — fail hard on undefined functions instead of generating runtime stubs
- [x] Mobile-responsive website (hamburger nav, internal scroll for code blocks, table responsiveness)
- [x] Browser REPL error display for invalid input
- [x] Consciousness engine in VM: KB pattern matching, factor graphs, workspace
- [x] Top-level mutual recursion via letrec-style group compilation

### Windows Platform Support (Added)
- [x] Native Windows build via MSYS2/MinGW64 (PR #9, mattneel)
- [x] UTF-8-safe REPL console output
- [x] Platform runtime abstraction layer

---

## Architecture Dependency Chain

```
v1.1 (COMPLETE)
 ├─ Consciousness engine (logic + inference + workspace)
 ├─ XLA/GPU backend
 ├─ Bytecode VM (production)
 ├─ Continuations + exact arithmetic
 └─ Web platform (WASM)
       │
v1.2 ──┤ Model serialization (requires tensors from v1.1)
       ├ Python bindings (requires stable API from v1.1)
       └ Per-thread arenas (requires OALR from v1.0)
             │
v1.3 ────────┤ R7RS library system (requires module system from v1.0)
             ├ String interpolation (parser extension)
             └ PGO (requires stable codegen from v1.1+)
                   │
v1.4 ──────────────┤ TCP/UDP + TLS (requires per-thread arenas from v1.2)
                   ├ Event loop (requires non-blocking I/O)
                   └ Linear resource types (requires HoTT from v1.0)
                         │
v1.5 ────────────────────┤ Symbol embeddings (requires tensors + KB from v1.1)
                         ├ Differentiable logic (requires AD + logic from v1.1)
                         └ LSTM/GRU (requires tensor backprop from v1.1)
                               │
v1.6 ──────────────────────────┤ Backward chaining (requires logic from v1.1)
                               ├ Knowledge graphs (requires KB + embeddings from v1.5)
                               └ Constraint solving (requires logic engine)
                                     │
v1.7 ────────────────────────────────┤ Neural-guided search (requires v1.5 bridge)
                                     ├ GNN (requires graph + tensor ops)
                                     └ Program synthesis (requires type system)
                                           │
v1.8 ──────────────────────────────────────┤ Windowing + event system (requires v1.4 I/O)
                                           ├ Real-time audio (requires signal from v1.1)
                                           └ Embedded targets (requires bare-metal LLVM)
                                                 │
v1.9 ────────────────────────────────────────────┤ Linear dependent types (requires HoTT)
                                                 ├ Effect types (requires type checker)
                                                 └ Algebraic effects (requires continuations)
                                                       │
v2.0 ──────────────────────────────────────────────────┤ Quantum types (requires linear dep types)
                                                       ├ Quantum gates + measurement
                                                       ├ Hybrid VQE/QAOA (requires AD + quantum)
                                                       └ Formal verification (requires dep types)
```

**Arbitrary-order AD (Taylor-tower) track:** The Taylor-tower campaign is
threaded through the existing version themes as enabling substrate, not
as a separate release ladder. P1 lands in v1.3.1, P2/P3 in v1.3.2,
P4/P6/P11 in the v1.4 infrastructure track alongside networking,
P5/P7/P9 in v1.5 intelligence, P10 bridges v1.5 to v1.7, P12 supports
v1.6 reasoning, and P8 feeds v2.0 formal verification of AD. See
[`docs/AD_CAMPAIGN.md`](docs/AD_CAMPAIGN.md).

---

## v1.2-scale (May 2026) - SHIPPED

**Focus:** Get models into production. Save them, load them, deploy them — and stop being surprised by edge cases.

- [x] Model serialization (`.eshkol-model` ESKB-extended binary format)
- [x] Stable C FFI header + Python bindings (pybind11; numpy zero-copy)
- [x] Per-thread arenas (safe concurrent memory allocation)
- [x] Deep recursion: 512 MB main-thread stack on Darwin/Linux/Windows
      (linker flags wired into both single-step and compiled-files
      link paths); 100K-frame recursion-depth check with typed
      exception
- [x] Image I/O (PNG/JPEG/WebP/BMP read/write/resize) — backed by
      native platform/system codec APIs (ImageIO/CoreGraphics on
      macOS, system libpng/libjpeg/libwebp on Linux, GDI+ on
      Windows) so the active backend no longer depends on vendored
      third-party media decoders
- [x] CSV/DataFrame (tabular data loading for ML pipelines)
- [x] Improved error messages with file:line:col + caret underlines
      (preserves newlines in stripped comments + cumulative file-line
      tracking across `parse_next_ast` calls; 5-case regression suite)
- [x] Terminal plotting (`sparkline`, `bar-chart` in pure Eshkol stdlib)
- [x] Codegen modularisation: `tensor_codegen.cpp` 19,940 → 1,280 lines
      (94% reduction); 13 focused per-domain split files.  The
      remaining `llvm_codegen.cpp` extractions
      (module_init_codegen.cpp, builtin_factory_codegen.cpp,
      repl_resolution_codegen.cpp) need the `EshkolLLVMCodeGen` class
      header exposed first and are tracked as v1.3 carry-forward.
- [x] v1.2 edge-case + security regression suite (62 tests) wired into
      `run_all_tests.sh` and a new `linux-x64-asan-ubsan` CI lane.
      Includes 3 shell-style tests for compile-time diagnostics.
- [x] Tagged release artifact contract: the GitHub release workflow
      validates the full 16-asset platform matrix before publishing
      (Linux x64/ARM64 lite/XLA/CUDA, macOS arm64/x64 lite/XLA,
      Windows x64/ARM64 lite/XLA/CUDA) and emits `SHA256SUMS.txt`;
      published as the `v1.2.3-scale` packaging closeout.
- [x] Stdlib correctness: user `(define (foo …))` after `(require
      stdlib)` cleanly shadows stdlib's `foo` at link time
      (LinkOnceODR linkage on stdlib functions) and at call-site
      lowering (variadic-info hygiene clears stale entries on
      redefine).
- [x] `--wasm` is self-contained: WASM emit no longer falls through
      to native clang++ link.
- [x] AD scalar derivative on inline lambdas: `(derivative
      (lambda (x) …) point)` inside a wrapper function correctly
      flows through the runtime closure dispatch.  AD value-typed
      captures pass LLVM IR verification when capturing
      function-parameter `tagged_value` Arguments.
- [x] M1 stdlib finalised: `core.json_schema` (Draft 7 subset),
      reflection (`procedure-arity`, `record-fields`, `describe`),
      memoization/LRU, PRNG seeding + deterministic replay, lazy
      streams (SRFI 41), time API (ISO-8601), regex capture groups,
      CLI argument parser, structured logging (JSON-L),
      Prometheus metrics, extra AD ops
      (atan2 / asin / acos / softmax / gelu / silu / sinh / cosh),
      priority queues / sets / deques.
- [x] Hardening: subprocess shell-injection fix (CRITICAL), Python
      FFI AST-injection fix (CRITICAL), 3 integer-overflow guards
      (HIGH), 4 path-traversal/TOCTOU/Windows-buffer fixes (HIGH),
      36 silent-swallow sites surfaced (HIGH), ReDoS protection +
      SQL-injection guards + URL validator (MEDIUM).

---

## v1.3-evolve (June 2026) - PLANNED

**Focus:** Make the language a joy to use day-to-day.

- [~] Full R7RS library system: `define-library` exports work end-to-end;
      `(rename (m) (a b))` import works.  `(prefix (m) p-)` currently
      requires an explicit `only` or `rename` clause — bare prefix
      over the module's whole export list is the remaining gap.
- [x] String interpolation (`~{expr}` within strings)
- [x] Named keyword arguments (`(f #:key value)`)
- [x] Pattern matching in `let` bindings (destructuring `let-match`)
- [~] Profile-guided optimization — build-time scaffold landed:
      `cmake -DESHKOL_PGO=generate` instruments, `-DESHKOL_PGO=use
      -DESHKOL_PGO_PROFILE=<merged.profdata>` consumes.  Workload
      selection + canonical merge step (the "what do we train on?"
      side) is the remaining gap; the codegen-side machinery is ready.
- [ ] Whole-program optimization (cross-module inlining and dead code elimination)
- [x] **Native media handling, no vendoring**: image I/O uses native
      platform/system codec APIs (ImageIO/CoreGraphics on macOS,
      system libpng/libjpeg/libwebp on Linux, GDI+ on Windows).
      Going forward the project does not vendor third-party media
      decoders.
- [x] AD `input2` plumbing for `tensor-matmul` — verified by
      `gradient` smoke probe: gradient flows to the kernel side
      (input2). Conv2d / batchnorm / layernorm / attention forward
      sites still leave `input2` null in their tape nodes and are
      tracked separately.
- [ ] Arbitrary-order AD Taylor-tower v1.3 track: P1 runtime
      `derivative^n` in v1.3.1 closes ESH-0118; P2 compile-time-K
      monomorphization and P3 JET8 subsumption land in v1.3.2, gated by
      `ad-depth` and `mono-equiv`.

---

## v1.4-connection (July 2026) - PLANNED

**Focus:** Connect to the outside world with compile-time safety.

- [ ] TCP/UDP sockets with linear resource types (guaranteed close)
- [ ] TLS/SSL via system libraries
- [ ] Non-blocking I/O with event loop (epoll/kqueue)
- [ ] Unix domain sockets for local IPC
- [ ] HTTP client (built on sockets + TLS)
- [ ] Linear types for all handles: `open → borrowed → closed` with compile-time tracking
- [ ] Borrow pattern for temporary resource access
- [ ] AD infrastructure track alongside networking: P4 GUW
      multivariate towers, P6 exact-coefficient towers, and P11
      tower-based user numerics, gated by `ad-exact` and `ad-numerics`.

---

## v1.5-intelligence (August 2026) - PLANNED

**Focus:** Neural and symbolic computation flow bidirectionally.

Informed by the [Neuro-Symbolic Architecture](docs/future/NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md).

- [ ] Symbol embeddings (learnable vector representations of KB symbols)
- [ ] Soft unification (differentiable similarity — gradients flow through matching)
- [ ] LSTM and GRU cells (standard recurrent neural architectures)
- [ ] Differentiable logic programs (gradients flow through rule application)
- [ ] Attention over knowledge base (neural query mechanism over symbolic facts)
- [ ] Gradient estimators for discrete operations (Gumbel-Softmax, straight-through)
- [ ] High-order AD for neuro-symbolic ML: P5 reverse-over-Taylor, P7
      tensor-valued towers, and P9 differentiable control flow provide
      the substrate for differentiable logic and recurrent/neural
      workloads, gated by `ad-reverse-highorder`, `ad-tensor-highorder`,
      and `ad-control-flow`.
- [ ] Checkpointed high-order reverse begins here as P10, with the
      memory-scaling work continuing into v1.7 synthesis.

---

## v1.6-reasoning (September 2026) - PLANNED

**Focus:** Make the logic engine production-grade.

- [ ] Backward chaining inference (Prolog-style goal-directed proof search with backtracking)
- [ ] Forward chaining inference (production rules with fixed-point derivation)
- [ ] Constraint solving (finite domain constraints, SAT solver integration)
- [ ] Knowledge graphs (RDF-style triple store with SPO/POS/OSP indexing)
- [ ] Knowledge graph embeddings (entity-relation-entity triples as learnable vectors)
- [ ] Sparse high-order AD tensors (P12) for knowledge-graph and
      relational workloads, gated by `ad-sparse` against dense recovery
      on tractable cases.

---

## v1.7-synthesis (October 2026) - PLANNED

**Focus:** Programs that write and improve programs.

- [ ] Neural-guided program search (beam search with neural scoring for candidate ranking)
- [ ] Type-directed synthesis holes (`??` syntax — compiler searches for well-typed completions)
- [ ] Graph Neural Networks (message passing, neighborhood aggregation, graph attention)
- [ ] Synthesis from input-output examples (inductive programming)
- [ ] Neural theorem provers (neural heuristic guides symbolic proof search)

---

## v1.8-platform (November 2026) - PLANNED

**Focus:** Eshkol runs on everything, controls everything.

Informed by the [Multimedia System Architecture](docs/future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md).

- [ ] Cross-platform windowing (X11/Wayland, Cocoa, Win32)
- [ ] Event system (keyboard, mouse, touch, window events)
- [ ] Real-time audio (CoreAudio, ALSA, WASAPI with callback-based I/O)
- [ ] MIDI input/output for instrument control
- [ ] Vulkan Compute for cross-platform GPU (beyond Metal/CUDA)
- [ ] Multi-GPU support (device selection, peer-to-peer transfer)
- [ ] Embedded cross-compilation (ARM bare-metal, RISC-V)

---

## v1.9-types (December 2026) - PLANNED

**Focus:** The type system becomes a proof system.

- [ ] Full dependent type enforcement (compile-time errors, not just warnings)
- [ ] Refinement types (`(Refine Integer (> x 0))` with SMT solver integration)
- [ ] Effect types (tracking `Pure`, `IO`, `State`, `Exception` at the type level)
- [ ] Algebraic effects and handlers (structured side-effect management)
- [ ] Row polymorphism for records (structural subtyping)
- [ ] Higher-rank types (rank-2 polymorphism for combinators)
- [ ] Session types for communication protocols

---

## v2.0-starlight (Q1 2027) - RESEARCH

**Focus:** Quantum computing meets formal verification.

Leverages OALR linear types (no-cloning theorem) and AD (variational circuits).

### Quantum Type System
- [ ] Qubit type with linear resource tracking (no-cloning enforced at compile time)
- [ ] Quantum register types `qreg<n>` with compile-time dimension
- [ ] `define-quantum-region` scoping for qubit allocation and deallocation

### Quantum Operations
- [ ] Gate primitives: H, CNOT, Rz, T, S, SWAP, Toffoli, arbitrary unitaries
- [ ] Measurement with classical outcome
- [ ] Circuit compilation and optimization (gate fusion, qubit mapping)
- [ ] AD integration for variational algorithms (parameter-shift rule)

### Hybrid Classical-Quantum
- [ ] Variational Quantum Eigensolver (VQE)
- [ ] Quantum Approximate Optimization Algorithm (QAOA)
- [ ] Quantum machine learning (parameterized circuits with AD)
- [ ] Integration with Moonlab quantum simulator

### Formal Verification
- [ ] Integration with proof assistants (Lean) for certified compilation
- [ ] Quantitative type theory for unified linear/quantum resource tracking
- [ ] Formal verification of automatic differentiation via Taylor
      models (P8), gated by `ad-validated-bounds` for sound enclosures
      and tightening with order.

---

## Release Timeline

| Version | Date | Theme | Key Deliverables |
|---------|------|-------|-----------------|
| **v1.1.13** | Apr 2026 | Accelerate | Windows ARM64, 16-lane release matrix, VM closure fixes, mobile site |
| **v1.2** | May 2026 | Scale | Model serialization, Python bindings, image I/O |
| **v1.3** | Jun 2026 | Evolve | R7RS libraries, string interpolation, PGO; arbitrary-order AD P1–P3 (closes ESH-0118) |
| **v1.4** | Jul 2026 | Connection | Networking, TLS, event loop, linear resource types; AD substrate P4/P6/P11 (GUW multivariate, exact-coefficient, user numerics) |
| **v1.5** | Aug 2026 | Intelligence | Symbol embeddings, differentiable logic, LSTM/GRU; high-order AD P5/P7/P9/P10 (reverse-over-Taylor, tensor towers, differentiable control flow) |
| **v1.6** | Sep 2026 | Reasoning | Backward chaining, constraint solving, knowledge graphs; sparse high-order AD tensors (P12) |
| **v1.7** | Oct 2026 | Synthesis | Neural-guided search, program synthesis, GNN |
| **v1.8** | Nov 2026 | Platform | Windowing, audio, Vulkan, embedded targets |
| **v1.9** | Dec 2026 | Types | Dependent types, effects, algebraic effects, session types |
| **v2.0** | Q1 2027 | Starlight | Quantum types, VQE/QAOA, formal verification; validated AD via Taylor models (P8) |

> **Arbitrary-order AD (Taylor-tower) track** threads phases P0–P12 through the
> version themes above as an enabling substrate — see
> [`docs/AD_CAMPAIGN.md`](docs/AD_CAMPAIGN.md) for the full phase → version →
> ICC-gate map. Each phase is gated by an `ad-*` ICC oracle criterion.

---

## Component Status

### Core Compiler
- [x] Parser - Complete
- [x] Type Checker - Complete
- [x] LLVM Backend - Complete (34,928 lines)
- [x] Module System - Complete
- [x] Macro System - Complete

### Automatic Differentiation
- [x] Forward Mode - Complete
- [x] Reverse Mode - Complete
- [x] Nested Gradients - Complete (32-level tape stack)
- [x] Vector Calculus - Complete (8 operators)

### Memory Management
- [x] Arena Allocation - Complete
- [x] OALR System - Complete
- [x] Ownership Tracking - Complete
- [x] Escape Analysis - Complete

### Standard Library (v1.1)
- [x] Core Functions (60+ list ops, 30+ string utils) - Complete
- [x] Math Library (linear algebra, statistics, ODE solvers) - Complete
- [x] Signal Processing (FFT, filters, window functions) - Complete
- [x] ML Library (optimizers, activations, normalization) - Complete
- [x] Web Platform (80+ DOM API functions, WASM target) - Complete
- [x] JSON/CSV/Base64 Support - Complete

### Development Tools
- [x] REPL with JIT (stdlib preloading, cross-eval persistence) - Complete
- [x] Compiler (eshkol-run, AOT + script mode) - Complete
- [x] Package Manager (eshkol-pkg, TOML manifest) - Complete
- [x] LSP Server (diagnostics, completion, hover) - Complete
- [x] VSCode Extension (syntax highlighting, LSP client) - Complete
- [x] Test Suite (35 suites, 438 tests) - Complete

### v1.1-accelerate (Complete)
- [x] XLA Backend (StableHLO/MLIR + LLVM-direct) - Complete
- [x] GPU Acceleration (Metal SF64 + CUDA cuBLAS) - Complete
- [x] SIMD Vectorization (SSE/AVX/NEON) - Complete
- [x] Parallel Primitives (work-stealing thread pool) - Complete
- [x] Exact Arithmetic (bignums + rationals + complex) - Complete
- [x] Consciousness Engine (logic, inference, workspace — 22 builtins) - Complete
- [x] ML Framework (75+ builtins: activations, losses, optimizers, CNN, transformers) - Complete
- [x] Signal Processing (FFT, filters, window functions) - Complete
- [x] R7RS Extensions (call/cc, dynamic-wind, bytevectors) - Complete

### v1.1.12 Additions
- [x] Production Bytecode VM (555+ builtins, 176/176 tests, dual number AD) - Complete
- [x] eshkol.ai Website (Eshkol→WASM, browser REPL, interactive tutorials) - Complete
- [x] GitHub Pages Deployment - Complete

### v1.1.13 Additions
- [x] Native Windows ARM64 (VS 2022 + ClangCL + LLVM 21 aarch64) - Complete
- [x] 16-lane release matrix with per-arch LLVM SDK caching - Complete
- [x] VM closure bug fixes (named-let nested closure PC + native 252 upvalue relay) - Complete
- [x] Windows setjmp hardening (x64 frameaddress, ARM64 sponentry) - Complete
- [x] Mobile-responsive website + browser REPL error display - Complete

### Planned (v1.2+)
- [ ] Model Serialization + Python Bindings — v1.2
- [ ] R7RS Library System + String Interpolation — v1.3
- [ ] Networking + Linear Resource Types — v1.4
- [ ] Neuro-Symbolic Bridge — v1.5
- [ ] Backward Chaining + Knowledge Graphs — v1.6
- [ ] Program Synthesis + Neural Search — v1.7
- [ ] Platform Abstraction (windows, audio, embedded) — v1.8
- [ ] Advanced Type Theory (dependent, effects, algebraic) — v1.9
- [ ] Quantum Computing + Formal Verification — v2.0

---

## Research Directions

**Active Research:**
- Polyhedral optimization for nested tensor loops (v1.3)
- Linear type systems for hardware resources (v1.4)
- Neuro-symbolic bridging — differentiable symbolic operations (v1.5)
- Effect systems for purity tracking and algebraic effects (v1.3/v2.0)

**Exploratory Research:**
- Quantum machine learning — AD through parameterized quantum circuits
- Probabilistic programming with exact inference via factor graphs
- Formal verification of automatic differentiation correctness
- Hardware-software co-design for quantum-classical hybrid systems
- Self-improving programs via gradient descent on code embeddings

---

## Community Engagement

**Open Source Development:**
- GitHub repository with MIT license
- Active issue tracking and PR reviews
- Quarterly release cycle
- Community contribution guidelines

**Academic Partnerships:**
- University curriculum integration
- Research collaborations
- Conference presentations
- Student project sponsorship

**Enterprise Support:**
- Professional consulting
- Custom feature development
- Training and workshops
- Priority support

---

## How to Contribute

We welcome contributions in all areas:

**Core Development:**
- Implement planned features
- Optimize existing code
- Fix bugs and issues
- Improve test coverage

**Research:**
- Explore new AD techniques
- Investigate type system extensions
- Study memory management innovations
- Publish findings

**Documentation:**
- Improve user guides
- Write tutorials
- Create examples
- Update specifications

**Ecosystem:**
- Develop libraries
- Create tools
- Build integrations
- Share use cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

---

*Last Updated: July 2026*

*The arbitrary-order automatic-differentiation (Taylor-tower) campaign — phases
P0–P12, spanning core high-order AD, exact-coefficient and tensor-valued towers,
GUW multivariate recovery, differentiable control flow, and validated Taylor
models — is threaded through the version themes above rather than run as a
separate release ladder. It is the differentiable-programming substrate for the
neuro-symbolic (v1.5–v1.7) and quantum/formal-verification (v2.0) arc. See
[`docs/AD_CAMPAIGN.md`](docs/AD_CAMPAIGN.md).*

*Eshkol v1.1-accelerate is complete with 47/47 roadmap items delivered plus the v1.1.12 and v1.1.13 additions (production VM, web platform, browser AD, Windows ARM64, mobile site). The roadmap progresses through data & deployment (v1.2-scale), language maturity (v1.3-evolve), networking & resources (v1.4-connection), neuro-symbolic intelligence (v1.5-intelligence), symbolic reasoning (v1.6-reasoning), program synthesis (v1.7-synthesis), platform & hardware (v1.8-platform), advanced type theory (v1.9-types), and quantum computing with formal verification (v2.0-starlight).*
