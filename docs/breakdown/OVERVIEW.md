# Eshkol Language Overview

## Table of Contents

- [Core Philosophy and Design Principles](#core-philosophy-and-design-principles)
- [High-Level Architecture](#high-level-architecture)
- [Key Features](#key-features)
- [v1.1-accelerate Feature Summary](#v11-accelerate-feature-summary)
- [When to Use Eshkol](#when-to-use-eshkol)
- [Comparison with Other Languages](#comparison-with-other-languages)
- [Production Readiness](#production-readiness)

---

## Core Philosophy and Design Principles

Eshkol is a **compiled programming language** targeting scientific computing, machine learning, and cognitive architectures. It reconciles three traditionally incompatible goals: Lisp's homoiconicity and functional purity, systems programming's deterministic memory management, and numerical computing's automatic differentiation requirements. Version 1.1-accelerate extends the core with GPU acceleration, parallel primitives, a consciousness engine, exact arithmetic, signal processing, first-class continuations, and a 75+ builtin ML framework.

### Architectural Constraints

The language design emerged from hard constraints:

**Memory determinism without garbage collection.** Scientific computing and real-time systems cannot tolerate GC pauses. Rather than adopting Rust's ownership system (which conflicts with Lisp's data-as-code philosophy), Eshkol implements OALR — a region-based system where heap objects carry 8-byte headers encoding subtype and lifecycle metadata. All allocations flow through a global arena with 8KB block granularity, providing O(1) amortized allocation with batch deallocation. The consolidation of 8 pointer types into 2 supertype categories (HEAP_PTR, CALLABLE) with subtype headers freed type tag space for the 7 new heap subtypes introduced in v1.1 (substitution, fact, knowledge base, factor graph, workspace, promise, continuation).

**Automatic differentiation as a first-class language feature.** Most AD systems bolt onto existing languages via operator overloading (C++) or metaprogramming (Python). Eshkol integrates AD into the type system and compiler:
- **Symbolic mode**: AST rewriting at compile-time using 12 differentiation rules
- **Forward mode**: 16-byte dual numbers `{value, derivative}` propagated through operators
- **Reverse mode**: Computational graph with 16 AD node types, 32-level tape stack for nested gradients

The `vref` operator is AD-aware: when extracting from tensors during gradient computation, it creates AD nodes; otherwise it's a simple pointer dereference. This context-sensitivity is achieved through runtime type inspection of closure arguments.

**Gradual typing without compromising performance.** The HoTT-inspired type checker performs inference via constraint generation and unification, but type mismatches emit warnings rather than errors. This preserves Scheme's exploratory programming model while enabling aggressive optimization when types are statically known. The 16-byte tagged values store an 8-bit type field; when the type is known at compile time, the compiler generates untagged LLVM IR, eliminating tagging overhead entirely.

**Cognitive computing as a compiler primitive.** v1.1 introduces a consciousness engine combining three theoretical frameworks — Robinson's unification (1965), Friston's Free Energy Principle (2010), and Baars' Global Workspace Theory (1988) — as 22 compiler-level builtins. This enables agent-based reasoning systems where logic programming, probabilistic inference, and attention-based module competition are first-class language operations, not library add-ons.

### Design Principles

1. **Homoiconicity as a compiler optimization target** — Lambda S-expressions are preserved in 40-byte closure structures, enabling runtime introspection and compilation. The lambda registry maps function pointers to their source representations. With `eval` support, programs can construct, transform, and execute code at runtime.

2. **Tagged values as the universal representation** — Every value is a 16-byte `{type:u8, flags:u8, reserved:u16, padding:u32, data:u64}` structure. Immediate values (integers, floats, booleans) store data inline; pointers store 64-bit addresses to objects with prepended headers. This uniformity simplifies the compiler but requires careful attention to alignment and cache behavior.

3. **Type consolidation via object headers** — The pointer consolidation eliminates specific pointer types (CONS_PTR, STRING_PTR, etc.) in favor of polymorphic HEAP_PTR and CALLABLE types. The 8-byte header at offset -8 from the data pointer stores the specific subtype (18 subtypes as of v1.1). This trades one pointer dereference for extensible type space.

4. **Modular code generation through callbacks** — The LLVM backend delegates to 21+ specialized modules via `std::function` callbacks. This inverts the typical dependency graph (modules call into main codegen rather than vice versa), enabling parallel development and incremental testing.

5. **Compilation to LLVM IR, not C** — The implementation targets LLVM IR directly, providing access to LLVM's optimization infrastructure (function inlining, loop vectorization, dead code elimination) and its comprehensive backend support for x86-64, ARM64, WebAssembly, and other architectures.

---

## High-Level Architecture

### Compilation Pipeline

The compiler executes a 5-phase pipeline:

**Phase 1: Macro Expansion** (861 lines in [macro_expander.cpp](lib/frontend/macro_expander.cpp))

Hygienic macro expansion via `syntax-rules` pattern matching. The system supports ellipsis (`...`) for repetition, nested patterns, and scope-safe renaming. Several R7RS derived forms — `case-lambda`, `parameterize`, `cond-expand`, `define-record-type` — are transformed during this phase or the subsequent parse phase.

```scheme
;; Input: (when (> x 0) (display x))
;; Pattern: (when test body ...)
;; Template: (if test (begin body ...) #f)
;; Output: (if (> x 0) (begin (display x)) #f)
```

**Phase 2: S-Expression Parsing** (7,540 lines in [parser.cpp](lib/frontend/parser.cpp))

Builds an AST from S-expressions. The parser is a recursive descent processor that handles:
- 94 operation types (see `eshkol_op_t` enum in [eshkol.h](inc/eshkol/eshkol.h))
- Variadic parameter encoding in lambda/define
- HoTT type annotation attachment to AST nodes
- Internal define → `letrec*` transformation (consecutive defines at body start only)
- `delay`/`delay-force` → promise constructor desugaring
- `define-record-type` → vector operation transformation
- Line/column tracking for error messages

Each AST node includes a `uint32_t inferred_hott_type` field packed as `[TypeId:16][universe:8][flags:8]`, set by the type checker.

**Phase 3: HoTT Type Checking** (1,999 lines in [type_checker.cpp](lib/types/type_checker.cpp))

Hindley-Milner-style inference with universe hierarchy extensions. The algorithm:

1. Generate constraints from AST traversal
2. Unify constraints using Robinson's algorithm
3. Apply type substitutions to AST
4. Emit warnings for unresolved constraints

Unlike traditional type checkers, Eshkol's is **non-blocking**: type errors don't prevent compilation. This enables rapid prototyping but requires runtime type guards for safety (via tagged values).

**Phase 4: LLVM IR Generation** (34,928 lines in [llvm_codegen.cpp](lib/backend/llvm_codegen.cpp) + 21 modules totaling ~232,000 lines)

Translates ASTs to LLVM IR. The modular architecture distributes code generation across specialized modules:

| Module | Lines | Responsibility |
|:---|---:|:---|
| [llvm_codegen.cpp](lib/backend/llvm_codegen.cpp) | 34,928 | Main codegen, dispatch, builtins |
| [tensor_codegen.cpp](lib/backend/tensor_codegen.cpp) | 19,187 | Tensor ops, ML builtins (75+ functions) |
| [autodiff_codegen.cpp](lib/backend/autodiff_codegen.cpp) | 3,694 | Forward/reverse mode AD |
| [string_io_codegen.cpp](lib/backend/string_io_codegen.cpp) | 2,975 | String, I/O, JSON, CSV operations |
| [parallel_llvm_codegen.cpp](lib/backend/parallel_llvm_codegen.cpp) | 2,401 | Work-stealing parallelism codegen |
| [arithmetic_codegen.cpp](lib/backend/arithmetic_codegen.cpp) | 2,332 | Numeric ops, bignum, rational, complex |
| [collection_codegen.cpp](lib/backend/collection_codegen.cpp) | 2,139 | Vector, list, hash table operations |
| [system_codegen.cpp](lib/backend/system_codegen.cpp) | 1,192 | System, environment, time, process |
| [binding_codegen.cpp](lib/backend/binding_codegen.cpp) | 1,178 | let/let\*/letrec/letrec\* with TCO |
| [thread_pool.cpp](lib/backend/thread_pool.cpp) | 1,131 | Work-stealing thread pool |
| [tensor_backward.cpp](lib/backend/tensor_backward.cpp) | 1,078 | Backward-mode AD gradients |
| [call_apply_codegen.cpp](lib/backend/call_apply_codegen.cpp) | 960 | Function calls, apply, partial application |
| [blas_backend.cpp](lib/backend/blas_backend.cpp) | 890 | BLAS dispatch, GPU cost model |
| [tagged_value_codegen.cpp](lib/backend/tagged_value_codegen.cpp) | 822 | Tagged value pack/unpack |
| [control_flow_codegen.cpp](lib/backend/control_flow_codegen.cpp) | 800 | if/cond/case/match/call-cc |
| [map_codegen.cpp](lib/backend/map_codegen.cpp) | 784 | map/for-each/fold with closures |
| [parallel_codegen.cpp](lib/backend/parallel_codegen.cpp) | 705 | parallel-map/fold/filter/for-each |
| [homoiconic_codegen.cpp](lib/backend/homoiconic_codegen.cpp) | 601 | Code-as-data, eval |
| [hash_codegen.cpp](lib/backend/hash_codegen.cpp) | 603 | Hash operations |
| [complex_codegen.cpp](lib/backend/complex_codegen.cpp) | 499 | Complex number ops (Smith's formula) |
| [tail_call_codegen.cpp](lib/backend/tail_call_codegen.cpp) | 497 | TCO transformation |

Additional backend components:
- **XLA/StableHLO**: 4,003 lines across 5 files — tensor compilation via MLIR
- **GPU/Metal**: 8,888 lines — Metal compute, SF64 software float64, CUDA stubs

Key code generation challenges:
- **Tagged value representation**: Map Eshkol's 16-byte tagged values to LLVM's `{i8, i8, i16, i32, i64}` struct type (5 fields — type, flags, reserved, padding, data at index 4)
- **Type dispatch**: Generate runtime type switches for polymorphic operations
- **Closure compilation**: Pack captured variables into arena-allocated environment structures
- **AD mode switching**: Generate different IR paths depending on whether AD is active
- **PHI node avoidance**: Loops calling `codegenClosureCall` use alloca+store/load instead of PHI nodes (closure calls create 15+ basic blocks, breaking PHI semantics)

**Phase 5: LLVM Optimization + Native Codegen**

Eshkol applies LLVM's standard optimization pipeline:
- Instruction combining and reassociation
- LICM (Loop-Invariant Code Motion)
- GVN (Global Value Numbering)
- Function inlining (threshold: 275 instructions)
- Loop unrolling and auto-vectorization
- Target-specific code generation (x86-64, ARM64, WebAssembly)

### Runtime Architecture

**Global Arena** ([arena_memory.cpp](lib/core/arena_memory.cpp), 4,972 lines):
- Single allocator for all heap objects
- 8KB minimum block size, doubling growth strategy
- No individual `free()`; memory reclaimed via arena reset
- Allocation is O(1) bump-pointer until block exhaustion
- 512MB stack via linker flags for deep recursion

**Object Header System** (8 bytes prepended to all heap data):
```c
struct {
    uint8_t subtype;     // 18 subtypes: cons, string, vector, tensor, closure,
                         //   continuation, promise, substitution, fact, KB,
                         //   factor graph, workspace, ...
    uint8_t flags;       // Linear, borrowed, shared, marked, exact
    uint16_t ref_count;  // For shared objects (0 = not ref-counted)
    uint32_t size;       // Object size excluding header
};
```

**Closure Environment Encoding** (40-byte closure structures):

Closures capture variables by storing **pointers** (not values) to the captured bindings, enabling mutable captures:

```scheme
(let ((counter 0))
  (define inc (lambda () (set! counter (+ counter 1)) counter))
  (inc)  ; Returns 1
  (inc)) ; Returns 2 (mutates same counter)
```

**Consciousness Engine Runtime** ([logic.cpp](lib/core/logic.cpp) 805 lines, [inference.cpp](lib/core/inference.cpp) 912 lines, [workspace.cpp](lib/core/workspace.cpp) 308 lines):

Three C runtime libraries implementing unification with substitutions, factor graph belief propagation, and global workspace softmax competition. LLVM codegen dispatches to these via tagged value calling conventions.

**Computational Tape (for reverse-mode AD)**:

A dynamic array of AD nodes allocated during forward pass, topologically sorted for backward pass. The tape stack depth of 32 enables computing derivatives-of-derivatives for meta-learning and Hessian calculations.

---

## Key Features

### Arena-Based Memory Management (OALR)

OALR (Ownership-Aware Lexical Regions) reconciles functional programming's desire for immutability with systems programming's need for explicit resource management:

- **Default mode:** Objects are arena-allocated with no lifetime tracking. Sound because arenas have well-defined lexical scope.
- **Linear types:** `(owned ...)` marks resources for exactly-once consumption via `(move ...)`. Prevents resource leaks at compile time.
- **Borrowing:** `(borrow value body)` provides temporary read-only access with compile-time safety checks.
- **Shared references:** `(shared ...)` activates reference counting via the header's 16-bit ref_count field.

### Type System

The type system operates on three levels:

**Runtime level:** 16-byte tagged values with an 8-bit type field. Types 0-7 store data directly (int64, double, bool, char, symbol, nil, complex); types 8-9 are consolidated pointers requiring header inspection for subtype. 18 heap subtypes and 3 callable subtypes as of v1.1.

**HoTT level:** Type expressions representing primitives, compounds (list, vector, tensor, arrow), polymorphic types (forall quantification), and a universe hierarchy (U_0 for values, U_1 for types, U_2 for type operators). Type checking generates constraints and solves via unification; failure produces warnings, not errors.

**Dependent level:** Tensor dimensions and array bounds are tracked as compile-time values when statically known. The dependent type checker validates dimension compatibility for tensor operations.

### Automatic Differentiation

Three modes integrated into the compiler:

**Symbolic mode** transforms the AST at compile time using Leibniz's chain rule. Produces optimal code with zero runtime overhead when the function is syntactically known.

**Forward mode** uses dual numbers encoding `x + x'ε` where `ε² = 0`. Efficient for functions R → R^n (one input, many outputs).

**Reverse mode** separates forward pass (compute values) from backward pass (compute gradients). The tape records all operations as AD nodes. Efficient for functions R^n → R (many inputs, one output) — the dominant pattern in machine learning. The 32-level tape stack supports second-order optimization methods (Hessians, natural gradient descent, K-FAC).

### Closure Implementation

40-byte structures: `{func_ptr:u64, env:ptr, sexpr_ptr:u64, return_type:u8, input_arity:u8, flags:u8, reserved:u8, hott_type_id:u32}`. Captures store pointers to variables (not values), enabling mutable state across invocations.

### Vector vs. Tensor Distinction

**Scheme vectors** (`HEAP_SUBTYPE_VECTOR`): heterogeneous, 16 bytes per element, any type. Operations: `vector`, `vector-ref`, `vector-set!`, `vector-length`, `vector-for-each`, `vector-map`.

**Tensors** (`HEAP_SUBTYPE_TENSOR`): homogeneous doubles stored as int64 bit patterns (8 bytes each). Multidimensional with shape metadata. 30+ operations including `tensor-dot`, `tensor-add`, `tensor-transpose`, `reshape`, `conv2d`, `matmul`. AD-aware: `vref` creates computational graph nodes during gradient computation.

---

## v1.1-accelerate Feature Summary

v1.1-accelerate adds eight major feature systems to the v1.0-foundation core:

### Machine Learning Framework (75+ Builtins)

A complete ML framework implemented as compiler-level builtins in [tensor_codegen.cpp](lib/backend/tensor_codegen.cpp) (19,187 lines), with SIMD acceleration and automatic GPU dispatch:

- **Activations** (16): relu, relu6, sigmoid, tanh, gelu, swish, mish, softmax, log-softmax, softplus, softsign, leaky-relu, prelu, elu, selu, celu
- **Loss functions** (14): mse-loss, mae-loss, cross-entropy-loss, bce-loss, huber-loss, kl-div-loss, hinge-loss, smooth-l1-loss, focal-loss, triplet-loss, contrastive-loss, label-smoothing-loss, cosine-embedding-loss
- **Optimizers** (5+3): sgd-step, adam-step, adamw-step, rmsprop-step, adagrad-step + zero-grad!, clip-grad-norm!, check-grad-health
- **Weight initializers** (5): xavier-uniform!, xavier-normal!, kaiming-uniform!, kaiming-normal!, lecun-normal!
- **LR schedulers** (4): linear-warmup-lr, step-decay-lr, exponential-decay-lr, cosine-annealing-lr
- **CNN layers** (7): conv1d, conv2d, conv3d, max-pool2d, avg-pool2d, batch-norm, layer-norm
- **Transformer operations** (8): scaled-dot-attention, multi-head-attention, positional-encoding, rotary-embedding, causal-mask, padding-mask, feed-forward, embedding
- **Data loading** (6): make-dataloader, dataloader-next, dataloader-reset!, dataloader-length, dataloader-has-next?, train-test-split

All ML builtins integrate with reverse-mode AD — calling `gradient` on any composition of these functions produces exact gradients.

### Consciousness Engine (22 Builtins)

Three theoretical frameworks unified as compiler primitives:

**Logic Programming** (Robinson's unification, 1965): `unify`, `walk`, `make-substitution`, `make-fact`, `make-kb`, `kb-assert!`, `kb-query`, `logic-var?`, `substitution?`, `kb?`, `fact?`

**Active Inference** (Friston's Free Energy Principle, 2010): `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `fg-update-cpt!`, `free-energy`, `expected-free-energy`, `factor-graph?`

**Global Workspace Theory** (Baars, 1988): `make-workspace`, `ws-register!`, `ws-step!`, `workspace?`

Implementation: [logic.cpp](lib/core/logic.cpp), [inference.cpp](lib/core/inference.cpp), [workspace.cpp](lib/core/workspace.cpp) (runtime); [llvm_codegen.cpp](lib/backend/llvm_codegen.cpp) (codegen dispatch).

### GPU Acceleration

Adaptive dispatch system with cost model calibration:

| Backend | Peak Performance | Overhead | Dispatch Range |
|:---|---:|---:|:---|
| SIMD (vectorized) | 25 GFLOPS | ~0 | ≤16 elements |
| cBLAS (Apple Accelerate AMX) | 1,100 GFLOPS | 5 us | 17 to ~1B elements |
| Metal GPU (SF64 software float64) | 200 GFLOPS | 200 us | >1B elements |

**SF64 (Software Float64):** Metal GPUs lack native float64 — SF64 emulates double precision using double-double arithmetic (two 32-bit mantissas combined for ~100-bit effective precision). Implemented in [metal_softfloat.h](lib/backend/gpu/metal_softfloat.h) (4,076 lines) and [gpu_memory.mm](lib/backend/gpu/gpu_memory.mm) (3,786 lines).

**Cost model dispatch** ([blas_backend.cpp](lib/backend/blas_backend.cpp)): Automatically selects the optimal backend based on tensor size and compute intensity. Configurable via environment variables (`ESHKOL_GPU_PRECISION`, `ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_GPU_PEAK_GFLOPS`).

### Parallel Computing

Work-stealing thread pool with parallel higher-order functions:

- `parallel-map` — applies function across vector elements in parallel
- `parallel-fold` — associative reduction with automatic work distribution
- `parallel-filter` — parallel predicate-based selection
- `parallel-for-each` — parallel side-effecting iteration

Implementation: [parallel_codegen.cpp](lib/backend/parallel_codegen.cpp) (705 lines), [parallel_llvm_codegen.cpp](lib/backend/parallel_llvm_codegen.cpp) (2,401 lines), [thread_pool.cpp](lib/backend/thread_pool.cpp) (1,131 lines). Worker functions use `LinkOnceODRLinkage` for safe parallel compilation.

### Signal Processing

FFT-based spectral analysis and digital filtering:

- **FFT/IFFT**: Cooley-Tukey radix-2 DIT algorithm, O(N log N)
- **Window functions**: Hamming, Hann, Blackman, Kaiser
- **Filter design**: Butterworth low-pass, high-pass, band-pass
- **FIR/IIR filters**: Direct-form implementation
- **Convolution**: Linear and fast (FFT-based) convolution

Implementation: [lib/signal/fft.esk](lib/signal/fft.esk), [lib/signal/filters.esk](lib/signal/filters.esk).

### Exact Arithmetic (Full Numeric Tower)

R7RS-compliant numeric tower with automatic precision promotion:

- **int64**: 64-bit signed integers (exact, immediate in tagged value)
- **bignum**: Arbitrary-precision integers (exact, heap-allocated, automatic overflow promotion)
- **rational**: Exact fractions as bignum pairs (always reduced via GCD)
- **double**: IEEE 754 64-bit floats (inexact)
- **complex**: Heap-allocated `{real:f64, imag:f64}` with Smith's formula division

Exactness tracking via `ESHKOL_FLAG_EXACT` in the tagged value flags byte. R7RS semantics: exact + exact = exact, exact + inexact = inexact.

### First-Class Continuations

- `call/cc` / `call-with-current-continuation` — single-shot continuations via setjmp/longjmp
- `dynamic-wind` — before/after thunks with proper unwinding on non-local exit
- `guard`/`raise` — R7RS exception handling with cond-style clauses
- `with-exception-handler` — low-level exception handler installation
- `delay`/`force`/`make-promise`/`delay-force` — full R7RS promise system with memoization

### Web Platform (WASM)

Compile Eshkol programs to WebAssembly with 73 DOM/Canvas/Event API bindings:

```bash
eshkol-run --wasm app.esk            # compile to WASM
```

The [lib/web/http.esk](lib/web/http.esk) module provides browser interop via an integer handle system:
- DOM element creation, tree manipulation, attributes, CSS classes
- Event handling (click, keyboard, mouse, timer events)
- Canvas 2D drawing API (rectangles, paths, arcs, text, transforms)
- Fetch API for HTTP requests
- LocalStorage for client-side persistence
- Window dimensions, scroll position, navigation

### Eval and Environments

Full R7RS `eval` with environment support ([llvm_codegen.cpp:16811](lib/backend/llvm_codegen.cpp#L16811)):

```scheme
(eval '(+ 1 2))                       ; => 3
(eval '(map car '((1 2) (3 4)))
      (scheme-report-environment 7))   ; => (1 3)
```

Three R7RS environment constructors: `interaction-environment`, `scheme-report-environment`, `null-environment`.

---

## When to Use Eshkol

### Ideal Use Cases

**Numerical optimization and scientific simulation:** The combination of automatic differentiation, efficient tensor operations, and deterministic memory makes Eshkol suitable for iterative algorithms (gradient descent, Newton's method, MCMC sampling) where garbage collection pauses would disrupt convergence.

**Neural network training and inference:** 75+ ML builtins with reverse-mode AD provide a complete training framework. Compiler-level integration eliminates Python interpreter overhead. The SIMD → cBLAS → GPU dispatch chain optimizes tensor operations across hardware.

**Cognitive architectures and agent systems:** The consciousness engine's 22 builtins enable logic-based reasoning (unification, knowledge bases), probabilistic inference (factor graphs, belief propagation, free energy minimization), and attention-based module competition (global workspace). These are first-class language operations, not library calls.

**Real-time signal processing:** FFT, window functions, and digital filters operate on tensors with zero GC pauses. Combined with parallel primitives for multi-channel processing.

**Embedded machine learning:** The absence of garbage collector, deterministic memory footprint, and native compilation make Eshkol viable for deploying ML models on resource-constrained devices where Python/TensorFlow are infeasible.

**Mathematical research and symbolic computing:** Homoiconicity enables symbolic manipulation of mathematical expressions. `eval` enables runtime code construction and execution. The lambda registry preserves source S-expressions in closures for introspection.

**Web applications:** WASM compilation with 73 DOM/Canvas bindings enables interactive browser applications. Combined with the ML framework for client-side inference.

### Limitations

**Not ideal for:**
- Enterprise web backends (no native HTTP server; web support is WASM/browser-only)
- Mobile applications (compilation targets desktop/server/WASM, no mobile SDK)
- Native GUI applications (no desktop GUI toolkit; graphics limited to WASM Canvas)
- Enterprise data processing (SQL/database integration not yet available)

**Ecosystem maturity:**
- Package ecosystem is growing (40 stdlib modules, eshkol-pkg package manager, git-based registry)
- IDE support includes LSP server (1,018 lines) with completions, hover, go-to-definition, and a VSCode extension
- Debugging support via `--dump-ir` and `--dump-ast` flags, plus REPL JIT for interactive development
- Multi-shot continuations not supported (single-shot via setjmp/longjmp covers most use cases)

---

## Comparison with Other Languages

### Eshkol vs. Julia

Julia targets numerical computing with JIT compilation and multiple dispatch. Eshkol compiles ahead-of-time with single dispatch.

**Memory:** Julia's GC causes unpredictable pauses (problematic for real-time systems). Eshkol's arena system provides deterministic allocation/deallocation.

**AD Integration:** Julia's Zygote operates via source-to-source transformation at the Julia IR level. Eshkol's AD is integrated into the compiler, operating at AST, dual number, and computational graph levels simultaneously.

**ML framework:** Julia relies on Flux.jl (library-level). Eshkol has 75+ compiler-level ML builtins with SIMD acceleration and automatic GPU dispatch.

**Type system:** Julia's dynamic types with JIT specialization vs. Eshkol's gradual types with AOT compilation. Julia optimizes for interactive performance; Eshkol optimizes for production deployment.

**Syntax:** Julia's mathematical notation `A * B` vs. Eshkol's S-expressions `(tensor-dot A B)`. Syntax preference is subjective, but S-expressions enable true homoiconicity and `eval`.

### Eshkol vs. Python + JAX/PyTorch

Python with JAX/PyTorch provides automatic differentiation via program transformation or operator overloading.

**Performance:** Python is interpreted; JAX JIT-compiles traced functions. Eshkol compiles to native code ahead-of-time. For cold-start performance (first execution), Eshkol is faster. For iterative workloads (training loops), JAX's XLA compilation may match Eshkol after warmup.

**ML framework:** PyTorch/JAX provide extensive model libraries and pretrained weights. Eshkol's 75+ ML builtins cover the core operations (activations, losses, optimizers, CNN, transformer) but lack the breadth of PyTorch's ecosystem.

**Cognitive computing:** Neither JAX nor PyTorch offer built-in logic programming, probabilistic inference, or attention-based competition. Eshkol's consciousness engine is unique.

**Memory model:** Python's GC can trigger at any time. JAX's device arrays are manually managed but host-side allocations are GC'd. Eshkol provides full control over allocations.

**Composability:** JAX transforms work on pure functions only; side effects break tracing. Eshkol's AD handles mutable state via pointers in closure environments.

**Ecosystem:** Python/JAX/PyTorch has vastly more libraries and community support.

### Eshkol vs. Scheme (Racket/Guile/Chez)

Eshkol is Scheme with different trade-offs:

**R7RS compliance:** Eshkol implements ~95% of R7RS-small (232 of 244 procedures). Full numeric tower, continuations, exceptions, promises, eval, records, bytevectors, and macros.

**Compilation:** Racket compiles to bytecode or uses a JIT. Guile interprets or JITs. Chez Scheme compiles to native code. Eshkol compiles to native via LLVM with access to LLVM's full optimization suite.

**Continuations:** Racket/Chez provide full multi-shot continuations. Eshkol provides single-shot via setjmp/longjmp (sufficient for non-local exit, insufficient for coroutines).

**Memory:** Both Racket and Guile use precise garbage collectors. Eshkol uses arenas. For programs with predictable lifetimes (scientific simulations), arenas are faster. For programs with complex object graphs (symbolic algebra), GC may be easier.

**Unique to Eshkol:** Three-mode automatic differentiation, 75+ ML builtins, consciousness engine, GPU acceleration, tensor operations, parallel primitives, signal processing, WASM compilation.

### Eshkol vs. Rust

Both pursue memory safety without garbage collection:

**Ownership:** Rust's borrow checker enforces exclusive mutability at compile time via affine types and lifetime annotations (strong guarantees, steep learning curve). Eshkol's OALR uses arena regions with optional linear types (simpler mental model, weaker guarantees).

**Performance:** Both compile to native code. Rust's zero-cost abstractions guarantee no runtime overhead. Eshkol's tagged values add overhead unless eliminated via type inference.

**Numerical computing:** Rust has no built-in AD, tensors, or ML framework (relies on libraries like Candle, Burn). Eshkol integrates all of these at the compiler level.

**Use case alignment:** Rust targets systems programming (operating systems, browsers, databases). Eshkol targets scientific computing (simulations, machine learning, cognitive architectures).

---

## Production Readiness

Eshkol v1.1-accelerate represents a **mature, production-ready implementation** for scientific computing, machine learning, and cognitive architecture deployment.

### Implementation Scale

| Component | Lines | Files |
|:---|---:|---:|
| LLVM backend (main + modules) | 80,873 | 21 |
| XLA/StableHLO backend | 4,003 | 5 |
| GPU/Metal backend | 8,888 | 4 |
| Frontend (parser, macro, types) | 10,400 | 3 |
| Runtime (arena, logic, inference, workspace) | 14,920 | 5 |
| REPL JIT | 3,110 | 3 |
| Tools (LSP, package manager) | 1,739 | 2 |
| **Total C/C++ compiler infrastructure** | **~232,000** | **43+** |
| Standard library (.esk) | ~2,500 | 40 modules |
| Test code (.esk) | 32,120 | 434 files |

### Test Coverage

- **38 test directories** organized by feature: autodiff, bignum, closures, collections, complex, control_flow, error_handling, features, gpu, integration, io, json, lists, logic, macros, memory, migration, ml, modules, neural, numeric, parallel, parser, rational, repl, signal, stdlib, string, system, tco, types, typesystem, web, xla, benchmark, codegen
- **434 test files** with 35 automated test suites via shell scripts
- **~32,000 lines** of test code

### Tooling

- **REPL JIT** ([repl_jit.cpp](lib/repl/repl_jit.cpp), 2,062 lines): LLVM OrcJIT with stdlib preloading, 237 precompiled functions, 305 globals
- **LSP server** ([eshkol_lsp.cpp](tools/lsp/eshkol_lsp.cpp), 1,018 lines): Completions, hover, go-to-definition, diagnostics, formatting
- **VSCode extension** ([tools/vscode-eshkol/](tools/vscode-eshkol/)): Syntax highlighting, LSP integration, build tasks
- **Package manager** ([eshkol_pkg.cpp](tools/pkg/eshkol_pkg.cpp), 721 lines): eshkol-pkg init/build/run/add/clean, TOML manifests, git-based registry
- **Docker images**: Debian debug/release, Ubuntu release, CUDA, XLA
- **CI/CD**: GitHub Actions for continuous integration and release automation
- **Homebrew**: `brew install eshkol` via tap

### Competitive Technical Advantages

**For gradient-based optimization and machine learning:** Three-mode AD with 32-level nested gradient support. 75+ compiler-level ML builtins covering the complete training pipeline (init → forward → loss → backward → clip → optimize → schedule). SIMD → cBLAS → GPU automatic dispatch.

**For cognitive and agent-based systems:** 22 consciousness engine builtins implementing logic programming, active inference, and global workspace theory. No other compiled language provides these as first-class operations.

**For real-time scientific applications:** Arena allocation provides microsecond-scale latency predictability impossible with garbage collectors. Applications requiring bounded worst-case execution time (robotics control loops, real-time signal processing, edge AI inference) can deploy Eshkol code where Python/Julia/JVM languages fail determinism requirements.

**For production deployment:** Ahead-of-time compilation to native binaries eliminates Python interpreter overhead (~10-100x for control flow), JIT warmup time, and runtime dependencies.

---

## See Also

- [Getting Started](GETTING_STARTED.md) — Installation, first programs
- [Scheme Compatibility](SCHEME_COMPATIBILITY.md) — R7RS compliance deep dive
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) — LLVM backend, compilation phases
- [Type System](TYPE_SYSTEM.md) — Tagged values, HoTT types, gradual typing
- [Memory Management](MEMORY_MANAGEMENT.md) — OALR system, object headers
- [Automatic Differentiation](AUTODIFF.md) — Three AD modes, tape architecture
- [Machine Learning](MACHINE_LEARNING.md) — 75+ ML builtins deep dive
- [Consciousness Engine](CONSCIOUSNESS_ENGINE.md) — Logic, active inference, global workspace
- [GPU Acceleration](GPU_ACCELERATION.md) — Metal, SF64, cost model
- [Parallel Computing](PARALLEL_COMPUTING.md) — Work-stealing, parallel primitives
- [Signal Processing](SIGNAL_PROCESSING.md) — FFT, filters, windows
- [Exact Arithmetic](EXACT_ARITHMETIC.md) — Numeric tower, bignum, rational
- [Continuations](CONTINUATIONS.md) — call/cc, dynamic-wind, guard/raise
- [Web Platform](WEB_PLATFORM.md) — WASM, DOM API, Canvas
- [Module System](MODULE_SYSTEM.md) — require/provide, precompiled stdlib
- [REPL JIT](REPL_JIT.md) — Interactive development
- [API Reference](../API_REFERENCE.md) — Complete function reference
