# Eshkol Design Document

## v1.1-accelerate

Eshkol is a compiled programming language for scientific computing, machine learning, and cognitive architectures. It compiles Scheme (R7RS) source through LLVM to native binaries, combining Lisp's homoiconicity with deterministic arena-based memory, compiler-integrated automatic differentiation, and a consciousness engine built on unification, active inference, and global workspace theory.

---

## Language Architecture

### Compilation Pipeline

The compiler executes five phases from source to native binary:

```
Source (.esk)
     |
     v
 Macro Expansion        syntax-rules, hygienic renaming, ellipsis patterns
     |                  case-lambda, parameterize, cond-expand, define-record-type
     v
 S-Expression Parsing   Recursive descent, 94 operation types
     |                  Internal define -> letrec* transformation
     v                  HoTT type annotation attachment, line/column tracking
 Annotated AST
     |
     v
 HoTT Type Checking     Bidirectional inference (synthesis + checking)
     |                  Constraint generation, Robinson unification
     v                  Gradual: warnings not errors, non-blocking
 Typed AST
     |
     v
 LLVM IR Generation     21 specialized codegen modules (~232,000 lines)
     |                  Tagged value lowering, closure compilation, AD dispatch
     v
 LLVM Optimization      Inlining, LICM, GVN, loop unrolling, auto-vectorization
     |
     v
 Native Code            x86-64, ARM64, WebAssembly via LLVM backends
     |
     v
 Linking                Runtime library, stdlib, BLAS, Metal/CUDA (optional)
     |
     v
 Executable / Shared Library / WASM
```

Compilation command: `eshkol-run file.esk -o binary`

### Modular Code Generation

The LLVM backend delegates to 21 specialized modules via `std::function` callbacks. This inverts the typical dependency graph: modules call into main codegen rather than vice versa, enabling parallel development and incremental testing.

| Module | Lines | Responsibility |
|:---|---:|:---|
| llvm_codegen.cpp | 34,928 | Main codegen, dispatch, builtins |
| tensor_codegen.cpp | 19,187 | Tensor ops, ML builtins (168 functions) |
| autodiff_codegen.cpp | 3,694 | Forward/reverse mode AD |
| string_io_codegen.cpp | 2,975 | String, I/O, JSON, CSV operations |
| parallel_llvm_codegen.cpp | 2,401 | Work-stealing parallelism codegen |
| arithmetic_codegen.cpp | 2,332 | Numeric ops, bignum, rational, complex |
| collection_codegen.cpp | 2,139 | Vector, list, hash table operations |
| system_codegen.cpp | 1,192 | System, environment, time, process |
| binding_codegen.cpp | 1,178 | let/let\*/letrec/letrec\* with TCO |
| thread_pool.cpp | 1,131 | Work-stealing thread pool |
| tensor_backward.cpp | 1,078 | Backward-mode AD gradients |
| call_apply_codegen.cpp | 960 | Function calls, apply, partial application |
| blas_backend.cpp | 890 | BLAS dispatch, GPU cost model |
| tagged_value_codegen.cpp | 822 | Tagged value pack/unpack |
| control_flow_codegen.cpp | 800 | if/cond/case/match/call-cc |
| map_codegen.cpp | 784 | map/for-each/fold with closures |
| parallel_codegen.cpp | 705 | parallel-map/fold/filter/for-each |
| homoiconic_codegen.cpp | 601 | Code-as-data, eval |
| hash_codegen.cpp | 603 | Hash operations |
| complex_codegen.cpp | 499 | Complex number ops (Smith's formula) |
| tail_call_codegen.cpp | 497 | TCO transformation |

Additional backends: XLA/StableHLO (4,003 lines across 5 files), GPU/Metal (8,888 lines across 4 files).

---

## Core Design

### Tagged Values

Every Eshkol value is a 16-byte structure:

```
{type:u8, flags:u8, reserved:u16, padding:u32, data:u64}
```

LLVM type: `{i8, i8, i16, i32, i64}` -- five fields, with data at index 4.

Immediate types (0-7) store data inline: int64, double, bool, char, symbol, nil, complex. Pointer types (8-9) consolidate all heap references into two supertypes -- HEAP_PTR and CALLABLE -- with an 8-byte object header at offset -8 storing the specific subtype. This consolidation freed type tag space for the 7 new heap subtypes added in v1.1 (substitution, fact, knowledge base, factor graph, workspace, promise, continuation), bringing the total to 18 heap subtypes and 3 callable subtypes.

When the type is known at compile time, the compiler generates untagged LLVM IR, eliminating tagging overhead entirely.

### Memory Management (OALR)

Ownership-Aware Lexical Regions provide deterministic memory without garbage collection:

- **Arena allocator**: Single global arena, 8KB minimum blocks, O(1) bump-pointer allocation, batch deallocation via arena reset
- **Object headers**: 8 bytes prepended to all heap data -- `{subtype:u8, flags:u8, ref_count:u16, size:u32}`
- **Closures**: 40-byte structures -- `{func_ptr:u64, env:ptr, sexpr_ptr:u64, return_type:u8, input_arity:u8, flags:u8, reserved:u8, hott_type_id:u32}`. Captures store pointers (not values) enabling mutable state
- **Escape analysis**: Compiler determines stack/region/shared allocation automatically
- **Linear types**: `(owned ...)` marks resources for exactly-once consumption; `(borrow value body)` for temporary read-only access; `(shared ...)` activates reference counting
- **Stack**: 512MB via linker flags for deep recursion, configurable via `ESHKOL_STACK_SIZE`

### HoTT Type System

Homotopy Type Theory foundations with bidirectional checking:

**Runtime level**: 16-byte tagged values with 8-bit type field. Types 0-7 are immediate; types 8-9 are consolidated pointers requiring header inspection.

**HoTT level**: Primitives, compounds (list, vector, tensor, arrow), polymorphic types (forall quantification), universe hierarchy (U_0 for values, U_1 for types, U_2 for type operators). Constraint generation + Robinson unification; failure produces warnings, not errors.

**Dependent level**: Tensor dimensions and array bounds tracked as compile-time values when statically known. Dimension compatibility validated for tensor operations.

### Homoiconicity

Lambda S-expressions are preserved in 40-byte closure structures. The lambda registry maps function pointers to their source representations. Full R7RS `eval` with three environment constructors (`interaction-environment`, `scheme-report-environment`, `null-environment`) enables runtime code construction and execution. Programs can introspect, transform, and compile code at native speed.

### Automatic Differentiation

Three modes integrated into the compiler, not bolted on as a library:

**Symbolic mode**: AST rewriting at compile time using 12 differentiation rules. Zero runtime overhead when the function is syntactically known.

**Forward mode**: 16-byte dual numbers `{value, derivative}` propagated through operators. Efficient for functions R -> R^n.

**Reverse mode**: Computational graph with 16 AD node types. The tape records all operations during forward pass; backward pass computes gradients via topological sort. 32-level tape stack enables nested gradients for Hessians, natural gradient descent, and meta-learning.

The `vref` operator is AD-aware: during gradient computation it creates computational graph nodes; otherwise it is a simple pointer dereference.

---

## v1.1-accelerate Features

### Machine Learning Framework (75+ Builtins)

A complete ML framework as compiler-level builtins in tensor_codegen.cpp with SIMD acceleration and automatic GPU dispatch:

- **Activations** (16): relu, relu6, sigmoid, tanh, gelu, swish, mish, softmax, log-softmax, softplus, softsign, leaky-relu, prelu, elu, selu, celu
- **Loss functions** (14): mse-loss, mae-loss, cross-entropy-loss, bce-loss, huber-loss, kl-div-loss, hinge-loss, smooth-l1-loss, focal-loss, triplet-loss, contrastive-loss, label-smoothing-loss, cosine-embedding-loss
- **Optimizers** (5+3): sgd-step, adam-step, adamw-step, rmsprop-step, adagrad-step + zero-grad!, clip-grad-norm!, check-grad-health
- **Weight initializers** (5): xavier-uniform!, xavier-normal!, kaiming-uniform!, kaiming-normal!, lecun-normal!
- **LR schedulers** (4): linear-warmup-lr, step-decay-lr, exponential-decay-lr, cosine-annealing-lr
- **CNN layers** (7): conv1d, conv2d, conv3d, max-pool2d, avg-pool2d, batch-norm, layer-norm
- **Transformer operations** (8): scaled-dot-attention, multi-head-attention, positional-encoding, rotary-embedding, causal-mask, padding-mask, feed-forward, embedding
- **Data loading** (6): make-dataloader, dataloader-next, dataloader-reset!, dataloader-length, dataloader-has-next?, train-test-split

All ML builtins integrate with reverse-mode AD -- `gradient` on any composition of these functions produces exact gradients.

### Consciousness Engine (22 Builtins)

Three theoretical frameworks unified as compiler primitives:

**Logic Programming** (Robinson's unification, 1965): `unify`, `walk`, `make-substitution`, `make-fact`, `make-kb`, `kb-assert!`, `kb-query`, `logic-var?`, `substitution?`, `kb?`, `fact?`

**Active Inference** (Friston's Free Energy Principle, 2010): `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `fg-update-cpt!`, `free-energy`, `expected-free-energy`, `factor-graph?`

**Global Workspace Theory** (Baars, 1988): `make-workspace`, `ws-register!`, `ws-step!`, `workspace?`

Runtime implementation in logic.cpp (805 lines), inference.cpp (912 lines), workspace.cpp (308 lines). LLVM codegen dispatches via tagged value calling conventions. Heap subtypes: SUBSTITUTION=12, FACT=13, KNOWLEDGE_BASE=15, FACTOR_GRAPH=16, WORKSPACE=17. Parser supports `?x` syntax for logic variables (R7RS compatible -- `?` is a valid identifier start character).

### GPU Acceleration

Adaptive dispatch with cost model calibration:

| Backend | Peak Performance | Overhead | Dispatch Range |
|:---|---:|---:|:---|
| SIMD (vectorized) | 25 GFLOPS | ~0 | <=16 elements |
| cBLAS (Apple Accelerate AMX) | 1,100 GFLOPS | 5 us | 17 to ~1B elements |
| Metal GPU (SF64 software float64) | 200 GFLOPS | 200 us | >1B elements |

**SF64 (Software Float64)**: Metal GPUs lack native float64. SF64 emulates double precision using double-double arithmetic (two 32-bit mantissas for ~100-bit effective precision). CUDA stubs enable NVIDIA GPU support.

Cost model in blas_backend.cpp selects the optimal backend by tensor size and compute intensity. Configurable via `ESHKOL_GPU_PRECISION`, `ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_GPU_PEAK_GFLOPS`.

### Parallel Computing

Work-stealing thread pool with parallel higher-order functions:

- `parallel-map` -- function application across vector elements in parallel
- `parallel-fold` -- associative reduction with automatic work distribution
- `parallel-filter` -- parallel predicate-based selection
- `parallel-for-each` -- parallel side-effecting iteration

Worker functions use `LinkOnceODRLinkage` for safe parallel compilation.

### Exact Arithmetic (Full Numeric Tower)

R7RS-compliant numeric tower with automatic precision promotion:

- **int64**: 64-bit signed integers (exact, immediate in tagged value)
- **bignum**: Arbitrary-precision integers (exact, heap-allocated, automatic overflow promotion)
- **rational**: Exact fractions as bignum pairs (always reduced via GCD)
- **double**: IEEE 754 64-bit floats (inexact)
- **complex**: Heap-allocated `{real:f64, imag:f64}` with Smith's formula division

Exactness tracked via `ESHKOL_FLAG_EXACT` in the tagged value flags byte. R7RS semantics: exact + exact = exact, exact + inexact = inexact.

### Signal Processing

FFT-based spectral analysis and digital filtering:

- FFT/IFFT (Cooley-Tukey radix-2 DIT, O(N log N))
- Window functions: Hamming, Hann, Blackman, Kaiser
- Filter design: Butterworth low-pass, high-pass, band-pass
- FIR/IIR direct-form implementation
- Linear and fast (FFT-based) convolution

### First-Class Continuations

- `call/cc` -- single-shot continuations via setjmp/longjmp
- `dynamic-wind` -- before/after thunks with proper unwinding on non-local exit
- `guard`/`raise` -- R7RS exception handling with cond-style clauses
- `with-exception-handler` -- low-level exception handler installation
- `delay`/`force`/`make-promise`/`delay-force` -- full R7RS promise system with memoization

### XLA Backend

StableHLO tensor compilation via MLIR (4,003 lines across 5 files). Enables accelerated tensor operations through XLA's optimization and dispatch infrastructure.

### Web Platform (WASM)

Compile to WebAssembly with 73 DOM/Canvas/Event API bindings:

```bash
eshkol-run --wasm app.esk
```

Provides DOM manipulation, event handling, Canvas 2D drawing, Fetch API, LocalStorage, and window management via an integer handle system.

### REPL JIT

Interactive development via LLVM OrcJIT (2,062 lines). Preloads 237 stdlib functions and 305 globals from precompiled `.o` + `.bc` metadata. Optimization level matched to precompiled objects to avoid ABI mismatches on struct argument passing.

### Package Manager

`eshkol-pkg` (721 lines) with TOML manifests and git-based registry. Commands: init, build, run, add, clean. 40 stdlib modules organized under core, math, signal, ml, and web namespaces with automatic recursive module discovery.

### Dual Backend Architecture

Eshkol has two production execution backends serving different purposes:

**LLVM Backend** (primary): Compiles to native ARM64/x86 binaries via LLVM IR. Uses 16-byte tagged values with 21 specialized codegen modules. This is the default path for `eshkol-run`.

**Bytecode VM** (complementary): 63-opcode register+stack interpreter (`eshkol_vm.c`, 8457 lines) with 250+ native call IDs covering the full language — arithmetic, closures, continuations, exception handling, tensors, complex/rational/bignum numbers, logic/inference/workspace, hash tables, bytevectors, parameters, and I/O. Compiles to ESKB binary format (section-based with LEB128 encoding, CRC32 checksums). Invoked via `eshkol-run input.esk -B output.eskb`.

**Weight Matrix Transformer**: Programs encoded as neural network weights (`weight_matrices.c`, 2299 lines). Architecture: d_model=36, 5 layers, FFN_DIM=512, 307K parameters. 25 core opcodes in weights, 38 via native dispatch. 3-way verification: reference interpreter = simulated transformer = matrix-based forward pass (55/55 tests). Exports QLMW binary format for qLLM loading.

The LLVM and VM backends share the same language semantics but use independent value representations. The VM exists for the qLLM/transformer weight pipeline and portable bytecode execution, not as a replacement for native compilation.

---

## Key Design Decisions

**LLVM IR instead of C.** Direct LLVM IR generation provides control over optimization, eliminates C compiler dependency, enables JIT compilation for the REPL, and gives access to LLVM's backend support for multiple architectures.

**Arena memory instead of GC.** Deterministic performance is critical for real-time systems, finance, and control loops. O(1) allocation, zero GC pauses, and cache-friendly sequential allocation. Provably safe through ownership analysis.

**Compiler-integrated AD instead of library.** Operates on AST, runtime, and LLVM IR simultaneously. No framework boundaries or tracing overhead. Works on any Eshkol function automatically with natural Scheme syntax.

**HoTT for type foundations.** Mathematical rigor via universe hierarchy. Gradual enforcement preserves Scheme's exploratory model while enabling aggressive optimization when types are known. Compile-time dimension checking for tensors.

**Homoiconicity as optimization target.** S-expressions preserved in closures enable runtime introspection, self-modifying systems, metaprogramming at native speed, and program synthesis.

**Pointer consolidation via headers.** Reducing 8 pointer types to 2 supertypes (HEAP_PTR, CALLABLE) with subtype headers traded one pointer dereference for extensible type space -- enabling v1.1's 7 new heap subtypes without exhausting the 8-bit type tag.

---

## Implementation Scale

| Component | Lines | Files |
|:---|---:|---:|
| LLVM backend (main + modules) | ~86,000 | 21 |
| Bytecode VM + runtime libs | ~41,000 | 29 |
| XLA/StableHLO backend | ~4,000 | 5 |
| GPU/Metal backend | ~8,900 | 4 |
| Frontend (parser, macro, types) | ~10,400 | 3 |
| Runtime (arena, logic, inference, workspace) | ~14,400 | 14 |
| REPL JIT | ~2,100 | 1 |
| Tools (LSP, package manager, VS Code) | ~2,800 | 3+ |
| Headers | ~16,000 | 52 |
| Weight matrix transformer | ~2,300 | 1 |
| **Total compiler infrastructure** | **~232,000** | **130+** |
| Standard library (.esk) | ~5,200 | 40 modules |
| Test code (.esk) | ~34,000 | 434 files |

35 automated test suites, 434 test files, 525+ assertions, 0 failures. Bytecode VM: 331/332 tests (99.7%). Weight matrices: 55/55 (3-way verified).

---

## Roadmap

**v1.1-accelerate** -- COMPLETE. GPU acceleration, parallel primitives, consciousness engine, exact arithmetic, signal processing, continuations, XLA backend, web platform, REPL JIT, package manager.

**v1.2-scale** -- Next. Distributed training framework, model deployment tools, multi-shot continuations, expanded GPU support, database integration.

---

*Eshkol v1.1-accelerate is a production compiler integrating automatic differentiation, deterministic memory management, homoiconic native code, GPU acceleration, cognitive computing primitives, and a dual backend architecture (LLVM + bytecode VM). The ~232,000-line codebase across 21 LLVM codegen modules, a 63-opcode bytecode VM, and a weight matrix transformer provides the foundation for the v1.2-scale release.*
