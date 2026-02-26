# Changelog

All notable changes to Eshkol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0-foundation] - 2025-12-12

### Production Release

Eshkol v1.0-foundation represents a complete, production-ready compiler with unprecedented integration of automatic differentiation, deterministic memory management, and homoiconic native code execution.

#### Core Compiler Implementation
- Modular LLVM backend with 15 specialized codegen modules
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
- LLVM 10.0+ (14+ recommended)
- CMake 3.14+
- C17 runtime, C++20 compiler
- readline (optional, for REPL features)

---

## [1.1.0-accelerate] - 2026-02-17

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
- All 35 bignum codegen gaps fixed
- Tensor bounds checking with runtime validation
- Metal buffer leak fix (@autoreleasepool)
- REPL complex type handling
- Module visibility enforcement
- 29 test suites passing (200+ test files)

---

## [Unreleased]

### v1.2-scale (Q2 2026) - Planned
- GPU acceleration enhancements (Vulkan Compute, multi-GPU)
- Distributed training framework (multi-node)
- Model serialization and deployment (ONNX, TensorFlow Lite, Core ML)
- Inference optimization (operator fusion, quantization, pruning)

### v1.5-intelligence (Q2-Q3 2026) - Planned
- Neuro-symbolic integration (logic programming, knowledge representation)
- Advanced neural primitives (attention, convolution, transformers)
- Symbolic reasoning with neural guidance

### v2.0-quantum (Q4 2026+) - Research
- Quantum computing integration
- Advanced type system (full dependent types, refinement types)
- Formal verification integration (Coq, Lean)
- Haskell integration
