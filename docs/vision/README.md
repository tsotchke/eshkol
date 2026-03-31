# Eshkol Vision Documentation

> **Note**: These documents describe v1.0-foundation capabilities. v1.1-accelerate adds
> GPU acceleration (Metal SF64 + CUDA), XLA backend (dual-mode StableHLO + LLVM-direct),
> consciousness engine (22 compiled primitives), parallel primitives (work-stealing
> thread pool), exact arithmetic (bignums + rationals), signal processing (FFT, filters),
> and first-class continuations (call/cc, dynamic-wind). See [CHANGELOG.md](../../CHANGELOG.md)
> and [V1.1 Scope](../V1.1_SCOPE.md).

This directory contains vision documents for Eshkol v1.0-architecture, grounded in the actual production compiler implementation.

## What is Eshkol v1.0-architecture?

Eshkol is a **production-ready Scheme dialect** with a sophisticated LLVM-based compiler (C/C++) that delivers:

- **Native code generation** via modular LLVM backend
- **Compiler-integrated automatic differentiation** (forward/reverse modes with nested gradient support)
- **Arena-based memory management** (OALR - Ownership-Aware Lexical Regions)
- **HoTT-inspired gradual type system** with bidirectional type checking
- **R7RS Scheme compatibility** with 550+ language features
- **Interactive REPL** with LLVM ORC JIT compilation
- **Quantum-inspired RNG** for high-quality stochastic computing

## Implementation Reality (v1.0-foundation)

**Production Compiler:**
- C/C++ implementation
- LLVM 17 backend with 21 specialized codegen modules
- Parser with HoTT type expression support
- Bidirectional type checker
- Arena memory runtime
- Quantum RNG (8-qubit simulation for randomness)

**Runtime Architecture:**
- 16-byte tagged values with consolidated type system
- 8-byte object headers for heap objects
- 32-byte cons cells with complete tagged values
- Lambda registry for homoiconic display
- Exception handling with R7RS guard/raise

**Actual Capabilities:**
- Gradient-based optimization (derivative, gradient, jacobian, hessian, divergence, curl, laplacian)
- Neural network training with automatic differentiation
- Numerical computing (determinants, matrix inversion, linear systems)
- Interactive development with persistent REPL state
- Module system with dependency resolution

## Vision Documents

### [Purpose and Vision](PURPOSE_AND_VISION.md)
Core mission focused on gradient-based optimization, neural network development, and scientific computing with deterministic memory management.

### [AI Focus](AI_FOCUS.md)
Production automatic differentiation system with forward-mode dual numbers, reverse-mode computational graphs, and nested gradient support up to 32 levels deep.

### [Scientific Computing](SCIENTIFIC_COMPUTING.md)
Arena memory architecture (OALR), N-dimensional tensors, linear algebra algorithms (LU decomposition, Gauss-Jordan), and quantum RNG for stochastic methods.

### [Technical White Paper](TECHNICAL_WHITE_PAPER.md)
Deep dive into LLVM backend architecture, tagged value system, closure compilation with environment encoding, AD tape stack, and object header design.

### [Differentiation Analysis](DIFFERENTIATION_ANALYSIS.md)
Honest comparison with Scheme, Julia, Python/JAX based on actual implemented differentiators: LLVM performance, arena memory, integrated AD, homoiconic closures.

### [Future Roadmap](FUTURE_ROADMAP.md)
Clear v1.0-foundation baseline (what exists NOW) followed by realistic post-v1.0 plans (GPU acceleration, distributed computing, quantum computing extensions).

## Key Technical Achievements (v1.0)

### Modular LLVM Backend
21 specialized codegen modules totaling 55,000+ lines (v1.1: ~232,000 total):
- `TaggedValueCodegen` - 16-byte value pack/unpack (812 lines)
- `AutodiffCodegen` - Dual numbers and AD graphs (1,766 lines)
- `FunctionCodegen` - Closures with capture (131 lines)
- `ArithmeticCodegen` - Polymorphic dispatch for int64/double/dual/tensor
- `ControlFlowCodegen` - Short-circuit and/or, cond, pattern matching
- `CollectionCodegen` - Cons cells, vectors with mixed types
- `TensorCodegen` - N-dimensional array operations
- `HashCodegen` - FNV-1a hash tables
- `StringIOCodegen` - Strings, ports, file I/O
- `TailCallCodegen` - Loop transformation for self-recursion
- Plus 5 more specialized modules

### Automatic Differentiation System
**Forward Mode (Dual Numbers):**
- Structure: `{double value; double derivative;}`
- Arithmetic rules for dual propagation
- Math functions: sin, cos, exp, log, sqrt, tan

**Reverse Mode (Computational Graphs):**
- AD nodes: `{type, value, gradient, input1, input2, id}`
- Tape stack for nested gradients (MAX_DEPTH=32)
- Backward pass with reverse topological traversal
- Operators: derivative, gradient, jacobian, hessian, divergence, curl, laplacian

### Memory Management (OALR)
**Arena Architecture:**
- Bump-pointer allocation in large blocks
- Scope-based cleanup (lexical regions)
- No garbage collection - deterministic timing
- Global arena + region stack (MAX_DEPTH=16)

**Escape Analysis:**
- `NO_ESCAPE` → stack allocation
- `RETURN_ESCAPE` → region allocation
- `CLOSURE_ESCAPE` / `GLOBAL_ESCAPE` → shared allocation

**Ownership Tracking:**
- States: UNOWNED, OWNED, MOVED, BORROWED
- Compile-time checks prevent use-after-move

### Tagged Value System
**Runtime Representation (16 bytes):**
```c
struct eshkol_tagged_value {
    uint8_t type;        // Type tag (0-255)
    uint8_t flags;       // Exactness, special flags
    uint16_t reserved;
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
    } data;
}
```

**Immediate Types (0-7):** NULL, INT64, DOUBLE, BOOL, CHAR, SYMBOL, DUAL_NUMBER

**Consolidated Types (8-9):** 
- HEAP_PTR (8) with subtypes: CONS, STRING, VECTOR, TENSOR, HASH, EXCEPTION, MULTI_VALUE
- CALLABLE (9) with subtypes: CLOSURE, LAMBDA_SEXPR, AD_NODE, PRIMITIVE, CONTINUATION

### Closure System
**Structure (40 bytes):**
```c
struct eshkol_closure {
    uint64_t func_ptr;              // Lambda function pointer
    eshkol_closure_env_t* env;      // Captured variables
    uint64_t sexpr_ptr;             // S-expression for display
    uint8_t return_type;
    uint8_t input_arity;
    uint8_t flags;                  // Variadic, etc.
    uint8_t reserved;
    uint32_t hott_type_id;
}
```

**Environment Encoding:**
- `num_captures` field packs: actual captures | (fixed_params << 16) | (is_variadic << 63)
- Flexible array of captured `eshkol_tagged_value_t` elements

## What v1.1-accelerate Added (Since v1.0)

- ✅ **GPU acceleration** — Metal (Apple Silicon, SF64 software float64) + CUDA (NVIDIA)
- ✅ **Parallel primitives** — work-stealing thread pool, `parallel-map`/`parallel-fold`/`parallel-filter`, `future`/`force`
- ✅ **Exact arithmetic** — arbitrary-precision integers (bignums) and rational numbers, full R7RS numeric tower
- ✅ **Consciousness engine** — 22 compiled primitives: logic programming, factor graphs, global workspace
- ✅ **Signal processing** — FFT/IFFT, window functions, FIR/IIR filters, Butterworth design
- ✅ **First-class continuations** — `call/cc`, `dynamic-wind`, `guard`/`raise`
- ✅ **ML framework** — 75+ builtins: activations, losses, optimizers, CNN layers, transformer ops
- ✅ **XLA backend** — dual-mode StableHLO + LLVM-direct for tensor acceleration
- ✅ **Web platform** — WASM compilation, 73 DOM API functions
- ✅ **Package manager** — `eshkol-pkg` with TOML manifest and registry support

### Not Yet Implemented (Planned for Future Releases)

- ❌ Distributed computing (v1.2-scale)
- ❌ Quantum computing primitives — qubits, gates, VQE (v2.0-starlight)
- ❌ Built-in plotting/visualization
- ❌ Full R7RS library system — `define-library`/`import` with renaming (v1.3-evolve)

See [Roadmap](../../ROADMAP.md) for planned development.

## Documentation Principles

All vision documents in this directory:
1. **Ground claims in actual implementation** - every feature referenced exists in the 80K-line codebase
2. **Include technical architecture** - how it actually works, not vague descriptions
3. **Provide working examples** - code that compiles and runs
4. **Separate present from future** - clear distinction between v1.0 and roadmap
5. **Reference implementation files** - link to actual source code

**Test Suite:**
- Autodiff tests: 50+ test files
- List operation tests: 120+ test files
- Neural network tests: many working examples
- Type system tests: comprehensive coverage

## See Also

- [`COMPLETE_LANGUAGE_SPECIFICATION.md`](../../COMPLETE_LANGUAGE_SPECIFICATION.md) - Technical specification of all 300+ language features
- [`ESHKOL_V1_LANGUAGE_REFERENCE.md`](../../ESHKOL_V1_LANGUAGE_REFERENCE.md) - User reference with examples
- [`docs/breakdown/`](../breakdown/) - Component-specific technical documentation
- [`docs/ESHKOL_V1_ARCHITECTURE.md`](../ESHKOL_V1_ARCHITECTURE.md) - Complete architecture overview

---

*This directory documents the Eshkol compiler vision. v1.1-accelerate is the current production release with GPU acceleration, consciousness engine, parallel primitives, exact arithmetic, dual backend architecture (LLVM + bytecode VM), and 75+ ML builtins. 434 tests across 35 suites.*
