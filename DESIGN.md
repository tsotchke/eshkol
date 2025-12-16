# Eshkol Design Document

This document outlines the core architecture and design decisions for Eshkol v1.0-foundation.

## Language Design

Eshkol is a production-grade Scheme dialect with LLVM-native performance, designed for gradient-based optimization and scientific computing. Core features:

- S-expression syntax (Scheme R7RS compatibility)
- HoTT-inspired gradual type system with bidirectional checking
- **LLVM IR generation** for native code compilation
- Arena-based memory management (OALR - zero garbage collection)
- Compiler-integrated automatic differentiation (forward, reverse, symbolic modes)

## Architecture Overview (v1.0-foundation)

```
Source (.esk)
     │
     ▼
┌──────────────────────┐
│  Parser              │  Recursive descent with HoTT type expressions
│  (parser.cpp)        │  Macro expansion (syntax-rules)
└──────────┬───────────┘
           │
           ▼
     AST + Source Locations
           │
           ▼
┌──────────────────────┐
│  Type Checker        │  Bidirectional HoTT type checking
│  (type_checker.cpp)  │  Type inference + annotations
└──────────┬───────────┘
           │
           ▼
   Annotated AST + Ownership Analysis
           │
           ▼
┌──────────────────────┐
│  LLVM Backend        │  15 Specialized Codegen Modules:
│  (llvm_codegen.cpp)  │  - TaggedValueCodegen
│                      │  - AutodiffCodegen
│                      │  - FunctionCodegen
│                      │  - ArithmeticCodegen
│                      │  - ControlFlowCodegen
│                      │  - CollectionCodegen
│                      │  - TensorCodegen
│                      │  - HashCodegen
│                      │  - StringIOCodegen
│                      │  - TailCallCodegen
│                      │  - SystemCodegen
│                      │  - HomoiconicCodegen
│                      │  - CallApplyCodegen
│                      │  - MapCodegen
│                      │  - BindingCodegen
└──────────┬───────────┘
           │
           ▼
       LLVM IR
           │
           ▼
┌──────────────────────┐
│  LLVM Optimizer      │  Standard LLVM optimization passes
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Code Generation     │  Native machine code for target architecture
└──────────┬───────────┘
           │
           ▼
     Object File (.o)
           │
           ▼
┌──────────────────────┐
│  Linker              │  Links with runtime and standard library
└──────────┬───────────┘
           │
           ▼
  Executable / Shared Library
```

## Core Components

### 1. Memory Management System (OALR)

**Ownership-Aware Lexical Regions** - deterministic memory without garbage collection:

- **Arena Allocator**: O(1) bump-pointer allocation, O(1) bulk deallocation
- **Lexical Scopes**: Memory tied to code structure (with-region syntax)
- **Escape Analysis**: Compiler determines stack/region/shared allocation automatically
- **Ownership Tracking**: Compile-time verification prevents use-after-move, double-free
- **Reference Counting**: Optional, only for shared/global escape cases
- **Zero GC Pauses**: Fully deterministic performance for real-time systems

### 2. HoTT-Based Gradual Type System

**Homotopy Type Theory foundations** with bidirectional checking:

- **Bidirectional Type Checking**: Synthesis (⇒) and checking (⇐) modes
- **Type Inference**: Local inference reduces annotation burden
- **Gradual Typing**: Warnings not errors - smooth transition from dynamic to static
- **Dependent Types**: Compile-time dimension checking for tensors/matrices
- **Universe Hierarchy**: Type₀, Type₁, Type₂, Typeω levels
- **Linear Types**: Resource consumption tracking (planned full enforcement v1.1)
- **Parametric Polymorphism**: Generic types with constraints (List<T>, Vector<T>)

### 3. Compilation Pipeline (LLVM-Based)

**v1.0-foundation actual implementation:**

1. **Lexical Analysis**: Token stream from S-expressions (supports type annotation syntax)
2. **Parsing**: Recursive descent → AST with source locations
3. **Macro Expansion**: syntax-rules with hygenic transformation
4. **Type Checking**: HoTT bidirectional inference and checking
5. **Ownership Analysis**: Escape analysis for allocation strategy
6. **LLVM IR Generation**: Modular backend with 15 specialized modules
7. **LLVM Optimization**: Standard optimization passes (inlining, dead code elimination, etc.)
8. **Code Generation**: Native machine code for target architecture
9. **Linking**: Combines with runtime library and dependencies

### 4. Automatic Differentiation System

**Compiler-integrated AD** - not a library, not a framework:

- **Symbolic Mode**: AST transformation during compilation (zero runtime overhead)
- **Forward Mode**: Dual number arithmetic for efficient f: ℝ → ℝⁿ
- **Reverse Mode**: Computational graph with tape stack for f: ℝⁿ → ℝ
- **Nested Gradients**: Up to 32 levels deep via global tape stack
- **Vector Calculus**: gradient, jacobian, hessian, divergence, curl, laplacian (8 operators)
- **Polymorphic**: Works with int64, double, dual numbers, AD nodes, tensors

## Key Design Decisions

### Why LLVM Instead of C?

**Decision:** Generate LLVM IR directly, not C source code

**Rationale:**
- Direct control over optimization
- Better integration with type system
- Eliminates C compiler dependency
- Enables JIT compilation (REPL)
- Cleaner code generation architecture

### Why Arena Memory Instead of GC?

**Decision:** OALR (Ownership-Aware Lexical Regions) instead of garbage collection

**Rationale:**
- Deterministic performance (critical for real-time, finance, control systems)
- O(1) allocation and deallocation
- Zero GC pauses eliminates tail latency
- Cache-friendly sequential allocation
- Provably safe through ownership analysis

### Why Compiler-Integrated AD?

**Decision:** AD as language primitive, not library

**Rationale:**
- Operates on AST, runtime, and LLVM IR simultaneously
- No framework boundaries or tracing overhead
- Works on any Eshkol function automatically
- Natural Scheme syntax (no special constructs)
- Enables nested gradients through tape stack

### Why HoTT for Type System?

**Decision:** Homotopy Type Theory foundations with gradual enforcement

**Rationale:**
- Mathematical rigor (type theory on solid foundations)
- Universe hierarchy enables dependent types
- Warnings not errors (smooth prototyping → production path)
- Compile-time dimension checking for tensors
- Supports both exploration and formal verification

### Why Homoiconicity Matters

**Decision:** Preserve S-expressions in compiled closures

**Rationale:**
- Enables runtime code introspection
- Self-modifying AI systems possible
- Metaprogramming at native speed
- Educational value (see compiled code structure)
- Research applications (program synthesis, meta-learning)

### Build System (CMake)

**Modern build infrastructure:**

- CMake 3.14+ for cross-platform support
- Ninja backend for fast parallel builds
- LLVM discovery and linking
- Separate targets for compiler, REPL, tests
- Docker containers for reproducible builds
- Package generation (Debian, Homebrew)

## Implementation Quality

### Production-Ready Codebase

- **Modular Architecture**: 15 specialized codegen modules for maintainability
- **Comprehensive Testing**: 170+ test files covering all features
- **Source Location Tracking**: Parser preserves line/column for error messages
- **Error Handling**: Exception system with guard/raise (R7RS semantics)
- **Memory Safety**: Ownership analysis prevents use-after-move at compile-time
- **Performance**: LLVM optimization passes applied automatically

### Ongoing Development

**v1.1-acceleration (Q1 2026):**
- XLA backend for accelerated tensor operations
- SIMD vectorization (SSE/AVX/NEON)
- Parallelism primitives (parallel-map, parallel-fold)

**v1.2-scale (Q2 2026):**
- GPU backends (CUDA, Metal, Vulkan)
- Distributed training framework
- Model deployment tools

**v1.5+ (Q2-Q3 2026):**
- Neuro-symbolic integration
- Quantum computing support
- Advanced type system features

See [ROADMAP.md](ROADMAP.md) and [docs/vision/FUTURE_ROADMAP.md](docs/vision/FUTURE_ROADMAP.md) for detailed plans.

---

*Eshkol v1.0-foundation represents a completed production compiler with unprecedented integration of automatic differentiation, deterministic memory management, and homoiconic native code execution. The modular LLVM architecture provides a solid foundation for future performance acceleration and advanced AI capabilities.*
