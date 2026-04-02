# Eshkol Programming Language Documentation

![Version](https://img.shields.io/badge/version-1.1.11--accelerate-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **Note**: This directory originally documented v1.0-foundation. For v1.1-accelerate
> features (GPU acceleration, XLA backend, parallel primitives, consciousness engine,
> exact arithmetic, signal processing, first-class continuations), see
> [CHANGELOG.md](../../CHANGELOG.md) and [V1.1 Scope](../V1.1_SCOPE.md).

## Introduction

Eshkol is a compiled programming language for scientific computing and machine learning. It reconciles Scheme's S-expression syntax and functional programming paradigm with C-level performance through LLVM-based native code generation. The language provides deterministic memory management via the OALR (Ownership-Aware Lexical Regions) system, a HoTT-inspired gradual type system, first-class functions with true homoiconicity, and compiler-integrated automatic differentiation.

**Design Focus**: Gradient-based optimization, numerical computing, and production machine learning deployment where Python's overhead and garbage collector pauses are unacceptable.

**Implementation**: ~232,000 lines of production C/C++ compiler infrastructure targeting LLVM IR.

## Quick Start

### Build from Source

```bash
# Clone repository
git clone https://github.com/tsotchke/eshkol.git
cd eshkol

# Build (requires LLVM 17, CMake 3.14+, C++20 compiler)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Executables in build/
build/eshkol-run   # AOT compiler
build/eshkol-repl  # JIT REPL
```

> **Requirements**: LLVM 17, CMake 3.14+, C++20 compiler (GCC 11+, Clang 14+), Ninja (recommended)

### Hello World

```scheme
;;; hello.esk
(display "Hello, Eshkol!")
(newline)
```

```bash
build/eshkol-run hello.esk
./hello
# Output: Hello, Eshkol!
```

## Documentation Index

### Language Fundamentals

- **[Overview](OVERVIEW.md)** — Design philosophy, technical architecture, competitive positioning
- **[Getting Started](GETTING_STARTED.md)** — Installation, build process, first programs
- **[Compilation Guide](COMPILATION_GUIDE.md)** — LLVM compilation, debugging, troubleshooting

### Core Architecture

- **[Compiler Architecture](COMPILER_ARCHITECTURE.md)** — LLVM backend, 21 specialized codegen modules, JIT system
- **[Type System](TYPE_SYSTEM.md)** — Runtime tagged values, HoTT gradual typing, formal inference rules, object headers
- **[Memory Management](MEMORY_MANAGEMENT.md)** — OALR, arena allocation, linear types, reference counting
- **[Function Composition](FUNCTION_COMPOSITION.md)** — Closures (40-byte structures), lambda registry, mutable captures
- **[Vector Operations](VECTOR_OPERATIONS.md)** — Scheme vectors (heterogeneous) vs. tensors (homogeneous, AD-aware)
- **[Automatic Differentiation](AUTODIFF.md)** — Three AD modes (symbolic, forward-mode dual numbers, reverse-mode computational graph)
- **[Scheme Compatibility](SCHEME_COMPATIBILITY.md)** — R7RS compliance, migration guide

### v1.1-accelerate Novel Features

- **[Consciousness Engine](CONSCIOUSNESS_ENGINE.md)** — Logic programming (unification, KB), active inference (factor graphs, belief propagation, free energy), global workspace theory
- **[Machine Learning](MACHINE_LEARNING.md)** — 75+ ML builtins: activations, losses, optimizers, CNN layers, transformer operations, training pipelines
- **[Signal Processing](SIGNAL_PROCESSING.md)** — FFT/IFFT (Cooley-Tukey), window functions, FIR/IIR filters, Butterworth design
- **[GPU Acceleration](GPU_ACCELERATION.md)** — Metal compute, SF64 software float64, CUDA, cost model dispatch
- **[Parallel Computing](PARALLEL_COMPUTING.md)** — Work-stealing thread pool, parallel-map/fold/filter, per-thread arena isolation
- **[Exact Arithmetic](EXACT_ARITHMETIC.md)** — Bignum/rational numeric tower, overflow detection, R7RS exactness
- **[Continuations](CONTINUATIONS.md)** — call/cc, dynamic-wind, guard/raise, exception handling
- **[XLA Backend](XLA_BACKEND.md)** — Tensor runtime operations, GPU dispatch, broadcasting, AD gradient support

### Platform and Tools

- **[Module System](MODULE_SYSTEM.md)** — require/provide, precompiled stdlib, LinkOnceODR user override
- **[REPL JIT](REPL_JIT.md)** — LLVM OrcJIT, CodeGenOptLevel matching, stdlib preloading, hot reload
- **[Web Platform](WEB_PLATFORM.md)** — WASM compilation, 73 DOM/Canvas/event API functions, integer handle system
- **[Package Manager](PACKAGE_MANAGER.md)** — eshkol-pkg CLI, TOML manifest, git-based registry
- **[Developer Tools](DEVELOPER_TOOLS.md)** — LSP server, VSCode extension, --dump-ir/--dump-ast debug flags
- **[Math Standard Library](MATH_STDLIB.md)** — Special functions, ODE solvers, statistics, PRNG and quantum RNG

## Reference Documentation

### Complete API Reference

- **[API Reference](../API_REFERENCE.md)** - All functions with signatures, examples from test suite
- **[Quickstart Tutorial](../QUICKSTART.md)** - 15-minute hands-on tutorial (lists → closures → tensors → gradients)
- **[Feature Matrix](../FEATURE_MATRIX.md)** - Implementation status table (✅ Implemented | ⚠️ Partial | ❌ Planned)

### Master Architecture Document

- **[Eshkol v1.1-Architecture](../ESHKOL_V1_ARCHITECTURE.md)** - Comprehensive technical architecture (consolidates 232,000-line deep dive)

## Technical Specifications

### Language Characteristics

- **Compilation**: LLVM IR → native code (x86-64, ARM64)
- **Memory Model**: OALR with global arena, 8-byte object headers, optional reference counting
- **Type System**: Gradual (HoTT-inspired warnings, not errors)
- **Tagged Values**: 16-byte uniform representation `{type:u8, flags:u8, reserved:u16, padding:u32, data:u64}`
- **Closures**: 40-byte structures with mutable captures (pointers to variables)
- **AD Integration**: Three modes with 32-level nested gradient support

### Implementation Statistics

- **Frontend**: S-expression parser (5,487 lines), macro expander (1,234 lines)
- **Type Checker**: HoTT-inspired gradual typing (1,561 lines)
- **LLVM Backend**: Main codegen (34,928 lines) + 21 modules (~20K lines)
- **Runtime**: Arena memory (3,210 lines), closure system, AD tape infrastructure
- **JIT**: REPL with LLVM OrcJIT (1,108 lines)

## How to Navigate

**New to Eshkol?**
1. Start with [Getting Started](GETTING_STARTED.md)
2. Read [Overview](OVERVIEW.md) for design philosophy
3. Try [Quickstart Tutorial](../QUICKSTART.md)

**Implementing numeric algorithms?**
1. [Automatic Differentiation](AUTODIFF.md) — Gradient computation
2. [Vector Operations](VECTOR_OPERATIONS.md) — Tensors and linear algebra
3. [Machine Learning](MACHINE_LEARNING.md) — Neural network training pipeline
4. [Math Standard Library](MATH_STDLIB.md) — Special functions, ODE solvers, statistics
5. [API Reference](../API_REFERENCE.md) — Complete function reference

**Building intelligent systems?**
1. [Consciousness Engine](CONSCIOUSNESS_ENGINE.md) — Logic programming, active inference, global workspace
2. [Signal Processing](SIGNAL_PROCESSING.md) — FFT, filters, spectral analysis
3. [Parallel Computing](PARALLEL_COMPUTING.md) — Work-stealing parallelism
4. [GPU Acceleration](GPU_ACCELERATION.md) — Metal/CUDA dispatch, SF64

**Understanding the implementation?**
1. [Compiler Architecture](COMPILER_ARCHITECTURE.md) — LLVM backend
2. [Type System](TYPE_SYSTEM.md) — Tagged values, HoTT types, formal inference rules
3. [Memory Management](MEMORY_MANAGEMENT.md) — OALR system internals
4. [Module System](MODULE_SYSTEM.md) — require/provide, precompiled stdlib
5. [REPL JIT](REPL_JIT.md) — Interactive development, hot reload
6. [Master Architecture](../ESHKOL_V1_ARCHITECTURE.md) — Complete technical deep dive

**Building for the web?**
1. [Web Platform](WEB_PLATFORM.md) — WASM compilation, DOM API
2. [Package Manager](PACKAGE_MANAGER.md) — Project setup, dependency management
3. [Developer Tools](DEVELOPER_TOOLS.md) — LSP, VSCode extension, debugging

**Contributing?**
- See [Scheme Compatibility](SCHEME_COMPATIBILITY.md) for R7RS compliance status
- Check [Feature Matrix](../FEATURE_MATRIX.md) for implementation status
- Review source code directly (all docs link to implementation files)

## License

Eshkol is licensed under the MIT License. See the LICENSE file in the repository root.
