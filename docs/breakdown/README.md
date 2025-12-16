# Eshkol Programming Language Documentation

![Version](https://img.shields.io/badge/version-1.0.0--foundation-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Introduction

Eshkol is a compiled programming language for scientific computing and machine learning. It reconciles Scheme's S-expression syntax and functional programming paradigm with C-level performance through LLVM-based native code generation. The language provides deterministic memory management via the OALR (Ownership-Aware Lexical Regions) system, a HoTT-inspired gradual type system, first-class functions with true homoiconicity, and compiler-integrated automatic differentiation.

**Design Focus**: Gradient-based optimization, numerical computing, and production machine learning deployment where Python's overhead and garbage collector pauses are unacceptable.

**Implementation**: 67,000+ lines of production C/C++ compiler infrastructure targeting LLVM IR.

## Quick Start

### Build from Source

```bash
# Clone repository
git clone https://github.com/tsotchke/eshkol.git
cd eshkol

# Build (requires LLVM 14+, CMake 3.16+, C++17 compiler)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Executables in build/
build/eshkol-run   # AOT compiler
build/eshkol-repl  # JIT REPL
```

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

## Core Documentation (docs/aidocs/)

### Language Fundamentals

- **[Overview](OVERVIEW.md)** - Design philosophy, technical architecture, competitive positioning
- **[Getting Started](GETTING_STARTED.md)** - Installation, build process, first programs
- **[Compilation Guide](COMPILATION_GUIDE.md)** - LLVM compilation, debugging, troubleshooting

### Type System and Memory

- **[Type System](TYPE_SYSTEM.md)** - Runtime tagged values, HoTT gradual typing, object headers
- **[Memory Management](MEMORY_MANAGEMENT.md)** - OALR, arena allocation, linear types, reference counting

### Core Features

- **[Function Composition](FUNCTION_COMPOSITION.md)** - Closures (32-byte structures), lambda registry, mutable captures
- **[Vector Operations](VECTOR_OPERATIONS.md)** - Scheme vectors (heterogeneous) vs. tensors (homogeneous, AD-aware)
- **[Automatic Differentiation](AUTODIFF.md)** - Three AD modes (symbolic, forward-mode dual numbers, reverse-mode computational graph)

### Implementation

- **[Compiler Architecture](COMPILER_ARCHITECTURE.md)** - LLVM backend, 19 modular codegen components, JIT system
- **[Scheme Compatibility](SCHEME_COMPATIBILITY.md)** - R5RS/R7RS compliance, migration guide

## Reference Documentation

### Complete API Reference

- **[API Reference](../API_REFERENCE.md)** - All functions with signatures, examples from test suite
- **[Quickstart Tutorial](../QUICKSTART.md)** - 15-minute hands-on tutorial (lists → closures → tensors → gradients)
- **[Feature Matrix](../FEATURE_MATRIX.md)** - Implementation status table (✅ Implemented | ⚠️ Partial | ❌ Planned)

### Master Architecture Document

- **[Eshkol v1.0-Architecture](../ESHKOL_V1_ARCHITECTURE.md)** - Comprehensive technical architecture (consolidates 67,000-line deep dive)

## Technical Specifications

### Language Characteristics

- **Compilation**: LLVM IR → native code (x86-64, ARM64)
- **Memory Model**: OALR with global arena, 8-byte object headers, optional reference counting
- **Type System**: Gradual (HoTT-inspired warnings, not errors)
- **Tagged Values**: 16-byte uniform representation `{type:u8, flags:u8, reserved:u16, data:u64}`
- **Closures**: 32-byte structures with mutable captures (pointers to variables)
- **AD Integration**: Three modes with 32-level nested gradient support

### Implementation Statistics

- **Frontend**: S-expression parser (5,487 lines), macro expander (1,234 lines)
- **Type Checker**: HoTT-inspired gradual typing (1,561 lines)
- **LLVM Backend**: Main codegen (27,079 lines) + 19 modules (~20K lines)
- **Runtime**: Arena memory (3,210 lines), closure system, AD tape infrastructure
- **JIT**: REPL with LLVM OrcJIT (1,108 lines)

## How to Navigate

**New to Eshkol?**
1. Start with [Getting Started](GETTING_STARTED.md)
2. Read [Overview](OVERVIEW.md) for design philosophy
3. Try [Quickstart Tutorial](../QUICKSTART.md)

**Implementing numeric algorithms?**
1. [Automatic Differentiation](AUTODIFF.md) - Gradient computation
2. [Vector Operations](VECTOR_OPERATIONS.md) - Tensors and linear algebra
3. [API Reference](../API_REFERENCE.md) - Function reference

**Understanding the implementation?**
1. [Compiler Architecture](COMPILER_ARCHITECTURE.md) - LLVM backend
2. [Type System](TYPE_SYSTEM.md) - Tagged values, HoTT types
3. [Memory Management](MEMORY_MANAGEMENT.md) - OALR system internals
4. [Master Architecture](../ESHKOL_V1_ARCHITECTURE.md) - Complete technical deep dive

**Contributing?**
- See [Scheme Compatibility](SCHEME_COMPATIBILITY.md) for R5RS/R7RS compliance status
- Check [Feature Matrix](../FEATURE_MATRIX.md) for implementation status
- Review source code directly (all docs link to implementation files)

## License

Eshkol is licensed under the MIT License. See the LICENSE file in the repository root.
