# Changelog

All notable changes to Eshkol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0-foundation] - 2025-12-XX

### Added

#### Type System
- HoTT-based type system with universe levels (U0, U1, U2)
- Bidirectional type checking with inference
- Type annotations: `(: name type)`, `(param : type)`
- Polymorphic types with `forall`
- Function types: `(-> arg-types return-type)`
- Product types (pairs) and sum types
- Path types for type equality
- Inductive types with pattern matching

#### Automatic Differentiation
- Forward-mode AD with dual numbers
- Reverse-mode AD with computation graph
- Vector calculus operators:
  - `gradient` - Compute gradient of scalar function
  - `jacobian` - Compute Jacobian matrix
  - `hessian` - Compute Hessian matrix
  - `divergence` - Vector field divergence
  - `curl` - Vector field curl (3D)
  - `laplacian` - Laplacian operator
- Neural network primitives (dense layers, activations)

#### Memory Management
- Arena-based allocation (OALR - Owned Arena with Linear References)
- Automatic memory cleanup without garbage collection
- Tail call optimization for recursive functions
- Efficient cons cell allocation

#### Language Features
- Scheme-compatible syntax (R7RS subset)
- Module system with `require`/`provide`
- Pattern matching
- First-class functions and closures
- Macros with `define-syntax`
- Quasiquotation

#### Standard Library
- 266 built-in functions
- List operations: map, filter, fold, reduce, etc.
- String operations: split, join, trim, case conversion
- Mathematical functions: trig, exp, log, linear algebra
- Tensor operations for numerical computing
- I/O operations: file read/write, display

#### Tools
- `eshkol-run` - Compiler and executor
- `eshkol-repl` - Interactive REPL with JIT
- Pre-compiled standard library for faster startup

### Platforms
- macOS 12+ (x86_64, arm64)
- Ubuntu 22.04+ (x86_64)
- Debian 12+ (x86_64)

### Dependencies
- LLVM 17
- CMake 3.14+
- C++20 compiler
- readline (optional, for REPL)

---

## [Unreleased]

### Planned
- TCP/UDP networking
- HTTP client/server
- Hash tables with O(1) lookup
- Concurrency with threads and channels
- Structured exception handling
