# Eshkol Development Roadmap

## Table of Contents
- [v1.0-Architecture Status](#v10-architecture-status)
- [Post-v1.0 Roadmap](#post-v10-roadmap)
- [Long-term Vision](#long-term-vision)
- [Release Philosophy](#release-philosophy)

---

## v1.0-Architecture Status

**Current Release**: v1.0.0-foundation

Eshkol v1.0-architecture represents a **production-ready foundation** for scientific computing and machine learning:

### Implemented Core Features

✅ **Language Core**
- Complete S-expression parser with 93 operators
- Hygienic macro system (`define-syntax`, `syntax-rules`)
- R5RS/R7RS Scheme compatibility (most features)
- Module system (`require`, `provide`)
- Exception handling (`guard`, `raise`, `error`)
- Multiple return values (`values`, `call-with-values`, `let-values`)

✅ **Type System**
- 16-byte tagged values with 8-bit type field
- Pointer consolidation (HEAP_PTR/CALLABLE/MEDIA) with 8-byte object headers
- HoTT-inspired gradual typing (warnings, not errors)
- 35+ builtin types in universe hierarchy
- Optional dependent types for tensor dimensions

✅ **Memory Management (OALR)**
- Global arena with 8KB blocks, automatic growth
- 8-byte object headers (subtype, flags, ref_count, size)
- Linear types (`owned`, `move`, `borrow`)
- Reference counting (`shared`, `weak-ref`)
- Compile-time ownership tracking

✅ **Automatic Differentiation**
- Symbolic differentiation (12 AST rewrite rules)
- Forward-mode AD (16-byte dual numbers)
- Reverse-mode AD (computational graph, 32-level tape stack)
- Vector calculus: gradient, jacobian, hessian, divergence, curl, laplacian
- AD-aware closures (context-sensitive execution)

✅ **Tensor Operations**
- 30+ tensor operations (zeros, ones, eye, arange, linspace)
- Linear algebra (tensor-dot, transpose, reshape)
- Element-wise operations (add, sub, mul, div)
- Reductions (sum, mean, max, min)
- AD integration (`vref` creates graph nodes)

✅ **Compiler Infrastructure**
- LLVM 14+ backend with 19 modular codegen components
- 5-phase compilation: macro expansion → parsing → type checking → LLVM IR → optimization
- AOT compiler (`eshkol-run`) and JIT REPL (`eshkol-repl`)
- Function cache for lambda deduplication
- Tail call optimization

✅ **Standard Library**
- List operations (map, filter, fold, take, drop, zip, etc.)
- Higher-order functions (compose, curry, flip, apply)
- String operations, file I/O
- Hash tables, JSON parsing
- CSV handling, Base64 encoding

✅ **Build System**
- CMake-based cross-platform build
- Docker containerization
- CI/CD pipelines
- Homebrew packaging

---

## Post-v1.0 Roadmap

### v1.1 - Ecosystem Foundations (Q1 2026)

**Package Management**
- Central package repository
- Dependency resolution
- Versioning system (`@1.0.0` syntax)
- Lock files for reproducible builds

**Developer Tooling**
- Language Server Protocol (LSP) implementation
- VS Code extension with syntax highlighting, autocomplete
- Source-level debugger (beyond GDB wrapper)
- Documentation generator from type annotations

**Language Features**
- `eval` implementation (leverage homoiconicity)
- Full `call/cc` support (currently AST-defined but not fully implemented)
- Pattern matching expansion (beyond basic `match`)

### v1.2 - Performance and Interoperability (Q1-2 2026)

**Optimization**
- Profile-guided optimization (PGO) integration
- Whole-program optimization across modules
- SIMD auto-vectorization improvements
- Specialized tensor operations for common ML patterns

**Foreign Function Interface (FFI)**
- C/C++ library integration
- Automatic binding generation from headers
- Zero-copy data exchange with NumPy, PyTorch
- Shared memory for inter-process communication

**GPU Acceleration**
- CUDA backend for tensor operations
- Metal backend (macOS)
- Vulkan compute shaders
- Automatic host/device memory management

### v1.3 - Advanced Type System (Q2 2026)

**Type System Enhancements**
- Effect system for side-effect tracking
- Row polymorphism for extensible records
- Refinement types (dependent types with predicates)
- Type-level computation (type families)

**Numeric Tower**
- Exact rational arithmetic (bignum library integration)
- Complex numbers (full implementation beyond AD context)
- Arbitrary precision floating point
- Symbolic algebra system

---

## Long-term Vision

### Scientific Computing Maturity

**Numerical Libraries** (Beyond v1.3)
- Differential equation solvers (ODE, PDE, SDE)
- Optimization algorithms (BFGS, L-BFGS, Adam, etc.)
- Monte Carlo methods (MCMC, Hamiltonian MC)
- Signal processing (FFT, filtering, wavelets)
- Statistical distributions and sampling

**Visualization**
- 2D plotting via OpenGL/Metal
- 3D scientific visualization
- Interactive notebooks (Jupyter-style)
- Real-time data streaming

### Machine Learning Ecosystem

**Neural Network Framework**
- High-level layer API (Dense, Conv2D, LSTM)
- Automatic model parallelism
- Distributed training (multi-GPU, multi-node)
- Model serialization and deployment

**Specialized Algorithms**
- Bayesian inference (variational, MCMC)
- Reinforcement learning primitives
- Meta-learning and few-shot learning
- Neural architecture search

### Systems Programming

**Embedded and Real-time**
- Bare-metal compilation (no OS)
- Hard real-time guarantees
- Interrupt handlers
- Memory-mapped I/O

**Multimedia**
- Audio processing (DSP, synthesis, analysis)
- Image processing (filters, transforms, codecs)
- Video encoding/decoding
- Real-time graphics

### Research Features

**Quantum Computing Integration**
- Quantum circuit simulation (32-qubit currently implemented)
- Variational quantum algorithms
- Quantum-classical hybrid algorithms
- Integration with quantum hardware backends

**Neuro-Symbolic AI**
- Logic programming integration
- Knowledge representation
- Reasoning with neural networks
- Differentiable theorem proving

---

## Release Philosophy

### API Stability

**v1.x releases** commit to:
- Backward compatibility for all documented APIs
- Deprecation cycles (minimum 6 months warning)
- Migration guides for breaking changes
- Semantic versioning (v1.MINOR.PATCH)

**Breaking changes** require:
- Major version bump (v2.0, v3.0, etc.)
- Comprehensive migration documentation
- Compatibility shims where feasible
- Community input via RFC process

### Feature Development Process

1. **Research** - Explore design space, review literature
2. **RFC** - Public request for comments on design
3. **Prototype** - Experimental implementation
4. **Testing** - Comprehensive test suite
5. **Documentation** - Technical docs, tutorials, examples
6. **Release** - Integration into stable branch

### Platform Support

**Tier 1** (fully supported, tested in CI):
- Linux x86-64 (Ubuntu 20.04+, Debian 11+)
- macOS x86-64 and ARM64 (macOS 11+)

**Tier 2** (best effort, community-tested):
- Windows x86-64 (Windows 10+, MSVC 2019+)
- FreeBSD, OpenBSD
- Linux ARM64 (Raspberry Pi 4+)
- Linux RISC-V

**Tier 3** (experimental, community-maintained):
- WebAssembly
- Embedded platforms (ARM Cortex-M)
- Other Unix-like systems

---

## Contributing

Eshkol development is open to contributions. Priority areas:

**High-impact contributions:**
- Standard library expansion (data structures, algorithms)
- Documentation improvements
- Test coverage expansion
- Performance benchmarking

**Advanced contributions:**
- LLVM optimization passes
- Type system enhancements
- GPU backend development
- FFI bindings for popular libraries

See CONTRIBUTING.md for guidelines.

---

## See Also

- [Feature Matrix](../FEATURE_MATRIX.md) - Implementation status of all features
- [Language Overview](OVERVIEW.md) - Design philosophy and technical positioning
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - Implementation details
