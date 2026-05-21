# Eshkol Development Roadmap (breakdown summary)

> **Position in the doc set:** this is the short-form roadmap surfaced
> from the per-subsystem `docs/breakdown/` documentation. For the
> canonical release plan, see [`ROADMAP.md`](../../ROADMAP.md) at the
> repo root; for engineering detail see
> [`COMPILER_ROADMAP.md`](../COMPILER_ROADMAP.md). When this document
> disagrees with `ROADMAP.md`, `ROADMAP.md` is correct.

## Table of Contents
- [Current Release](#current-release)
- [Released](#released)
- [Forward Roadmap](#forward-roadmap)
- [Long-term Vision](#long-term-vision)
- [Release Philosophy](#release-philosophy)

---

## Current Release

**Current Release:** v1.2.1-scale (2026-05-20). See [`../../RELEASE_NOTES.md`](../../RELEASE_NOTES.md) for the closeout and [`../../RELEASE_READINESS_REPORT.md`](../../RELEASE_READINESS_REPORT.md) for the v1.2 acceptance evidence (37-suite aggregate gate, 87 passing edge / security tests, zero outstanding audit blockers).

The remainder of this document is organised as: (1) a recap of features delivered across the four shipped lines (`v1.0-foundation`, `v1.1-accelerate`, `v1.2-scale`, `v1.2.1-scale`); (2) the forward roadmap for `v1.3-evolve` and beyond; and (3) long-term vision items that remain research-scope.

---

## Released

### v1.0-foundation

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
- LLVM 21 backend with roughly thirty modular codegen components
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

### v1.1-accelerate (Q1 2026, COMPLETE)

**Package management**
- ✅ Central package repository (`eshkol-pkg` with nine commands)
- ✅ Dependency resolution
- ✅ Versioning syntax (`@1.0.0`)
- ✅ Lock files for reproducible builds

**Developer tooling**
- ✅ Language Server Protocol (`eshkol-lsp`)
- ✅ VS Code extension with syntax highlighting, autocomplete, hover docs
- DWARF source-level debug info (`-g`); visual debugger remains a v1.3+ item
- Documentation generator from type annotations remains a v1.3+ item

**Language features**
- ✅ `eval` (via REPL JIT)
- ✅ Full `call/cc` with `dynamic-wind` (first-class continuations)
- ✅ Pattern matching (`match`) shipped; richer guard syntax in v1.3+

### v1.2-scale (Q1-2 2026, COMPLETE)

**Numeric tower**
- ✅ Exact bignum arithmetic with rational + double dispatch
- ✅ Full complex numbers (Smith's-formula division, magnitude overflow handling)
- ✅ Arbitrary-precision rationals with automatic int64 demotion
- Arbitrary-precision floating point and symbolic algebra remain v1.3+

**GPU acceleration**
- ✅ Metal backend (macOS) with SF64 / DF64 / F32 / FP24 / FP53 precision tiers
- ✅ CUDA backend (cuBLAS dispatch, cost-model gate)
- ✅ XLA / StableHLO backend with dual JIT / batch paths
- Vulkan compute shaders remain v1.3+

**Optimisation**
- ✅ SIMD vectorisation (auto-vectorised inner loops, SLP)
- ✅ Work-stealing parallel-map / parallel-fold / parallel-filter (4-12× on 24 cores)
- ✅ Specialised tensor operations for activations, convolutions, transformers
- Profile-guided optimisation and whole-program optimisation remain v1.3+

**Foreign function interface**
- ✅ Native HTTP client (libcurl), SQLite, subprocess, filesystem-watch FFI surfaces
- ✅ Agent FFI link wiring via `ESHKOL_HOST_AGENT_FFI_LINK_ARGS`
- Automatic binding generation from C headers and zero-copy NumPy / PyTorch bridges remain v1.3+

**Consciousness engine**
- ✅ Logic kernel (`logic.cpp` — unification, substitutions, knowledge base)
- ✅ Active inference (`inference.cpp` — factor graphs, belief propagation, free / expected free energy)
- ✅ Global workspace (`workspace.cpp` — modules, softmax competition, learnable CPTs)
- ✅ Twenty-two LLVM-codegen builtins covering the above

**Hardening (v1.2.1)**
- ✅ Stdlib `LinkOnceODR` linkage so user definitions can shadow stdlib symbols
- ✅ Parser line-tracking through `(load …)` and per-form REPL streams
- ✅ Closure variable capture in `dynamic-wind` / `call-cc` / `guard` / `raise`
- ✅ TCO context preservation across nested `letrec`
- ✅ Test-script grep tightened against `Failed: 0` false positives

## Forward Roadmap

### v1.3-evolve (target Q3 2026)

**Type system**
- Effect system for side-effect tracking
- Row polymorphism for extensible records
- Refinement / predicate types
- Type-level computation (type families)
- Visual debugger UI on top of existing DWARF data
- Documentation generator from type annotations

**Backends and runtime**
- Profile-guided optimisation (PGO)
- Whole-program optimisation across modules
- Vulkan compute shaders for non-NVIDIA / non-Apple GPUs
- Automatic C-header binding generation
- Zero-copy bridge surfaces for NumPy and PyTorch

**Bytecode VM as transformer (research track)**
- Stage 3 of the VM memory-ops → weight-matrix programme: lift the remaining 24 of 26 delegated opcodes into analytic weight constructions; raise `d_model` from 128 to 256; add Layers 6/7. See [`VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md`](VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md).

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
