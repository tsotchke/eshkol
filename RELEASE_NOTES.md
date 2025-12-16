# Eshkol v1.0.0-foundation - Production Release

**Release Date**: December 12, 2025

We are proud to announce the **production release** of Eshkol v1.0-foundation - a complete, production-ready programming language that delivers unprecedented integration of compiler-level automatic differentiation, deterministic arena memory management, and homoiconic native code execution.

## What is Eshkol?

Eshkol is a production-grade Scheme dialect built on LLVM infrastructure, designed for gradient-based optimization, neural network development, and scientific computing. It combines functional programming elegance with native performance while eliminating garbage collection entirely.

## v1.0-foundation Achievements

### Complete Production Compiler

Eshkol v1.0-foundation delivers a fully functional compiler with:

- **Modular LLVM backend** with 15 specialized code generation modules
- **HoTT-inspired gradual type system** with bidirectional type checking
- **Comprehensive parser** supporting S-expressions, type annotations, pattern matching, and macros
- **Ownership and escape analysis** for automatic allocation strategy optimization
- **Module system** with dependency resolution and circular dependency detection
- **Interactive REPL** with LLVM ORC JIT compilation
- **170+ test files** providing comprehensive verification

### Compiler-Integrated Automatic Differentiation

First-class AD system operating at compiler, runtime, and LLVM IR levels:

- **Forward-mode AD** using dual number arithmetic
- **Reverse-mode AD** with computational graph and tape stack
- **Symbolic AD** through AST transformation
- **Nested gradients** up to 32 levels deep
- **8 vector calculus operators**: derivative, gradient, jacobian, hessian, divergence, curl, laplacian, directional-derivative
- **Polymorphic implementation** supporting int64, double, dual numbers, AD nodes, and tensors

### Deterministic Memory Management (OALR)

Zero garbage collection with ownership-aware lexical regions:

- **Arena allocation** with O(1) bump-pointer allocation
- **Escape analysis** automatically determining stack/region/shared allocation
- **with-region syntax** for lexical memory scopes  
- **Ownership tracking** preventing use-after-move at compile time
- **Fully deterministic** - zero GC pauses for real-time applications

### Comprehensive Language Features

**300+ language elements including:**
- 39 special forms (define, lambda, let/let*/letrec, if/cond/case/match, etc.)
- 60+ list operations with full Scheme compatibility
- 30+ string utilities
- 25+ tensor operations
- 10 hash table operations
- Complete I/O system with ports and exception handling
- Hygienic macros (syntax-rules)
- Pattern matching with 7 pattern types
- Multiple return values (values, call-with-values, let-values)

### Rich Standard Library

Modular library organization with pure Eshkol implementations:

- **stdlib.esk** - Central module re-exporting core functionality
- **math.esk** - Linear algebra (det, inv, solve), numerical integration, root finding, statistics
- **core.functional** - compose, curry, flip combinators
- **core.list** - higher-order functions, transformations, queries, sorting
- **core.strings** - extended string manipulation
- **core.json** - JSON parsing and serialization
- **core.data** - CSV processing, Base64 encoding

### Production-Ready Infrastructure

- **Cross-platform**: macOS (Intel/Apple Silicon), Linux (x86_64/ARM64)
- **Docker containers**: Debian and Ubuntu images
- **CMake build system**: Modern, maintainable build infrastructure
- **Comprehensive documentation**: Language specification, user reference, API docs
- **Package generation**: Homebrew formula, Debian packages

## Installation

### Quick Start

```bash
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run a program
build/eshkol-run tests/neural/nn_working.esk

# Start interactive REPL
build/eshkol-repl
```

### System Requirements

- **LLVM** 10.0+ (14+ recommended)
- **CMake** 3.14+
- **C17/C++20 compiler** (GCC 8+, Clang 6+)
- **readline** (optional, for REPL enhancements)

## Example: Neural Network Training

```scheme
(require stdlib)

;; Sigmoid activation
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))

;; Mean squared error loss
(define (mse-loss pred target)
  (let ((diff (- pred target)))
    (* 0.5 (* diff diff))))

;; Forward pass
(define (forward weights bias input)
  (sigmoid (+ (tensor-dot weights input) bias)))

;; Compute loss gradient for backpropagation
(define (loss-gradient weights bias input target)
  (gradient 
    (lambda (params)
      (mse-loss 
        (forward (vref params 0) (vref params 1) input)
        target))
    (vector weights bias)))

;; Training works - automatic differentiation handles the calculus
```

## What Makes v1.0-foundation Special

### 1. Compiler-Integrated AD - Not a Library

Unlike JAX, PyTorch, or TensorFlow, Eshkol's automatic differentiation is built into the **compiler itself**, operating on AST, runtime values, and LLVM IR simultaneously. This enables differentiation of **any** Eshkol function without framework constraints or graph tracing overhead.

### 2. Homoiconic Native Code

Lambdas compile to LLVM-native code but retain their source S-expressions in closure structures, enabling both **runtime introspection** and **native performance** - a combination no other compiled language achieves.

### 3. Zero Garbage Collection

Arena-based memory management provides **fully deterministic** performance without GC pauses, making Eshkol suitable for real-time systems, trading algorithms, and control systems where predictable timing is critical.

### 4. Production-Quality Implementation

This isn't a research prototype - it's a complete compiler with comprehensive testing, thorough documentation, and a clear architectural foundation for future expansion.

## Documentation

- **[Language Specification](COMPLETE_LANGUAGE_SPECIFICATION.md)** - Complete technical specification
- **[Language Reference](ESHKOL_V1_LANGUAGE_REFERENCE.md)** - User-focused reference with examples
- **[Vision Documents](docs/vision/)** - Purpose, competitive analysis, roadmap
- **[Architecture Guide](docs/ESHKOL_V1_ARCHITECTURE.md)** - Technical architecture overview
- **[API Reference](docs/API_REFERENCE.md)** - Comprehensive function documentation
- **[Quickstart](docs/QUICKSTART.md)** - Hands-on tutorial

## Known Limitations

v1.0-foundation focuses on core capabilities. These features are planned for upcoming releases:

- **GPU acceleration** - Planned v1.2 (Q2 2026) with CUDA/Metal/Vulkan backends
- **Multi-threading** - Planned v1.1 (Q1 2026) with parallelism primitives
- **Distributed computing** - Planned v1.2 (Q2 2026)
- **XLA integration** - In active development for v1.1
- **SIMD vectorization** - In active development for v1.1

See [ROADMAP.md](ROADMAP.md) and [docs/vision/FUTURE_ROADMAP.md](docs/vision/FUTURE_ROADMAP.md) for detailed development plans.

## Next Steps

### For Users

1. **Explore the REPL**: `build/eshkol-repl`
2. **Try the examples**: `build/eshkol-run tests/autodiff/*.esk`
3. **Read the docs**: Start with [ESHKOL_V1_LANGUAGE_REFERENCE.md](ESHKOL_V1_LANGUAGE_REFERENCE.md)
4. **Experiment with AD**: The automatic differentiation system is production-ready

### For Contributors

1. **Review architecture**: [docs/ESHKOL_V1_ARCHITECTURE.md](docs/ESHKOL_V1_ARCHITECTURE.md)
2. **Check the roadmap**: [ROADMAP.md](ROADMAP.md) for v1.1/v1.2 plans
3. **See contribution guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
4. **Join development**: XLA backend and SIMD vectorization are priorities for v1.1

### For Researchers

1. **Study the AD implementation**: [docs/vision/TECHNICAL_WHITE_PAPER_V1.md](docs/vision/ADDENDUM_TECHNICAL_WHITE_PAPER_V1.md)
2. **Examine memory architecture**: [docs/breakdown/MEMORY_MANAGEMENT.md](docs/breakdown/MEMORY_MANAGEMENT.md)
3. **Analyze type system**: [docs/breakdown/TYPE_SYSTEM.md](docs/breakdown/TYPE_SYSTEM.md)
4. **Explore homoiconic closures**: [docs/vision/AI_FOCUS.md](docs/vision/AI_FOCUS.md)

## Acknowledgments

Eshkol v1.0-foundation represents years of research and implementation, synthesizing ideas from:
- **Scheme** for elegant functional programming
- **LLVM** for world-class code generation
- **Homotopy Type Theory** for rigorous type foundations
- **Region-based memory** research for deterministic allocation

We thank early testers and contributors who provided valuable feedback during development.

## License

Eshkol is released under the **MIT License** - see [LICENSE](LICENSE) for details.

## Contact

- **GitHub Repository**: https://github.com/tsotchke/eshkol
- **Issues**: Bug reports and feature requests
- **Discussions**: Technical questions and community engagement

---

**Eshkol v1.0-foundation** establishes a new standard for programming languages combining automatic differentiation, deterministic memory, and homoiconic native code. This is not a preview - this is only the beginning. Eshkol has a production-grade compiler ready for gradient-based computing, neural network development, and scientific applications where mathematical correctness and performance are non-negotiable.

*Where mathematical elegance meets uncompromising performance.*
