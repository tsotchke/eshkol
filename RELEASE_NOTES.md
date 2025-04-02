# Eshkol 0.1.0-alpha - Early Developer Preview

**Release Date**: April 2, 2025

We're excited to announce the first public release of Eshkol, a high-performance LISP-like language designed for scientific computing and AI applications. This Early Developer Preview is intended for developers interested in exploring the language and potentially contributing to its development.

## Overview

Eshkol combines the elegant, expressive syntax of Scheme with the raw performance of C. It's designed specifically for scientific computing and artificial intelligence applications, delivering a balance between developer productivity and computational efficiency.

This Early Developer Preview showcases the core vision and capabilities of Eshkol, while being transparent about its current limitations and ongoing development.

## What's Included

### Core Language Features (65% Complete)
- Basic Scheme syntax and core special forms
- Lambda expressions with lexical scoping
- Core list operations (cons, car, cdr, etc.)
- Arena-based memory management

### Type System (55% Complete)
- Optional static type annotations
- Basic type inference
- Three typing approaches: implicit, inline explicit, and separate declarations
- Compile-time type checking (partial)

### Scientific Computing (70% Complete)
- Vector operations with optimized implementations
- Vector calculus operations (gradient, divergence, curl, laplacian)
- Automatic differentiation (forward and reverse mode)
- SIMD optimizations (partial)

### Function Composition (75% Complete)
- Basic function composition with JIT compilation
- Support for both x86-64 and ARM64 architectures
- Proper handling of closure calling conventions

### Development Tools (80% Complete)
- MCP tools for code analysis
- Type analysis tools
- Binding and lambda analysis
- AST visualization
- Closure memory visualization
- Mutual recursion analysis

### VSCode Integration
- Syntax highlighting
- Language configuration
- Enhanced editing experience

## Installation

### Prerequisites
- C/C++ Compiler (GCC 9.0+ or Clang 10.0+)
- CMake (version 3.12 or higher)
- Node.js (for MCP tools, version 14.0+ recommended)

### Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/tsotchke/eshkol.git
   cd eshkol
   ```

2. Create a build directory and build the project:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make
   ```

3. Install the MCP tools (optional, for development tools):
   ```bash
   cd ../eshkol-tools
   npm install
   ```

### Running Your First Eshkol Program

Create a file named `hello.esk`:

```scheme
(define (main)
  (display "Hello, Eshkol!"))
```

Compile and run:

```bash
./eshkol hello.esk
```

## Documentation

Comprehensive documentation is available to help you learn and master Eshkol:

- [README.md](README.md) - Project overview and quick start
- [ROADMAP.md](ROADMAP.md) - Development roadmap and priorities
- [KNOWN_ISSUES.md](docs/scheme_compatibility/KNOWN_ISSUES.md) - Current limitations and workarounds
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute to Eshkol

Additional documentation is available in the `docs/` directory:

- [Type System Documentation](docs/type_system/TYPE_SYSTEM.md)
- [Scheme Compatibility](docs/scheme_compatibility/SCHEME_COMPATIBILITY.md)
- [Vision and Purpose](docs/vision/PURPOSE_AND_VISION.md)

## Examples

The `examples/` directory contains sample programs demonstrating Eshkol's capabilities:

- Basic examples: `hello.esk`, `factorial.esk`, `fibonacci.esk`
- Type system examples: `implicit_typed.esk`, `inline_typed.esk`, `separate_typed.esk`
- Scientific computing examples: `vector_calculus.esk`, `autodiff_example.esk`
- Function composition examples: `function_composition.esk`, `advanced_function_composition.esk`

## Known Issues and Limitations

This Early Developer Preview has several known issues and limitations:

- **Function Composition**: Complex composition chains may not work correctly in all cases
- **Type System**: Type inference for autodiff functions is incomplete
- **Vector Types**: Vector return types not correctly handled in all cases
- **Tail Call Optimization**: Not yet implemented
- **Standard Library**: Many Scheme standard library functions not yet implemented

For a complete list of known issues and workarounds, see [KNOWN_ISSUES.md](docs/scheme_compatibility/KNOWN_ISSUES.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the development roadmap, including:

- Version 0.2.0-alpha (Function Composition Update) - Target: Q2 2025
- Version 0.3.0-alpha (Type System Update) - Target: Q3 2025
- Version 0.4.0-alpha (Performance Update) - Target: Q4 2025
- Version 1.0.0 (Full Release) - Target: Q3-Q4 2026

## Contributing

We welcome contributions to help us achieve these roadmap goals! See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

Eshkol is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

We would like to thank all the early testers and contributors who have helped shape Eshkol into what it is today.

## Contact

- GitHub Issues: For bug reports and feature requests
- GitHub Discussions: For general questions and community discussions

---

**Note**: This is an Early Developer Preview release and is not intended for production use. We appreciate your understanding and feedback as we continue to develop Eshkol.
