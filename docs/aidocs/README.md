# Eshkol Programming Language Documentation

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Introduction

Eshkol is a high-performance programming language that combines the elegance and expressiveness of Scheme with the performance and efficiency of C. It features arena-based memory management, a gradual typing system, powerful function composition capabilities, and built-in support for scientific computing and automatic differentiation.

The language is designed to be both approachable for beginners and powerful enough for advanced users, with a focus on performance-critical applications in scientific computing, machine learning, and systems programming.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/openSVM/eshkol.git
cd eshkol

# Build the compiler
make

# Add to your PATH
export PATH=$PATH:$(pwd)/bin
```

### Hello World Example

```scheme
;; hello.esh
(println "Hello, Eshkol!")
```

Compile and run:

```bash
eshkol compile hello.esh
./hello
```

## Table of Contents

### Core Documentation

- [Overview](OVERVIEW.md) - High-level overview of the Eshkol language
- [Memory Management](MEMORY_MANAGEMENT.md) - Arena-based allocation system
- [Type System](TYPE_SYSTEM.md) - Gradual typing and type inference
- [Function Composition](FUNCTION_COMPOSITION.md) - Closures and function composition
- [Automatic Differentiation](AUTODIFF.md) - Built-in autodiff capabilities

### Coming Soon

The following documentation is under development and will be available in future updates:

- Getting Started - Installation and first steps
- Vector Operations - SIMD-optimized vector math
- Compiler Architecture - How the Eshkol compiler works
- Scheme Compatibility - R5RS and R7RS-small support
- Knowledge Graph - Relationships between language components
- Compilation Guide - Building Eshkol programs
- eBPF Guide - Compiling for eBPF targets
- Roadmap - Future development plans

### Examples and Tutorials

The following examples are under development and will be available in future updates:

- BTree Map - BTreeMap implementation
- Trie - Trie data structure implementation
- Sorting Algorithms - QuickSort and MergeSort
- Compression - Simple compression algorithms
- Neural Network - Neural network implementation
- Gradient Descent - Optimization with autodiff

## How to Navigate This Documentation

This documentation is organized to support different learning paths:

1. **New to Eshkol?** Start with the [Overview](OVERVIEW.md) and check back soon for the Getting Started guide.
2. **Interested in language features?** Explore the core documentation sections on memory management, type system, etc.
3. **Looking for examples?** Check back soon for the examples directory with complete implementations of common algorithms and data structures.
4. **Want to contribute?** Check back soon for the Roadmap for development priorities and opportunities to help.

Each document includes diagrams (using Mermaid) to visualize concepts, code examples to demonstrate usage, and explanations of the underlying principles.

## Building the Documentation

The documentation uses Mermaid for diagrams. The diagrams are embedded directly in the Markdown files and can be viewed in any Markdown viewer that supports Mermaid (such as GitHub or modern Markdown editors).

If you want to generate static images from the diagrams, you can use the Mermaid CLI:

```bash
# Install Mermaid CLI (requires Node.js)
npm install -g @mermaid-js/mermaid-cli

# Generate diagram images
mmdc -i docs/aidocs/AUTODIFF.md -o diagrams/autodiff.png
```

## Contributing

We welcome contributions to both the Eshkol language and its documentation. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please ensure your documentation follows the established format and includes appropriate diagrams and examples.

## License

Eshkol is licensed under the MIT License. See the LICENSE file for details.
