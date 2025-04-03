# Eshkol Programming Language Documentation

## Introduction

Eshkol is a high-performance programming language that combines the elegance and expressiveness of Scheme with the performance and efficiency of C. It features arena-based memory management, a gradual typing system, powerful function composition capabilities, and built-in support for scientific computing and automatic differentiation.

The language is designed to be both approachable for beginners and powerful enough for advanced users, with a focus on performance-critical applications in scientific computing, machine learning, and systems programming.

## Table of Contents

### Core Documentation

- [Overview](OVERVIEW.md) - High-level overview of the Eshkol language
- [Getting Started](GETTING_STARTED.md) - Installation and first steps
- [Memory Management](MEMORY_MANAGEMENT.md) - Arena-based allocation system
- [Type System](TYPE_SYSTEM.md) - Gradual typing and type inference
- [Function Composition](FUNCTION_COMPOSITION.md) - Closures and function composition
- [Automatic Differentiation](AUTODIFF.md) - Built-in autodiff capabilities
- [Vector Operations](VECTOR_OPERATIONS.md) - SIMD-optimized vector math
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - How the Eshkol compiler works
- [Scheme Compatibility](SCHEME_COMPATIBILITY.md) - R5RS and R7RS-small support
- [Knowledge Graph](KNOWLEDGE_GRAPH.md) - Relationships between language components
- [Compilation Guide](COMPILATION_GUIDE.md) - Building Eshkol programs
- [eBPF Guide](EBPF_GUIDE.md) - Compiling for eBPF targets
- [Roadmap](ROADMAP.md) - Future development plans

### Examples and Tutorials

- [BTree Map](examples/BTREE_MAP.md) - BTreeMap implementation
- [Trie](examples/TRIE.md) - Trie data structure implementation
- [Sorting Algorithms](examples/SORTING.md) - QuickSort and MergeSort
- [Compression](examples/COMPRESSION.md) - Simple compression algorithms
- [Neural Network](examples/NEURAL_NETWORK.md) - Neural network implementation
- [Gradient Descent](examples/GRADIENT_DESCENT.md) - Optimization with autodiff

## How to Navigate This Documentation

This documentation is organized to support different learning paths:

1. **New to Eshkol?** Start with the [Overview](OVERVIEW.md) and [Getting Started](GETTING_STARTED.md) guides.
2. **Interested in language features?** Explore the core documentation sections on memory management, type system, etc.
3. **Looking for examples?** Check out the examples directory for complete implementations of common algorithms and data structures.
4. **Want to contribute?** See the [Roadmap](ROADMAP.md) for development priorities and opportunities to help.

Each document includes diagrams (using Mermaid) to visualize concepts, code examples to demonstrate usage, and explanations of the underlying principles.

## Building the Documentation

The documentation uses Mermaid for diagrams. The diagrams are embedded directly in the Markdown files and can be viewed in any Markdown viewer that supports Mermaid (such as GitHub or modern Markdown editors).

If you want to generate static images from the diagrams, you can use the Mermaid CLI:

```bash
# Install Mermaid CLI (requires Node.js)
npm install -g @mermaid-js/mermaid-cli

# Generate diagram images
npx @mermaid-js/mermaid-cli -i docs/aidocs/OVERVIEW.md -o docs/aidocs/overview_diagram.png
```

Note: The diagrams in this documentation are designed to be viewed directly in Markdown and don't require image generation for normal usage.

## License

The Eshkol language and this documentation are licensed under [LICENSE INFORMATION].
