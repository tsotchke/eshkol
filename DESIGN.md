# Eshkol Design Document

This document outlines the architecture and design decisions for the Eshkol programming language.

## Language Design

Eshkol is a Scheme-like language with C-level performance, designed specifically for scientific computing applications. It features:

- S-expression syntax (LISP/Scheme style)
- Optional static type annotations
- Direct compilation to C
- High-performance memory management
- Scientific computing optimizations

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│  Middle-End │────▶│   Backend   │────▶│  Generated  │
│             │     │             │     │             │     │     Code     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                    │
      ▼                   ▼                   ▼                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Lexer    │     │    Type     │     │    Code     │     │      C      │
│    Parser   │     │  Inference  │     │  Generation │     │    Code     │
│     AST     │     │Optimization │     │   Runtime   │     │     GCC     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Core Components

### 1. Memory Management System

The memory management system is designed for high performance and efficiency:

- **Arena Allocator**: Fast allocation with bulk deallocation
- **Object Pools**: Efficient allocation for frequently used objects
- **Region-Based Allocation**: Memory management for computation phases
- **Reference Counting**: For shared data structures
- **Optional Manual Memory Management**: For performance-critical code

### 2. Type System

The type system balances flexibility and performance:

- **Static Type Inference**: Determine types without annotations where possible
- **Optional Type Annotations**: Explicit type declarations when needed
- **Specialized Numeric Types**: Optimized for scientific computing
- **Gradual Typing**: Mix typed and untyped code
- **Scientific Types**: Vectors, matrices, and other domain-specific types

### 3. Compilation Pipeline

The compilation process follows these steps:

1. **Lexical Analysis**: Convert source to tokens
2. **Parsing**: Build abstract syntax tree
3. **Type Inference/Checking**: Apply type system
4. **Optimization**: Apply performance optimizations
5. **Code Generation**: Generate C code
6. **Compilation**: Use GCC to compile to machine code

### 4. Scientific Computing Features

Specialized features for scientific computing:

- **SIMD Support**: Vector instructions for array operations
- **Parallel Processing**: Multi-threading and work distribution
- **Specialized Containers**: Efficient vectors and matrices
- **Numerical Algorithms**: Optimized implementations
- **Library Integration**: Interface with existing scientific libraries

## Design Decisions

### Memory Management Approach

We chose a hybrid memory management approach:

- **Region-based allocation** for temporary calculations
- **Reference counting** for shared data structures
- **Manual memory management** for performance-critical code

This approach provides both safety and performance, allowing efficient memory use without garbage collection pauses.

### Type System Design

The type system is designed to:

- Provide C-like performance through specialization
- Maintain Scheme-like flexibility
- Support scientific computing with specialized types
- Allow gradual adoption of types

### C Interoperability

C interoperability is a core feature:

- Direct C function calls with no overhead
- Memory layout compatibility
- Inline C code support
- C ABI compliance

### Build System

We use CMake with Ninja for:

- Cross-platform compatibility
- Fast, parallel builds
- Good IDE integration
- Scalability for large projects

## Implementation Strategy

The implementation follows an incremental approach:

1. **Core Infrastructure**: Memory management, string handling, error reporting
2. **Frontend**: Lexer, parser, AST
3. **Backend**: C code generation, compilation pipeline
4. **Type System**: Type inference, checking, specialization
5. **Optimization**: Performance optimizations
6. **Scientific Features**: SIMD, parallelism, specialized types
