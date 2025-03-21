# Initial Design Conversation for Eshkol

This document contains the initial planning conversation for the Eshkol programming language. It captures the design decisions, architecture discussions, and implementation planning that formed the foundation of the project.

## Key Design Decisions

1. **Language Design**: Scheme-like syntax with optional static typing, compiling to C
2. **Memory Management**: Hybrid approach with arena allocation, region-based memory, and reference counting
3. **Type System**: Gradual typing with specialization for scientific computing
4. **Compilation Strategy**: Direct translation to C with optimization passes
5. **Project Structure**: Modular design with clear separation of concerns
6. **Build System**: CMake with Ninja for cross-platform compatibility
7. **File Extensions**: `.esk`, `.eskh`, `.eskir`, `.eskc`, etc.

## Implementation Plan

We decided on an incremental approach with always-working prototypes:

1. **Core Infrastructure**: Memory management, string handling, error reporting
2. **Frontend**: Lexer, parser, AST
3. **Backend**: C code generation, compilation pipeline
4. **Type System**: Type inference, checking, specialization
5. **Optimization**: Performance optimizations
6. **Scientific Features**: SIMD, parallelism, specialized types

## Memory Management Design

We designed a hybrid memory management system that includes:

- **Arena Allocator**: Fast allocation with bulk deallocation
- **Object Pools**: Efficient allocation for frequently used objects
- **Region-Based Allocation**: Memory management for computation phases
- **Reference Counting**: For shared data structures
- **Optional Manual Memory Management**: For performance-critical code

## Type System Design

The type system is designed to balance flexibility and performance:

- **Static Type Inference**: Determine types without annotations where possible
- **Optional Type Annotations**: Explicit type declarations when needed
- **Specialized Numeric Types**: Optimized for scientific computing
- **Gradual Typing**: Mix typed and untyped code
- **Scientific Types**: Vectors, matrices, and other domain-specific types

## Higher-Order Function Handling

For higher-order functions and closures, we designed:

- **Closure Representation**: Efficient representation of closures
- **Environment Capture**: Optimized environment capture
- **Function Specialization**: Specialized versions of higher-order functions
- **Inlining**: Aggressive inlining of known functions

## C Interoperability

C interoperability is a core feature:

- **Foreign Function Interface**: Call C functions directly
- **Data Representation**: Compatible memory layout
- **Inline C Code**: Embed C code within Eshkol
- **ABI Compatibility**: Match platform-specific calling conventions

## Next Steps

The immediate next steps are:

1. Set up project structure and build system
2. Implement core memory management (arena allocator)
3. Create basic lexer and parser
4. Implement simple C code generation
5. Create a minimal end-to-end compilation pipeline

This document serves as a reference for the project's original design intentions and can be consulted when making future design decisions.
