# Eshkol Architecture Documentation

This directory indexes Eshkol's architecture documentation.

## Primary Architecture Documents

### Comprehensive Technical Architecture

**[Eshkol v1.1-Architecture](../ESHKOL_V1_ARCHITECTURE.md)** - Master architecture document covering the entire system based on ~232,000 line code analysis. Includes:
- Compilation pipeline (5 phases)
- Runtime type system (tagged values, object headers)
- Memory management (OALR, arena allocation)
- Automatic differentiation (3 modes)
- Closure system (40-byte structures)
- LLVM backend architecture (21 modules)
- v1.1 extensions: XLA backend, GPU acceleration, consciousness engine, parallel primitives, exact arithmetic, signal processing

### Component Architecture

**[Compiler Architecture](../breakdown/COMPILER_ARCHITECTURE.md)** - LLVM backend details:
- Modular code generation (19 files, callback pattern)
- Type mapping (Eshkol → LLVM IR)
- JIT compilation (REPL)
- Optimization strategy (LLVM passes)

**[Type System Architecture](../breakdown/TYPE_SYSTEM.md)** - Triple-layer type system:
- Runtime types (8-bit tags, object headers)
- HoTT compile-time types (gradual typing)
- Dependent types (dimension checking)

**[Memory Architecture](../breakdown/MEMORY_MANAGEMENT.md)** - OALR system:
- Arena allocation (8KB blocks, global arena)
- Object headers (8 bytes prepended to all heap objects)
- Ownership modes (default, owned, move, borrow, shared, weak)

**[AD Architecture](../breakdown/AUTODIFF.md)** - Automatic differentiation:
- Symbolic mode (AST rewriting, 12 rules)
- Forward mode (dual numbers, 16 bytes)
- Reverse mode (computational graph, 32-level tape)

## See Also

- [API Reference](../API_REFERENCE.md) - Complete function reference
- [Feature Matrix](../FEATURE_MATRIX.md) - Implementation status
- [Overview](../aidocs/OVERVIEW.md) - Design philosophy and positioning
