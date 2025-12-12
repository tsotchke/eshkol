# Eshkol Components Documentation

This directory indexes component-level documentation for Eshkol's implementation.

## Compiler Components

For detailed component documentation, see:

**[Compiler Architecture](../aidocs/COMPILER_ARCHITECTURE.md)** - Complete pipeline documentation:
- Frontend: Macro expansion, parsing, type checking
- Backend: LLVM IR generation, optimization, native codegen
- Modular architecture: 19 specialized codegen files with callback pattern

## Implementation Components

### Frontend

**Parser** - [`lib/frontend/parser.cpp`](../../lib/frontend/parser.cpp)
- S-expression parsing (recursive descent)
- AST construction with 93 operators
- Line/column tracking for error messages

**Macro Expander** - [`lib/frontend/macro_expander.cpp`](../../lib/frontend/macro_expander.cpp)
- Hygienic macro expansion
- Pattern matching with ellipsis
- Symbol table for hygiene

### Type System

**HoTT Type Checker** - [`lib/types/type_checker.cpp`](../../lib/types/type_checker.cpp)
- Constraint generation and unification
- Gradual typing (warnings, not errors)
- Universe hierarchy (𝒰₀, 𝒰₁, 𝒰₂)

**Type Structures** - [`lib/types/hott_types.cpp`](../../lib/types/hott_types.cpp)
- Type expression manipulation
- 35+ builtin types

**Dependent Types** - [`lib/types/dependent.cpp`](../../lib/types/dependent.cpp)
- Compile-time value tracking
- Tensor dimension checking

### Backend (LLVM Code Generation)

**Main Codegen** - [`lib/backend/llvm_codegen.cpp`](../../lib/backend/llvm_codegen.cpp)
- AST → LLVM IR translation
- Orchestrates 19 specialized modules

**Specialized Modules**:
- Arithmetic: [`arithmetic_codegen.cpp`](../../lib/backend/arithmetic_codegen.cpp)
- Collections: [`collection_codegen.cpp`](../../lib/backend/collection_codegen.cpp)
- Tensors: [`tensor_codegen.cpp`](../../lib/backend/tensor_codegen.cpp)
- Autodiff: [`autodiff_codegen.cpp`](../../lib/backend/autodiff_codegen.cpp)
- Functions: [`function_codegen.cpp`](../../lib/backend/function_codegen.cpp)
- (...14 more modules)

### Runtime System

**Arena Memory** - [`lib/core/arena_memory.cpp`](../../lib/core/arena_memory.cpp)
- Global arena allocation
- Object header management
- Display system

**AST Manipulation** - [`lib/core/ast.cpp`](../../lib/core/ast.cpp)
- AST data structures
- Symbolic differentiation helpers

**JIT Compiler** - [`lib/repl/repl_jit.cpp`](../../lib/repl/repl_jit.cpp)
- LLVM OrcJIT integration
- Interactive REPL

## See Also

- [Compiler Architecture](../aidocs/COMPILER_ARCHITECTURE.md) - High-level overview
- [Master Architecture](../ESHKOL_V1_ARCHITECTURE.md) - Complete technical deep dive
