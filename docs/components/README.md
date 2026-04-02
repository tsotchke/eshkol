# Eshkol Components Documentation

This directory indexes component-level documentation for Eshkol's implementation.

## Compiler Components

For detailed component documentation, see:

**[Compiler Architecture](../breakdown/COMPILER_ARCHITECTURE.md)** - Complete pipeline documentation:
- Frontend: Macro expansion, parsing, type checking
- Backend: LLVM IR generation, optimization, native codegen
- Modular architecture: 21 specialized codegen files with callback pattern

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
- Orchestrates 21 specialized modules

**Specialized Modules**:
- Arithmetic: [`arithmetic_codegen.cpp`](../../lib/backend/arithmetic_codegen.cpp)
- Collections: [`collection_codegen.cpp`](../../lib/backend/collection_codegen.cpp)
- Tensors: [`tensor_codegen.cpp`](../../lib/backend/tensor_codegen.cpp)
- Autodiff: [`autodiff_codegen.cpp`](../../lib/backend/autodiff_codegen.cpp)
- Functions: [`function_codegen.cpp`](../../lib/backend/function_codegen.cpp)
- System: [`system_codegen.cpp`](../../lib/backend/system_codegen.cpp) — parallel, continuations, signal processing
- BLAS/GPU dispatch: [`blas_backend.cpp`](../../lib/backend/blas_backend.cpp)
- Strings/IO: [`string_io_codegen.cpp`](../../lib/backend/string_io_codegen.cpp)
- Control flow: [`control_flow_codegen.cpp`](../../lib/backend/control_flow_codegen.cpp)
- Bindings: [`binding_codegen.cpp`](../../lib/backend/binding_codegen.cpp)
- Hash tables: [`hash_codegen.cpp`](../../lib/backend/hash_codegen.cpp)
- Homoiconic: [`homoiconic_codegen.cpp`](../../lib/backend/homoiconic_codegen.cpp)
- Tail calls: [`tail_call_codegen.cpp`](../../lib/backend/tail_call_codegen.cpp)
- Tagged values: [`tagged_value_codegen.cpp`](../../lib/backend/tagged_value_codegen.cpp)
- Call/Apply: [`call_apply_codegen.cpp`](../../lib/backend/call_apply_codegen.cpp)
- Map operations: [`map_codegen.cpp`](../../lib/backend/map_codegen.cpp)
- Memory: [`memory_codegen.cpp`](../../lib/backend/memory_codegen.cpp)
- Complex numbers: [`complex_codegen.cpp`](../../lib/backend/complex_codegen.cpp)
- Parallel: [`parallel_codegen.cpp`](../../lib/backend/parallel_codegen.cpp), [`parallel_llvm_codegen.cpp`](../../lib/backend/parallel_llvm_codegen.cpp)
- Tensor backward: [`tensor_backward.cpp`](../../lib/backend/tensor_backward.cpp)

### v1.1 Runtime Components

**Consciousness Engine**:
- Logic: [`lib/core/logic.cpp`](../../lib/core/logic.cpp) — unification, substitutions, KB
- Inference: [`lib/core/inference.cpp`](../../lib/core/inference.cpp) — factor graphs, belief propagation
- Workspace: [`lib/core/workspace.cpp`](../../lib/core/workspace.cpp) — global workspace, softmax competition

**GPU Backends**:
- Metal: [`lib/backend/gpu/gpu_memory.mm`](../../lib/backend/gpu/gpu_memory.mm) — Apple Silicon, SF64
- CUDA: [`lib/backend/gpu/gpu_memory_cuda.cpp`](../../lib/backend/gpu/gpu_memory_cuda.cpp)

**XLA Backend**: [`lib/backend/xla/xla_runtime.cpp`](../../lib/backend/xla/xla_runtime.cpp)

### Runtime System

**Arena Memory** - [`lib/core/arena_memory.cpp`](../../lib/core/arena_memory.cpp)
- Global arena allocation
- Object header management
- Display system

**Bignum/Rational** - [`lib/core/bignum.cpp`](../../lib/core/bignum.cpp), [`lib/core/rational.cpp`](../../lib/core/rational.cpp)
- Arbitrary-precision integers and exact fractions

**AST Manipulation** - [`lib/core/ast.cpp`](../../lib/core/ast.cpp)
- AST data structures
- Symbolic differentiation helpers

**JIT Compiler** - [`lib/repl/repl_jit.cpp`](../../lib/repl/repl_jit.cpp)
- LLVM OrcJIT integration
- Interactive REPL with stdlib preloading

## See Also

- [Compiler Architecture](../breakdown/COMPILER_ARCHITECTURE.md) - High-level overview
- [Master Architecture](../ESHKOL_V1_ARCHITECTURE.md) - Complete technical deep dive
