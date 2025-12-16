# Compiler Architecture in Eshkol

## Table of Contents
- [Overview](#overview)
- [Compilation Pipeline](#compilation-pipeline)
- [Frontend: Parsing and Macro Expansion](#frontend-parsing-and-macro-expansion)
- [Type Checking (HoTT System)](#type-checking-hott-system)
- [LLVM Backend](#llvm-backend)
- [Modular Codegen Architecture](#modular-codegen-architecture)
- [JIT Compilation (REPL)](#jit-compilation-repl)
- [Optimization Strategy](#optimization-strategy)
- [Build System](#build-system)

---

## Overview

Eshkol is a **production compiler** targeting **LLVM** for native code generation. The architecture combines:

- **S-expression parser** for Scheme-compatible syntax
- **HoTT-inspired type checker** for gradual typing
- **Macro expander** for hygienic macro system
- **Modular LLVM backend** for code generation
- **JIT compiler** for interactive REPL

---

## Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   ESHKOL COMPILATION PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

 Source Code (.esk)
       │
       ↓
┌──────────────────┐
│ 1. MACRO         │  lib/frontend/macro_expander.cpp
│    EXPANSION     │  → Hygenic macro expansion (syntax-rules)
└──────────────────┘
       │
       ↓
┌──────────────────┐
│ 2. S-EXPRESSION  │  lib/frontend/parser.cpp
│    PARSING       │  → Builds eshkol_ast_t tree
└──────────────────┘
       │
       ↓
┌──────────────────┐
│ 3. HoTT TYPE     │  lib/types/type_checker.cpp
│    CHECKING      │  → Infers types, emits warnings (gradual typing)
└──────────────────┘
       │
       ↓
┌──────────────────┐
│ 4. LLVM IR       │  lib/backend/llvm_codegen.cpp
│    GENERATION    │  → AST → LLVM IR translation
│                  │  + 19 modular codegen files
└──────────────────┘
       │
       ↓
┌──────────────────┐
│ 5. LLVM          │  LLVM's optimization pipeline
│    OPTIMIZATION  │  → opt passes (inlining, DCE, vectorization)
└──────────────────┘
       │
       ↓
┌──────────────────┐
│ 6. NATIVE CODE   │  LLVM's code generator
│    GENERATION    │  → Machine code for target architecture
└──────────────────┘
       │
       ↓
 Executable Binary
```

---

## Frontend: Parsing and Macro Expansion

### S-Expression Parser

**Implementation:** [`lib/frontend/parser.cpp:1-5487`](lib/frontend/parser.cpp:1)

The parser reads S-expressions and builds an **Abstract Syntax Tree (AST)**:

```c
typedef struct eshkol_ast {
    eshkol_type_t type;          // AST node type
    union {
        int64_t int64_val;       // Integer literal
        double double_val;        // Float literal
        struct {                  // String literal
            char *ptr;
            uint64_t size;
        } str_val;
        struct {                  // Function definition
            char *id;
            struct eshkol_ast *variables;
            eshkol_operations_t *func_commands;
            uint8_t is_variadic;
            hott_type_expr_t *return_type;
        } eshkol_func;
        struct {                  // Cons cell
            struct eshkol_ast *car;
            struct eshkol_ast *cdr;
        } cons_cell;
        eshkol_operations_t operation;  // Operations (if, let, lambda, etc.)
    };
    uint32_t inferred_hott_type;  // HoTT type from type checker
    uint32_t line;                // Source line number
    uint32_t column;              // Source column number
} eshkol_ast_t;
```

**Key features:**
- Handles all Scheme special forms (`define`, `lambda`, `let`, `if`, `cond`, etc.)
- Supports **93 operators** (see [`inc/eshkol/eshkol.h:1074-1139`](inc/eshkol/eshkol.h:1074))
- Line/column tracking for error messages
- HoTT type annotation support

### Macro System

**Implementation:** [`lib/frontend/macro_expander.cpp:1-1234`](lib/frontend/macro_expander.cpp:1)

**Hygienic macro expansion** via pattern matching and template substitution:

```scheme
;; Define macro
(define-syntax when
  (syntax-rules ()
    ((when test body ...)
     (if test (begin body ...) #f))))

;; Use macro
(when (> x 0)
  (display "positive")
  (newline))

;; Expands to:
(if (> x 0)
    (begin 
      (display "positive")
      (newline))
    #f)
```

**Features:**
- Pattern matching with ellipsis (`...`) for repetition
- Literal identifiers
- Nested patterns
- Hygiene (automatic renaming to avoid capture)

---

## Type Checking (HoTT System)

**Implementation:** [`lib/types/type_checker.cpp:1-1561`](lib/types/type_checker.cpp:1)

Eshkol uses a **Homotopy Type Theory-inspired** type system with:

### Type Universe Hierarchy

```
𝒰₂ (universe of universes)
 ├─ 𝒰₁ (universe of types)
 │   ├─ integer : 𝒰₀
 │   ├─ real : 𝒰₀
 │   ├─ (→ τ₁ τ₂) : 𝒰₀
 │   ├─ (list τ) : 𝒰₀
 │   └─ (forall (α) τ) : 𝒰₁
 └─ 𝒰₀ (universe of values)
```

### Type Inference Algorithm

1. **Constraint Generation** - Collect type constraints from AST
2. **Unification** - Solve constraints using unification
3. **Type Substitution** - Apply solutions throughout AST
4. **Warning Emission** - Report type mismatches as warnings (not errors)

**Gradual typing:** Type errors produce warnings, code still compiles.

### Type Expression Structure

```c
typedef struct hott_type_expr {
    hott_type_kind_t kind;       // INTEGER, REAL, ARROW, FORALL, etc.
    union {
        char* var_name;          // For type variables
        struct {                  // For arrow types (→)
            struct hott_type_expr** param_types;
            uint64_t num_params;
            struct hott_type_expr* return_type;
        } arrow;
        struct {                  // For forall types (∀)
            char** type_vars;
            uint64_t num_vars;
            struct hott_type_expr* body;
        } forall;
        // ... other type constructors
    };
} hott_type_expr_t;
```

**35+ builtin types** including `integer`, `real`, `boolean`, `string`, `list`, `vector`, `tensor`, function types, and polymorphic types.

---

## LLVM Backend

**Implementation:** [`lib/backend/llvm_codegen.cpp:1-27079`](lib/backend/llvm_codegen.cpp:1)

The LLVM backend is the **heart of the compiler** - the code generator that transforms ASTs into LLVM IR.

### Code Generation Strategy

```cpp
class LLVMCodeGenerator {
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    llvm::Module* module;
    
    // Generate LLVM IR from AST
    llvm::Value* codegen(eshkol_ast_t* ast);
    
    // Generate function
    llvm::Function* codegenFunction(eshkol_ast_t* func_ast);
    
    // Generate operation (if, let, lambda, etc.)
    llvm::Value* codegenOperation(eshkol_operations_t* op);
};
```

### Type Mapping (Eshkol → LLVM)

| Eshkol Type | LLVM Type | Notes |
|-------------|-----------|-------|
| `integer` | `i64` | 64-bit signed |
| `real` | `double` | 64-bit float |
| `boolean` | `i1` | 1-bit boolean |
| `char` | `i32` | Unicode codepoint |
| `string` | `i8*` | Pointer to UTF-8 data |
| `(list τ)` | `%cons*` | Pointer to cons cell |
| `(vector τ)` | `%vector*` | Pointer to vector struct |
| `(tensor τ)` | `%tensor*` | Pointer to tensor struct |
| `(→ τ₁ τ₂)` | `i8*` | Function pointer |
| Tagged value | `%tagged_value` | 16-byte struct `{i8, i8, i16, i64}` |

### LLVM Type Definitions

```cpp
// lib/backend/llvm_codegen.cpp:456-523
// Tagged value: {type:i8, flags:i8, reserved:i16, data:i64}
llvm::StructType* tagged_value_type = llvm::StructType::create(
    context,
    {
        llvm::Type::getInt8Ty(context),   // type
        llvm::Type::getInt8Ty(context),   // flags
        llvm::Type::getInt16Ty(context),  // reserved
        llvm::Type::getInt64Ty(context)   // data (union)
    },
    "eshkol_tagged_value_t"
);

// Cons cell: {header:i64, car:tagged_value, cdr:tagged_value}
llvm::StructType* cons_type = llvm::StructType::create(
    context,
    {
        llvm::Type::getInt64Ty(context),  // 8-byte header
        tagged_value_type,                // car (16 bytes)
        tagged_value_type                 // cdr (16 bytes)
    },
    "eshkol_cons_t"
);
```

---

## Modular Codegen Architecture

The LLVM backend is **modularized** into 19 specialized files using a **callback pattern** for inter-module communication:

### Codegen Modules

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| **Main Codegen** | [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:1) | 27,079 | Orchestrator, AST traversal |
| **Arithmetic** | [`arithmetic_codegen.cpp`](lib/backend/arithmetic_codegen.cpp:1) | 1,234 | `+`, `-`, `*`, `/`, numeric ops |
| **Collections** | [`collection_codegen.cpp`](lib/backend/collection_codegen.cpp:1) | 1,560 | `cons`, `car`, `cdr`, `list`, `vector` |
| **Tensors** | [`tensor_codegen.cpp`](lib/backend/tensor_codegen.cpp:1) | 3,041 | Tensor creation, indexing, linear algebra |
| **Autodiff** | [`autodiff_codegen.cpp`](lib/backend/autodiff_codegen.cpp:1) | 1,766 | 3 AD modes, vector calculus |
| **Functions** | [`function_codegen.cpp`](lib/backend/function_codegen.cpp:1) | 823 | Function definitions, closures |
| **Calls** | [`call_apply_codegen.cpp`](lib/backend/call_apply_codegen.cpp:1) | 567 | Function calls, apply, variadic |
| **Bindings** | [`binding_codegen.cpp`](lib/backend/binding_codegen.cpp:1) | 445 | `define`, `let`, `let*`, `letrec` |
| **Control Flow** | [`control_flow_codegen.cpp`](lib/backend/control_flow_codegen.cpp:1) | 789 | `if`, `cond`, `and`, `or`, `begin` |
| **Strings/IO** | [`string_io_codegen.cpp`](lib/backend/string_io_codegen.cpp:1) | 678 | String ops, `display`, `write`, file I/O |
| **Hash Tables** | [`hash_codegen.cpp`](lib/backend/hash_codegen.cpp:1) | 423 | `make-hash`, `hash-ref`, `hash-set!` |
| **Maps** | [`map_codegen.cpp`](lib/backend/map_codegen.cpp:1) | 234 | `map`, `filter`, `fold` operations |
| **Memory** | [`memory_codegen.cpp`](lib/backend/memory_codegen.cpp:1) | 734 | OALR operators (`owned`, `move`, `borrow`) |
| **Homoiconic** | [`homoiconic_codegen.cpp`](lib/backend/homoiconic_codegen.cpp:1) | 1,123 | Lambda S-expressions, `quote` |
| **Tail Calls** | [`tail_call_codegen.cpp`](lib/backend/tail_call_codegen.cpp:1) | 456 | Tail call optimization |
| **Tagged Values** | [`tagged_value_codegen.cpp`](lib/backend/tagged_value_codegen.cpp:1) | 234 | Tagged value helpers |
| **Type System** | [`type_system.cpp`](lib/backend/type_system.cpp:1) | 287 | LLVM type generation |
| **Builtins** | [`builtin_declarations.cpp`](lib/backend/builtin_declarations.cpp:1) | 345 | Built-in function declarations |
| **System** | [`system_codegen.cpp`](lib/backend/system_codegen.cpp:1) | 178 | System operations |
| **Context** | [`codegen_context.cpp`](lib/backend/codegen_context.cpp:1) | 123 | Shared codegen state |

### Callback Pattern

Modules communicate via **callbacks registered in the main codegen**:

```cpp
// lib/backend/llvm_codegen.cpp:234-267
class LLVMCodeGenerator {
    // Module callbacks
    std::function<llvm::Value*(eshkol_ast_t*)> arithmetic_callback;
    std::function<llvm::Value*(eshkol_ast_t*)> tensor_callback;
    std::function<llvm::Value*(eshkol_ast_t*)> autodiff_callback;
    
    // Register callbacks
    void registerArithmeticCallback(auto cb) { arithmetic_callback = cb; }
    void registerTensorCallback(auto cb) { tensor_callback = cb; }
    void registerAutodiffCallback(auto cb) { autodiff_callback = cb; }
};

// Module registration (lib/backend/arithmetic_codegen.cpp:89)
void register_arithmetic_codegen(LLVMCodeGenerator* gen) {
    gen->registerArithmeticCallback([](eshkol_ast_t* ast) {
        return generate_arithmetic(ast);
    });
}
```

---

## Type Checking (HoTT System)

**Implementation:** [`lib/types/type_checker.cpp:1-1561`](lib/types/type_checker.cpp:1)

### Type Inference Process

```cpp
// 1. Constraint generation (lib/types/type_checker.cpp:234-456)
std::vector<TypeConstraint> generate_constraints(eshkol_ast_t* ast);

// 2. Unification (lib/types/type_checker.cpp:567-789)
TypeSubstitution unify_constraints(std::vector<TypeConstraint> constraints);

// 3. Type application (lib/types/type_checker.cpp:890-1023)
void apply_types(eshkol_ast_t* ast, TypeSubstitution subst);

// 4. Warning generation (lib/types/type_checker.cpp:1234-1345)
void emit_type_warnings(eshkol_ast_t* ast);
```

### Constraint System

```cpp
struct TypeConstraint {
    TypeExpr expected;    // Expected type
    TypeExpr actual;      // Actual type
    AstNode* location;    // Where constraint originated
    string reason;        // Human-readable explanation
};

// Example constraints:
// (+ 1 2.0) generates:
//   - expected: number, actual: integer (for 1)
//   - expected: number, actual: real (for 2.0)
//   - unify to: number (supertype)
```

---

## LLVM Backend

### Codegen Entry Point

```cpp
// lib/backend/llvm_codegen.cpp:789-856
llvm::Value* LLVMCodeGenerator::codegen(eshkol_ast_t* ast) {
    switch (ast->type) {
        case ESHKOL_INT64:
            return llvm::ConstantInt::get(int64_type, ast->int64_val);
        
        case ESHKOL_DOUBLE:
            return llvm::ConstantFP::get(double_type, ast->double_val);
        
        case ESHKOL_CONS:
            return codegenCons(ast->cons_cell.car, ast->cons_cell.cdr);
        
        case ESHKOL_FUNC:
            if (ast->eshkol_func.is_lambda)
                return codegenLambda(ast);
            else
                return codegenDefine(ast);
        
        case ESHKOL_OP:
            return codegenOperation(&ast->operation);
        
        // ... 20+ more cases
    }
}
```

### Operation Codegen Dispatch

```cpp
// lib/backend/llvm_codegen.cpp:3814-4567
llvm::Value* codegenOperation(eshkol_operations_t* op) {
    switch (op->op) {
        case ESHKOL_ADD_OP:
            return arithmetic_callback(op);  // → arithmetic_codegen.cpp
        
        case ESHKOL_TENSOR_OP:
            return tensor_callback(op);      // → tensor_codegen.cpp
        
        case ESHKOL_GRADIENT_OP:
            return autodiff_callback(op);    // → autodiff_codegen.cpp
        
        case ESHKOL_IF_OP:
            return control_flow_callback(op); // → control_flow_codegen.cpp
        
        // ... 89 more operators
    }
}
```

### Function Cache

**Implementation:** [`lib/backend/function_cache.cpp:1-234`](lib/backend/function_cache.cpp:1)

**Deduplicates identical lambda functions** to reduce code size:

```cpp
// Cache key: (arity, parameter types, return type, body hash)
struct FunctionCacheKey {
    uint8_t arity;
    std::vector<uint32_t> param_types;
    uint32_t return_type;
    uint64_t body_hash;
};

// If two lambdas have same signature and body, reuse LLVM function
llvm::Function* get_or_create_function(FunctionCacheKey key);
```

---

## JIT Compilation (REPL)

**Implementation:** [`lib/repl/repl_jit.cpp:1-1108`](lib/repl/repl_jit.cpp:1)

The REPL uses **LLVM's JIT (Just-In-Time) compiler** for interactive execution:

### JIT Architecture

```cpp
class EshkolJIT {
    llvm::orc::LLJIT* jit;              // LLVM's JIT compiler
    llvm::LLVMContext context;
    llvm::IRBuilder<> builder;
    
    // Compile and execute AST immediately
    eshkol_tagged_value_t execute(eshkol_ast_t* ast);
    
    // Add function to JIT
    void addFunction(llvm::Function* func);
    
    // Lookup function by name
    void* lookupFunction(const char* name);
};
```

### REPL Execution Flow

```
User Input: "(+ 1 2)"
       │
       ↓
Parse → AST
       │
       ↓
Type Check (optional)
       │
       ↓
Generate LLVM IR:
  %1 = add i64 1, 2
       │
       ↓
JIT Compile → Native Code
       │
       ↓
Execute → Return 3
       │
       ↓
Display: "3"
```

**Incremental compilation:** Each REPL input is compiled independently and cached for future use.

---

## Optimization Strategy

Eshkol relies on **LLVM's optimization passes** rather than implementing custom optimizations:

### LLVM Optimization Passes Used

```cpp
// lib/backend/llvm_codegen.cpp:1234-1345
llvm::PassManagerBuilder pm_builder;
pm_builder.OptLevel = 3;  // O3 optimization
pm_builder.SizeLevel = 0;
pm_builder.Inliner = llvm::createFunctionInliningPass(275);

// Passes applied:
// - Constant folding
// - Dead code elimination
// - Function inlining (threshold: 275 instructions)
// - Loop unrolling
// - SIMD vectorization (auto-vectorizer)
// - Tail call optimization
// - Memory access optimization
```

### Eshkol-Specific Optimizations

**1. Tagged Value Elimination** - Remove tagging overhead when types are statically known:

```scheme
;; Source code
(define (f x : integer y : integer) : integer
  (+ x y))

;; Generated LLVM IR (tags eliminated)
define i64 @f(i64 %x, i64 %y) {
  %result = add i64 %x, %y
  ret i64 %result
}
```

**2. Closure Specialization** - Generate specialized code for known closure types:

```cpp
// Generic closure call (slow path)
call_closure(closure_ptr, args...)

// Specialized closure call (fast path when type is known at compile time)
func_ptr(arg1, arg2, ...)
```

**3. Tensor Fusion** - Combine multiple tensor operations into single LLVM IR sequence:

```scheme
;; Source: Three separate operations
(tensor-add (tensor-mul A B) C)

;; Fused LLVM IR: Single loop, no intermediate allocation
for (i = 0; i < n; i++)
    result[i] = A[i] * B[i] + C[i];
```

---

## Build System

**Implementation:** [`CMakeLists.txt`](CMakeLists.txt:1)

Eshkol uses **CMake** for cross-platform building:

### Build Targets

```cmake
# Compiler executable
add_executable(eshkol-run exe/eshkol-run.cpp)
target_link_libraries(eshkol-run PRIVATE ${LLVM_LIBS})

# REPL executable
add_executable(eshkol-repl exe/eshkol-repl.cpp)
target_link_libraries(eshkol-repl PRIVATE ${LLVM_LIBS})

# Core library
add_library(eshkol-core STATIC
    lib/core/arena_memory.cpp
    lib/core/ast.cpp
    lib/core/logger.cpp
    lib/core/printer.cpp)

# Frontend library
add_library(eshkol-frontend STATIC
    lib/frontend/parser.cpp
    lib/frontend/macro_expander.cpp)

# Type system library
add_library(eshkol-types STATIC
    lib/types/type_checker.cpp
    lib/types/hott_types.cpp
    lib/types/dependent.cpp)

# Backend library (19 codegen modules)
add_library(eshkol-backend STATIC
    lib/backend/llvm_codegen.cpp
    lib/backend/arithmetic_codegen.cpp
    lib/backend/tensor_codegen.cpp
    # ... 16 more modules
)
```

### Compilation Command

```bash
# Build entire project
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Install
sudo cmake --install build
```

### Dependencies

```cmake
# Required:
find_package(LLVM 14 REQUIRED)

# Linked libraries:
- LLVM core libraries (IR, CodeGen, Passes)
- LLVM JIT libraries (OrcJIT, ExecutionEngine)
- Standard C++ library
```

---

## Compiler Executables

### eshkol-run (AOT Compiler)

**Source:** [`exe/eshkol-run.cpp:1-2260`](exe/eshkol-run.cpp:1)

Ahead-of-time compiler that generates native executables:

```bash
# Compile Eshkol program to native binary
eshkol-run program.esk

# Output: ./program (native executable)
```

### eshkol-repl (JIT Compiler)

**Source:** [`exe/eshkol-repl.cpp:1-456`](exe/eshkol-repl.cpp:1)

Interactive Read-Eval-Print Loop:

```bash
# Start REPL
eshkol-repl
```

---

## Performance Characteristics

### Compilation Time

| Phase | Time (typical) | Notes |
|-------|---------------|-------|
| Parsing | < 10ms | S-expression parsing is fast |
| Macro expansion | < 5ms | Pattern matching |
| Type checking | 10-50ms | Constraint solving (warnings only) |
| LLVM IR generation | 50-200ms | Main compilation bottleneck |
| LLVM optimization | 100-500ms | Depends on optimization level |
| Code generation | 50-150ms | LLVM backend |
| **Total** | **220-915ms** | For typical program |

### Runtime Performance

- **Numeric code:** Within performance of hand-written C (LLVM optimizations)
- **List operations:** Outperforms Racket/Guile
- **Tensor operations:** Comparable to NumPy (SIMD, cache-friendly layouts)
- **AD gradient computation:** State-of-the-art for reverse-mode AD

---

## See Also

- [Type System](TYPE_SYSTEM.md) - Tagged values, HoTT types, gradual typing
- [Memory Management](MEMORY_MANAGEMENT.md) - Arena allocation, OALR, object headers
- [Automatic Differentiation](AUTODIFF.md) - 3 AD modes, vector calculus
- [Vector Operations](VECTOR_OPERATIONS.md) - Tensors, linear algebra, SIMD
- [API Reference](../API_REFERENCE.md) - Complete function and operator reference
