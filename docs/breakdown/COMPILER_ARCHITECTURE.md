# Compiler Architecture in Eshkol (v1.1-accelerate)

## Table of Contents

- [Overview](#overview)
- [Compilation Pipeline](#compilation-pipeline)
- [Frontend: Parsing and Macro Expansion](#frontend-parsing-and-macro-expansion)
- [Type Checking (HoTT System)](#type-checking-hott-system)
- [LLVM Backend](#llvm-backend)
- [Tagged Value Representation](#tagged-value-representation)
- [Modular Codegen Architecture](#modular-codegen-architecture)
- [v1.1-accelerate Feature Codegen](#v11-accelerate-feature-codegen)
- [Optimization Pipeline](#optimization-pipeline)
- [JIT Compilation (REPL)](#jit-compilation-repl)
- [Build System](#build-system)
- [Compiler Executables](#compiler-executables)
- [Runtime Architecture](#runtime-architecture)

---

## Overview

Eshkol is a **production compiler** targeting **LLVM IR** for native code generation. The v1.1-accelerate release extends the v1.0 foundation with GPU acceleration, parallel primitives, a consciousness engine, exact arithmetic (full R7RS numeric tower), first-class continuations, signal processing, a 75+ builtin ML framework, and WebAssembly output.

The architecture combines:

- **S-expression parser** for R7RS-compatible Scheme syntax
- **Hygienic macro expander** via `syntax-rules` pattern matching
- **HoTT-inspired type checker** for gradual typing (warnings, not errors)
- **Modular LLVM backend** with 21 specialized codegen modules (~232,000 lines)
- **JIT compiler** for interactive REPL via LLVM OrcJIT (LLJIT)
- **Pre-compiled standard library** (40 modules, compiled to `stdlib.o`)
- **GPU dispatch** with SIMD/cBLAS/Metal cost model selection
- **Work-stealing thread pool** for parallel higher-order functions

---

## Compilation Pipeline

The compiler executes a 5-phase pipeline. Source files (`.esk`) enter at Phase 1 and produce a native executable binary at the output.

```
 Source Code (.esk)
       |
       v
+------------------+
| 1. MACRO         |  lib/frontend/macro_expander.cpp (861 lines)
|    EXPANSION     |  Hygienic expansion via syntax-rules
+------------------+
       |
       v
+------------------+
| 2. S-EXPRESSION  |  lib/frontend/parser.cpp (7,540 lines)
|    PARSING       |  Builds eshkol_ast_t tree, 94 operation types
+------------------+
       |
       v
+------------------+
| 3. HoTT TYPE     |  lib/types/type_checker.cpp (1,999 lines)
|    CHECKING      |  Constraint generation + unification (non-blocking)
+------------------+
       |
       v
+------------------+
| 4. LLVM IR       |  lib/backend/llvm_codegen.cpp (34,928 lines)
|    GENERATION    |  AST -> LLVM IR via 21 codegen modules (~232,000 lines)
+------------------+
       |
       v
+------------------+
| 5. LLVM          |  LLVM PassBuilder + TargetMachine
|    OPT + CODEGEN |  New PassManager pipeline, then native code emission
+------------------+
       |
       v
 Executable Binary (or .o / .wasm)
```

**Key distinction from v1.0:** The v1.1 pipeline adds GPU kernel compilation (Metal/CUDA), parallel worker function generation with `LinkOnceODRLinkage`, consciousness engine runtime dispatch, and exact arithmetic promotion paths.

---

## Frontend: Parsing and Macro Expansion

### Macro System

**Implementation:** [`lib/frontend/macro_expander.cpp`](lib/frontend/macro_expander.cpp) (861 lines)

Hygienic macro expansion runs before parsing. The system supports:

- Pattern matching with ellipsis (`...`) for repetition
- Literal identifiers for keyword-like matching
- Nested patterns and template substitution
- Hygiene via automatic renaming to prevent variable capture

```scheme
;; Define macro
(define-syntax when
  (syntax-rules ()
    ((when test body ...)
     (if test (begin body ...) #f))))

;; Usage expands hygienically
(when (> x 0) (display "positive") (newline))
;; => (if (> x 0) (begin (display "positive") (newline)) #f)
```

Several R7RS derived forms (`case-lambda`, `parameterize`, `cond-expand`, `define-record-type`) are desugared during macro expansion or the subsequent parse phase.

### S-Expression Parser

**Implementation:** [`lib/frontend/parser.cpp`](lib/frontend/parser.cpp) (7,540 lines)

The parser is a recursive descent processor that builds an AST from S-expressions:

```c
// inc/eshkol/eshkol.h
typedef struct eshkol_ast {
    eshkol_type_t type;          // AST node type (ESHKOL_INT64, ESHKOL_FUNC, ESHKOL_OP, etc.)
    union {
        int64_t int64_val;       // Integer literal
        double double_val;       // Float literal
        struct { char *ptr; uint64_t size; } str_val;   // String literal
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
    uint32_t inferred_hott_type;  // HoTT type [TypeId:16][universe:8][flags:8]
    uint32_t line;
    uint32_t column;
} eshkol_ast_t;
```

**Key responsibilities:**
- 94 operation types (see `eshkol_op_t` enum in [`inc/eshkol/eshkol.h`](inc/eshkol/eshkol.h))
- Internal define to `letrec*` transformation (consecutive defines at body start only)
- `delay`/`delay-force` desugaring to promise constructors
- `define-record-type` to vector operation transformation
- `?x` syntax for logic variables (`ESHKOL_LOGIC_VAR_OP`)
- Line/column tracking for error messages
- HoTT type annotation attachment to AST nodes

**Critical invariant:** The `letrec*` transformation only collects consecutive `define` forms from the beginning of a body. Once a non-define expression appears, collection stops. This preserves side-effect ordering per R7RS semantics.

---

## Type Checking (HoTT System)

**Implementation:** [`lib/types/type_checker.cpp`](lib/types/type_checker.cpp) (1,999 lines)

Eshkol uses a Homotopy Type Theory-inspired type system with a universe hierarchy:

```
U_2 (universe of universes)
 +-- U_1 (universe of types)
 |    +-- integer : U_0
 |    +-- real : U_0
 |    +-- (-> t1 t2) : U_0
 |    +-- (list t) : U_0
 |    +-- (forall (a) t) : U_1
 +-- U_0 (universe of values)
```

### Type Inference Algorithm

1. **Constraint Generation** -- Collect type constraints from AST traversal
2. **Unification** -- Solve constraints using Robinson's unification algorithm
3. **Type Substitution** -- Apply solutions throughout the AST
4. **Warning Emission** -- Report type mismatches as warnings (not errors)

**Gradual typing:** Type errors produce warnings, code still compiles. This preserves Scheme's exploratory programming model while enabling aggressive optimization when types are statically known. The 16-byte tagged values store an 8-bit type field; when the type is known at compile time, the compiler generates untagged LLVM IR, eliminating tagging overhead.

### Type Expression Structure

```c
typedef struct hott_type_expr {
    hott_type_kind_t kind;       // INTEGER, REAL, ARROW, FORALL, etc.
    union {
        char* var_name;          // For type variables
        struct {                 // For arrow types
            struct hott_type_expr** param_types;
            uint64_t num_params;
            struct hott_type_expr* return_type;
        } arrow;
        struct {                 // For forall types
            char** type_vars;
            uint64_t num_vars;
            struct hott_type_expr* body;
        } forall;
        // ... other type constructors
    };
} hott_type_expr_t;
```

35+ built-in types including `integer`, `real`, `boolean`, `string`, `list`, `vector`, `tensor`, `complex`, `rational`, `bignum`, function types, and polymorphic types.

---

## LLVM Backend

**Implementation:** [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) (34,928 lines)

The LLVM backend is the heart of the compiler. It translates ASTs to LLVM IR and orchestrates 21 specialized codegen modules.

### Code Generator Class Structure

The main code generator holds `std::unique_ptr` member objects for each module -- not `std::function` callbacks:

```cpp
// lib/backend/llvm_codegen.cpp (simplified)
class LLVMCodeGenerator {
    // LLVM infrastructure
    std::unique_ptr<LLVMContext> context;
    std::unique_ptr<Module> module;
    std::unique_ptr<IRBuilder<>> builder;

    // Shared infrastructure
    std::unique_ptr<eshkol::TypeSystem> types;       // LLVM type cache
    std::unique_ptr<eshkol::FunctionCache> funcs;    // Lazy-loaded C library functions
    std::unique_ptr<eshkol::MemoryCodegen> mem;      // Arena allocation declarations
    std::unique_ptr<eshkol::CodegenContext> ctx_;     // Shared state for all modules

    // Codegen modules (owned unique_ptrs, NOT std::function callbacks)
    std::unique_ptr<eshkol::TaggedValueCodegen> tagged_;     // Pack/unpack tagged values
    std::unique_ptr<eshkol::BuiltinDeclarations> builtins_;  // Runtime function decls
    std::unique_ptr<eshkol::ComplexCodegen> complex_;        // Complex arithmetic
    std::unique_ptr<eshkol::ArithmeticCodegen> arith_;       // Polymorphic arithmetic
    std::unique_ptr<eshkol::CallApplyCodegen> call_apply_;   // Function calls, apply
    std::unique_ptr<eshkol::MapCodegen> map_;                // map/for-each/fold
    std::unique_ptr<eshkol::ControlFlowCodegen> flow_;       // if/cond/case/match/call-cc
    std::unique_ptr<eshkol::StringIOCodegen> strio_;         // String/IO operations
    std::unique_ptr<eshkol::CollectionCodegen> coll_;        // List/vector operations
    std::unique_ptr<eshkol::FunctionCodegen> func_;          // Lambda/closure codegen
    std::unique_ptr<eshkol::TensorCodegen> tensor_;          // Tensor/ML builtins
    std::unique_ptr<eshkol::AutodiffCodegen> autodiff_;      // Three AD modes
    std::unique_ptr<eshkol::BindingCodegen> binding_;        // let/letrec/define/set!
    std::unique_ptr<eshkol::HomoiconicCodegen> homoiconic_;  // quote, S-expressions, eval
    std::unique_ptr<eshkol::TailCallCodegen> tailcall_;      // TCO transformation
    std::unique_ptr<eshkol::SystemCodegen> system_;          // System/env/time/process
    std::unique_ptr<eshkol::HashCodegen> hash_;              // Hash table operations
    std::unique_ptr<eshkol::ParallelCodegen> parallel_;      // Parallel primitives
};
```

Each module is constructed via `std::make_unique` during initialization. Modules receive a `CodegenContext&` reference for shared LLVM state, and inter-module calls use **static wrapper callbacks** (via a `ControlFlowCallbacks` helper class), not `std::function` lambdas. For example:

```cpp
// Module initialization (lib/backend/llvm_codegen.cpp:2370+)
tensor_ = std::make_unique<eshkol::TensorCodegen>(*ctx_, *tagged_, *mem);
tensor_->setCodegenCallbacks(
    ControlFlowCallbacks::codegenASTWrapper,
    ControlFlowCallbacks::codegenTypedASTWrapper,
    ControlFlowCallbacks::typedToTaggedWrapper,
    this  // opaque pointer back to main codegen
);
```

This pattern inverts the typical dependency graph -- modules call back into the main codegen through thin static wrappers rather than holding direct references to each other.

---

## Tagged Value Representation

Every Eshkol value at runtime is a 16-byte tagged struct with **5 fields**:

```
+--------+--------+----------+---------+----------+
| type   | flags  | reserved | padding |   data   |
| (i8)   | (i8)   | (i16)    | (i32)   | (i64)    |
+--------+--------+----------+---------+----------+
  idx 0    idx 1    idx 2      idx 3     idx 4
```

### LLVM Type Definition

```cpp
// inc/eshkol/backend/type_system.h — TypeSystem class
// The tagged value struct is {i8, i8, i16, i32, i64} = 16 bytes
llvm::StructType* tagged_value_type = llvm::StructType::create(
    context,
    {
        llvm::Type::getInt8Ty(context),   // idx 0: type tag (0-10, 16-19)
        llvm::Type::getInt8Ty(context),   // idx 1: flags (EXACT, LINEAR, etc.)
        llvm::Type::getInt16Ty(context),  // idx 2: reserved
        llvm::Type::getInt32Ty(context),  // idx 3: padding (alignment)
        llvm::Type::getInt64Ty(context)   // idx 4: data (int64/double bits/pointer)
    },
    "eshkol_tagged_value_t"
);
```

### Field Index Constants

```cpp
// inc/eshkol/backend/type_system.h
static constexpr unsigned TAGGED_VALUE_TYPE_IDX = 0;     // uint8_t type
static constexpr unsigned TAGGED_VALUE_FLAGS_IDX = 1;    // uint8_t flags
static constexpr unsigned TAGGED_VALUE_RESERVED_IDX = 2; // uint16_t reserved
static constexpr unsigned TAGGED_VALUE_PADDING_IDX = 3;  // uint32_t padding
static constexpr unsigned TAGGED_VALUE_DATA_IDX = 4;     // int64_t data
```

**Critical:** The data field is at index **4**, not 3. Using index 3 accesses the padding field, which was the root cause of multiple null dereference bugs in v1.0.

### Type Tags

| Tag | Name | Data Encoding |
|:---:|:---|:---|
| 0 | `ESHKOL_VALUE_NULL` | Null/empty value |
| 1 | `ESHKOL_VALUE_INT64` | 64-bit signed integer (inline) |
| 2 | `ESHKOL_VALUE_DOUBLE` | Double as int64 bit pattern (inline) |
| 3 | `ESHKOL_VALUE_BOOL` | Boolean #t/#f (inline) |
| 4 | `ESHKOL_VALUE_CHAR` | Unicode codepoint (inline) |
| 5 | `ESHKOL_VALUE_SYMBOL` | Interned symbol pointer |
| 6 | `ESHKOL_VALUE_DUAL_NUMBER` | Forward-mode AD dual number pointer |
| 7 | `ESHKOL_VALUE_COMPLEX` | Complex number `{real:f64, imag:f64}` pointer |
| 8 | `ESHKOL_VALUE_HEAP_PTR` | Heap pointer (subtype in 8-byte object header) |
| 9 | `ESHKOL_VALUE_CALLABLE` | Callable pointer (closure, lambda-sexpr, ad-node) |
| 10 | `ESHKOL_VALUE_LOGIC_VAR` | Logic variable `?x` (var_id as int64) |

Tags 0-7 are immediate values (data stored inline). Tags 8-9 are consolidated pointer types -- the specific type is determined by reading the 8-byte object header at `pointer - 8`.

### Heap Subtypes (20 as of v1.1)

```c
// inc/eshkol/eshkol.h
HEAP_SUBTYPE_CONS         = 0,   // Cons cell (pair/list node)
HEAP_SUBTYPE_STRING       = 1,   // String (UTF-8 with length)
HEAP_SUBTYPE_VECTOR       = 2,   // Heterogeneous vector (16 bytes/element)
HEAP_SUBTYPE_TENSOR       = 3,   // N-dimensional numeric tensor (8 bytes/element)
HEAP_SUBTYPE_MULTI_VALUE  = 4,   // Multiple return values container
HEAP_SUBTYPE_HASH         = 5,   // Hash table / dictionary
HEAP_SUBTYPE_EXCEPTION    = 6,   // Exception object
HEAP_SUBTYPE_RECORD       = 7,   // User-defined record type
HEAP_SUBTYPE_BYTEVECTOR   = 8,   // Raw byte vector (R7RS)
HEAP_SUBTYPE_PORT         = 9,   // I/O port
HEAP_SUBTYPE_SYMBOL       = 10,  // Interned symbol
HEAP_SUBTYPE_BIGNUM       = 11,  // Arbitrary-precision integer
HEAP_SUBTYPE_SUBSTITUTION = 12,  // Logic engine: immutable binding map
HEAP_SUBTYPE_FACT         = 13,  // Logic engine: predicate + arguments
HEAP_SUBTYPE_KNOWLEDGE_BASE = 15, // Logic engine: collection of facts
HEAP_SUBTYPE_FACTOR_GRAPH = 16,  // Active inference: factor graph
HEAP_SUBTYPE_WORKSPACE    = 17,  // Global workspace: cognitive competition
HEAP_SUBTYPE_PROMISE      = 18,  // Lazy promise (delay/force)
HEAP_SUBTYPE_RATIONAL     = 19,  // Exact rational number
```

Subtypes 12-19 are new in v1.1-accelerate. The consolidation of 8 legacy pointer types (CONS_PTR, STRING_PTR, etc.) into `HEAP_PTR` + subtype headers freed the type tag space for these extensions.

---

## Modular Codegen Architecture

The LLVM backend distributes code generation across 21 specialized modules totaling approximately 232,000 lines. Each module is a C++ class instantiated as a `std::unique_ptr` member of the main `LLVMCodeGenerator`.

### Complete Module Table

| Module | Source File | Lines | Responsibility |
|:---|:---|---:|:---|
| **Main Codegen** | [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) | 34,928 | Orchestrator, AST dispatch, builtins, consciousness engine |
| **Tensor** | [`tensor_codegen.cpp`](lib/backend/tensor_codegen.cpp) | 19,187 | Tensor ops, 75+ ML builtins, BLAS/GPU dispatch |
| **Autodiff** | [`autodiff_codegen.cpp`](lib/backend/autodiff_codegen.cpp) | 3,694 | Forward/reverse/symbolic AD modes |
| **String/IO** | [`string_io_codegen.cpp`](lib/backend/string_io_codegen.cpp) | 2,975 | String ops, display/write, file I/O, JSON, CSV |
| **Parallel LLVM** | [`parallel_llvm_codegen.cpp`](lib/backend/parallel_llvm_codegen.cpp) | 2,401 | Work-stealing parallelism LLVM IR generation |
| **Arithmetic** | [`arithmetic_codegen.cpp`](lib/backend/arithmetic_codegen.cpp) | 2,332 | +, -, *, /, bignum, rational, complex dispatch |
| **Collections** | [`collection_codegen.cpp`](lib/backend/collection_codegen.cpp) | 2,139 | Vector, list, cons, bytevector operations |
| **System** | [`system_codegen.cpp`](lib/backend/system_codegen.cpp) | 1,192 | System, environment, time, process, eval support |
| **Bindings** | [`binding_codegen.cpp`](lib/backend/binding_codegen.cpp) | 1,178 | let/let*/letrec/letrec* with TCO context save/restore |
| **Thread Pool** | [`thread_pool.cpp`](lib/backend/thread_pool.cpp) | 1,131 | Work-stealing thread pool runtime |
| **Tensor Backward** | [`tensor_backward.cpp`](lib/backend/tensor_backward.cpp) | 1,078 | Backward-mode AD gradient computation for tensors |
| **Call/Apply** | [`call_apply_codegen.cpp`](lib/backend/call_apply_codegen.cpp) | 960 | Function calls, apply, partial application, variadic |
| **BLAS Backend** | [`blas_backend.cpp`](lib/backend/blas_backend.cpp) | 890 | BLAS dispatch, GPU cost model calibration |
| **Tagged Values** | [`tagged_value_codegen.cpp`](lib/backend/tagged_value_codegen.cpp) | 822 | Pack/unpack tagged values, type extraction |
| **Control Flow** | [`control_flow_codegen.cpp`](lib/backend/control_flow_codegen.cpp) | 800 | if/cond/case/match/when/unless/call-cc/guard |
| **Map** | [`map_codegen.cpp`](lib/backend/map_codegen.cpp) | 784 | map/for-each/fold with closure dispatch |
| **Parallel** | [`parallel_codegen.cpp`](lib/backend/parallel_codegen.cpp) | 705 | parallel-map/fold/filter/for-each runtime |
| **Homoiconic** | [`homoiconic_codegen.cpp`](lib/backend/homoiconic_codegen.cpp) | 601 | Code-as-data, quote, lambda S-expressions, eval |
| **Hash** | [`hash_codegen.cpp`](lib/backend/hash_codegen.cpp) | 603 | make-hash, hash-ref, hash-set!, hash-for-each |
| **Complex** | [`complex_codegen.cpp`](lib/backend/complex_codegen.cpp) | 499 | Complex number arithmetic (Smith's formula division) |
| **Tail Calls** | [`tail_call_codegen.cpp`](lib/backend/tail_call_codegen.cpp) | 497 | TCO transformation, trampoline runtime |

**Additional backend components (not in the 21-module count):**

| Component | Source Files | Lines | Purpose |
|:---|:---|---:|:---|
| Type System | [`type_system.cpp`](lib/backend/type_system.cpp) | ~300 | LLVM type creation and caching |
| Codegen Context | [`codegen_context.cpp`](lib/backend/codegen_context.cpp) | ~200 | Shared state for module communication |
| Function Cache | [`function_cache.cpp`](lib/backend/function_cache.cpp) | ~250 | Lazy-loaded C library function declarations |
| Builtin Declarations | [`builtin_declarations.cpp`](lib/backend/builtin_declarations.cpp) | ~350 | Runtime function declarations (deep_equal, display, registry) |
| Memory Codegen | [`memory_codegen.cpp`](lib/backend/memory_codegen.cpp) | ~300 | Arena allocation IR generation |
| CPU Features | [`cpu_features.cpp`](lib/backend/cpu_features.cpp) | ~100 | SIMD capability detection |
| XLA/StableHLO | 5 files in `lib/backend/xla/` | 4,003 | Tensor compilation via MLIR pipeline |
| GPU/Metal | `lib/backend/gpu/gpu_memory.mm`, `metal_softfloat.h` | 8,888 | Metal compute, SF64 software float64, CUDA stubs |

### Module Initialization Order

Modules are initialized in dependency order during `LLVMCodeGenerator` construction:

```
TypeSystem -> MemoryCodegen -> CodegenContext -> TaggedValueCodegen
  -> BuiltinDeclarations -> TensorCodegen -> AutodiffCodegen
  -> ComplexCodegen -> ArithmeticCodegen -> CallApplyCodegen
  -> MapCodegen -> ControlFlowCodegen -> StringIOCodegen
  -> BindingCodegen -> CollectionCodegen -> HomoiconicCodegen
  -> TailCallCodegen -> SystemCodegen -> HashCodegen -> ParallelCodegen
```

ArithmeticCodegen depends on TensorCodegen, AutodiffCodegen, and ComplexCodegen because polymorphic arithmetic must dispatch to tensor element-wise ops, dual number propagation, and complex arithmetic paths.

---

## v1.1-accelerate Feature Codegen

### GPU Dispatch (SIMD -> cBLAS -> Metal)

**Implementation:** [`blas_backend.cpp`](lib/backend/blas_backend.cpp) (890 lines), [`gpu_memory.mm`](lib/backend/gpu/gpu_memory.mm) (3,786 lines)

The cost model selects the optimal compute backend based on tensor dimensions:

| Backend | Peak Performance | Overhead | When Selected |
|:---|---:|---:|:---|
| SIMD (vectorized) | 25 GFLOPS | ~0 | 16 or fewer elements |
| cBLAS (Apple Accelerate AMX) | 1,100 GFLOPS | 5 us | 17 to ~1B elements |
| Metal GPU (SF64 software float64) | 200 GFLOPS | 200 us | >1B elements (GPU genuinely faster) |

**SF64 (Software Float64):** Metal GPUs lack native float64. SF64 emulates double precision using double-double arithmetic (two 32-bit mantissas for ~100-bit effective precision). Implemented in [`metal_softfloat.h`](lib/backend/gpu/metal_softfloat.h) (4,076 lines).

**Cost model calibration:** Measured values are `blas_peak_gflops=1100` (Apple AMX) and `gpu_peak_gflops=200` (SF64). Configurable via environment variables `ESHKOL_BLAS_PEAK_GFLOPS` and `ESHKOL_GPU_PEAK_GFLOPS`.

The Metal shader source is embedded at build time via a CMake custom command that converts `metal_softfloat.h` into a C string literal (`metal_sf64_embedded.inc`), ensuring the GPU kernel and host code stay in sync.

### Parallel Primitives

**Implementation:** [`parallel_codegen.cpp`](lib/backend/parallel_codegen.cpp) (705 lines), [`parallel_llvm_codegen.cpp`](lib/backend/parallel_llvm_codegen.cpp) (2,401 lines), [`thread_pool.cpp`](lib/backend/thread_pool.cpp) (1,131 lines)

Four parallel higher-order functions with work-stealing scheduling:

```scheme
(parallel-map f vec)        ; Apply f to each element in parallel
(parallel-fold f init vec)  ; Associative reduction with work distribution
(parallel-filter pred vec)  ; Parallel predicate-based selection
(parallel-for-each f vec)   ; Parallel side-effecting iteration
```

Worker functions use `LinkOnceODRLinkage` to prevent duplicate symbol errors when multiple compilation units contain parallel constructs. The same linkage is used for all stdlib symbols to allow user-defined functions to override stdlib definitions.

### Consciousness Engine Codegen

**Runtime:** [`lib/core/logic.cpp`](lib/core/logic.cpp) (805 lines), [`lib/core/inference.cpp`](lib/core/inference.cpp) (912 lines), [`lib/core/workspace.cpp`](lib/core/workspace.cpp) (308 lines)

**Codegen:** Dispatched directly from [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) (not a separate module)

22 builtins across three theoretical frameworks:

**Logic Programming (Robinson's unification):** `unify`, `walk`, `make-substitution`, `make-fact`, `make-kb`, `kb-assert!`, `kb-query`, `logic-var?`, `substitution?`, `kb?`, `fact?`

**Active Inference (Free Energy Principle):** `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `fg-update-cpt!`, `free-energy`, `expected-free-energy`, `factor-graph?`

**Global Workspace Theory:** `make-workspace`, `ws-register!`, `ws-step!`, `workspace?`

`ws-step!` is fully implemented: LLVM codegen generates a loop that calls registered module closures via `codegenClosureCall`, then C runtime helpers (`eshkol_ws_make_content_tensor`, `eshkol_ws_step_finalize`) handle tensor wrapping and softmax broadcast for competitive attention.

Logic variables use syntax `?x` (parsed as `ESHKOL_LOGIC_VAR_OP`), which is R7RS-compatible since `?` is a valid identifier start character in Scheme.

### Exact Arithmetic Dispatch

**Implementation:** [`arithmetic_codegen.cpp`](lib/backend/arithmetic_codegen.cpp) (2,332 lines)

The full R7RS numeric tower with automatic precision promotion:

```
int64 -> bignum -> rational -> double -> complex
         (overflow)  (exact div)  (inexact)  (make-rectangular)
```

All arithmetic operations (`+`, `-`, `*`, `/`, comparison, `abs`, `min`, `max`, `expt`) dispatch through a type-based cascade:

1. Check for dual number (AD active) -- propagate derivatives
2. Check for `HEAP_PTR` -- inspect subtype for bignum/rational
3. Check for `DOUBLE` -- IEEE 754 arithmetic
4. Check for `INT64` -- native integer ops with overflow check
5. Overflow promotes int64 to bignum via C runtime (`eshkol_bignum_binary_tagged`)

**Exactness tracking:** The `ESHKOL_FLAG_EXACT` flag in the tagged value's flags byte propagates through operations. R7RS semantics: exact + exact = exact, exact + inexact = inexact.

**Runtime functions for exact arithmetic:**
- `eshkol_bignum_binary_tagged` -- bignum +, -, *, div, mod, gcd
- `eshkol_bignum_compare_tagged` -- exact comparison
- `eshkol_bignum_pow_tagged` -- exact exponentiation via repeated squaring
- `eshkol_rational_compare_tagged_ptr` -- rational comparison
- `eshkol_bignum_to_string` / `eshkol_string_to_number_tagged` -- conversion

### First-Class Continuations

**Implementation:** [`control_flow_codegen.cpp`](lib/backend/control_flow_codegen.cpp) (800 lines), with `call/cc` and `dynamic-wind` dispatch in [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

- `call/cc` -- single-shot continuations via setjmp/longjmp
- `dynamic-wind` -- before/after thunks with proper unwinding on non-local exit
- `guard`/`raise` -- R7RS exception handling with cond-style clauses
- `with-exception-handler` -- low-level exception handler installation
- `delay`/`force`/`make-promise`/`delay-force` -- full R7RS promise system with memoization

Continuations are `HEAP_PTR` objects with `HEAP_SUBTYPE_PROMISE` (for promises) or stored as callable closures (for `call/cc`).

### Machine Learning Framework (75+ Builtins)

**Implementation:** [`tensor_codegen.cpp`](lib/backend/tensor_codegen.cpp) (19,187 lines), [`tensor_backward.cpp`](lib/backend/tensor_backward.cpp) (1,078 lines)

Categories: activations (16), loss functions (14), optimizers (5+3), weight initializers (5), LR schedulers (4), CNN layers (7), transformer operations (8), data loading (6), plus tensor creation/manipulation ops.

All ML builtins integrate with reverse-mode AD -- calling `gradient` on any composition produces exact gradients via the computational tape.

---

## Optimization Pipeline

Eshkol uses LLVM's **New Pass Manager** (not the deprecated Legacy PassManagerBuilder):

```cpp
// lib/backend/llvm_codegen.cpp:119-141
static void optimizeModule(llvm::Module& module, llvm::TargetMachine* TM) {
    auto opt_level = getPassBuilderOptLevel();
    if (opt_level == llvm::OptimizationLevel::O0) return;

    llvm::PassBuilder PB(TM);

    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(opt_level);
    MPM.run(module, MAM);
}
```

### Optimization Levels

| Flag | LLVM Level | Passes |
|:---|:---|:---|
| `-O0` | `OptimizationLevel::O0` | No optimization (default for stdlib.o) |
| `-O1` | `OptimizationLevel::O1` | Basic: constant folding, DCE |
| `-O2` | `OptimizationLevel::O2` | Full: inlining, GVN, LICM, SROA |
| `-O3` | `OptimizationLevel::O3` | Aggressive: loop unrolling, auto-vectorization |

### Eshkol-Specific Optimization Patterns

**Tagged Value Elimination:** When types are statically known from HoTT inference, the compiler generates untagged LLVM IR:

```scheme
;; Typed function -- tags eliminated
(define (add x : integer y : integer) : integer (+ x y))
;; => define i64 @add(i64 %x, i64 %y) { %r = add i64 %x, %y; ret i64 %r }
```

**PHI Node Avoidance:** Loops that call `codegenClosureCall` (which creates 15+ basic blocks) use alloca + store/load instead of PHI nodes. PHI nodes require exactly one entry per predecessor block, but closure calls break this invariant by inserting intermediate blocks.

```cpp
// CORRECT pattern for loops with closure calls:
Value* counter_ptr = builder->CreateAlloca(int64_type);  // alloca
builder->CreateStore(ConstantInt::get(int64_type, 0), counter_ptr);
// ... in loop body:
Value* counter = builder->CreateLoad(int64_type, counter_ptr);
// ... call closure ...
builder->CreateStore(new_counter, counter_ptr);

// INCORRECT (breaks with closure calls):
// PHINode* counter = builder->CreatePHI(int64_type, 2);
```

**Block Capture After IR-Generating Calls:** Any function that may create basic blocks (e.g., `extractDoubleFromTagged`, `packDoubleToTaggedValue`, `isHeapSubtype`) can shift the builder's insert point. Code that creates PHI nodes or conditional branches must capture the actual exit block with `builder->GetInsertBlock()` after such calls, not before.

---

## JIT Compilation (REPL)

**Implementation:** [`lib/repl/repl_jit.cpp`](lib/repl/repl_jit.cpp) (~2,062 lines)

The REPL uses **LLVM's LLJIT** (via OrcJIT v2) for interactive execution.

### JIT Architecture

```
User Input: "(+ 1 2)"
       |
       v
  Parse -> AST
       |
       v
  Type Check (optional)
       |
       v
  Generate LLVM IR (same codegen as AOT)
       |
       v
  LLJIT Compile -> Native Code (in-process)
       |
       v
  Execute -> Return 3
       |
       v
  Display: "3"
```

### Critical: CodeGenOptLevel Matching

The JIT's `CodeGenOptLevel` **must** match the optimization level used to compile `stdlib.o`:

```cpp
// lib/repl/repl_jit.cpp:132-144
auto jtmb = orc::JITTargetMachineBuilder::detectHost();
// detectHost() defaults to CodeGenOptLevel::Default (-O2)
// BUT stdlib.o is compiled at -O0
// On ARM64, -O2 generates different struct arg stack layouts for
// {i8, i8, i16, i32, i64} -- 3rd+ arg received as all-zeros
jtmb->setCodeGenOptLevel(CodeGenOptLevel::None);  // Match stdlib.o
```

Without this fix, the 3rd and subsequent tagged value arguments to stdlib functions arrive as all-zeros on ARM64, because the ABI for struct passing differs between optimization levels.

### Stdlib Preloading

The REPL preloads stdlib via two mechanisms:

1. **`addObjectFile`** -- Loads `stdlib.o` directly into the JIT for instant availability of all 237+ stdlib functions and 305 globals
2. **`.bc` metadata** -- Parses `stdlib.bc` (bitcode) to dynamically discover all exported function and global names, enabling the JIT to resolve stdlib symbols without hardcoded lists

```cpp
// Symbol resolution chain:
// 1. JIT-compiled user code (current session)
// 2. stdlib.o (preloaded object file)
// 3. DynamicLibrarySearchGenerator (host process symbols)
auto generator = orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
    jit_->getDataLayout().getGlobalPrefix());
```

### Force-Loading for JIT Symbol Resolution

The REPL executable links with `-force_load` (macOS) or `--whole-archive` (Linux) to prevent the linker from dead-stripping archive members that are only referenced by JIT-compiled code at runtime:

```cmake
# CMakeLists.txt (macOS)
target_link_options(eshkol-repl PRIVATE
    "-Wl,-force_load" "$<TARGET_FILE:eshkol-static>")
target_link_options(eshkol-repl PRIVATE "-Wl,-export_dynamic")
```

Combined with `-export_dynamic`, this makes `DynamicLibrarySearchGenerator` auto-resolve ALL runtime symbols -- new runtime functions added to the compiler are automatically available in the REPL without manual `ADD_SYMBOL` registration.

---

## Build System

**Implementation:** [`CMakeLists.txt`](CMakeLists.txt)

Eshkol uses CMake (minimum 3.14) with C17/C++20 standards.

### Build Modes

The build system supports two LLVM integration modes:

**Lite Build (default):** Uses the system LLVM 21 toolchain on macOS/Linux and the LLVM 21 CMake package on native Windows.

**XLA Build:** Set `STABLEHLO_ROOT` to use StableHLO's bundled LLVM/MLIR. Enables full XLA backend with StableHLO dialect support for tensor compilation via MLIR.

### Library Structure

```cmake
# Static library containing ALL backend sources (auto-discovered)
file(GLOB_RECURSE LIB_SRC RELATIVE ${CMAKE_SOURCE_DIR} "lib/*.c*")
list(FILTER LIB_SRC EXCLUDE REGEX "lib/repl/.*")         # REPL separate
list(FILTER LIB_SRC EXCLUDE REGEX "lib/backend/gpu/.*")   # GPU added explicitly

add_library(eshkol-static STATIC ${LIB_SRC})
# GPU sources added per-platform (Metal .mm on macOS, stub .cpp on Linux)
target_sources(eshkol-static PRIVATE ${GPU_SOURCES})
```

### Build Targets

| Target | Source | Description |
|:---|:---|:---|
| `eshkol-static` | `lib/**/*.cpp` | Static library with all compiler + runtime code |
| `eshkol-run` | `exe/eshkol-run.cpp` | AOT compiler (also supports `-e`/`--eval` JIT mode) |
| `eshkol-repl` | `exe/eshkol-repl.cpp` | Interactive REPL with LLJIT |
| `eshkol-repl-lib` | `lib/repl/*.cpp` | JIT library (linked into both `eshkol-run` and `eshkol-repl`) |
| `stdlib` | `lib/stdlib.esk` | Pre-compiled standard library (`stdlib.o`) |
| `eshkol-pkg` | `tools/pkg/eshkol_pkg.cpp` | Package manager |
| `eshkol-lsp` | `tools/lsp/eshkol_lsp.cpp` | LSP server |
| `metal_sf64_shader` | `lib/backend/gpu/metal_softfloat.h` | Embedded Metal shader (macOS only) |

### Platform-Specific Configuration

**BLAS Acceleration:**
- macOS: Apple Accelerate framework (built-in, AMX-optimized)
- Linux: OpenBLAS via `find_package(BLAS)` or `pkg-config`

**GPU Acceleration:**
- macOS: Metal framework + MetalPerformanceShaders (Objective-C++ via `gpu_memory.mm`)
- Linux: CUDA Toolkit (`gpu_memory_cuda.cpp` + `gpu_cuda_kernels.cu`)
- No GPU: Stub (`gpu_memory_stub.cpp`) so API symbols resolve at link time

**Stack Size:**
- macOS: `-Wl,-stack_size,0x20000000` (512MB)
- Linux: `-Wl,-z,stacksize=536870912` (512MB)
- Runtime override via `ESHKOL_STACK_SIZE` environment variable

### Stdlib Compilation

The standard library is compiled as a custom build step:

```cmake
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stdlib.o
    COMMAND $<TARGET_FILE:eshkol-run> --shared-lib -o "stdlib" "lib/stdlib.esk"
    DEPENDS eshkol-run lib/stdlib.esk
)
```

Library mode (`--shared-lib`) uses `LinkOnceODRLinkage` for all symbols, allowing user programs to override any stdlib function without duplicate symbol errors.

### Build Commands

```bash
# Standard build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Build with XLA support
cmake -B build -DESHKOL_XLA_ENABLED=ON -DSTABLEHLO_ROOT=/path/to/stablehlo
cmake --build build -j$(nproc)

# Quick rebuild (from build directory)
cd build && make -j8

# Install
sudo cmake --install build
```

### Optional Features

```cmake
option(ESHKOL_BUILD_TESTS "Build tests" ON)
option(ESHKOL_BLAS_ENABLED "Enable BLAS acceleration" ON)
option(ESHKOL_GPU_ENABLED "Enable GPU acceleration" ON)
option(ESHKOL_XLA_ENABLED "Enable XLA backend" OFF)
option(BUILD_REPL "Build the interactive REPL" ON)
option(ESHKOL_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(ESHKOL_ENABLE_UBSAN "Enable UB Sanitizer" OFF)
```

---

## Compiler Executables

### eshkol-run (AOT Compiler + Eval)

**Source:** [`exe/eshkol-run.cpp`](exe/eshkol-run.cpp)

Ahead-of-time compiler that produces native executables:

```bash
# Compile and run
eshkol-run program.esk -o program
./program

# Compile to WebAssembly
eshkol-run --wasm program.esk

# Library mode (for stdlib)
eshkol-run --shared-lib -o stdlib lib/stdlib.esk

# JIT eval mode (uses REPL JIT internally)
eshkol-run -e '(display (+ 1 2))'

# Debug flags
eshkol-run --dump-ir program.esk     # Print LLVM IR
eshkol-run --dump-ast program.esk    # Print AST
```

The `require` system handles module discovery: `(require stdlib)` links with pre-compiled `stdlib.o`, while `(require core.list.transform)` loads a specific sub-module. Module discovery uses `collect_all_submodules()` which recursively parses source files -- no hardcoded prefix lists.

### eshkol-repl (JIT REPL)

**Source:** [`exe/eshkol-repl.cpp`](exe/eshkol-repl.cpp)

Interactive Read-Eval-Print Loop with readline support:

```bash
eshkol-repl
> (define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
> (fib 30)
832040
> (gradient (lambda (x) (* x x)) 3.0)
6.0
```

### eshkol-lsp (Language Server)

**Source:** [`tools/lsp/eshkol_lsp.cpp`](tools/lsp/eshkol_lsp.cpp) (1,018 lines)

LSP server providing completions, hover, go-to-definition, diagnostics, and formatting for IDE integration (VSCode extension available).

### eshkol-pkg (Package Manager)

**Source:** [`tools/pkg/eshkol_pkg.cpp`](tools/pkg/eshkol_pkg.cpp) (721 lines)

Package manager with TOML manifests and git-based registry: `eshkol-pkg init/build/run/add/clean`.

---

## Runtime Architecture

### Global Arena

**Implementation:** [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp) (~4,972 lines)

- Single allocator for all heap objects
- 8KB minimum block size, doubling growth strategy
- O(1) bump-pointer allocation until block exhaustion
- No individual `free()` -- memory reclaimed via arena reset
- Thread-safe for parallel primitives

### Object Header System

Every heap-allocated object has an 8-byte header prepended at `pointer - 8`:

```c
struct eshkol_object_header_t {
    uint8_t subtype;     // 20 subtypes (cons, string, vector, tensor, closure, ...)
    uint8_t flags;       // Linear, borrowed, shared, marked, exact
    uint16_t ref_count;  // For shared objects (0 = not ref-counted)
    uint32_t size;       // Object size excluding header
};
```

The subtype field determines the actual type of `HEAP_PTR` and `CALLABLE` tagged values. Type checking at runtime reads the header at `pointer - 8`, costing one memory dereference but providing extensible type space.

### Closure Structure (40 bytes)

```c
struct {
    void*    func_ptr;       // Function pointer (8 bytes)
    void*    env;            // Captured environment pointer (8 bytes)
    void*    sexpr_ptr;      // Lambda S-expression for homoiconicity (8 bytes)
    uint8_t  return_type;    // Return type tag
    uint8_t  input_arity;    // Number of parameters
    uint8_t  flags;          // Variadic, tail-recursive, etc.
    uint8_t  reserved;
    uint32_t hott_type_id;   // HoTT type identifier
};
```

Closures capture variables by storing **pointers** to bindings (not copies), enabling mutation through `set!`.

### Tensor vs. Vector Distinction

**Scheme vectors** (`HEAP_SUBTYPE_VECTOR`): Heterogeneous arrays of 16-byte tagged values. Any Eshkol value can be stored. Created via `(vector 1 "hello" #t)`.

**Tensors** (`HEAP_SUBTYPE_TENSOR`): Homogeneous arrays of doubles stored as int64 bit patterns (8 bytes each). Multidimensional with shape metadata. Created via `#(1.0 2.0 3.0)` or `(make-tensor (list 4 4) 0.0)`.

The distinction matters for iteration builtins: `vector-for-each` and `vector-map` must check the header subtype at `ptr - 8` to determine element size and dispatch correctly.

### Computational Tape (Reverse-Mode AD)

A dynamic array of AD nodes allocated during the forward pass, topologically sorted for the backward pass. The 32-level tape stack enables computing nested derivatives (Hessians, meta-learning gradients).

### Consciousness Engine Runtime

Three C runtime libraries:
- **Logic** ([`lib/core/logic.cpp`](lib/core/logic.cpp), 805 lines): Robinson's unification, substitution environments, knowledge base query
- **Inference** ([`lib/core/inference.cpp`](lib/core/inference.cpp), 912 lines): Factor graph construction, belief propagation, free energy computation
- **Workspace** ([`lib/core/workspace.cpp`](lib/core/workspace.cpp), 308 lines): Module registration, softmax competitive attention, step execution

LLVM codegen dispatches to these runtime functions via tagged value calling conventions, with heap subtypes 12-17 identifying consciousness engine objects.

---

## See Also

- [Overview](OVERVIEW.md) -- High-level architecture and feature summary
- [Type System](TYPE_SYSTEM.md) -- Tagged values, HoTT types, gradual typing
- [Memory Management](MEMORY_MANAGEMENT.md) -- OALR system, arena allocation, object headers
- [Automatic Differentiation](AUTODIFF.md) -- Three AD modes, tape architecture
- [Machine Learning](MACHINE_LEARNING.md) -- 75+ ML builtins deep dive
- [Consciousness Engine](CONSCIOUSNESS_ENGINE.md) -- Logic, active inference, global workspace
- [GPU Acceleration](GPU_ACCELERATION.md) -- Metal, SF64, cost model
- [Parallel Computing](PARALLEL_COMPUTING.md) -- Work-stealing, parallel primitives
- [Exact Arithmetic](EXACT_ARITHMETIC.md) -- Numeric tower, bignum, rational
- [Continuations](CONTINUATIONS.md) -- call/cc, dynamic-wind, guard/raise
- [Module System](MODULE_SYSTEM.md) -- require/provide, precompiled stdlib
- [REPL JIT](REPL_JIT.md) -- Interactive development
- [API Reference](../API_REFERENCE.md) -- Complete function reference
