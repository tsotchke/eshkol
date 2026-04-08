# Eshkol System Architecture Reference

**Version**: 1.1.11
**Release**: v1.1-accelerate
**Date**: March 2026
**Status**: Production-ready compiler with GPU acceleration, consciousness engine, and exact arithmetic

> **Note**: This document describes the **actual implemented system** based on comprehensive code analysis. Features marked as "planned" or "future" are documented separately in roadmap documents.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Memory Architecture (OALR)](#memory-architecture-oalr)
4. [Type System (Triple-Layer)](#type-system-triple-layer)
5. [Automatic Differentiation](#automatic-differentiation)
6. [Closure System](#closure-system)
7. [N-Dimensional Tensors](#n-dimensional-tensors)
8. [Compilation Pipeline](#compilation-pipeline)
9. [Module System](#module-system)
10. [REPL/JIT System](#repljit-system)
11. [Standard Library](#standard-library)
12. [Code Organization](#code-organization)
13. [Performance Characteristics](#performance-characteristics)
14. [v1.1 Architecture Extensions](#v11-architecture-extensions)

---

## Executive Summary

Eshkol is a production-grade compiler implementing a Scheme-like language with:

- **Automatic differentiation** (3 modes: symbolic, forward, reverse)
- **N-dimensional tensors** with comprehensive linear algebra operations
- **Arena-based memory management** (OALR - Ownership-Aware Lexical Regions)
- **LLVM backend** for native code generation
- **Interactive REPL** with JIT compilation
- **Module system** with dependency resolution
- **Pattern matching** with recursive patterns
- **Homoiconic** code-as-data representation

### Key Statistics

| Metric | Value |
|--------|-------|
| Total codebase | ~232,000 lines |
| LLVM backend | 21 codegen modules, ~86,000 lines |
| Bytecode VM | 63 opcodes, 250+ native calls, ~41,000 lines |
| Main codegen | 35,074 lines ([`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:1)) |
| Parser | 7,551 lines ([`lib/frontend/parser.cpp`](../lib/frontend/parser.cpp:1)) |
| Memory manager | 4,972 lines ([`lib/core/arena_memory.cpp`](../lib/core/arena_memory.cpp:1)) |
| Weight matrix transformer | 2,299 lines, 55/55 tests, 3-way verified |
| Test suite | 438 tests across 35 suites (525+ assertions, 0 failures) |

---

## System Overview

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    ESHKOL USER PROGRAMS                         │
│         (Scheme syntax with autodiff & tensor operations)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   STANDARD LIBRARY (Eshkol)                     │
│  stdlib.esk, math.esk, core/{functional,list,logic}/*.esk       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                 COMPILER FRONTEND (C++/LLVM)                    │
│  Parser (5.5K) → Macro Expander (579) → Type Checker (1.6K)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                 COMPILER BACKEND (C++/LLVM)                     │
│  Main Codegen (35K) + 21 Specialized Modules (20K)              │
│  • Arithmetic  • Autodiff  • Tensor  • Collection  • Complex    │
│  • Control Flow  • Binding  • Call/Apply  • Map  • Parallel     │
│  • Homoiconic  • String/IO  • Hash  • Tail Call  • Memory       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    RUNTIME SYSTEM (C)                           │
│  Arena Memory (3.2K) + Display + Deep Equality + Exceptions     │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  NATIVE EXECUTABLE (via LLVM)                   │
│              OR   REPL/JIT (LLVM ORC, 1.1K lines)               │
└─────────────────────────────────────────────────────────────────┘
```

### Design Philosophy

1. **Correctness First**: Production-grade compiler with comprehensive error handling
2. **Zero-Cost Abstractions**: Type information erased at runtime when possible
3. **Gradual Typing**: Optional type annotations, dynamic fallback
4. **Memory Safety**: Arena-based allocation eliminates GC pauses
5. **Performance**: LLVM optimization passes, native code generation

---

## Memory Architecture (OALR)

**Implementation**: [`lib/core/arena_memory.cpp`](../lib/core/arena_memory.cpp:1) (3,210 lines)

### Core Principles

Eshkol uses **Ownership-Aware Lexical Regions** (OALR) instead of garbage collection:

- **Lexical scoping**: Memory freed when leaving scope
- **Ownership tracking**: Compile-time analysis prevents leaks
- **Arena allocation**: Bump-pointer allocation (extremely fast)
- **Deterministic cleanup**: No GC pauses

### Global Arena Architecture

```c
// Single global arena shared across all functions
arena_t* __global_arena;

// Default configuration
#define DEFAULT_BLOCK_SIZE 8192  // 8KB blocks
```

**Key insight**: Eshkol uses a hybrid arena model — a global arena with scope tracking for the main thread, plus per-thread arenas (1 MB, lazily allocated via `thread_local`) for parallel workers. This avoids both fragmentation and contention.

### Object Header System

**ALL heap objects** have an 8-byte header prepended:

```c
typedef struct eshkol_object_header {
    uint8_t  subtype;    // Type discrimination within HEAP_PTR/CALLABLE
    uint8_t  flags;      // GC marks, linear type status, lifecycle
    uint16_t ref_count;  // Reference counting for shared objects  
    uint32_t size;       // Object data size (excluding header)
} eshkol_object_header_t;

_Static_assert(sizeof(eshkol_object_header_t) == 8, "Must be 8 bytes");
```

**Access pattern**:
```c
#define ESHKOL_GET_HEADER(ptr) \
    ((eshkol_object_header_t*)((uint8_t*)ptr - 8))
```

The header is at **offset -8** from the data pointer returned by allocators.

### Phase 3B: Tagged Cons Cells

**Modern cons cell layout** (32 bytes, cache-aligned):

```c
typedef struct arena_tagged_cons_cell {
    eshkol_tagged_value_t car;  // 16 bytes - Complete tagged value
    eshkol_tagged_value_t cdr;  // 16 bytes - Complete tagged value
} arena_tagged_cons_cell_t;     // Total: 32 bytes
```

This replaces the old union-based design, enabling:
- Direct tagged value storage (no extra indirection)
- Simpler type checking (car/cdr have full type information)
- Better cache performance (32 bytes = 1/2 cache line)

### Allocation Functions

```c
// Generic allocation with header
void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                  uint8_t subtype, uint8_t flags);

// Cons cells with header (HEAP_PTR type, HEAP_SUBTYPE_CONS)
arena_tagged_cons_cell_t* arena_allocate_cons_with_header(arena_t* arena);

// Strings with header (HEAP_PTR type, HEAP_SUBTYPE_STRING)
char* arena_allocate_string_with_header(arena_t* arena, size_t length);

// Vectors with header (HEAP_PTR type, HEAP_SUBTYPE_VECTOR)
void* arena_allocate_vector_with_header(arena_t* arena, size_t capacity);

// Closures with header (CALLABLE type, CALLABLE_SUBTYPE_CLOSURE)
eshkol_closure_t* arena_allocate_closure_with_header(
    arena_t* arena, uint64_t func_ptr, size_t num_captures,
    uint64_t sexpr_ptr, uint64_t return_type_info);

// Tensors with header (HEAP_PTR type, HEAP_SUBTYPE_TENSOR)
eshkol_tensor_t* arena_allocate_tensor_with_header(arena_t* arena);

// AD nodes with header (CALLABLE type, CALLABLE_SUBTYPE_AD_NODE)
ad_node_t* arena_allocate_ad_node_with_header(arena_t* arena);
```

### Reference Counting (for shared ownership)

```c
typedef struct eshkol_shared_header {
    void (*destructor)(void*);   // Custom cleanup function
    uint32_t ref_count;          // Strong references
    uint32_t weak_count;         // Weak references
    uint8_t flags;               // Marked, deallocated flags
    uint8_t value_type;          // Type of shared value
    uint16_t reserved;           // Alignment
    uint32_t reserved2;          // Total: 24 bytes aligned
} eshkol_shared_header_t;
```

Operations:
- `shared_allocate()` - Create ref-counted object
- `shared_retain()` - Increment ref count
- `shared_release()` - Decrement, free when zero
- `weak_ref_create()` - Create weak reference
- `weak_ref_upgrade()` - Promote to strong reference

---

## Type System (Triple-Layer)

Eshkol uses **three layers** of type information for different purposes:

### Layer 1: Runtime Types (Tagged Values)

**Implementation**: [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h:1) (1,497 lines)

```c
typedef struct eshkol_tagged_value {
    uint8_t type;        // eshkol_value_type_t (0-255)
    uint8_t flags;       // Exactness, indirection flags
    uint16_t reserved;   // Alignment
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
        uint64_t raw_val;  // For efficient copying
    } data;
} eshkol_tagged_value_t;

_Static_assert(sizeof(eshkol_tagged_value_t) == 16, "Must be 16 bytes");
```

**Type Encoding** (eshkol_value_type_t):

**Immediate Values** (0-7) - No heap allocation:
```c
ESHKOL_VALUE_NULL (0)
ESHKOL_VALUE_INT64 (1)
ESHKOL_VALUE_DOUBLE (2)
ESHKOL_VALUE_BOOL (3)
ESHKOL_VALUE_CHAR (4)
ESHKOL_VALUE_SYMBOL (5)
ESHKOL_VALUE_DUAL_NUMBER (6)
```

**Consolidated Pointer Types** (8-9) - M1 Migration COMPLETE:
```c
ESHKOL_VALUE_HEAP_PTR (8)    // All heap data objects
    Subtypes: CONS, STRING, VECTOR, TENSOR, HASH, EXCEPTION, RECORD, etc.
    
ESHKOL_VALUE_CALLABLE (9)     // All callable objects
    Subtypes: CLOSURE, LAMBDA_SEXPR, AD_NODE, PRIMITIVE, CONTINUATION
```

**Legacy Types** (32-40) - Retained ONLY for display backward compatibility:
```c
ESHKOL_VALUE_CONS_PTR (32)
ESHKOL_VALUE_STRING_PTR (33)
ESHKOL_VALUE_VECTOR_PTR (34)
ESHKOL_VALUE_TENSOR_PTR (35)
ESHKOL_VALUE_HASH_PTR (39)
ESHKOL_VALUE_CLOSURE_PTR (38)
// ... (deprecated, use consolidated types in new code)
```

**Critical**: New code MUST use consolidated types (8-9) with subtypes. Legacy types exist only for the display system.

### Layer 2: Compile-Time Types (HoTT)

**Implementation**: [`lib/types/hott_types.cpp`](../lib/types/hott_types.cpp:1) (682 lines), [`lib/types/type_checker.cpp`](../lib/types/type_checker.cpp:1) (1,561 lines)

**Universe Hierarchy**:
```scheme
𝒰₂ (Propositions)
  ├── Eq, <, Bounded, Subtype (proof types, erased at runtime)
  
𝒰₁ (Type Constructors)
  ├── List, Vector, Tensor, Function, Pair, Closure
  ├── DualNumber, ADNode, HashTable
  ├── Handle, Buffer, Stream (planned)
  
𝒰₀ (Ground Types)
  ├── Value (top type)
  │   ├── Number
  │   │   ├── Integer → Int64, Natural
  │   │   └── Real → Float64
  │   ├── Text
  │   │   ├── String
  │   │   └── Char
  │   ├── Boolean
  │   ├── Null
  │   └── Symbol
```

**Type ID Encoding** (32-bit):
```c
typedef struct {
    uint16_t id;           // Type identifier (65,536 unique types)
    uint8_t  level;        // Universe level (0-255)
    uint8_t  flags;        // TYPE_FLAG_EXACT, TYPE_FLAG_LINEAR, etc.
} TypeId;
```

**35+ Built-in Types** organized in supertype hierarchies:
- Numeric tower: `Number` → `Integer`/`Real`
- Collections: `List<T>`, `Vector<T>`, `Tensor<T,Shape>`
- Functions: Π-types (dependent function types)
- Proofs: `Eq`, `<`, `Bounded` (erased at runtime)

**Current Status**: Type checker produces **warnings only**, doesn't block compilation (gradual typing).

### Layer 3: Dependent Types

**Implementation**: [`lib/types/dependent.cpp`](../lib/types/dependent.cpp:1) (440 lines)

**Compile-Time Value Tracking**:
```c
typedef struct CTValue {
    enum { Nat, Expr, Bool, Unknown } kind;
    uint64_t nat_val;
    const eshkol_ast_t* expr;
    bool bool_val;
} CTValue;
```

**Dimension Checking**:
```c
// Verifies: index < bound at compile time
DimensionChecker::Result checkBounds(
    const CTValue& idx, const CTValue& bound, const std::string& context);

// Verifies: left_cols == right_rows for matrix multiply
DimensionChecker::Result checkMatMulDimensions(
    const DependentType& left, const DependentType& right);
```

**Usage**: Tensor operations use dependent types to verify dimension compatibility.

---

## Automatic Differentiation

**Implementation**: [`lib/backend/autodiff_codegen.cpp`](../lib/backend/autodiff_codegen.cpp:1) (1,766 lines), [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:1) (lines 17750-23200)

Eshkol provides **three modes** of automatic differentiation, each optimized for different use cases:

### Mode 1: Symbolic Differentiation

**Compile-time AST transformation**:

```scheme
(diff (* x x) x)  ; Compiles to: (* 2 x)
(diff (+ (* a x) b) x)  ; Compiles to: a
```

**Implementation**: [`buildSymbolicDerivative()`](../lib/backend/llvm_codegen.cpp:17750) in llvm_codegen.cpp

**12 Differentiation Rules**:
- Constants → 0
- Variables → 1 or 0
- Addition → sum of derivatives
- Product → product rule: `d(f·g) = f'·g + f·g'`
- Quotient → quotient rule: `d(f/g) = (f'·g - f·g')/g²`
- Chain rule for: sin, cos, exp, log, sqrt, pow

**Advantages**:
- Zero runtime cost
- Produces simplified expressions
- Useful for formula manipulation

### Mode 2: Forward-Mode AD (Dual Numbers)

**Runtime dual number arithmetic**:

```c
typedef struct eshkol_dual_number {
    double value;       // Primal value f(x)
    double derivative;  // Tangent f'(x)
} eshkol_dual_number_t;

_Static_assert(sizeof(eshkol_dual_number_t) == 16, "Exact size required");
```

**Arithmetic Rules**:
```c
// Addition: (a, a') + (b, b') = (a+b, a'+b')
// Multiplication (product rule): (a, a') * (b, b') = (a*b, a'*b + a*b')
// Sin (chain rule): sin(a, a') = (sin(a), a'*cos(a))
```

**Usage**:
```scheme
(derivative (lambda (x) (* x x x)) 2.0)  ; → 12.0
```

**Advantages**:
- Efficient for f: ℝ → ℝⁿ (few inputs, many outputs)
- Exact derivatives in one pass
- Supports all math operations

### Mode 3: Reverse-Mode AD (Computational Graphs)

**Graph-based backpropagation**:

```c
typedef struct ad_node {
    uint32_t type;          // AD_NODE_CONSTANT, ADD, MUL, SIN, etc.
    double value;           // Forward pass value
    double gradient;        // Backward pass gradient
    ad_node_t* input1;      // First input node (or null)
    ad_node_t* input2;      // Second input node (or null)
    uint32_t id;            // Unique node ID
    uint32_t padding;       // Alignment
} ad_node_t;

_Static_assert(sizeof(ad_node_t) == 48, "Expected size");
```

**16 Operation Types**:
```c
AD_NODE_CONSTANT, AD_NODE_VARIABLE,
AD_NODE_ADD, AD_NODE_SUB, AD_NODE_MUL, AD_NODE_DIV,
AD_NODE_SIN, AD_NODE_COS, AD_NODE_EXP, AD_NODE_LOG,
AD_NODE_POW, AD_NODE_NEG, AD_NODE_ABS
```

**Tape Structure** (for graph recording):
```c
typedef struct ad_tape {
    ad_node_t** nodes;      // Array of node pointers
    uint64_t num_nodes;     // Current node count
    uint64_t capacity;      // Array capacity
    ad_node_t** variables;  // Array of variable nodes
    uint64_t num_variables; // Variable count
} ad_tape_t;
```

**Usage**:
```scheme
(gradient (lambda (v) (sin (vref v 0))) (vector 1.0))  ; → (vector 0.5403...)
```

**Advantages**:
- Efficient for f: ℝⁿ → ℝ (many inputs, few outputs)
- Scales to large neural networks
- Supports nested gradients (∂²f/∂x²)

### Nested Gradient Support

**32-level tape stack** for computing derivatives of derivatives:

```c
ad_tape_t* __ad_tape_stack[32];  // Stack of tapes
uint64_t __ad_tape_depth;         // Current nesting depth
ad_node_t* __outer_ad_node_stack[16];  // Outer AD nodes for double backward
```

**Example**:
```scheme
;; Second derivative: ∂²f/∂x²
(gradient 
  (lambda (x) 
    (vref (gradient f (vector x)) 0))
  (vector x0))
```

### Vector Calculus Operations

**7 Vector Calculus Operators** (implemented in [`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:21000)):

```scheme
;; Gradient: ∇f: ℝⁿ → ℝⁿ (vector of partial derivatives)
(gradient f (vector x y z))

;; Jacobian: J: ℝⁿ → ℝᵐˣⁿ (matrix of all partial derivatives)
(jacobian F (vector x y))

;; Hessian: H: ℝⁿ → ℝⁿˣⁿ (matrix of second derivatives)
(hessian f (vector x y))

;; Divergence: ∇·F: ℝⁿ → ℝ (sum of diagonal Jacobian elements)
(divergence F (vector x y z))

;; Curl: ∇×F: ℝ³ → ℝ³ (3D rotation operator)
(curl F (vector x y z))

;; Laplacian: ∇²f: ℝⁿ → ℝ (sum of diagonal Hessian elements)
(laplacian f (vector x y))

;; Directional derivative: D_v f = ∇f · v
(directional-deriv f (vector x y) (vector dx dy))
```

All implemented and tested in [`tests/autodiff/phase4_vector_calculus_test.esk`](../tests/autodiff/phase4_vector_calculus_test.esk:1).

---

## Closure System

**Implementation**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:15000), [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h:1)

### Closure Structure (40 bytes)

```c
typedef struct eshkol_closure {
    uint64_t func_ptr;              // Function pointer (8 bytes)
    eshkol_closure_env_t* env;      // Environment pointer (8 bytes)
    uint64_t sexpr_ptr;             // S-expression for homoiconicity (8 bytes)
    uint8_t return_type;            // Return type category (1 byte)
    uint8_t input_arity;            // Input parameter count (1 byte)
    uint8_t flags;                  // CLOSURE_FLAG_VARIADIC, etc. (1 byte)
    uint8_t reserved;               // Alignment (1 byte)
    uint32_t hott_type_id;          // HoTT type ID for return (4 bytes)
} eshkol_closure_t;                 // Total: 40 bytes
```

### Environment (Packed Format)

```c
typedef struct eshkol_closure_env {
    uint64_t num_captures;           // Packed: [captures:16][params:16][variadic:1]
    eshkol_tagged_value_t captures[];// Flexible array of captured values
} eshkol_closure_env_t;
```

**Packed field encoding** (num_captures):
- Bits 0-15: Actual capture count (0-65535)
- Bits 16-31: Fixed parameter count (0-65535)  
- Bit 63: Variadic flag (0=fixed arity, 1=variadic)

**Access macros**:
```c
CLOSURE_ENV_GET_NUM_CAPTURES(packed)   // Extract bits 0-15
CLOSURE_ENV_GET_FIXED_PARAMS(packed)   // Extract bits 16-31
CLOSURE_ENV_IS_VARIADIC(packed)        // Test bit 63
```

### Mutable Captures (Critical Feature)

Closures store **pointers** to captured variables, not values:

```scheme
;; Mutable capture example
(define (make-counter initial)
  (let ((count initial))
    (lambda ()
      (set! count (+ count 1))  ; Mutates captured variable
      count)))

(define counter (make-counter 0))
(counter)  ; → 1
(counter)  ; → 2
(counter)  ; → 3
```

**Implementation**: Captured variables are allocated as `GlobalVariable` or arena storage, closure stores pointers to these locations. `set!` writes through the pointer.

### Variadic Functions

**Rest parameters**:
```scheme
(define (variadic-fn a b . rest)
  (list a b rest))

(variadic-fn 1 2 3 4 5)  ; → (1 2 (3 4 5))
```

**Closure call dispatch** ([`llvm_codegen.cpp:12000`](../lib/backend/llvm_codegen.cpp:12000)):
- Extracts variadic flag from packed `num_captures`
- Builds rest list from extra arguments
- Switches on capture count (0-32) for efficient dispatch

### Return Type Categories

**8 categories** for optimization:
```c
CLOSURE_RETURN_UNKNOWN (0)
CLOSURE_RETURN_SCALAR (1)
CLOSURE_RETURN_VECTOR (2)
CLOSURE_RETURN_LIST (3)
CLOSURE_RETURN_BOOL (4)
CLOSURE_RETURN_STRING (5)
CLOSURE_RETURN_FUNCTION (6)
CLOSURE_RETURN_VOID (7)
```

Enables type-directed optimizations in higher-order functions.

---

## N-Dimensional Tensors

**Implementation**: [`lib/backend/tensor_codegen.cpp`](../lib/backend/tensor_codegen.cpp:1) (3,041 lines)

### Tensor Structure

```c
typedef struct eshkol_tensor {
    uint64_t* dimensions;     // Array of dimension sizes (8 bytes)
    uint64_t  num_dimensions; // Rank (8 bytes)
    int64_t*  elements;       // Elements as int64 bit patterns (8 bytes)
    uint64_t  total_elements; // Product of dimensions (8 bytes)
} eshkol_tensor_t;           // Total: 32 bytes (cache-aligned)
```

**Storage Convention**: Elements stored as `int64_t` containing **bit patterns** of doubles.

**CRITICAL**: Always use `bitcast`, never `fptosi`:
```c
// CORRECT: bitcast preserves bit pattern
double value = 3.14;
int64_t bits = *reinterpret_cast<int64_t*>(&value);

// WRONG: fptosi truncates to integer
int64_t wrong = static_cast<int64_t>(value);  // → 3 (loses precision!)
```

### Tensor Operations (30+)

**Creation**:
```scheme
(zeros n)              ; Or (zeros (list m n)) for 2D
(ones n)
(eye n)                ; Identity matrix
(arange start end step)
(linspace start end num)
(reshape tensor dim1 dim2 ...)
```

**Arithmetic** (element-wise):
```scheme
(tensor-add A B)
(tensor-sub A B)
(tensor-mul A B)
(tensor-div A B)
```

**Linear Algebra**:
```scheme
(tensor-dot A B)       ; 1D: dot product, 2D: matrix multiply
(transpose M)          ; 2D matrix transpose
(norm v)               ; Euclidean norm
(trace M)              ; Sum of diagonal elements
```

**Indexing**:
```scheme
(tensor-get T i j)     ; 2D indexing
(vref v i)             ; 1D shorthand (AD-aware!)
(tensor-set! T i j val); Mutable update
```

**Reductions**:
```scheme
(tensor-sum T)
(tensor-mean T)
(tensor-reduce T fn init)  ; Custom reduction
```

### N-D Slicing Support

**Partial indexing** returns view tensors (zero-copy):

```scheme
;; For tensor with shape [4, 5, 6]
(tensor-get T 2)      ; Returns slice [5, 6] (view into row 2)
(tensor-get T 2 3)    ; Returns slice [6] (view into row 2, col 3)
(tensor-get T 2 3 4)  ; Returns scalar element
```

**Implementation**: Computes linear offset, creates new tensor struct pointing into original data.

---

## Compilation Pipeline

**Implementation**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:1) (main engine)

### 5-Phase Process

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: MACRO EXPANSION                                        │
│ • Process all define-syntax forms                               │
│ • Expand macro invocations in ASTs                              │
│ • Filter out define-syntax (no runtime code)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ PHASE 2: HOTT TYPE CHECKING                                     │
│ • Optional annotations processed                                │
│ • Bidirectional type inference                                  │
│ • Warnings only (gradual typing)                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ PHASE 3: LLVM IR GENERATION                                     │
│ Order CRITICAL for correctness:                                 │
│ 1. Function declarations (all top-level and nested)             │
│ 2. Global variable pre-declarations (forward references)        │
│ 3. Top-level lambda pre-generation (for user main)              │
│ 4. Function definitions (body compilation)                      │
│ 5. Main function creation with global init                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ PHASE 4: OPTIMIZATION                                           │
│ • LLVM optimization passes                                      │
│ • Tail call optimization                                        │
│ • Dead code elimination                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ PHASE 5: CODE GENERATION                                        │
│ • Object file (.o)  OR                                          │
│ • Executable (with stdlib.o)  OR                                │
│ • Shared library (.so/.dylib)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Special Forms (70+)

**Core**: `define`, `define-type`, `define-syntax`, `set!`, `lambda`, `let`, `let*`, `letrec`, `if`, `cond`, `case`, `match`, `and`, `or`, `when`, `unless`, `do`

**Quotation**: `quote`, `quasiquote`, `unquote`, `unquote-splicing`

**Functions**: `call`, `apply`, `compose`, `values`, `call-with-values`

**Memory**: `with-region`, `owned`, `move`, `borrow`, `shared`, `weak-ref`

**Autodiff**: `diff`, `derivative`, `gradient`, `jacobian`, `hessian`, `divergence`, `curl`, `laplacian`, `directional-deriv`

**Exceptions**: `guard`, `raise`

**Modules**: `require`, `provide` (new), `import` (legacy)

**Pattern Matching**: Recursive patterns with `match`

---

## Module System

**Implementation**: [`exe/eshkol-run.cpp`](../exe/eshkol-run.cpp:1) (2,260 lines)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ ModuleDependencyResolver (DFS-based cycle detection)            │
│ • Topological sort for load order                               │
│ • Circular dependency detection with error reporting            │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ ModuleSymbolTable (Export tracking & name mangling)             │
│ • Public symbols: exported via (provide ...)                    │
│ • Private symbols: mangled as __module_name__symbol             │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ Module Resolution (Path search with precedence)                 │
│ 1. Current directory                                            │
│ 2. Library path (lib/)                                          │
│ 3. $ESHKOL_PATH environment variable (colon-separated)          │
└─────────────────────────────────────────────────────────────────┘
```

### Syntax

```scheme
;; Import a module
(require core.functional.compose)

;; Export symbols
(provide compose ∘ pipe >>)

;; Module structure
;; lib/core/functional/compose.esk
(provide compose)
(define (compose f g)
  (lambda (x) (f (g x))))
```

### Symbol Resolution

**Symbolic names** → **file paths**:
```
core.functional.compose → lib/core/functional/compose.esk
data.json → lib/data/json.esk
```

**Private symbols** get mangled to avoid collisions:
```scheme
;; In module test.modules.mod_a:
(define helper ...)  ; Not in provides

;; Mangled to:
__test_modules_mod_a__helper
```

### Pre-Compiled Modules

**stdlib.o linking**:
- Compiler detects `(require stdlib)` or `(require core.*)`
- Auto-links pre-compiled stdlib.o if found
- Skips recompiling modules present in .o file
- External declarations for exported symbols

---

## REPL/JIT System

**Implementation**: [`lib/repl/repl_jit.cpp`](../lib/repl/repl_jit.cpp:1) (1,108 lines), [`exe/eshkol-repl.cpp`](../exe/eshkol-repl.cpp:1) (1,051 lines)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Interactive REPL                             │
│ • Readline integration (history, tab completion)                │
│ • Multi-line editing with paren balance                         │
│ • Command system (:help, :load, :type, etc.)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    JIT Compiler (LLVM ORC)                      │
│ • Incremental compilation per expression                        │
│ • Thread-safe context (shared across modules)                   │
│ • Runtime symbol registration (100+ functions)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              Cross-Module Symbol Persistence                    │
│ • Symbol table: var_name → address                              │
│ • Function table: lambda_name → (address, arity)                │
│ • S-expression cache: var_name_sexpr → value                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                 Shared Arena (Persistent)                       │
│ • __repl_shared_arena persists across evaluations               │
│ • S-expressions remain valid between calls                      │
│ • Lambda registry maintains function → S-expr mappings          │
└─────────────────────────────────────────────────────────────────┘
```

### REPL Commands (12)

```
:help, :h       Show help
:quit, :q       Exit REPL
:load, :l       Load file
:reload, :r     Reload last file
:env, :e        Show defined symbols
:type, :t       Show type info
:ast            Show AST structure
:doc, :d        Show documentation
:time           Time expression execution
:history        Show command history
:clear          Clear screen
:stdlib         Load standard library
```

### Symbol Persistence Mechanism

**Problem**: JIT modules are transient. How do later modules reference earlier functions?

**Solution**: Global registration:

```cpp
// After JIT compilation, register symbols:
eshkol_repl_register_symbol(var_name, address);
eshkol_repl_register_function(lambda_name, address, arity);
eshkol_repl_register_lambda_name(var_name, lambda_name);
eshkol_repl_register_sexpr(sexpr_name, value);
```

**Before compiling new module**: Inject external declarations:

```cpp
void injectPreviousSymbols(Module* module) {
    for (auto& [var_name, lambda_info] : defined_lambdas_) {
        // Create external function declaration
        FunctionType* func_type = /* infer from arity */;
        Function::Create(func_type, ExternalLinkage, lambda_name, module);
    }
}
```

This enables:
```scheme
eshkol> (define (square x) (* x x))
eshkol> (define (cube x) (* x (square x)))  ; References earlier function
eshkol> (cube 3)
27
```

---

## Standard Library

**Implementation**: 25 `.esk` files in [`lib/`](../lib/)

### Module Organization

```
lib/
├── stdlib.esk              # Re-exports all modules
├── math.esk                # Numerical algorithms
│
├── core/
│   ├── io.esk              # I/O wrappers
│   ├── json.esk            # JSON parsing
│   ├── strings.esk         # String utilities
│   │
│   ├── control/
│   │   └── trampoline.esk  # Deep recursion support
│   │
│   ├── data/
│   │   ├── csv.esk         # CSV parser
│   │   └── base64.esk      # Base64 encoding
│   │
│   ├── functional/
│   │   ├── compose.esk     # Function composition
│   │   ├── curry.esk       # Currying utilities
│   │   └── flip.esk        # Argument flipping
│   │
│   ├── list/               # 8 modules
│   │   ├── compound.esk    # cadr, caddr, cadddr, etc.
│   │   ├── convert.esk     # list->vector, string->list
│   │   ├── generate.esk    # iota, repeat, range
│   │   ├── higher_order.esk# map, filter, fold
│   │   ├── query.esk       # member, assoc, length
│   │   ├── search.esk      # binary-search, find
│   │   ├── sort.esk        # quicksort, mergesort
│   │   └── transform.esk   # reverse, append, zip
│   │
│   ├── logic/
│   │   ├── boolean.esk     # and, or, not combinators
│   │   ├── predicates.esk  # Type predicates
│   │   └── types.esk       # Type checking utilities
│   │
│   └── operators/
│       ├── arithmetic.esk  # First-class +, -, *, /
│       └── compare.esk     # First-class <, >, =, etc.
```

### Math Library Highlights

**[`lib/math.esk`](../lib/math.esk:1)** (441 lines):

```scheme
;; Linear algebra
(define (det M n)           ; Determinant via LU decomposition
(define (inv M n)           ; Matrix inverse via Gauss-Jordan
(define (solve A b n)       ; Solve Ax = b
(define (cross u v)         ; Cross product (3D)
(define (dot u v)           ; Dot product
(define (normalize v)       ; Unit vector

;; Eigenvalues
(define (power-iteration A n max-iters tolerance)

;; Numerical methods
(define (integrate f a b n)       ; Simpson's rule
(define (newton f df x0 tol iters); Newton-Raphson

;; Statistics
(define (variance v)
(define (std v)
(define (covariance u v)
```

All implemented in **pure Eshkol** using tensor operations and autodiff.

---

## Code Organization

### Directory Structure

```
eshkol/
├── CMakeLists.txt          # Build system (281 lines)
├── README.md               # Project overview
├── LICENSE                 # MIT license
│
├── inc/eshkol/             # Public headers
│   ├── eshkol.h            # Main header (1,497 lines)
│   ├── llvm_backend.h      # Backend API (173 lines)
│   ├── logger.h            # Logging system
│   │
│   ├── backend/            # Backend module headers
│   │   ├── arithmetic_codegen.h
│   │   ├── autodiff_codegen.h
│   │   ├── tensor_codegen.h
│   │   ├── collection_codegen.h
│   │   ├── control_flow_codegen.h
│   │   ├── binding_codegen.h
│   │   ├── call_apply_codegen.h
│   │   ├── map_codegen.h
│   │   ├── homoiconic_codegen.h
│   │   ├── string_io_codegen.h
│   │   ├── hash_codegen.h
│   │   ├── tail_call_codegen.h
│   │   ├── type_system.h
│   │   ├── tagged_value_codegen.h
│   │   ├── memory_codegen.h
│   │   ├── builtin_declarations.h
│   │   ├── function_cache.h
│   │   ├── codegen_context.h
│   │   └── function_codegen.h
│   │
│   ├── frontend/
│   │   └── macro_expander.h
│   │
│   └── types/
│       ├── hott_types.h
│       ├── type_checker.h
│       └── dependent.h
│
├── lib/                    # Implementation
│   ├── stdlib.esk          # Standard library (42 lines, re-exports)
│   ├── math.esk            # Math library (441 lines)
│   │
│   ├── backend/            # 21 backend modules (~20K lines)
│   │   ├── llvm_codegen.cpp      # Main engine (27,079 lines!)
│   │   ├── arithmetic_codegen.cpp# Polymorphic arithmetic (1,478 lines)
│   │   ├── autodiff_codegen.cpp  # AD operations (1,766 lines)
│   │   ├── tensor_codegen.cpp    # Tensor ops (3,041 lines)
│   │   ├── collection_codegen.cpp# Lists/vectors (1,560 lines)
│   │   ├── control_flow_codegen.cpp # if/cond/and/or (780 lines)
│   │   ├── binding_codegen.cpp   # define/let/set! (910 lines)
│   │   ├── call_apply_codegen.cpp# apply & closures (821 lines)
│   │   ├── map_codegen.cpp       # Higher-order map (779 lines)
│   │   ├── homoiconic_codegen.cpp# Quote & S-expr (599 lines)
│   │   ├── string_io_codegen.cpp # Strings & I/O (1,935 lines)
│   │   ├── hash_codegen.cpp      # Hash tables (603 lines)
│   │   ├── tail_call_codegen.cpp # TCO infra (299 lines)
│   │   ├── type_system.cpp       # LLVM types (95 lines)
│   │   ├── tagged_value_codegen.cpp # Pack/unpack (812 lines)
│   │   ├── memory_codegen.cpp    # Arena decls (255 lines)
│   │   ├── builtin_declarations.cpp # Runtime funcs (148 lines)
│   │   ├── function_cache.cpp    # C library (161 lines)
│   │   ├── codegen_context.cpp   # Shared state (217 lines)
│   │   └── function_codegen.cpp  # Lambda/closure (131 lines)
│   │
│   ├── core/               # Core runtime (C)
│   │   ├── arena_memory.cpp # Memory manager (3,210 lines)
│   │   ├── arena_memory.h   # Memory header (469 lines)
│   │   ├── ast.cpp          # AST manipulation (562 lines)
│   │   ├── logger.cpp       # Logging
│   │   ├── printer.cpp      # Display system
│   │   └── *.esk            # Stdlib modules (25 files)
│   │
│   ├── frontend/
│   │   ├── parser.cpp       # S-expr parser (5,487 lines)
│   │   └── macro_expander.cpp # Macro system (579 lines)
│   │
│   ├── types/
│   │   ├── hott_types.cpp   # HoTT types (682 lines)
│   │   ├── type_checker.cpp # Type inference (1,561 lines)
│   │   └── dependent.cpp    # Dependent types (440 lines)
│   │
│   ├── repl/
│   │   ├── repl_jit.cpp     # JIT compiler (1,108 lines)
│   │   └── repl_utils.h     # REPL utilities
│   │
│   └── quantum/
│       ├── quantum_rng.c    # Quantum RNG (560 lines)
│       └── quantum_rng.h    # RNG header
│
├── exe/
│   ├── eshkol-run.cpp      # Compiler executable (2,260 lines)
│   └── eshkol-repl.cpp     # REPL executable (1,051 lines)
│
└── tests/                  # 400+ test files
    ├── autodiff/           # AD tests (40+ files)
    ├── lists/              # List tests (60+ files)
    ├── tensors/            # Tensor tests
    ├── neural/             # Neural network tests
    ├── types/              # Type system tests
    ├── features/           # Feature tests
    ├── memory/             # Memory tests
    ├── modules/            # Module tests
    └── stdlib/             # Stdlib tests
```

### Backend Modular Refactoring

**Status**: 21 modules extracted from monolithic codegen

**Callback Pattern** for inter-module communication:
```cpp
// Modules can't directly call codegen functions (circular dependency)
// Instead: callback pointers

// In ArithmeticCodegen:
typedef llvm::Value* (*CodegenASTCallback)(const void* ast, void* context);
CodegenASTCallback codegen_ast_callback_;
void* callback_context_;

// Usage:
llvm::Value* arg_value = codegen_ast_callback_(ast, callback_context_);
```

**Remaining Work**: Some modules (FunctionCodegen) have stub implementations, full logic still in main codegen.

---

## Performance Characteristics

### Memory Allocation

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Arena allocate | O(1) | Bump-pointer |
| Scope push/pop | O(1) | Linked list |
| Cons cell alloc | O(1) | Header-aware, 32 bytes |
| String alloc | O(n) | Copy + header, n = length |
| Tensor alloc | O(n·m) | Dims array + elements, varies |

### Autodiff Performance

| Mode | Forward Pass | Backward Pass | Memory | Best For |
|------|--------------|---------------|--------|----------|
| Symbolic | O(n) compile | N/A | O(n) AST | f: ℝ → ℝ, simple |
| Forward | O(n) | N/A | O(n) duals | f: ℝ → ℝⁿ |
| Reverse | O(n) | O(n) | O(n) nodes | f: ℝⁿ → ℝ |

Where n = number of operations.

### Compilation Speed

**Measured on MacBook Pro M1**:
- Simple expression (1 + 2): ~50ms
- Moderate function (fibonacci): ~150ms
- Complex autodiff (neural net): ~500ms
- Full program with stdlib: ~2-5 seconds

**Bottleneck**: LLVM optimization passes (can be reduced with -O0)

---

## Build System

**Implementation**: [`CMakeLists.txt`](../CMakeLists.txt:1) (281 lines)

### Requirements

- **CMake**: 3.14 or higher
- **C Compiler**: C17 support required
- **C++ Compiler**: C++20 support required
- **LLVM**: Always required (core dependency)
- **Optional**: Readline (for REPL history/completion)

### Build Targets

```cmake
# Libraries
eshkol-static       # Core compiler (no main)
eshkol-repl-lib     # REPL with JIT (optional)

# Executables
eshkol-run          # Compiler: .esk → executable
eshkol-repl         # Interactive REPL

# Special
stdlib.o            # Pre-compiled standard library
```

### Build Commands

```bash
# Standard build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# With REPL
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target eshkol-repl

# Run tests
cd build && ctest
```

### Symbol Export

**Critical for REPL/JIT**:
```cmake
# Export dynamic symbols for runtime lookup
target_link_options(eshkol-static PUBLIC
    $<$<PLATFORM_ID:Linux>:-Wl,--export-dynamic>
    $<$<PLATFORM_ID:Darwin>:-Wl,-export_dynamic>
)
```

This makes arena functions, autodiff tape operations, etc. available to JIT-compiled code.

---

## Testing

### Test Suite Organization

**438 tests** across 35 suites:

| Category | Count | Purpose |
|----------|-------|---------|
| autodiff/ | 40+ | All 3 AD modes, vector calculus |
| lists/ | 60+ | Cons cells, map, filter, fold |
| tensors/ | 15+ | Creation, arithmetic, linear algebra |
| neural/ | 10+ | Neural network operations |
| types/ | 10+ | HoTT type system |
| features/ | 20+ | Language features, closures |
| memory/ | 10+ | Arena allocation, OALR |
| modules/ | 10+ | Module system, imports |
| stdlib/ | 15+ | Standard library functions |
| system/ | 10+ | Hash tables, I/O |

### Running Tests

```bash
# All tests
./scripts/run_all_tests.sh

# Specific category
./scripts/run_autodiff_tests.sh
./scripts/run_list_tests.sh
./scripts/run_tensor_tests.sh

# With output capture
./scripts/run_autodiff_tests_with_output.sh
```

### Test Validation

Each test verifies:
- ✅ Correct results
- ✅ Type safety
- ✅ Memory cleanup (no leaks)
- ✅ Error handling (for failure tests)

---

## What's NOT in v1.1

These features are **designed but not implemented**. See roadmap documents for details:

- **Native Quantum Types**: Qubit, qreg, quantum gates (design in `QUANTUM_STOCHASTIC_COMPUTING_ARCHITECTURE.md`)
  - **What IS implemented**: Quantum RNG ([`lib/quantum/quantum_rng.c`](../lib/quantum/quantum_rng.c:1))

- **Multimedia**: Windows, graphics, audio, GPIO (design in `MULTIMEDIA_SYSTEM_ARCHITECTURE.md`)

- **Full Linear Types**: Compile-time enforcement of use-exactly-once (partial: warnings only)

- **Dependent Types with Proofs**: Full proof terms (partial: dimension checking only)

**Note**: Logic programming, previously listed as unimplemented, shipped in v1.1 as part of the Consciousness Engine (see [v1.1 Architecture Extensions](#v11-architecture-extensions)).

**See**: [`ROADMAP.md`](ROADMAP.md) for future releases.

---

## References

### Primary Source Files (analyzed in detail)

- [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h:1) - Main system header (1,497 lines)
- [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp:1) - Core codegen (27,079 lines)
- [`lib/core/arena_memory.cpp`](../lib/core/arena_memory.cpp:1) - Memory manager (3,210 lines)
- [`lib/frontend/parser.cpp`](../lib/frontend/parser.cpp:1) - S-expr parser (5,487 lines)
- [`lib/types/type_checker.cpp`](../lib/types/type_checker.cpp:1) - Type inference (1,561 lines)
- [`lib/repl/repl_jit.cpp`](../lib/repl/repl_jit.cpp:1) - JIT compiler (1,108 lines)
- [`exe/eshkol-run.cpp`](../exe/eshkol-run.cpp:1) - Compiler executable (2,260 lines)

### Design Documents (future features)

- `docs/QUANTUM_STOCHASTIC_COMPUTING_ARCHITECTURE.md` - Quantum computing roadmap
- `docs/NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md` - Logic programming & KB
- `docs/MULTIMEDIA_SYSTEM_ARCHITECTURE.md` - Graphics/audio/hardware
- `docs/HOTT_TYPE_SYSTEM_EXTENSION.md` - Full HoTT integration

### Related Documentation

- [`API_REFERENCE.md`](API_REFERENCE.md) - Complete API with examples
- [`QUICKSTART.md`](QUICKSTART.md) - 15-minute getting started guide
- [`API_REFERENCE.md`](API_REFERENCE.md) - Complete API documentation (700+ builtins)
- [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) - Current limitations and planned features
- [`QUICKSTART.md`](QUICKSTART.md) - Getting started guide

---

## v1.1 Architecture Extensions

The v1.1-accelerate release adds six major subsystems to the compiler and runtime. Each integrates with the existing LLVM codegen pipeline, arena memory, and module system described above.

---

### XLA Backend (Dual-Mode)

**Implementation**: [`lib/backend/xla/`](../lib/backend/xla/)

Eshkol v1.1 provides an optional XLA compilation path for tensor-heavy workloads. The backend operates in dual mode:

1. **StableHLO/MLIR path**: Emits StableHLO dialect operations for dynamic shapes, broadcasting, and large tensor programs. The [`stablehlo_emitter.cpp`](../lib/backend/xla/stablehlo_emitter.cpp:1) translates Eshkol tensor AST nodes into StableHLO IR. [`xla_compiler.cpp`](../lib/backend/xla/xla_compiler.cpp:1) lowers StableHLO through the MLIR pipeline to executable code.

2. **LLVM-direct fallback**: Operations that do not benefit from XLA overhead (small tensors, scalar-heavy code) remain on the standard LLVM codegen path. The cost model in [`xla_codegen.cpp`](../lib/backend/xla/xla_codegen.cpp:1) selects the appropriate backend per operation.

**Key files**:

| File | Role |
|------|------|
| [`xla_codegen.cpp`](../lib/backend/xla/xla_codegen.cpp:1) | Top-level dispatch, cost model, LLVM integration |
| [`stablehlo_emitter.cpp`](../lib/backend/xla/stablehlo_emitter.cpp:1) | AST → StableHLO dialect translation |
| [`xla_compiler.cpp`](../lib/backend/xla/xla_compiler.cpp:1) | StableHLO → executable lowering pipeline |
| [`xla_runtime.cpp`](../lib/backend/xla/xla_runtime.cpp:1) | Runtime buffer management, execution |
| [`xla_memory.cpp`](../lib/backend/xla/xla_memory.cpp:1) | XLA-specific memory allocation and lifetime |
| [`xla_types.cpp`](../lib/backend/xla/xla_types.cpp:1) | Eshkol type ↔ XLA element type mapping |

**ORC JIT integration**: StableHLO programs are compiled at runtime via LLVM ORC, enabling the REPL to execute XLA-accelerated tensor code without ahead-of-time compilation.

---

### GPU Acceleration

**Implementation**: [`lib/backend/gpu/`](../lib/backend/gpu/)

Two GPU backends provide hardware-accelerated tensor operations:

**Metal (Apple Silicon)**:
- [`gpu_memory.mm`](../lib/backend/gpu/gpu_memory.mm:1): Objective-C++ Metal API integration
- Software float64 (SF64) emulation — Metal lacks native float64 support
- [`metal_softfloat.h`](../lib/backend/gpu/metal_softfloat.h:1): IEEE 754 double-precision arithmetic in Metal shading language
- Shader source embedded at build time; no runtime file I/O

**CUDA**:
- [`gpu_memory_cuda.cpp`](../lib/backend/gpu/gpu_memory_cuda.cpp:1): CUDA API integration
- [`gpu_cuda_kernels.cu`](../lib/backend/gpu/gpu_cuda_kernels.cu:1): Native float64 kernels, cuBLAS for matrix operations
- [`gpu_memory_stub.cpp`](../lib/backend/gpu/gpu_memory_stub.cpp:1): No-op stub for builds without GPU support

**Cost-Model Dispatch**:

The runtime selects the execution path based on tensor element count:

```
SIMD vectorization     ← element count ≥ 64
cBLAS (Accelerate/MKL) ← element count ≥ 64, matmul-class operations
GPU offload            ← element count ≥ 100,000
```

Peak GFLOPS parameters calibrated on Apple M1:
- `blas_peak_gflops = 1100` (measured via Apple Accelerate AMX, sustained ~1.2 TFLOPS for matmul up to 15000x15000)
- `gpu_peak_gflops = 200` (SF64 emulation overhead on Metal)

Dispatch proceeds SIMD → cBLAS → GPU; the GPU path is selected only when the cost model predicts it will outperform the CPU path for the given operation and size.

---

### Consciousness Engine

**Implementation**: [`lib/core/logic.cpp`](../lib/core/logic.cpp:1), [`lib/core/inference.cpp`](../lib/core/inference.cpp:1), [`lib/core/workspace.cpp`](../lib/core/workspace.cpp:1)

Three interconnected subsystems implement a compiled consciousness architecture:

#### Logic Programming

**Files**: [`inc/eshkol/core/logic.h`](../inc/eshkol/core/logic.h:1), [`lib/core/logic.cpp`](../lib/core/logic.cpp:1)

- First-order unification (Martelli-Montanari algorithm) with occurs check and triangular substitution chains
- Knowledge base with fact assertion and pattern-matching conjunctive query
- Walk operation: chain dereferencing through substitution bindings to ground values
- Parser extension: `?x` syntax produces `ESHKOL_LOGIC_VAR_OP` AST nodes (R7RS-compatible — `?` is a valid identifier start character)

#### Active Inference

**Files**: [`inc/eshkol/core/inference.h`](../inc/eshkol/core/inference.h:1), [`lib/core/inference.cpp`](../lib/core/inference.cpp:1)

- Factor graphs: bipartite graph G = (V, F, E) with discrete variable nodes and factor nodes
- Conditional probability tables (CPTs) as flat log-probability tensors indexed by joint state assignment
- Sum-product belief propagation in log-space (`fg-infer!`) with configurable max iterations
- CPT mutation for online learning (`fg-update-cpt!` — replaces CPT, resets all messages, beliefs reconverge on next inference pass)
- Variational free energy: F = E_q[log q(s)] - E_q[log p(o, s)]
- Expected free energy (EFE) for action selection: G = E_q[log q(s') - log p(o', s')]
- Observation format: `#(var_index observed_state)` pairs, not state vectors

#### Global Workspace

**Files**: [`inc/eshkol/core/workspace.h`](../inc/eshkol/core/workspace.h:1), [`lib/core/workspace.cpp`](../lib/core/workspace.cpp:1)

- Module registration with closure-based content generators
- Softmax competition across modules (temperature-controlled)
- Winner content broadcasting to all registered modules
- `ws-step!` fully compiled: LLVM codegen loop calls closures via `codegenClosureCall`; C runtime helpers (`eshkol_ws_make_content_tensor`, `eshkol_ws_step_finalize`) handle tensor wrapping and softmax broadcast

**22 Compiled Primitives**:

| Category | Primitives |
|----------|-----------|
| Logic | `unify`, `walk`, `make-substitution`, `make-fact`, `make-kb`, `kb-assert!`, `kb-query`, `logic-var?`, `substitution?`, `kb?`, `fact?` |
| Inference | `make-factor-graph`, `fg-add-factor!`, `fg-infer!`, `fg-update-cpt!`, `free-energy`, `expected-free-energy`, `factor-graph?` |
| Workspace | `make-workspace`, `ws-register!`, `ws-step!`, `workspace?` |

**Heap Subtypes**: SUBSTITUTION=12, FACT=13, KNOWLEDGE\_BASE=15, FACTOR\_GRAPH=16, WORKSPACE=17. Type tag: `ESHKOL_VALUE_LOGIC_VAR` = 10.

---

### Parallel Worker Thread Pool

**Implementation**: [`lib/backend/parallel_llvm_codegen.cpp`](../lib/backend/parallel_llvm_codegen.cpp:1)

Work-stealing deque architecture for data-parallel operations:

- Hardware-aware thread count (defaults to `std::thread::hardware_concurrency()`)
- Task granularity control to amortize dispatch overhead
- Per-thread arena isolation — no cross-thread arena contention

**Primitives**:

```scheme
(parallel-map f lst)           ; Map with work distribution across threads
(parallel-fold f init lst)     ; Parallel reduction with associative combiner
(parallel-filter pred lst)     ; Concurrent predicate evaluation
(parallel-for-each f lst)      ; Side-effecting parallel traversal
(parallel-execute thunk1 ...)  ; Concurrent evaluation of independent thunks
(future expr)                  ; Deferred concurrent computation
(force future)                 ; Block until result available
```

**Linkage**: Worker function symbols use `LinkOnceODRLinkage` to prevent duplicate symbol errors when linking parallel-compiled modules with stdlib.o. This matches the stdlib linkage convention established for all library symbols.

---

### Exact Arithmetic Runtime

Two exact numeric types extend the R7RS numeric tower beyond machine integers and IEEE 754 doubles.

#### Bignum (Arbitrary-Precision Integers)

**Implementation**: [`inc/eshkol/core/bignum.h`](../inc/eshkol/core/bignum.h:1), [`lib/core/bignum.cpp`](../lib/core/bignum.cpp:1)

- Sign-magnitude representation with dynamic limb array
- Automatic int64 → bignum promotion on overflow; demotion back to int64 when result fits
- C runtime dispatch replaces ~1300 lines of inline LLVM IR

**Runtime entry points**:

| Function | Purpose |
|----------|---------|
| `eshkol_bignum_binary_tagged` | +, -, *, /, modulo on tagged values |
| `eshkol_bignum_compare_tagged` | Exact comparison (avoids extractAsDouble precision loss) |
| `eshkol_bignum_pow` | Exponentiation via repeated squaring |
| `eshkol_bignum_to_string` | Decimal string conversion |
| `eshkol_bignum_from_string` | Parse arbitrarily long integer literals |
| `eshkol_is_bignum_tagged` | Type predicate for dispatch |

**Codegen helpers**: `emitBignumBinaryCall`, `emitBignumCompareCall`, `emitIsBignumCheck` in [`arithmetic_codegen.cpp`](../lib/backend/arithmetic_codegen.cpp:1).

#### Rational (Exact Fractions)

**Implementation**: [`inc/eshkol/core/rational.h`](../inc/eshkol/core/rational.h:1), [`lib/core/rational.cpp`](../lib/core/rational.cpp:1)

- GCD-reduced canonical form with positive denominator invariant
- Heap-allocated as `HEAP_PTR` with subtype discrimination
- `eshkol_rational_compare_tagged_ptr` for comparison dispatch in [`arithmetic_codegen.cpp`](../lib/backend/arithmetic_codegen.cpp:1)

#### Numeric Tower

The full R7RS numeric tower as implemented:

```
int64 < bignum < rational < double < complex
       exact                inexact
```

Mixed-exactness arithmetic follows R7RS: exact + inexact produces inexact. The `exact?` and `inexact?` predicates reflect this at runtime via the tagged value flags byte.

---

### Signal Processing Stdlib

**Implementation**: [`lib/signal/`](../lib/signal/)

A compiled DSP library providing 13 signal processing functions:

```scheme
;; FFT/IFFT (Cooley-Tukey radix-2 decimation-in-time)
(fft signal)
(ifft spectrum)

;; Window functions
(hamming-window n)
(hann-window n)
(blackman-window n)
(kaiser-window n beta)

;; Filtering
(convolution signal kernel)
(butterworth-lowpass order cutoff sample-rate)
(butterworth-highpass order cutoff sample-rate)

;; Spectral analysis
(power-spectrum signal)
(magnitude-spectrum signal)
(phase-spectrum signal)
```

All functions operate on Eshkol tensors. The module compiles to `stdlib.o` and is available via `(require signal.filters)`.

---

### REPL JIT Enhancements

**Implementation**: [`lib/repl/repl_jit.cpp`](../lib/repl/repl_jit.cpp:1), [`exe/eshkol-repl.cpp`](../exe/eshkol-repl.cpp:1)

v1.1 resolves several production issues in the interactive JIT:

**Stdlib hot-loading**: Pre-compiled `stdlib.o` is loaded via `addObjectFile` (no recompilation). Symbol discovery uses `.bc` metadata extracted at build time, exposing 237 functions and 305 globals to the JIT symbol resolver.

**ABI-correct optimization level**: `JITTargetMachineBuilder::setCodeGenOptLevel(CodeGenOptLevel::None)` matches the `-O0` level used to compile `stdlib.o`. On ARM64, mismatched optimization levels produce different stack layouts for the `{i8,i8,i16,i32,i64}` tagged value struct, causing the 3rd+ function argument to arrive as zero. This was the root cause of stdlib functions with 3+ arguments returning incorrect results in the REPL.

**Crash recovery**: Signal handlers for `SIGSEGV`, `SIGFPE`, and `SIGBUS` catch runtime faults in JIT-compiled code and return control to the REPL prompt rather than terminating the process.

**Archive linking**: `-force_load` on macOS and `--whole-archive` on Linux prevent the linker from dead-stripping archive members (e.g., XLA runtime functions) that are referenced only by JIT-compiled code at runtime. Combined with `-export_dynamic`, all runtime symbols are visible to the ORC `DynamicLibrarySearchGenerator`.

### v1.1 Feature Integration

**Consciousness Engine in the Runtime Layer.** The consciousness engine (logic inference, active inference, and global workspace) integrates into the runtime as a set of 22 builtin functions backed by three C++ modules: `logic.h/logic.cpp` (unification, substitution, knowledge base), `inference.h/inference.cpp` (factor graph belief propagation, free energy minimization), and `workspace.h/workspace.cpp` (module registration, softmax competition). These are not separate subsystems -- they use the same arena allocator and tagged value representation as the core runtime, with dedicated heap subtypes (SUBSTITUTION=12, FACT=13, KNOWLEDGE_BASE=15, FACTOR_GRAPH=16, WORKSPACE=17) and a logic variable type tag (ESHKOL_VALUE_LOGIC_VAR=10). The `?x` syntax for logic variables is parsed as `ESHKOL_LOGIC_VAR_OP`, remaining compatible with R7RS since `?` is a valid identifier start character. Workspace stepping (`ws-step!`) uses `codegenClosureCall` in LLVM IR to invoke module closures, while C runtime helpers handle tensor wrapping and softmax broadcast.

**GPU Dispatch in the Compilation Pipeline.** GPU acceleration is transparent to the compilation pipeline. The LLVM codegen emits calls to XLA C runtime functions (e.g., `eshkol_xla_matmul`, `eshkol_xla_elementwise`) when tensor sizes exceed the XLA threshold (default 100K elements). These runtime functions internally probe GPU availability via `eshkol_gpu_should_use()` and wrap host pointers into Metal buffers using `newBufferWithBytesNoCopy` for zero-copy access on Apple Silicon's unified memory. The cost model in `blas_backend.cpp` selects between scalar, SIMD (NEON 4x4 micro-kernel / AVX 4x8 micro-kernel), cBLAS (Apple Accelerate AMX at ~1100 GFLOPS), and GPU (Metal sf64 compute shaders at ~200 GFLOPS) based on estimated execution time. GPU dispatch adds 200 microseconds of overhead per operation, so it is reserved for matrices exceeding ~1 billion output elements. The decision is entirely runtime -- the compiled binary contains all code paths, and environment variables (`ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_GPU_PEAK_GFLOPS`, `ESHKOL_GPU_MATMUL_THRESHOLD`) allow tuning without recompilation.

**Parallel Execution in the Memory Model.** Eshkol's arena-based memory model supports parallel execution through per-worker arena allocation and `LinkOnceODRLinkage` for parallel worker functions to prevent duplicate symbol conflicts at link time. The parallel primitives (`parallel-map`, `parallel-for`, etc.) partition work across OS threads, each operating on independent arena segments. Tensor operations that internally parallelize (e.g., GPU compute kernels, cBLAS) are safe because they operate on pre-allocated contiguous buffers -- the arena allocator is only invoked to allocate result tensors before the parallel kernel launches. The REPL JIT uses `-force_load` / `--whole-archive` on the static library and matches the compilation's `CodeGenOptLevel::None` to avoid ABI divergence in struct passing on ARM64, which is critical for correct tagged value transmission across the JIT boundary.

**Exact Arithmetic in the Numeric Tower.** The numeric tower extends from fixnums through bignums, rationals, and complex to tensors, with all transitions handled by the tagged value system's 16-byte `{type:8, flags:8, reserved:16, padding:32, data:64}` representation. Bignum operations dispatch through C runtime functions (`eshkol_bignum_binary_tagged`, `eshkol_bignum_compare_tagged`) that examine the type tag at index 0 and operate on GMP-backed arbitrary-precision integers stored as heap pointers. R7RS exactness semantics are preserved: mixed exact/inexact operations promote to inexact (e.g., bignum + double returns double), while `expt` with exact integer arguments and non-negative exponent uses repeated squaring (`eshkol_bignum_pow`) to return an exact bignum result. The rational type stores numerator/denominator bignums and dispatches through `eshkol_rational_compare_tagged_ptr` for comparisons. All numeric types are checked in `ArithmeticCodegen::compare()`, `abs()`, `min/max`, and `pow()` to prevent precision loss from fallthrough to double paths.

---

*This document reflects the v1.1-accelerate release. All claims are verified against actual source code. For questions or corrections, see [`CONTRIBUTING.md`](../CONTRIBUTING.md).*