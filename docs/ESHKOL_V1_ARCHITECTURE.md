---
kind: reference
status: current
owner-area: build
since: v1.0.0
sources:
  - inc/eshkol/eshkol.h
  - inc/eshkol/types/type_relation.h
  - inc/eshkol/backend/libm_codegen.h
  - inc/eshkol/backend/static_callee_binding.h
  - inc/eshkol/backend/closure_capture_scope.h
  - exe/eshkol-run.cpp
  - lib/backend/llvm_codegen.cpp
  - lib/types/type_checker.cpp
  - CMakeLists.txt
---
# Eshkol System Architecture Reference

**Version**: v1.3.5-evolve
**Release**: v1.3.5-evolve
**Date**: September 2026
**Status**: Production-ready compiler with GPU acceleration, consciousness engine, and exact arithmetic

> **Note**: This document describes the **actual implemented system** based on comprehensive code analysis. Features marked as "planned" or "future" are documented separately in roadmap documents.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Bytecode VM Architecture](#bytecode-vm-architecture)
4. [Memory Architecture (OALR)](#memory-architecture-oalr)
5. [Type System (Triple-Layer)](#type-system-triple-layer)
6. [Automatic Differentiation](#automatic-differentiation)
7. [Closure System](#closure-system)
8. [N-Dimensional Tensors](#n-dimensional-tensors)
9. [Compilation Pipeline](#compilation-pipeline)
10. [Module System](#module-system)
11. [REPL/JIT System](#repljit-system)
12. [Standard Library](#standard-library)
13. [Code Organization](#code-organization)
14. [Performance Characteristics](#performance-characteristics)
15. [v1.1 Architecture Extensions](#v11-architecture-extensions)

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
| Total backend (`lib/backend/`) | ~220,211 lines indexed |
| LLVM backend | 39 codegen modules, 118,470 lines |
| Bytecode VM | 66 core opcodes, 722 VM-table builtins, ~57,650 lines |
| Main codegen | 47,107 lines ([`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)) |
| Parser | 11,691 lines ([`lib/frontend/parser.cpp`](../lib/frontend/parser.cpp)) |
| Memory manager | 4,259 lines ([`lib/core/runtime_arena_core.cpp`](../lib/core/runtime_arena_core.cpp) and its `runtime_*` siblings) |
| Weight matrix transformer | ~7,400 lines, 127/127 inline + 124/124 traced, 3-way verified |
| Test suite | 1,020 self-reported tests across 46 suites (0 failures; see [TEST_COVERAGE.md](TEST_COVERAGE.md)) |

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

## Bytecode VM Architecture

The bytecode backend is a unity-build C system rooted at
[`lib/backend/eshkol_vm.c`](../lib/backend/eshkol_vm.c). Its interpreter is
decomposed by responsibility while retaining the existing static symbols,
public C API, and include order:

| Responsibility | Module | Implemented boundary |
|---|---|---|
| Dispatch loop | `vm_run.c` | Fetch/decode and computed-goto or switch dispatch |
| Value operations | `vm_ops.c` | Comparison, pair, vector, and operand-stack bodies |
| Frames and closures | `vm_frame.c` | Upvalues, closure creation, and returns |
| Non-local control | `vm_control.c` | Continuations, dynamic-wind, and exception handlers |
| Execution limits | `vm_limits.c` | Instruction ceiling and cooperative timeout polling |
| VM lifecycle | `vm_lifecycle.c` | Instance creation/destruction and test-program helpers |

The neighboring modules retain their existing ownership boundaries: builtin
trampolines and native dispatch are in `vm_native.c`, error-object support is
in `vm_error.c`, and coverage hooks plus public ESKB/VM entry points are in
`eshkol_vm.c`. The unity hub includes the extracted modules before
`vm_run.c`, after the native and region-evacuation components they call.
Handlers whose threaded and switch paths intentionally differ remain inline
in `vm_run.c`, so this structural change does not alter behavior.

---

## Memory Architecture (OALR)

**Implementation**: [`lib/core/runtime_arena_core.cpp`](../lib/core/runtime_arena_core.cpp) and its `runtime_arena_*` / `runtime_regions` / `runtime_*_alloc` siblings (18,367 lines total), against the [`lib/core/arena_memory.h`](../lib/core/arena_memory.h) interface (1,156 lines)

### Core Principles

Eshkol uses **Ownership-Aware Lexical Regions** (OALR) instead of garbage collection:

- **Lexical scoping**: Memory freed when leaving scope, on both engines; the
  bytecode VM reclaims through its Stage-1 region evacuator and has no automatic
  per-loop nursery, so a resident VM loop needs an explicit `with-region`
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

**Heap subtypes have one definition.** `heap_subtype_t` in [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h) is the only place a heap subtype is declared: 25 members (values 0-13 and 15-25; 14 is reserved). Each member carries an interior-pointer tag on its declaration line. `[DEEPWALK]` (16 members) means the region evacuator must walk the object's interior tagged values and pointers when the object escapes a dying region. `[LEAF]` (9 members) means a contiguous header-plus-payload copy is sound, because the object holds no interior region pointer. Every `[DEEPWALK]` member has a native evacuation handler in `evac_kind_for` ([`lib/core/runtime_regions.cpp`](../lib/core/runtime_regions.cpp)); a leaf copy does not count as a deep walk. `eshkol_heap_subtype_is_declared()` is an exhaustive switch over the same enum with no `default:`, so a member added to the enum and not to the predicate is a compile error, and a subtype byte read from a header that is not a declared member takes a loud fallback that names the value. The architecture model grades the tags against the evacuator's actual `case` arms (`INV-oalr-interior-pointer-deepwalk`, and `INV-vm-region-evac-subtype-total` for the VM evacuator) (since v1.3.5; [ADR 0014](design/adr/0014-release-invariant-contracts.md)).

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

**Implementation**: [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h) (3,759 lines)

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

**Implementation**: [`lib/types/hott_types.cpp`](../lib/types/hott_types.cpp) (1,130 lines), [`lib/types/type_checker.cpp`](../lib/types/type_checker.cpp) (6,061 lines)

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

**Current Status**: Type checker produces **warnings only** and does not block compilation (gradual typing), with one deliberate exception: a value carrying `TYPE_FLAG_LINEAR` (`Qubit`, `Handle`, `Stream`) is enforced. Cloning one is a compile-time error in the default build on both engines and no artifact is written (v1.3.5-evolve, #471).

#### The Type Relation

**Implementation**: [`inc/eshkol/types/type_relation.h`](../inc/eshkol/types/type_relation.h) (170 lines), [`lib/types/type_relation.cpp`](../lib/types/type_relation.cpp) (433 lines) (since v1.3.5; [ADR 0013](design/adr/0013-gradual-type-relation.md))

`TypeRelation` is the sole owner of every judgment that compares, combines or prints types. It is a lightweight view over one `TypeEnvironment`: it reads and extends the environment's interned signatures, pairs and sums, fills the environment's subtype cache, and keys nothing by expression or binding (per-expression facts belong to the semantic identity tables, not to this module). The lattice has `Value` at the top and `Never` at the bottom; between them sit the nominal graph, tracked pairs `Pair<A, B>` (a bare `Pair` is `Pair<Value, Value>`), sums `(+ A B ...)`, and function signatures `(-> A... R)` below the generic procedure types `Function` and `Closure`.

| Operation | Judgment |
|-----------|----------|
| `isSubtype(sub, super)` | Static subtyping `A <: B` (cached). Arrows are contravariant in parameters and covariant in results; pair components are covariant; a union fits when every source arm fits the target; an unresolved `Invalid` carries no evidence |
| `isConsistent(a, b)` | Siek-Taha consistency `A ~ B`: equal up to `Value`, structurally inside arrows and pairs. Symmetric, not transitive, never cached |
| `isConsistentSubtype(sub, super)` | Consistent subtyping, the gradual check applied to procedure arguments and return annotations |
| `compatibility(from, to)`, `accepts(from, to)` | Flow evidence (`RelationEvidence`): `Identity`, `Upcast`, `Dynamic`, `Numeric` or `Incompatible`. Every kind except `Incompatible` is accepted; the kind names the coercion a typed intermediate form makes explicit |
| `castable(actual, ascribed)` | The `(the T e)` rule: the ascription is accepted unless it is a provable contradiction, so overlapping types are castable |
| `join(a, b)`, `joinAll(types)`, `meet(a, b)` | Least upper and greatest lower bounds. Both recurse through unions, pairs and arrows, then use the nominal graph. `Never` is the join identity; disjoint concrete types meet at `Never` |
| `narrow(current, proven)` | The type a successful runtime test proves: the meet, or the proven type when the meet is empty |
| `widen(slot, incoming, policy)` | Inferred-slot widening under one of two policies (`WidenPolicy`). `InferenceSlot` (a named-let parameter) treats a join that reaches `Value` as a conflict and keeps the slot's type, except where either side is `Boolean`. `AdoptTop` (a `do` variable) adopts every join |
| `pairProjection(pair, side)` | The type `car` or `cdr` yields: a tracked component, a `List` tail, else `Value` |
| `print(type)` | The one printing function: `(-> Number Int64)`, `Pair<A, B>`, `(+ A B)`, `Function` |

`TypeEnvironment` keeps thin facades for its existing callers (`isSubtype`, `leastCommonSupertype`, `getTypeName`, `getFunctionTypeName`). Each one constructs a `TypeRelation` and delegates; none contains a rule of its own. The checker calls `TypeRelation` directly. `tests/types/type_relation_test.cpp` is the direct contract for the module (arrow variance, dynamic components, disjoint joins and meets, pair covariance, printing, both widening policies), compiled and run by `scripts/run_cpp_type_tests.sh`.

#### Checker Data Flow

The checker ([`lib/types/type_checker.cpp`](../lib/types/type_checker.cpp)) is bidirectional (`synthesize` and `check`) and runs as continuation tasks, so its native stack use does not grow with expression nesting. Three properties define how types flow through a program (since v1.3.5):

- **Every control form is synthesised.** Each evaluated subexpression of `cond`, `case`, `match`, `when`, `unless`, `do`, `guard`, `and`, `or`, `set!`, `begin` and body sequences, quasiquote escapes, `dynamic-wind`, `call/cc`, `values`, `call-with-values`, `let-values`, `raise`, `with-region`, the ownership forms, tensor literals, computed callees, and the function, point, direction and order operands of the differentiation operators reaches the checker, so an argument error inside any of them produces the same diagnostic as the same call at top level. Each branch body is checked in its own scope; a multi-branch form's type is the `join` of its branches plus the value the form yields when no branch runs (`#f` for `cond`, `case`, `when` and `unless`).
- **Recursive slots are typed by fixpoint.** A named-let parameter takes the join of its seed and every argument passed at a call that resolves to that loop's own binding (`LoopFrame` records the arguments per parameter). The body is checked in speculative passes: diagnostics are held (`emitDiagnostic`), and when a pass widens a parameter, `rollbackSpeculation` undoes the pass's diagnostics, recorded errors, linearity count and linear-usage counters before the body is checked again. The pass that widens nothing is the real check, and `releaseSpeculation` prints what it held. Annotated and linear bindings are never widened, and a parameter only moves up its finite supertype chain, so the iteration terminates. The same frame serves the result type of every recursive procedure (named let, function define, `letrec` lambda), bounded at 8 passes (`kMaxRecursionPasses`), after which the result is `Value`. A `do` variable is the fixpoint join of its init and its step under the `AdoptTop` policy.
- **One relation at every site.** Branch-producing forms use `join`; return annotations and procedure arguments use consistent subtyping; `the` uses `castable`; diagnostics print types through `TypeRelation::print`.

The user-facing account of annotations, inference and diagnostics is the [Gradual Typing guide](guide/GRADUAL_TYPING.md).

### Layer 3: Dependent Types

**Implementation**: [`lib/types/dependent.cpp`](../lib/types/dependent.cpp) (534 lines)

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

**Implementation**: [`lib/backend/autodiff_codegen.cpp`](../lib/backend/autodiff_codegen.cpp) (14,938 lines), with reverse-mode AD dispatch sites inside [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)

Eshkol provides **three modes** of automatic differentiation, each optimized for different use cases:

### Mode 1: Symbolic Differentiation

**Compile-time AST transformation**:

```scheme
(diff (* x x) x)  ; Compiles to: (* 2 x)
(diff (+ (* a x) b) x)  ; Compiles to: a
```

**Implementation**: [`buildSymbolicDerivative()`](../lib/backend/llvm_codegen.cpp) in llvm_codegen.cpp

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
    double derivative;  // Tangent f'(x); the complete native jet has six more slots
    double e2, e12;     // Independent second direction and mixed coefficient
    double ep, ep1, ep2, ep12; // Reverse-seed derivative jet
} eshkol_dual_number_t;

_Static_assert(sizeof(eshkol_dual_number_t) == 64, "Exact mixed-mode jet size required");
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

**7 Vector Calculus Operators** (implemented in [`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)):

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
(directional-derivative f (vector x y) (vector dx dy))
```

All implemented and tested in [`tests/autodiff/phase4_vector_calculus_test.esk`](../tests/autodiff/phase4_vector_calculus_test.esk).

---

## Closure System

**Implementation**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp), [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h)

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
- Bits 0-31: Actual capture count
- Bits 32-47: Fixed parameter count (0-65535)
- Bit 63: Variadic flag (0=fixed arity, 1=variadic)

**Access macros**:
```c
CLOSURE_ENV_GET_NUM_CAPTURES(packed)   // Extract bits 0-31
CLOSURE_ENV_GET_FIXED_PARAMS(packed)   // Extract bits 32-47
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

### Static Callee Bindings

**Implementation**: [`inc/eshkol/backend/static_callee_binding.h`](../inc/eshkol/backend/static_callee_binding.h) (247 lines) (since v1.3.5; [ADR 0015](design/adr/0015-static-callee-binding-identity.md))

Binding a variable to a lambda records a static alias, `<name>_func` (inside a function also `<function>.<name>_func`), so direct calls, `apply`, `map`, `reduce`, `remove` and the differentiation operators can call the LLVM function without going through the runtime closure value. An alias is a binding fact, and the header is the one place its validity is decided:

1. **The binding keeps its value.** A binding that is the target of a `set!` anywhere in its scope, or a top-level name that is defined more than once, gets no alias (`bindStaticCallee`). The top-level reassignment and redefinition facts come from the compiler's lexical mutation analysis over the fully expanded unit and are computed before code generation (`collectRedefinedTopLevelNames`, the memoised reassignment query, and the per-body `flat_mutation_targets_` summaries in [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)). The analysis uses the existing source binding names and storage locations, and it reads an `extern` declaration as signature metadata, not as a call expression.
2. **The alias belongs to the binding the name denotes at the use site.** Each alias records the LLVM storage of the binding that created it (`recordStaticCallee`), and `staticCalleeHiddenByRuntimeBinding` accepts an alias only when that storage is the one the name currently denotes. A sibling `let`, a parameter or a loop variable of the same name therefore hides an older alias.

A call through a mutable binding evaluates the binding's current value and dispatches through the closure ABI; an unmodified binding keeps direct-call resolution. `inheritStaticCalleeCapture` propagates an alias into a nested function only for a capture proven immutable.

### Capture Resolution for Direct Calls

**Implementation**: [`inc/eshkol/backend/closure_capture_scope.h`](../inc/eshkol/backend/closure_capture_scope.h) (182 lines), `AutodiffCodegen::appendDifferentiandCaptures` in [`lib/backend/autodiff_codegen.cpp`](../lib/backend/autodiff_codegen.cpp) (since v1.3.5)

A lambda with free variables lowers to an LLVM function that takes one pointer parameter per captured variable, appended after its user parameters. A site that calls such a function directly has to supply those pointers itself, and two rules keep that sound:

1. **A captured value comes only from the function being emitted** (one of its own arguments or instructions) or from the module (a global, constant or function). `valueUsableInFunction` is the structural check every capture site applies. A value that belongs to an enclosing function is reached through the current function's own capture pointer; where no such pointer exists, code generation stops with a diagnostic that names the variable, the owning function, the function being emitted and the source location.
2. **A callee reached through a name takes its captures from the closure object.** `emitClosureCaptureArguments` reads them from the closure's environment under the closure-call ABI, including the environment-pointer form used above 64 captures, which is what the runtime closure call does. An inline lambda created at the call site resolves its captures by name.

`appendDifferentiandCaptures` is the single resolver for a differentiated closure's captures. `derivative`, `derivative-n`, `taylor`, every `gradient` path, `jacobian`, `hessian`, and through them `divergence`, `curl`, `laplacian` and `directional-derivative`, all use it. `MapCodegen` applies the same two rules for `map` and multi-list `map`, and `reduce` dispatches on the closure value.

### Variadic Functions

**Rest parameters**:
```scheme
(define (variadic-fn a b . rest)
  (list a b rest))

(variadic-fn 1 2 3 4 5)  ; → (1 2 (3 4 5))
```

**Closure call dispatch** ([`llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)):
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

**Implementation**: [`lib/backend/tensor_codegen.cpp`](../lib/backend/tensor_codegen.cpp) (2,012-line dispatcher; per-domain ops in thirteen `tensor_*_codegen.cpp` siblings totalling 23,389 lines)

### Tensor Structure

```c
typedef struct eshkol_tensor {
    uint64_t* dimensions;     // Array of dimension sizes (8 bytes)
    uint64_t  num_dimensions; // Rank (8 bytes)
    int64_t*  elements;       // Elements as int64 bit patterns (8 bytes)
    uint64_t  total_elements; // Product of dimensions (8 bytes)
    uint64_t  dtype;          // Tensor dtype tag (8 bytes)
} eshkol_tensor_t;           // Total: 40 bytes (8-byte aligned)
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

**Implementation**: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) (main engine)

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

### Source Locations and Node Identity

**Implementation**: `eshkol_ast_t` in [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h), [`lib/frontend/parser.cpp`](../lib/frontend/parser.cpp), [`inc/eshkol/frontend/node_identity.h`](../inc/eshkol/frontend/node_identity.h) (since v1.3.5)

- **Every AST node is born with a location.** In C++, `eshkol_ast_t` carries default member initialisers: `line` and `column` are copied from a thread-local birth location (`eshkol_ast_birth_location`), and `source_file_id` and `node_id` start at 0. The parser, the macro expander and codegen each open an `EshkolAstBirthLocationScope` for the form they are processing, so a node synthesised while handling that form (internal-define `letrec*`, body sequences, named-let and `do` lowering, record-type expansion, nodes built by hand in codegen) inherits the location of the form it came from. A node's own stamp overrides its birth location, and outside any scope the location is 0/0, which means "no originating form". Raw arena storage is constructed through `eshkol_ast_construct_array`, so it follows the same rule. The initialisers, the scope class and the constructor helper sit inside `#ifdef __cplusplus`; the struct layout is identical in C and C++ and the type stays trivially copyable.
- **The reader keeps line and column per input stream.** The cumulative position lives in `std::ios_base` storage on the stream (`xalloc` slots), bound to the reader only for the duration of one call (`StreamPositionBinding`). A nested read of a required module, which the AOT driver performs while it is still reading the parent, cannot disturb the parent's position; a fresh stream starts at line 1, and the position dies with the stream.
- **One node-identity key.** The parser's `eshkol_ast_t::node_id`, the NodeId allocator (`eshkol_node_id_new`) and the semantic queries (`eshkol_binding_id_for_node`, `eshkol_typed_expr_info`) share the `eshkol_node_id_t` key type. The AST field is a 32-bit alias of it, which preserves the public field width and layout ([ADR 0014](design/adr/0014-release-invariant-contracts.md)). Payload (source spans, binding identity, typed-expression facts) lives in side tables keyed on this value, not in fields on the node.

Deterministic locations are what make language-coverage records reproducible; `language_coverage_determinism_test` holds that property (see [TESTING.md](TESTING.md#language-coverage-instrumentation)).

### Driver Analyses Before Codegen

**Implementation**: `OwnershipAnalyzer` and `EscapeAnalyzer` in [`exe/eshkol-run.cpp`](../exe/eshkol-run.cpp)

Ownership analysis and escape analysis traverse the AST iteratively. Each analyzer keeps an explicit stack of work records (`WorkItem`): an `AST` record visits a node, and continuation records (`LET_BINDING`, `EXIT_SCOPE`, `UNBORROW` for ownership; `LET_BINDING`, `EXIT_SCOPE`, `LET_EXIT`, `LAMBDA_EXIT` for escape) run the step that follows a child's traversal. The records preserve the traversal order and the scope cleanup of a recursive walk, so diagnostics are unchanged, while native stack use is independent of source nesting. Together with the explicit continuation stacks in the parser, the type checker and codegen, this lets deeply nested source compile ahead of time within the default 8 MiB stack: the `parser_stack_compile` CTest compiles and runs 16,000 levels of nesting through the JIT and AOT with the stack limit fixed at 8 MiB, and `ownership_nested_diagnostics` holds the analyzers' diagnostics on nested input (since v1.3.5).

### AST ownership

Every phase above reads the same AST, so its data lives for the whole
compilation:

- **Node identity and spans**: `NodeId -> SourceSpan`, minted by the parser
  ([`node_identity.h`](../inc/eshkol/frontend/node_identity.h), ADR-0000
  Stage 1).
- **String payloads**: identifiers, literal text, operation names, rest
  parameters, type-variable names, and every name that expansion, renaming,
  the driver, the REPL or codegen synthesizes. These have one owner, a
  process-rooted chunked arena
  ([`ast_strings.h`](../inc/eshkol/frontend/ast_strings.h),
  [ADR-0021](design/adr/0021-ast-string-owner.md)). Producers allocate from
  it, and no consumer frees an individual string. `eshkol-run` releases it
  when `main()` returns, and the REPL releases it in its ordered exit.
- **Node storage** (`eshkol_ast_t` arrays) is still reclaimed only when the
  process exits (epic #182).

### Special Forms (70+)

**Core**: `define`, `define-type`, `define-syntax`, `set!`, `lambda`, `let`, `let*`, `letrec`, `if`, `cond`, `case`, `match`, `and`, `or`, `when`, `unless`, `do`

**Quotation**: `quote`, `quasiquote`, `unquote`, `unquote-splicing`

**Functions**: `call`, `apply`, `compose`, `values`, `call-with-values`

**Memory**: `with-region`, `owned`, `move`, `borrow`, `shared`, `weak-ref`

**Autodiff**: `diff`, `derivative`, `gradient`, `jacobian`, `hessian`, `divergence`, `curl`, `laplacian`, `directional-derivative`

**Exceptions**: `guard`, `raise`

**Modules**: `require`, `provide` (new), `import` (legacy)

**Pattern Matching**: Recursive patterns with `match`

---

## Module System

**Implementation**: [`exe/eshkol-run.cpp`](../exe/eshkol-run.cpp) (6,099 lines)

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
│ 1. The requiring file's directory                               │
│ 2. The project root (current directory)                         │
│ 3. $ESHKOL_PATH / -I directories (the explicit override)        │
│ 4. Library path (lib/): beside the compiler, then system        │
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
core.json → lib/core/json.esk
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

**Implementation**: [`lib/repl/repl_jit.cpp`](../lib/repl/repl_jit.cpp) (4,679 lines), [`exe/eshkol-repl.cpp`](../exe/eshkol-repl.cpp) (1,743 lines)

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

**[`lib/math.esk`](../lib/math.esk)** (412 lines):

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

The indented tree is an illustrative layout snapshot; its per-file size annotations are historical and may not match current sources.

```
eshkol/
├── CMakeLists.txt          # Build system (6,484 lines)
├── README.md               # Project overview
├── LICENSE                 # MIT license
│
├── inc/eshkol/             # Public headers
│   ├── eshkol.h            # Main header (3,759 lines)
│   ├── llvm_backend.h      # Backend API (432 lines)
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
│   ├── stdlib.esk          # Standard library (149 lines, re-exports)
│   ├── math.esk            # Math library (412 lines)
│   │
│   ├── backend/            # 35 codegen modules (~106.5K lines)
│   │   ├── llvm_codegen.cpp      # Main engine (44,003 lines)
│   │   ├── arithmetic_codegen.cpp# Polymorphic arithmetic (4,012 lines)
│   │   ├── autodiff_codegen.cpp  # AD operations (14,545 lines)
│   │   ├── tensor_codegen.cpp    # Tensor-op dispatcher (2,012 lines); per-domain in tensor_*_codegen.cpp
│   │   ├── collection_codegen.cpp# Lists/vectors (3,173 lines)
│   │   ├── control_flow_codegen.cpp # if/cond/and/or (1,107 lines)
│   │   ├── binding_codegen.cpp   # define/let/set! (1,662 lines)
│   │   ├── call_apply_codegen.cpp# apply & closures (1,270 lines)
│   │   ├── map_codegen.cpp       # Higher-order map (1,142 lines)
│   │   ├── homoiconic_codegen.cpp# Quote & S-expr (706 lines)
│   │   ├── string_io_codegen.cpp # Strings & I/O (3,860 lines)
│   │   ├── hash_codegen.cpp      # Hash tables (671 lines)
│   │   ├── tail_call_codegen.cpp # TCO infra (748 lines)
│   │   ├── type_system.cpp       # LLVM types (187 lines)
│   │   ├── tagged_value_codegen.cpp # Pack/unpack (807 lines)
│   │   ├── memory_codegen.cpp    # Arena decls (329 lines)
│   │   ├── builtin_declarations.cpp # Runtime funcs (148 lines)
│   │   ├── function_cache.cpp    # C library (173 lines)
│   │   ├── codegen_context.cpp   # Shared state (377 lines)
│   │   └── function_codegen.cpp  # Lambda/closure (209 lines)
│   │
│   ├── core/               # Core runtime (C)
│   │   ├── runtime_arena_core.cpp # Arena runtime core (720 lines)
│   │   ├── runtime_regions.cpp  # OALR regions (2,296 lines)
│   │   ├── arena_memory.h   # Memory header (1,041 lines)
│   │   ├── ast.cpp          # AST manipulation (653 lines)
│   │   ├── logger.cpp       # Logging
│   │   ├── printer.cpp      # Display system
│   │   └── *.esk            # Stdlib modules (33 files)
│   │
│   ├── frontend/
│   │   ├── parser.cpp       # S-expr parser (11,115 lines)
│   │   └── macro_expander.cpp # Macro system (1,658 lines)
│   │
│   ├── types/
│   │   ├── hott_types.cpp   # HoTT types (1,247 lines)
│   │   ├── type_checker.cpp # Type inference (3,910 lines)
│   │   └── dependent.cpp    # Dependent types (534 lines)
│   │
│   ├── repl/
│   │   ├── repl_jit.cpp     # JIT compiler (4,679 lines)
│   │   └── repl_utils.h     # REPL utilities
│   │
│   └── quantum/
│       ├── quantum_rng.c    # Quantum RNG (614 lines)
│       └── quantum_rng.h    # RNG header
│
├── exe/
│   ├── eshkol-run.cpp      # Compiler executable (5,820 lines)
│   └── eshkol-repl.cpp     # REPL executable (1,088 lines)
│
└── tests/                  # 1,600+ test files
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

**Status**: 35 `*codegen*.cpp` translation units under `lib/backend/`; the extraction from the original monolith is ongoing, not complete

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

### Shared Codegen Facilities

Three header-only facilities under `inc/eshkol/backend/` each own one decision that several codegen modules need, so no module carries a private copy of the rule (since v1.3.5):

| Header | Lines | The decision it owns |
|--------|------:|----------------------|
| [`libm_codegen.h`](../inc/eshkol/backend/libm_codegen.h) | 204 | How codegen obtains a libm function |
| [`static_callee_binding.h`](../inc/eshkol/backend/static_callee_binding.h) | 247 | When a variable may be resolved to an `llvm::Function` at compile time (see [Static Callee Bindings](#static-callee-bindings)) |
| [`closure_capture_scope.h`](../inc/eshkol/backend/closure_capture_scope.h) | 182 | Where a statically resolved closure's captures come from (see [Capture Resolution for Direct Calls](#capture-resolution-for-direct-calls)) |

**libm access.** `eshkol::libm_codegen::unary` and `binary` are the single way codegen obtains a libm function. They answer with the LLVM intrinsic (`llvm.exp.f64`, `llvm.log.f64`, `llvm.pow.f64`, ...) wherever the LLVM major being built against has one; an intrinsic name is reserved, so no module symbol, including a user program's own `(define (exp x) ...)`, can shadow it. Where no intrinsic exists (`tanh` before LLVM 19, `atan2` before LLVM 20), the helper looks the function up by name and verifies the found function's type against the signature about to be emitted; on a mismatch it declares a distinctly named function (`eshkol_libm_<name>`), so a collision is a link failure and never a call through the wrong ABI. `BuiltinFactoryCodegen` declares the libm names that have no intrinsic on any supported LLVM (`asinh`, `acosh`, `atanh`, `cbrt`, `fmod`, `remainder`, `nextafter`) at module initialisation, before any user definition or lowering runs. `AutodiffCodegen::getMathFunc` and `EshkolLLVMCodeGen::mathFunc` route the scalar-math and AD paths through the same helper, verify a cached row's type before reuse, and never return null.

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

**Implementation**: [`CMakeLists.txt`](../CMakeLists.txt) (10,748 lines)

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

**1,020 self-reported tests** across 46 suites (the aggregate `scripts/run_all_tests.sh` run; figures in [TEST_COVERAGE.md](TEST_COVERAGE.md)). Representative categories:

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
- Correct results
- Type safety
- Memory cleanup (no leaks)
- Error handling (for failure tests)

---

## What's NOT in v1.1

These features are **designed but not implemented**. See roadmap documents for details:

- **Native Quantum Types**: Qubit, qreg, quantum gates (design in `QUANTUM_STOCHASTIC_COMPUTING_ARCHITECTURE.md`)
  - **What IS implemented**: Quantum RNG ([`lib/quantum/quantum_rng.c`](../lib/quantum/quantum_rng.c))

- **Multimedia**: Windows, graphics, audio, GPIO (design in `MULTIMEDIA_SYSTEM_ARCHITECTURE.md`)

- **Full Linear Types**: Compile-time enforcement of use-exactly-once (partial: warnings only)

- **Dependent Types with Proofs**: Full proof terms (partial: dimension checking only)

**Note**: Logic programming, previously listed as unimplemented, shipped in v1.1 as part of the Consciousness Engine (see [v1.1 Architecture Extensions](#v11-architecture-extensions)).

**See**: [`ROADMAP.md`](../ROADMAP.md) for future releases.

---

## References

### Primary Source Files (analyzed in detail)

- [`inc/eshkol/eshkol.h`](../inc/eshkol/eshkol.h) - Main system header (3,759 lines)
- [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp) - Core codegen (47,107 lines)
- [`lib/core/runtime_arena_core.cpp`](../lib/core/runtime_arena_core.cpp) - Arena runtime core (1226 lines; 4,259 across all `runtime_*` memory modules)
- [`lib/frontend/parser.cpp`](../lib/frontend/parser.cpp) - S-expr parser (11,691 lines)
- [`lib/types/type_checker.cpp`](../lib/types/type_checker.cpp) - Type inference (6,061 lines)
- [`lib/repl/repl_jit.cpp`](../lib/repl/repl_jit.cpp) - JIT compiler (4,679 lines)
- [`exe/eshkol-run.cpp`](../exe/eshkol-run.cpp) - Compiler executable (6,099 lines)
- [`lib/types/type_relation.cpp`](../lib/types/type_relation.cpp) - Gradual type relation (433 lines)

### Forward-looking design documents

- [`future/NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md`](future/NEURO_SYMBOLIC_COMPLETE_ARCHITECTURE.md) — neuro-symbolic stack beyond v1.2
- [`future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md`](future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md) — multimedia / graphics / audio roadmap
- (Quantum-stochastic and full HoTT integration documents were planned but are not in the public tree; the project memory tracks both as v2.0+ research items.)

### Related Documentation

- [`API_REFERENCE.md`](API_REFERENCE.md) - Complete API reference with examples
- [`QUICKSTART.md`](QUICKSTART.md) - 15-minute getting-started guide
- [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) - Current limitations and roadmap items

---

## v1.1 Architecture Extensions

The v1.1-accelerate release adds six major subsystems to the compiler and runtime. Each integrates with the existing LLVM codegen pipeline, arena memory, and module system described above.

---

### XLA Backend (Dual-Mode)

**Implementation**: [`lib/backend/xla/`](../lib/backend/xla/)

Eshkol v1.1 provides an optional XLA compilation path for tensor-heavy workloads. The backend operates in dual mode:

1. **StableHLO/MLIR path**: Emits StableHLO dialect operations for dynamic shapes, broadcasting, and large tensor programs. The [`stablehlo_emitter.cpp`](../lib/backend/xla/stablehlo_emitter.cpp) translates Eshkol tensor AST nodes into StableHLO IR. [`xla_compiler.cpp`](../lib/backend/xla/xla_compiler.cpp) lowers StableHLO through the MLIR pipeline to executable code.

2. **LLVM-direct fallback**: Operations that do not benefit from XLA overhead (small tensors, scalar-heavy code) remain on the standard LLVM codegen path. The cost model in [`xla_codegen.cpp`](../lib/backend/xla/xla_codegen.cpp) selects the appropriate backend per operation.

**Key files**:

| File | Role |
|------|------|
| [`xla_codegen.cpp`](../lib/backend/xla/xla_codegen.cpp) | Top-level dispatch, cost model, LLVM integration |
| [`stablehlo_emitter.cpp`](../lib/backend/xla/stablehlo_emitter.cpp) | AST → StableHLO dialect translation |
| [`xla_compiler.cpp`](../lib/backend/xla/xla_compiler.cpp) | StableHLO → executable lowering pipeline |
| [`xla_runtime.cpp`](../lib/backend/xla/xla_runtime.cpp) | Runtime buffer management, execution |
| [`xla_memory.cpp`](../lib/backend/xla/xla_memory.cpp) | XLA-specific memory allocation and lifetime |
| [`xla_types.cpp`](../lib/backend/xla/xla_types.cpp) | Eshkol type ↔ XLA element type mapping |

**ORC JIT integration**: StableHLO programs are compiled at runtime via LLVM ORC, enabling the REPL to execute XLA-accelerated tensor code without ahead-of-time compilation.

---

### GPU Acceleration

**Implementation**: [`lib/backend/gpu/`](../lib/backend/gpu/)

Two GPU backends provide hardware-accelerated tensor operations:

**Metal (Apple Silicon)**:
- [`gpu_memory.mm`](../lib/backend/gpu/gpu_memory.mm): Objective-C++ Metal API integration
- Software float64 (SF64) emulation — Metal lacks native float64 support
- [`metal_softfloat.h`](../lib/backend/gpu/metal_softfloat.h): IEEE 754 double-precision arithmetic in Metal shading language
- Shader source embedded at build time; no runtime file I/O

**CUDA**:
- [`gpu_memory_cuda.cpp`](../lib/backend/gpu/gpu_memory_cuda.cpp): CUDA API integration
- [`gpu_cuda_kernels.cu`](../lib/backend/gpu/gpu_cuda_kernels.cu): Native float64 kernels, cuBLAS for matrix operations
- [`gpu_memory_stub.cpp`](../lib/backend/gpu/gpu_memory_stub.cpp): No-op stub for builds without GPU support

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

**Implementation**: [`lib/core/logic.cpp`](../lib/core/logic.cpp), [`lib/core/inference.cpp`](../lib/core/inference.cpp), [`lib/core/workspace.cpp`](../lib/core/workspace.cpp)

Three interconnected subsystems implement a compiled consciousness architecture:

#### Logic Programming

**Files**: [`inc/eshkol/core/logic.h`](../inc/eshkol/core/logic.h), [`lib/core/logic.cpp`](../lib/core/logic.cpp)

- First-order unification (Martelli-Montanari algorithm) with occurs check and triangular substitution chains
- Knowledge base with fact assertion and pattern-matching conjunctive query
- Walk operation: chain dereferencing through substitution bindings to ground values
- Parser extension: `?x` syntax produces `ESHKOL_LOGIC_VAR_OP` AST nodes (R7RS-compatible — `?` is a valid identifier start character)

#### Active Inference

**Files**: [`inc/eshkol/core/inference.h`](../inc/eshkol/core/inference.h), [`lib/core/inference.cpp`](../lib/core/inference.cpp)

- Factor graphs: bipartite graph G = (V, F, E) with discrete variable nodes and factor nodes
- Conditional probability tables (CPTs) as flat log-probability tensors indexed by joint state assignment
- Sum-product belief propagation in log-space (`fg-infer!`) with configurable max iterations
- CPT mutation for online learning (`fg-update-cpt!` — replaces CPT, resets all messages, beliefs reconverge on next inference pass)
- Variational free energy: F = E_q[log q(s)] - E_q[log p(o, s)]
- Expected free energy (EFE) for action selection: G = E_q[log q(s') - log p(o', s')]
- Observation format: `#(var_index observed_state)` pairs, not state vectors

#### Global Workspace

**Files**: [`inc/eshkol/core/workspace.h`](../inc/eshkol/core/workspace.h), [`lib/core/workspace.cpp`](../lib/core/workspace.cpp)

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

**Implementation**: [`lib/backend/parallel_llvm_codegen.cpp`](../lib/backend/parallel_llvm_codegen.cpp)

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

**Implementation**: [`inc/eshkol/core/bignum.h`](../inc/eshkol/core/bignum.h), [`lib/core/bignum.cpp`](../lib/core/bignum.cpp)

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

**Codegen helpers**: `emitBignumBinaryCall`, `emitBignumCompareCall`, `emitIsBignumCheck` in [`arithmetic_codegen.cpp`](../lib/backend/arithmetic_codegen.cpp).

#### Rational (Exact Fractions)

**Implementation**: [`inc/eshkol/core/rational.h`](../inc/eshkol/core/rational.h), [`lib/core/rational.cpp`](../lib/core/rational.cpp)

- GCD-reduced canonical form with positive denominator invariant
- Heap-allocated as `HEAP_PTR` with subtype discrimination
- `eshkol_rational_compare_tagged_ptr` for comparison dispatch in [`arithmetic_codegen.cpp`](../lib/backend/arithmetic_codegen.cpp)

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

**Implementation**: [`lib/repl/repl_jit.cpp`](../lib/repl/repl_jit.cpp), [`exe/eshkol-repl.cpp`](../exe/eshkol-repl.cpp)

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

*This document reflects the v1.3.5-evolve release. All claims are verified against actual source code. For questions or corrections, see [`CONTRIBUTING.md`](../CONTRIBUTING.md).*
