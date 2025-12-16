# Type System in Eshkol

## Table of Contents
- [Type System in Eshkol](#type-system-in-eshkol)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Runtime Type System](#runtime-type-system)
    - [Immediate Values](#immediate-values)
    - [Pointer Types with Headers](#pointer-types-with-headers)
      - [HEAP\_PTR (Type 8)](#heap_ptr-type-8)
      - [CALLABLE (Type 9)](#callable-type-9)
    - [Object Header System](#object-header-system)
  - [HoTT Compile-Time Type System](#hott-compile-time-type-system)
    - [Type Hierarchy](#type-hierarchy)
    - [Type Expression Syntax](#type-expression-syntax)
    - [Type Annotations](#type-annotations)
  - [Gradual Typing](#gradual-typing)
  - [Type Checking Examples](#type-checking-examples)
    - [Checking Object Types](#checking-object-types)
    - [Compatibility Macros](#compatibility-macros)
  - [Implementation Details](#implementation-details)
    - [Tagged Value Size](#tagged-value-size)
    - [Exactness Flags](#exactness-flags)
    - [Type System Source Files](#type-system-source-files)
  - [See Also](#see-also)

---

## Overview

Eshkol features a sophisticated **triple-layer type system** combining runtime tagged values, compile-time HoTT-inspired type checking, and optional dependent types:

1. **Runtime Types**: 8-bit type tags with 8-byte object headers for heap-allocated values
2. **HoTT Types**: Compile-time type inference producing warnings (gradual typing)
3. **Dependent Types**: Optional compile-time value tracking for array bounds and dimensions

All three layers work together to provide performance, safety, and expressiveness for scientific computing.

---

## Runtime Type System

The runtime type system uses **16-byte tagged values** (`eshkol_tagged_value_t`) where every value carries an 8-bit type tag:

```c
typedef struct eshkol_tagged_value {
    uint8_t type;        // Value type (eshkol_value_type_t)
    uint8_t flags;       // Exactness and other flags
    uint16_t reserved;   // Reserved for future use
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;   // Pointer to heap object
        uint64_t raw_val;
    } data;
} eshkol_tagged_value_t;  // 16 bytes total
```

### Immediate Values

**Types 0-7** store data directly in the tagged value (no heap allocation):

| Type | Value | Description | Storage |
|------|-------|-------------|---------|
| [`ESHKOL_VALUE_NULL`](inc/eshkol/eshkol.h:56) | 0 | Empty list `()` | No data needed |
| [`ESHKOL_VALUE_INT64`](inc/eshkol/eshkol.h:57) | 1 | 64-bit signed integer | `data.int_val` |
| [`ESHKOL_VALUE_DOUBLE`](inc/eshkol/eshkol.h:58) | 2 | Double-precision float | `data.double_val` |
| [`ESHKOL_VALUE_BOOL`](inc/eshkol/eshkol.h:59) | 3 | Boolean `#t`/`#f` | `data.int_val` (0/1) |
| [`ESHKOL_VALUE_CHAR`](inc/eshkol/eshkol.h:60) | 4 | Unicode character | `data.int_val` (codepoint) |
| [`ESHKOL_VALUE_SYMBOL`](inc/eshkol/eshkol.h:61) | 5 | Interned symbol | `data.int_val` (symbol ID) |
| [`ESHKOL_VALUE_DUAL_NUMBER`](inc/eshkol/eshkol.h:62) | 6 | Forward-mode AD dual | `data.ptr_val` → 16-byte `{value, derivative}` |
| *(Reserved)* | 7 | Future use | — |

**Example: Creating immediate values**

```c
// Integer (exact)
eshkol_tagged_value_t num = eshkol_make_int64(42, true);
// num.type = ESHKOL_VALUE_INT64
// num.data.int_val = 42

// Double (inexact)
eshkol_tagged_value_t pi = eshkol_make_double(3.14159);
// pi.type = ESHKOL_VALUE_DOUBLE
// pi.data.double_val = 3.14159
```

### Pointer Types with Headers

**Types 8-9** are consolidated pointer types. The [`data.ptr_val`](inc/eshkol/eshkol.h:126) points to a heap object with an 8-byte header **prepended** to the data:

#### HEAP_PTR (Type 8)

[`ESHKOL_VALUE_HEAP_PTR`](inc/eshkol/eshkol.h:69) consolidates all data structures. The header's **subtype field** distinguishes them:

| Subtype | Constant | Eshkol Type | Implementation |
|---------|----------|-------------|----------------|
| 0 | [`HEAP_SUBTYPE_CONS`](inc/eshkol/eshkol.h:291) | Cons cell (pair) | [`lib/backend/collection_codegen.cpp`](lib/backend/collection_codegen.cpp:120) |
| 1 | [`HEAP_SUBTYPE_STRING`](inc/eshkol/eshkol.h:292) | UTF-8 string | [`lib/backend/string_io_codegen.cpp`](lib/backend/string_io_codegen.cpp:45) |
| 2 | [`HEAP_SUBTYPE_VECTOR`](inc/eshkol/eshkol.h:293) | Scheme vector (heterogeneous) | [`lib/backend/collection_codegen.cpp`](lib/backend/collection_codegen.cpp:890) |
| 3 | [`HEAP_SUBTYPE_TENSOR`](inc/eshkol/eshkol.h:294) | N-dimensional numeric tensor | [`lib/backend/tensor_codegen.cpp`](lib/backend/tensor_codegen.cpp:150) |
| 4 | [`HEAP_SUBTYPE_MULTI_VALUE`](inc/eshkol/eshkol.h:295) | Multiple return values | [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp:5234) |
| 5 | [`HEAP_SUBTYPE_HASH`](inc/eshkol/eshkol.h:296) | Hash table | [`lib/backend/hash_codegen.cpp`](lib/backend/hash_codegen.cpp:78) |
| 6 | [`HEAP_SUBTYPE_EXCEPTION`](inc/eshkol/eshkol.h:297) | Exception object | [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp:1456) |
| 7 | [`HEAP_SUBTYPE_RECORD`](inc/eshkol/eshkol.h:298) | User-defined record | *(Planned)* |
| 8 | [`HEAP_SUBTYPE_BYTEVECTOR`](inc/eshkol/eshkol.h:299) | Raw byte vector (R7RS) | *(Planned)* |
| 9 | [`HEAP_SUBTYPE_PORT`](inc/eshkol/eshkol.h:300) | I/O port | *(Planned)* |

#### CALLABLE (Type 9)

[`ESHKOL_VALUE_CALLABLE`](inc/eshkol/eshkol.h:70) consolidates all function-like objects:

| Subtype | Constant | Description | Implementation |
|---------|----------|-------------|----------------|
| 0 | [`CALLABLE_SUBTYPE_CLOSURE`](inc/eshkol/eshkol.h:309) | Compiled closure (func_ptr + env) | [`lib/backend/function_codegen.cpp`](lib/backend/function_codegen.cpp:234) |
| 1 | [`CALLABLE_SUBTYPE_LAMBDA_SEXPR`](inc/eshkol/eshkol.h:310) | Lambda as data (homoiconic) | [`lib/backend/homoiconic_codegen.cpp`](lib/backend/homoiconic_codegen.cpp:156) |
| 2 | [`CALLABLE_SUBTYPE_AD_NODE`](inc/eshkol/eshkol.h:311) | Autodiff computation node | [`lib/backend/autodiff_codegen.cpp`](lib/backend/autodiff_codegen.cpp:89) |
| 3 | [`CALLABLE_SUBTYPE_PRIMITIVE`](inc/eshkol/eshkol.h:312) | Built-in primitive function | *(Planned)* |
| 4 | [`CALLABLE_SUBTYPE_CONTINUATION`](inc/eshkol/eshkol.h:313) | First-class continuation | *(Planned)* |

### Object Header System

Every heap-allocated object has an **8-byte header at offset -8** from the data pointer:

```c
typedef struct eshkol_object_header {
    uint8_t  subtype;      // Type-specific subtype (0-255)
    uint8_t  flags;        // GC mark, linear, borrowed, etc.
    uint16_t ref_count;    // Reference count (0 = not ref-counted)
    uint32_t size;         // Object size in bytes (excluding header)
} eshkol_object_header_t;  // 8 bytes
```

**Memory Layout Example (Cons Cell):**

```
┌────────────────────────┬──────────────────────────────────┐
│  Object Header (8B)    │  Cons Data (32B)                 │
├────────────────────────┼──────────────────────────────────┤
│ subtype: 0 (CONS)      │ car: eshkol_tagged_value_t (16B) │
│ flags: 0x00            │ cdr: eshkol_tagged_value_t (16B) │
│ ref_count: 0           │                                  │
│ size: 32               │                                  │
└────────────────────────┴──────────────────────────────────┘
         ↑                        ↑
      Header            data.ptr_val points here
```

**Accessing Header from Pointer:**

```c
// Get header from data pointer
void* data_ptr = tagged_val.data.ptr_val;
eshkol_object_header_t* header = ESHKOL_GET_HEADER(data_ptr);
uint8_t subtype = header->subtype;

// Check if it's a cons cell
if (tagged_val.type == ESHKOL_VALUE_HEAP_PTR && subtype == HEAP_SUBTYPE_CONS) {
    // It's a cons cell
    eshkol_cons_t* cons = (eshkol_cons_t*)data_ptr;
}
```

**Object Flags:**

```c
#define ESHKOL_OBJ_FLAG_MARKED    0x01  // GC mark bit
#define ESHKOL_OBJ_FLAG_LINEAR    0x02  // Linear type (must consume exactly once)
#define ESHKOL_OBJ_FLAG_BORROWED  0x04  // Currently borrowed (temporary access)
#define ESHKOL_OBJ_FLAG_CONSUMED  0x08  // Linear value has been consumed
#define ESHKOL_OBJ_FLAG_SHARED    0x10  // Reference-counted shared object
#define ESHKOL_OBJ_FLAG_WEAK      0x20  // Weak reference
#define ESHKOL_OBJ_FLAG_PINNED    0x40  // Pinned in memory (no relocation)
#define ESHKOL_OBJ_FLAG_EXTERNAL  0x80  // External resource (needs cleanup)
```

---

## HoTT Compile-Time Type System

Eshkol includes a **Homotopy Type Theory-inspired** compile-time type system (implementation: [`lib/types/type_checker.cpp`](lib/types/type_checker.cpp:1)). It performs type inference and produces **warnings** (not errors) for type mismatches.

### Type Hierarchy

```
Universe 𝒰₂
  ├─ 𝒰₁ (types of types)
  │   ├─ 𝒰₀ (base types)
  │   │   ├─ integer
  │   │   ├─ real
  │   │   ├─ boolean
  │   │   ├─ string
  │   │   ├─ char
  │   │   ├─ symbol
  │   │   ├─ null
  │   │   ├─ (list τ)
  │   │   ├─ (vector τ)
  │   │   ├─ (tensor τ)
  │   │   └─ (→ τ₁ τ₂ ... τₙ)  ; function types
  │   └─ Polymorphic types
  │       └─ (forall (α β ...) τ)
  └─ Type operators
```

### Type Expression Syntax

```scheme
;; Primitive types
integer        ; 64-bit signed integer
real           ; double-precision float
number         ; supertype of integer and real
boolean        ; #t or #f
string         ; UTF-8 string
char           ; Unicode character
symbol         ; Interned symbol

;; Compound types
(-> τ₁ τ₂ ... τₙ τᵣ)    ; Function type (last is return type)
(list τ)                 ; Homogeneous list
(vector τ)               ; Heterogeneous Scheme vector
(tensor τ)               ; N-dimensional numeric tensor
(pair τ₁ τ₂)             ; Cons pair

;; Polymorphic types
(forall (α β) (-> α β β))  ; Generic function

;; Top and bottom
any            ; Top type (accepts anything)
nothing        ; Bottom type (uninhabited)
```

### Type Annotations

**Inline parameter types:**

```scheme
(define (add-ints x : integer y : integer) : integer
  (+ x y))
```

**Separate type declarations:**

```scheme
(: factorial (-> integer integer))
(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))
```

**Lambda type annotations:**

```scheme
(lambda (x : real) : real
  (* x x))
```

---

## Gradual Typing

Eshkol implements **gradual typing**: type annotations are optional, and type mismatches produce **warnings** (not errors). Code compiles and runs even with type warnings.

```scheme
;; No annotations - type inference works
(define (double x)
  (* x 2))

;; Partial annotations - mix typed and untyped
(define (process data : list)
  (map double data))

;; Full annotations - maximum safety
(define (safe-divide : (-> real real (maybe real)))
  (lambda (a : real b : real)
    (if (= b 0.0)
        nothing
        (just (/ a b)))))
```

**Type Inference Example:**

```scheme
;; Compiler infers:
;; double : (-> number number)
(define (double x) (* x 2))

;; Compiler infers:
;; make-point : (-> real real (pair real real))
(define (make-point x y)
  (cons x y))
```

---

## Type Checking Examples

### Checking Object Types

```c
// Check immediate types (no pointer dereference)
bool is_int = (val.type == ESHKOL_VALUE_INT64);
bool is_double = (val.type == ESHKOL_VALUE_DOUBLE);

// Check consolidated types
bool is_heap = (val.type == ESHKOL_VALUE_HEAP_PTR);
bool is_callable = (val.type == ESHKOL_VALUE_CALLABLE);

// Check specific heap subtype
if (val.type == ESHKOL_VALUE_HEAP_PTR) {
    uint8_t subtype = ESHKOL_GET_SUBTYPE((void*)val.data.ptr_val);
    if (subtype == HEAP_SUBTYPE_CONS) {
        // It's a cons cell
    } else if (subtype == HEAP_SUBTYPE_TENSOR) {
        // It's a tensor
    }
}

// Check specific callable subtype
if (val.type == ESHKOL_VALUE_CALLABLE) {
    uint8_t subtype = ESHKOL_GET_SUBTYPE((void*)val.data.ptr_val);
    if (subtype == CALLABLE_SUBTYPE_CLOSURE) {
        // It's a closure
        eshkol_closure_t* closure = (eshkol_closure_t*)val.data.ptr_val;
    }
}
```

### Compatibility Macros

For backward compatibility with the display system, Eshkol provides compatibility macros that check **both** new consolidated types and legacy types:

```c
// Check if value is a cons (new or legacy format)
bool is_cons = ESHKOL_IS_CONS_COMPAT(val);

// Check if value is a closure (new or legacy format)
bool is_closure = ESHKOL_IS_CLOSURE_COMPAT(val);
```

---

## Implementation Details

### Tagged Value Size

```c
_Static_assert(sizeof(eshkol_tagged_value_t) <= 16,
               "Tagged value must fit in 16 bytes for efficiency");
```

**Memory Layout:**

```
Offset  Size  Field
------  ----  -----
0       1     type
1       1     flags
2       2     reserved
4       4     (padding)
8       8     data (union)
------  ----
Total:  16 bytes
```

### Exactness Flags

Scheme distinguishes **exact** (integer) from **inexact** (floating-point) numbers:

```c
#define ESHKOL_VALUE_EXACT_FLAG   0x10
#define ESHKOL_VALUE_INEXACT_FLAG 0x20

// Check exactness
bool is_exact = (val.flags & ESHKOL_VALUE_EXACT_FLAG);
bool is_inexact = (val.flags & ESHKOL_VALUE_INEXACT_FLAG);
```

### Type System Source Files

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Runtime Types** | [`inc/eshkol/eshkol.h`](inc/eshkol/eshkol.h:48-547) | 500 | Type definitions, macros |
| **HoTT Type Checker** | [`lib/types/type_checker.cpp`](lib/types/type_checker.cpp:1) | 1,561 | Type inference engine |
| **HoTT Types** | [`lib/types/hott_types.cpp`](lib/types/hott_types.cpp:1) | 892 | Type expression manipulation |
| **Dependent Types** | [`lib/types/dependent.cpp`](lib/types/dependent.cpp:1) | 423 | Compile-time value tracking |
| **Type Codegen** | [`lib/backend/type_system.cpp`](lib/backend/type_system.cpp:1) | 287 | LLVM type generation |

---

## See Also

- [Memory Management (OALR System)](MEMORY_MANAGEMENT.md) - Arena allocation, object headers, lifetimes
- [Vector Operations](VECTOR_OPERATIONS.md) - Scheme vectors vs. tensors
- [Automatic Differentiation](AUTODIFF.md) - Dual numbers, AD nodes, computational graphs
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - Type checking pipeline, LLVM codegen
- [API Reference](../API_REFERENCE.md) - Complete function reference with types
