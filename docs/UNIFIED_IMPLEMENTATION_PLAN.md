# Eshkol Unified Implementation Plan
## Foundation for Multimedia Scientific Computing

**Document Version**: 1.0
**Created**: December 2025
**Purpose**: Combined roadmap merging pointer consolidation, language features, and multimedia system

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Vision](#2-architecture-vision)
3. [Phase Structure](#3-phase-structure)
4. [Detailed Implementation Schedule](#4-detailed-implementation-schedule)
5. [Dependency Graph](#5-dependency-graph)
6. [File Change Summary](#6-file-change-summary)
7. [Testing Strategy](#7-testing-strategy)
8. [Checkpoints and Milestones](#8-checkpoints-and-milestones)

---

## 1. Overview

This plan unifies three separate implementation efforts into a coherent roadmap:

| Component | Source Document | Purpose |
|-----------|-----------------|---------|
| Pointer Consolidation | `POINTER_CONSOLIDATION_IMPLEMENTATION.md` | Free type slots, establish subtype infrastructure |
| Language Features | `IMPLEMENTATION_PLAN_V1.md` (Phases 2-6) | Complete R7RS-compatible features |
| Multimedia System | `MULTIMEDIA_SYSTEM_ARCHITECTURE.md` | Resource-safe multimedia primitives |

### 1.1 Current State

- **Phase 1 Complete**: Exception handling (`guard`, `raise`, `error`)
- **Type Slots Exhausted**: 4-bit type field (0-15) fully used
- **Foundation Needed**: Subtype system required for future expansion

### 1.2 End State

After completing this plan:

```
Type System (4-bit primary):
├── 0: NIL
├── 1: BOOLEAN
├── 2: INTEGER
├── 3: (freed)
├── 4: REAL
├── 5: (freed)
├── 6: (freed)
├── 7: (freed)
├── 8: HEAP_PTR ──────────────────┐
│       ├── CONS (0)              │
│       ├── STRING (1)            │
│       ├── VECTOR (2)            │
│       ├── TENSOR (3)            │
│       ├── MULTI_VALUE (4)       │  Subtypes
│       ├── HASH (5)              │  (8-bit)
│       ├── EXCEPTION (6)         │
│       ├── RECORD (7)            │
│       ├── BYTEVECTOR (8)        │
│       └── PORT (9)              │
├── 9: CALLABLE ──────────────────┤
│       ├── CLOSURE (0)           │
│       ├── LAMBDA_SEXPR (1)      │
│       └── AD_NODE (2)           │
├── 10: CHAR                      │
├── 11: SYMBOL                    │
├── 12: (freed)
├── 13: (freed)
├── 14: (freed)
├── 15: (freed)
├── 16: HANDLE ───────────────────┐
│       ├── WINDOW (0)            │
│       ├── GL_CONTEXT (1)        │
│       ├── AUDIO_DEVICE (2)      │  Multimedia
│       └── ... (255 subtypes)    │  Subtypes
├── 17: BUFFER ───────────────────┤
├── 18: STREAM ───────────────────┤
└── 19: EVENT ────────────────────┘

Language Features:
├── Multiple Return Values (values, call-with-values, let-values)
├── Quasiquotation (`, ,, ,@)
├── Pattern Matching (match)
├── Basic Macros (define-syntax, syntax-rules)
└── Enhanced Error Messages
```

---

## 2. Architecture Vision

### 2.1 Type System Evolution

**Before Consolidation:**
```c
typedef enum {
    ESHKOL_VALUE_NIL         = 0,
    ESHKOL_VALUE_BOOLEAN     = 1,
    ESHKOL_VALUE_INTEGER     = 2,
    ESHKOL_VALUE_CONS_PTR    = 3,   // Separate type
    ESHKOL_VALUE_REAL        = 4,
    ESHKOL_VALUE_AD_NODE_PTR = 5,   // Separate type
    ESHKOL_VALUE_TENSOR_PTR  = 6,   // Separate type
    ESHKOL_VALUE_LAMBDA_SEXPR= 7,   // Separate type
    ESHKOL_VALUE_STRING_PTR  = 8,   // Separate type
    ESHKOL_VALUE_CHAR        = 10,
    ESHKOL_VALUE_SYMBOL      = 11,
    ESHKOL_VALUE_CLOSURE_PTR = 12,  // Separate type
    ESHKOL_VALUE_HASH_PTR    = 14,  // Separate type
    ESHKOL_VALUE_EXCEPTION   = 15,  // Separate type
} eshkol_value_type_t;
// NO ROOM FOR NEW TYPES!
```

**After Consolidation:**
```c
typedef enum {
    ESHKOL_VALUE_NIL         = 0,
    ESHKOL_VALUE_BOOLEAN     = 1,
    ESHKOL_VALUE_INTEGER     = 2,
    // 3: freed
    ESHKOL_VALUE_REAL        = 4,
    // 5-7: freed
    ESHKOL_VALUE_HEAP_PTR    = 8,   // Consolidated heap types
    ESHKOL_VALUE_CALLABLE    = 9,   // Consolidated callable types
    ESHKOL_VALUE_CHAR        = 10,
    ESHKOL_VALUE_SYMBOL      = 11,
    // 12-15: freed for future use
    ESHKOL_VALUE_HANDLE      = 16,  // Multimedia: resources
    ESHKOL_VALUE_BUFFER      = 17,  // Multimedia: memory
    ESHKOL_VALUE_STREAM      = 18,  // Multimedia: I/O
    ESHKOL_VALUE_EVENT       = 19,  // Multimedia: events
} eshkol_value_type_t;
```

### 2.2 Object Header System

All heap-allocated objects gain a standard header:

```c
typedef struct eshkol_object_header {
    uint8_t  subtype;      // Distinguishes types within HEAP_PTR/CALLABLE
    uint8_t  flags;        // GC marks, linear status, etc.
    uint16_t reserved;     // Alignment / future use
    uint32_t size;         // Object size in bytes
} eshkol_object_header_t;

// Memory layout:
// [header (8 bytes)][object data (variable)]
```

### 2.3 Subtype Enumerations

```c
// HEAP_PTR subtypes (consolidated from separate types)
typedef enum {
    HEAP_SUBTYPE_CONS        = 0,   // Cons cell (pair/list node)
    HEAP_SUBTYPE_STRING      = 1,   // String (UTF-8 with length)
    HEAP_SUBTYPE_VECTOR      = 2,   // Heterogeneous vector
    HEAP_SUBTYPE_TENSOR      = 3,   // N-dimensional numeric tensor
    HEAP_SUBTYPE_MULTI_VALUE = 4,   // Multiple return values container
    HEAP_SUBTYPE_HASH        = 5,   // Hash table / dictionary
    HEAP_SUBTYPE_EXCEPTION   = 6,   // Exception object
    HEAP_SUBTYPE_RECORD      = 7,   // User-defined record type
    HEAP_SUBTYPE_BYTEVECTOR  = 8,   // Raw byte vector (R7RS)
    HEAP_SUBTYPE_PORT        = 9,   // I/O port
    // Reserved: 10-255
} heap_subtype_t;

// CALLABLE subtypes
typedef enum {
    CALLABLE_SUBTYPE_CLOSURE     = 0,  // Compiled closure
    CALLABLE_SUBTYPE_LAMBDA_SEXPR= 1,  // Lambda as data (homoiconicity)
    CALLABLE_SUBTYPE_AD_NODE     = 2,  // Autodiff computation node
    CALLABLE_SUBTYPE_PRIMITIVE   = 3,  // Built-in primitive
    CALLABLE_SUBTYPE_CONTINUATION= 4,  // First-class continuation (future)
    // Reserved: 5-255
} callable_subtype_t;

// HANDLE subtypes (multimedia resources)
typedef enum {
    HANDLE_SUBTYPE_WINDOW       = 0,
    HANDLE_SUBTYPE_GL_CONTEXT   = 1,
    HANDLE_SUBTYPE_AUDIO_DEVICE = 2,
    HANDLE_SUBTYPE_MIDI_PORT    = 3,
    HANDLE_SUBTYPE_CAMERA       = 4,
    HANDLE_SUBTYPE_SOCKET       = 5,
    HANDLE_SUBTYPE_FRAMEBUFFER  = 6,
    // Reserved: 7-255
} handle_subtype_t;
```

---

## 3. Phase Structure

### 3.1 Visual Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UNIFIED IMPLEMENTATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FOUNDATION LAYER (Pointer Consolidation 1-3)                               │
│  ├── F1: Header Infrastructure                                              │
│  ├── F2: Type Enum Refactoring                                              │
│  └── F3: Arena Memory Updates                                               │
│                                                                              │
│  FEATURE LAYER (Language Features)                                          │
│  ├── L1: Multiple Return Values                                             │
│  ├── L2: Quasiquotation                                                     │
│  ├── L3: Pattern Matching                                                   │
│  └── L4: Basic Macros                                                       │
│                                                                              │
│  MIGRATION LAYER (Pointer Consolidation 4-6)                                │
│  ├── M1: Codegen Migration                                                  │
│  ├── M2: Display System Updates                                             │
│  └── M3: Testing & Validation                                               │
│                                                                              │
│  MULTIMEDIA LAYER                                                           │
│  ├── MM1: Handle System                                                     │
│  ├── MM2: Buffer System                                                     │
│  ├── MM3: Stream System                                                     │
│  ├── MM4: Event System                                                      │
│  └── MM5: Platform Abstraction Layer                                        │
│                                                                              │
│  POLISH LAYER                                                               │
│  └── P1: Error Messages & Final Polish                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase Dependencies

```
F1 (Headers) ─────┬─────────────────────────────────────────────────────┐
                  │                                                      │
F2 (Type Enum) ───┼─── F3 (Arena) ───┬─── L1 (Multi-Values) ────────────┤
                  │                   │                                  │
                  │                   ├─── L2 (Quasiquote) ──────────────┤
                  │                   │                                  │
                  │                   └─── L3 (Pattern Match) ──┬────────┤
                  │                                             │        │
                  │                                 L4 (Macros) ┘        │
                  │                                                      │
                  └─── M1 (Codegen Migration) ──────────────────────────┼─┐
                                                                        │ │
                       M2 (Display) ────────────────────────────────────┘ │
                                                                          │
                       M3 (Testing) ──────────────────────────────────────┘
                            │
                            ▼
                       MM1-MM5 (Multimedia)
                            │
                            ▼
                       P1 (Polish)
```

---

## 4. Detailed Implementation Schedule

### FOUNDATION LAYER

---

#### F1: Header Infrastructure
**Ref**: `POINTER_CONSOLIDATION_IMPLEMENTATION.md` Phase 1

**Objective**: Add object header system without breaking existing code

**Tasks**:
1. Add header structures to `eshkol.h`
2. Add subtype enumerations
3. Create header accessor macros
4. Add forward compatibility layer

**Files Modified**:
| File | Changes |
|------|---------|
| `inc/eshkol/eshkol.h` | Add `eshkol_object_header_t`, subtype enums, macros |

**Code to Add** (`eshkol.h`):
```c
// ═══════════════════════════════════════════════════════════
// OBJECT HEADER (prepended to all heap objects)
// ═══════════════════════════════════════════════════════════
typedef struct eshkol_object_header {
    uint8_t  subtype;
    uint8_t  flags;
    uint16_t reserved;
    uint32_t size;
} eshkol_object_header_t;

#define ESHKOL_OBJ_FLAG_MARKED    0x01
#define ESHKOL_OBJ_FLAG_LINEAR    0x02
#define ESHKOL_OBJ_FLAG_BORROWED  0x04
#define ESHKOL_OBJ_FLAG_CONSUMED  0x08

// Header access macros
#define ESHKOL_GET_HEADER(ptr) \
    ((eshkol_object_header_t*)((uint8_t*)(ptr) - sizeof(eshkol_object_header_t)))

#define ESHKOL_GET_SUBTYPE(ptr) \
    (ESHKOL_GET_HEADER(ptr)->subtype)

#define ESHKOL_GET_DATA_PTR(header_ptr) \
    ((void*)((uint8_t*)(header_ptr) + sizeof(eshkol_object_header_t)))

// Subtype enumerations (as shown in Section 2.3)
// ... heap_subtype_t, callable_subtype_t, handle_subtype_t ...
```

**Validation**:
- [ ] All existing tests pass (no behavior change yet)
- [ ] Header structures compile correctly
- [ ] Macros work with test pointers

---

#### F2: Type Enum Refactoring
**Ref**: `POINTER_CONSOLIDATION_IMPLEMENTATION.md` Phase 2

**Objective**: Add consolidated type values while keeping old ones for compatibility

**Tasks**:
1. Add `ESHKOL_VALUE_HEAP_PTR` (8) - already exists as STRING_PTR!
2. Add `ESHKOL_VALUE_CALLABLE` (9) - new
3. Reserve multimedia types (16-19)
4. Create compatibility macros

**Files Modified**:
| File | Changes |
|------|---------|
| `inc/eshkol/eshkol.h` | Add new type values, compatibility layer |

**Code to Add**:
```c
// New consolidated types (in addition to existing)
// Note: 8 (STRING_PTR) will be repurposed to HEAP_PTR
// Note: 9 was unused, now CALLABLE
#define ESHKOL_VALUE_HEAP_PTR   8   // Replaces STRING_PTR
#define ESHKOL_VALUE_CALLABLE   9   // New consolidated callable

// Future multimedia types (reserve now)
#define ESHKOL_VALUE_HANDLE     16
#define ESHKOL_VALUE_BUFFER     17
#define ESHKOL_VALUE_STREAM     18
#define ESHKOL_VALUE_EVENT      19

// ═══════════════════════════════════════════════════════════
// COMPATIBILITY MACROS (old type → new type + subtype)
// Use these during migration, remove after consolidation complete
// ═══════════════════════════════════════════════════════════

// Type checking that works with both old and new representations
#define ESHKOL_IS_CONS(val) \
    ((val).type == ESHKOL_VALUE_CONS_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_CONS))

#define ESHKOL_IS_STRING(val) \
    ((val).type == ESHKOL_VALUE_STRING_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_STRING))

#define ESHKOL_IS_VECTOR(val) \
    ((val).type == ESHKOL_VALUE_VECTOR_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_VECTOR))

#define ESHKOL_IS_TENSOR(val) \
    ((val).type == ESHKOL_VALUE_TENSOR_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_TENSOR))

#define ESHKOL_IS_HASH(val) \
    ((val).type == ESHKOL_VALUE_HASH_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_HASH))

#define ESHKOL_IS_EXCEPTION(val) \
    ((val).type == ESHKOL_VALUE_EXCEPTION || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_EXCEPTION))

#define ESHKOL_IS_CLOSURE(val) \
    ((val).type == ESHKOL_VALUE_CLOSURE_PTR || \
     ((val).type == ESHKOL_VALUE_CALLABLE && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == CALLABLE_SUBTYPE_CLOSURE))

#define ESHKOL_IS_LAMBDA_SEXPR(val) \
    ((val).type == ESHKOL_VALUE_LAMBDA_SEXPR || \
     ((val).type == ESHKOL_VALUE_CALLABLE && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == CALLABLE_SUBTYPE_LAMBDA_SEXPR))

#define ESHKOL_IS_AD_NODE(val) \
    ((val).type == ESHKOL_VALUE_AD_NODE_PTR || \
     ((val).type == ESHKOL_VALUE_CALLABLE && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == CALLABLE_SUBTYPE_AD_NODE))

// NEW: Multi-value check (only new format)
#define ESHKOL_IS_MULTI_VALUE(val) \
    ((val).type == ESHKOL_VALUE_HEAP_PTR && \
     ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_MULTI_VALUE)
```

**Validation**:
- [ ] All existing tests pass (compatibility macros work)
- [ ] New type values don't conflict
- [ ] Both old and new representations can be checked

---

#### F3: Arena Memory Updates
**Ref**: `POINTER_CONSOLIDATION_IMPLEMENTATION.md` Phase 3

**Objective**: Add allocation functions that prepend headers

**Tasks**:
1. Create `arena_alloc_with_header()` function
2. Create typed allocation helpers
3. Ensure alignment correctness

**Files Modified**:
| File | Changes |
|------|---------|
| `lib/core/arena_memory.cpp` | Add new allocation functions |
| `inc/eshkol/core/arena_memory.h` | Declare new functions |

**Code to Add** (`arena_memory.cpp`):
```cpp
// Allocate object with header prepended
void* arena_alloc_with_header(
    eshkol_arena_t* arena,
    size_t data_size,
    uint8_t subtype,
    uint8_t flags
) {
    size_t total_size = sizeof(eshkol_object_header_t) + data_size;

    // Ensure 8-byte alignment
    total_size = (total_size + 7) & ~7;

    void* raw = eshkol_arena_allocate(arena, total_size);
    if (!raw) return NULL;

    // Initialize header
    eshkol_object_header_t* header = (eshkol_object_header_t*)raw;
    header->subtype = subtype;
    header->flags = flags;
    header->reserved = 0;
    header->size = (uint32_t)data_size;

    // Return pointer to data (after header)
    return (void*)((uint8_t*)raw + sizeof(eshkol_object_header_t));
}

// Typed allocation helpers
void* arena_alloc_cons(eshkol_arena_t* arena) {
    return arena_alloc_with_header(arena, sizeof(eshkol_cons_t),
                                   HEAP_SUBTYPE_CONS, 0);
}

void* arena_alloc_string(eshkol_arena_t* arena, size_t length) {
    return arena_alloc_with_header(arena, sizeof(eshkol_string_t) + length + 1,
                                   HEAP_SUBTYPE_STRING, 0);
}

void* arena_alloc_vector(eshkol_arena_t* arena, size_t capacity) {
    return arena_alloc_with_header(arena,
                                   sizeof(eshkol_vector_t) + capacity * sizeof(eshkol_tagged_value_t),
                                   HEAP_SUBTYPE_VECTOR, 0);
}

void* arena_alloc_multi_value(eshkol_arena_t* arena, size_t count) {
    return arena_alloc_with_header(arena,
                                   sizeof(size_t) + count * sizeof(eshkol_tagged_value_t),
                                   HEAP_SUBTYPE_MULTI_VALUE, 0);
}

void* arena_alloc_closure(eshkol_arena_t* arena) {
    return arena_alloc_with_header(arena, sizeof(eshkol_closure_t),
                                   CALLABLE_SUBTYPE_CLOSURE, 0);
}
```

**Validation**:
- [ ] Header alignment is correct (8-byte)
- [ ] Subtype can be read back from allocated objects
- [ ] All existing tests pass

---

### FEATURE LAYER

---

#### L1: Multiple Return Values
**Ref**: `IMPLEMENTATION_PLAN_V1.md` Phase 2

**Objective**: Implement `values`, `call-with-values`, `let-values`

**Tasks**:
1. Add multi-value structure
2. Implement `values` builtin
3. Implement `call-with-values`
4. Add `let-values` and `let*-values` special forms
5. Fix `apply` to handle multiple leading arguments

**Files Modified**:
| File | Changes |
|------|---------|
| `inc/eshkol/eshkol.h` | Add `ESHKOL_LET_VALUES_OP`, multi-value struct |
| `lib/frontend/parser.cpp` | Parse `let-values`, `call-with-values` |
| `lib/backend/llvm_codegen.cpp` | Codegen for multi-value operations |

**Multi-Value Structure**:
```c
// Uses HEAP_SUBTYPE_MULTI_VALUE = 4 (already reserved)
typedef struct eshkol_multi_value {
    size_t count;
    eshkol_tagged_value_t values[];  // Flexible array member
} eshkol_multi_value_t;
```

**Implementation Pattern**:
```scheme
;; (values v1 v2 ...) -> packs into multi-value object
;; Allocated with HEAP_PTR type, HEAP_SUBTYPE_MULTI_VALUE subtype

;; (call-with-values producer consumer)
;; 1. Call producer
;; 2. If result is multi-value, unpack and apply to consumer
;; 3. If single value, apply consumer with one arg

;; (let-values (((a b) (values 1 2))) body)
;; Expands to call-with-values with binding lambda
```

**Test Cases** (`tests/features/multivalue_test.esk`):
```scheme
;;; Test values and call-with-values
(define result1
  (call-with-values
    (lambda () (values 1 2 3))
    (lambda (a b c) (+ a b c))))
(display "Test 1: ") (display result1) (newline)  ; Should be 6

;;; Test single value passthrough
(define result2
  (call-with-values
    (lambda () 42)
    (lambda (x) (* x 2))))
(display "Test 2: ") (display result2) (newline)  ; Should be 84

;;; Test let-values
(define result3
  (let-values (((q r) (values 10 3)))
    (+ q r)))
(display "Test 3: ") (display result3) (newline)  ; Should be 13
```

**Validation**:
- [ ] `values` creates multi-value object
- [ ] `call-with-values` unpacks correctly
- [ ] `let-values` binds correctly
- [ ] Single values pass through unchanged
- [ ] All existing tests pass

---

#### L2: Quasiquotation
**Ref**: `IMPLEMENTATION_PLAN_V1.md` Phase 3

**Objective**: Implement `` ` ``, `,`, `,@`

**Tasks**:
1. Add lexer tokens: `BACKQUOTE`, `COMMA`, `COMMA_AT`
2. Add parser rules for quasiquote nesting
3. Implement expansion logic in codegen
4. Handle nested quasiquotes correctly

**Files Modified**:
| File | Changes |
|------|---------|
| `lib/frontend/parser.cpp` | New tokens, quasiquote parsing |
| `lib/backend/llvm_codegen.cpp` | Quasiquote expansion |
| `inc/eshkol/eshkol.h` | New op types |

**New Operations**:
```c
ESHKOL_QUASIQUOTE_OP,        // `expr
ESHKOL_UNQUOTE_OP,           // ,expr
ESHKOL_UNQUOTE_SPLICING_OP,  // ,@expr
```

**Expansion Rules**:
```
`x                    → 'x (quote x)
`(a ,b c)             → (list 'a b 'c)
`(a ,@b c)            → (append (list 'a) b (list 'c))
``(a ,(+ 1 2))        → `(a ,(+ 1 2))  (nested, stay quoted)
``(a ,,(+ 1 2))       → `(a ,3)        (double unquote evaluates)
```

**Test Cases** (`tests/features/quasiquote_test.esk`):
```scheme
;;; Basic quasiquote
(define t1 `(a b c))
(display "Test 1: ") (display t1) (newline)  ; (a b c)

;;; Unquote
(define x 5)
(define t2 `(1 ,x 3))
(display "Test 2: ") (display t2) (newline)  ; (1 5 3)

;;; Unquote-splicing
(define lst '(2 3 4))
(define t3 `(1 ,@lst 5))
(display "Test 3: ") (display t3) (newline)  ; (1 2 3 4 5)

;;; Nested quasiquote
(define t4 ``(a ,(+ 1 2)))
(display "Test 4: ") (display t4) (newline)  ; `(a ,(+ 1 2))
```

**Validation**:
- [ ] Simple quasiquote equals quote
- [ ] Unquote evaluates expression
- [ ] Unquote-splicing flattens list
- [ ] Nested quasiquotes preserve structure
- [ ] All existing tests pass

---

#### L3: Pattern Matching
**Ref**: `IMPLEMENTATION_PLAN_V1.md` Phase 4

**Objective**: Implement `match` expression

**Tasks**:
1. Define pattern AST representation
2. Add `match` to parser
3. Implement pattern compilation to decision tree
4. Add pattern types: literal, variable, wildcard, cons, list, predicate, guard

**Files Modified**:
| File | Changes |
|------|---------|
| `inc/eshkol/eshkol.h` | Pattern types, match op |
| `lib/frontend/parser.cpp` | Parse match clauses |
| `lib/backend/llvm_codegen.cpp` | Pattern compilation |

**Pattern Types**:
```c
typedef enum {
    PATTERN_LITERAL,      // 42, "hello", #t
    PATTERN_VARIABLE,     // x, y (binds to value)
    PATTERN_WILDCARD,     // _ (matches anything, no binding)
    PATTERN_CONS,         // (cons h t)
    PATTERN_LIST,         // (list a b c)
    PATTERN_VECTOR,       // #(a b c)
    PATTERN_PREDICATE,    // (? number?) or (? number? n)
    PATTERN_QUOTE,        // 'symbol
    PATTERN_AND,          // (and p1 p2)
    PATTERN_OR,           // (or p1 p2)
} pattern_type_t;
```

**Syntax**:
```scheme
(match expr
  (pattern body ...)
  (pattern (when guard) body ...)
  (_ default-body ...))
```

**Test Cases** (`tests/features/match_test.esk`):
```scheme
;;; Literal matching
(define t1 (match 2
             (1 'one)
             (2 'two)
             (_ 'other)))
(display "Test 1: ") (display t1) (newline)  ; two

;;; Variable binding
(define t2 (match 5
             (x (* x 2))))
(display "Test 2: ") (display t2) (newline)  ; 10

;;; List destructuring
(define t3 (match '(1 2 3)
             ((list a b c) (+ a b c))))
(display "Test 3: ") (display t3) (newline)  ; 6

;;; Cons pattern
(define t4 (match '(1 2 3)
             ((cons h t) t)))
(display "Test 4: ") (display t4) (newline)  ; (2 3)

;;; Predicate pattern
(define t5 (match 4
             ((? odd?) 'odd)
             ((? even?) 'even)))
(display "Test 5: ") (display t5) (newline)  ; even

;;; Guard
(define t6 (match 100
             ((? number? n) (when (> n 50)) 'big)
             ((? number?) 'small)))
(display "Test 6: ") (display t6) (newline)  ; big
```

**Validation**:
- [ ] Literal patterns work
- [ ] Variable binding works
- [ ] List/cons destructuring works
- [ ] Predicate patterns work
- [ ] Guards filter correctly
- [ ] No-match raises error
- [ ] All existing tests pass

---

#### L4: Basic Macros
**Ref**: `IMPLEMENTATION_PLAN_V1.md` Phase 5

**Objective**: Implement `define-syntax` with `syntax-rules`

**Tasks**:
1. Design macro representation
2. Implement `define-syntax` parsing
3. Implement pattern matching for macro patterns
4. Implement template instantiation
5. Handle `...` ellipsis patterns
6. Add macro expansion pass before codegen

**Files Modified**:
| File | Changes |
|------|---------|
| `inc/eshkol/eshkol.h` | Macro types |
| `lib/frontend/parser.cpp` | Parse define-syntax |
| `lib/frontend/macro_expander.cpp` | NEW: Expansion pass |
| `lib/backend/llvm_codegen.cpp` | Call expander before codegen |

**Syntax**:
```scheme
(define-syntax name
  (syntax-rules (literal ...)
    ((pattern1) template1)
    ((pattern2) template2)
    ...))
```

**Implementation**:
```cpp
// New macro expansion pass
class MacroExpander {
public:
    eshkol_ast_t expand(const eshkol_ast_t& ast);

private:
    std::map<std::string, macro_def_t> macros_;

    eshkol_ast_t expandMacroCall(const macro_def_t& macro,
                                  const eshkol_ast_t& call);
    bool matchPattern(const pattern_t* pat,
                      const eshkol_ast_t& input,
                      std::map<std::string, eshkol_ast_t*>& bindings);
    eshkol_ast_t instantiateTemplate(const eshkol_ast_t* tmpl,
                                     const std::map<std::string, eshkol_ast_t*>& bindings);
};
```

**Test Cases** (`tests/features/macro_test.esk`):
```scheme
;;; Simple macro
(define-syntax my-when
  (syntax-rules ()
    ((my-when test body ...)
     (if test (begin body ...) #f))))

(define t1 (my-when #t 1 2 3))
(display "Test 1: ") (display t1) (newline)  ; 3

(define t2 (my-when #f 1 2 3))
(display "Test 2: ") (display t2) (newline)  ; #f

;;; Macro with literals
(define-syntax my-case
  (syntax-rules (else)
    ((my-case key (else result))
     result)
    ((my-case key ((atoms ...) result) clause ...)
     (if (memq key '(atoms ...))
         result
         (my-case key clause ...)))))

;;; Let as macro
(define-syntax my-let
  (syntax-rules ()
    ((my-let ((var val) ...) body ...)
     ((lambda (var ...) body ...) val ...))))

(define t3 (my-let ((x 1) (y 2)) (+ x y)))
(display "Test 3: ") (display t3) (newline)  ; 3
```

**Validation**:
- [ ] Simple pattern macros work
- [ ] Ellipsis (`...`) expands correctly
- [ ] Literal identifiers match exactly
- [ ] Nested macros expand correctly
- [ ] All existing tests pass

---

### MIGRATION LAYER

---

#### M1: Codegen Migration
**Ref**: `POINTER_CONSOLIDATION_IMPLEMENTATION.md` Phase 4

**Objective**: Migrate all type checks and allocations to new system

**Tasks**:
1. Replace old type checks with compatibility macros
2. Update allocation sites to use header functions
3. Update type tag creation
4. Run tests after each file

**Files to Migrate** (in order):
1. `lib/backend/llvm_codegen.cpp` (~449 occurrences)
2. `lib/backend/collection_codegen.cpp`
3. `lib/backend/tensor_codegen.cpp`
4. `lib/backend/string_io_codegen.cpp`
5. `lib/backend/hash_codegen.cpp`
6. `lib/backend/function_codegen.cpp`
7. `lib/backend/autodiff_codegen.cpp`
8. Additional codegen files as needed

**Migration Pattern**:
```cpp
// BEFORE:
if (val.type == ESHKOL_VALUE_CONS_PTR) { ... }

// AFTER:
if (ESHKOL_IS_CONS(val)) { ... }

// BEFORE (allocation):
auto* cons = (eshkol_cons_t*)arena_allocate(arena, sizeof(eshkol_cons_t));
result.type = ESHKOL_VALUE_CONS_PTR;

// AFTER:
auto* cons = (eshkol_cons_t*)arena_alloc_cons(arena);
result.type = ESHKOL_VALUE_HEAP_PTR;  // Subtype in header
```

**Validation**:
- [ ] Run full test suite after each file
- [ ] No regressions allowed
- [ ] Track migration progress per file

---

#### M2: Display System Updates
**Ref**: `POINTER_CONSOLIDATION_IMPLEMENTATION.md` Phase 5

**Objective**: Update value display to use subtypes

**Tasks**:
1. Update display codegen for HEAP_PTR types
2. Update display codegen for CALLABLE types
3. Add subtype names for debugging

**Files Modified**:
| File | Changes |
|------|---------|
| `lib/backend/llvm_codegen.cpp` | Display switch cases |

**Display Logic**:
```cpp
case ESHKOL_VALUE_HEAP_PTR: {
    uint8_t subtype = ESHKOL_GET_SUBTYPE((void*)val.data.ptr_val);
    switch (subtype) {
        case HEAP_SUBTYPE_CONS:
            displayCons(val);
            break;
        case HEAP_SUBTYPE_STRING:
            displayString(val);
            break;
        case HEAP_SUBTYPE_VECTOR:
            displayVector(val);
            break;
        // ... etc
    }
    break;
}
```

**Validation**:
- [ ] All types display correctly
- [ ] Debug output shows subtype names
- [ ] All existing tests pass

---

#### M3: Testing & Validation
**Ref**: `POINTER_CONSOLIDATION_IMPLEMENTATION.md` Phase 6

**Objective**: Comprehensive testing of consolidated type system

**Tasks**:
1. Run all test suites
2. Create migration validation tests
3. Run memory tests
4. Performance comparison

**Test Suites**:
```bash
./scripts/run_tests_with_output.sh
./scripts/run_stdlib_tests.sh
./scripts/run_features_tests.sh
./scripts/run_autodiff_tests.sh
./scripts/run_memory_tests.sh
./scripts/run_list_tests.sh
```

**Migration Validation Test** (`tests/types/consolidation_test.esk`):
```scheme
;;; Test that all consolidated types work correctly

;; Heap types
(define c (cons 1 2))
(display "cons: ") (display c) (newline)

(define s "hello")
(display "string: ") (display s) (newline)

(define v (vector 1 2 3))
(display "vector: ") (display v) (newline)

(define h (make-hash-table))
(hash-set! h 'key 'value)
(display "hash: ") (display (hash-ref h 'key)) (newline)

;; Callable types
(define f (lambda (x) (* x 2)))
(display "closure: ") (display (f 5)) (newline)

;; Multi-value (new)
(define mv (call-with-values
             (lambda () (values 1 2 3))
             list))
(display "multi-value: ") (display mv) (newline)

(display "All consolidation tests passed!")
(newline)
```

**Validation**:
- [ ] All 253+ existing tests pass
- [ ] Memory tests show no leaks
- [ ] Performance within 5% of baseline

---

### MULTIMEDIA LAYER

---

#### MM1: Handle System
**Ref**: `MULTIMEDIA_SYSTEM_ARCHITECTURE.md` Section 3

**Objective**: Implement linear resource handles

**Tasks**:
1. Add HANDLE type (16) to type enum
2. Add handle structure with subtype
3. Implement linear ownership tracking
4. Add handle creation/destruction
5. Add platform-specific handle operations

**Key Structures**:
```c
typedef struct eshkol_handle {
    handle_subtype_t subtype;
    uint8_t flags;               // LINEAR, CONSUMED, etc.
    void* platform_handle;       // OS-specific handle
    void (*destructor)(void*);   // Cleanup function
} eshkol_handle_t;
```

**Linear Type Enforcement**:
- Handles must be consumed exactly once
- Compiler error if handle escapes scope unconsumed
- `handle-consume!` marks handle as used

---

#### MM2: Buffer System
**Ref**: `MULTIMEDIA_SYSTEM_ARCHITECTURE.md` Section 4

**Objective**: Implement zero-copy buffer management

**Tasks**:
1. Add BUFFER type (17) to type enum
2. Implement buffer allocation with alignment
3. Add buffer views (slices without copy)
4. Add memory mapping support
5. Implement buffer operations

**Key Structures**:
```c
typedef struct eshkol_buffer {
    buffer_subtype_t subtype;
    uint8_t flags;
    size_t size;
    size_t alignment;
    void* data;
    eshkol_buffer_t* parent;  // For views
} eshkol_buffer_t;
```

---

#### MM3: Stream System
**Ref**: `MULTIMEDIA_SYSTEM_ARCHITECTURE.md` Section 5

**Objective**: Implement async I/O streams

**Tasks**:
1. Add STREAM type (18) to type enum
2. Implement stream state machine
3. Add stream combinators (map, filter, merge)
4. Add backpressure handling
5. Implement async read/write

---

#### MM4: Event System
**Ref**: `MULTIMEDIA_SYSTEM_ARCHITECTURE.md` Section 6

**Objective**: Implement event loop and callbacks

**Tasks**:
1. Add EVENT type (19) to type enum
2. Implement event queue
3. Add event handlers
4. Add timer events
5. Implement poll/select integration

---

#### MM5: Platform Abstraction Layer
**Ref**: `MULTIMEDIA_SYSTEM_ARCHITECTURE.md` Section 8

**Objective**: Cross-platform multimedia support

**Tasks**:
1. Create platform detection
2. Implement macOS backend (Metal, CoreAudio)
3. Implement Linux backend (Vulkan, ALSA/PulseAudio)
4. Implement Windows backend (D3D12, WASAPI)
5. Create unified API layer

---

### POLISH LAYER

---

#### P1: Error Messages & Final Polish
**Ref**: `IMPLEMENTATION_PLAN_V1.md` Phase 6

**Objective**: Production-quality error messages

**Tasks**:
1. Add source context to all errors
2. Implement error hints
3. Add color output for terminal
4. Update documentation
5. Final integration testing

**Example Error Format**:
```
Error at line 42, column 15 in file.esk:
  Type mismatch in function application
  Expected: (-> integer integer integer)
  Got:      (-> string integer integer)

  41 | (define (add x y)
  42 |   (+ x "hello"))
             ^^^^^^^
  Hint: The second argument should be an integer, not a string.
```

---

## 5. Dependency Graph

```
                              ┌─────────────────┐
                              │    START        │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  F1: Headers    │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  F2: Type Enum  │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  F3: Arena Mem  │
                              └────────┬────────┘
                                       │
              ┌────────────────┬───────┴───────┬────────────────┐
              │                │               │                │
     ┌────────▼────────┐ ┌─────▼─────┐  ┌──────▼──────┐ ┌───────▼───────┐
     │ L1: Multi-Value │ │L2: Quasi- │  │L3: Pattern  │ │M1: Codegen    │
     │                 │ │   quote   │  │   Match     │ │   Migration   │
     └────────┬────────┘ └─────┬─────┘  └──────┬──────┘ └───────┬───────┘
              │                │               │                │
              │                │       ┌───────▼───────┐        │
              │                │       │  L4: Macros   │        │
              │                │       └───────┬───────┘        │
              │                │               │                │
              └────────────────┴───────┬───────┴────────────────┘
                                       │
                              ┌────────▼────────┐
                              │  M2: Display    │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  M3: Testing    │
                              └────────┬────────┘
                                       │
              ┌────────────────┬───────┴───────┬────────────────┐
              │                │               │                │
     ┌────────▼────────┐ ┌─────▼─────┐  ┌──────▼──────┐ ┌───────▼───────┐
     │  MM1: Handle    │ │MM2: Buffer│  │ MM3: Stream │ │  MM4: Event   │
     └────────┬────────┘ └─────┬─────┘  └──────┬──────┘ └───────┬───────┘
              │                │               │                │
              └────────────────┴───────┬───────┴────────────────┘
                                       │
                              ┌────────▼────────┐
                              │  MM5: Platform  │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  P1: Polish     │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │    COMPLETE     │
                              └─────────────────┘
```

---

## 6. File Change Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `lib/frontend/macro_expander.cpp` | Macro expansion pass |
| `inc/eshkol/frontend/macro_expander.h` | Macro expander interface |
| `lib/multimedia/handle.cpp` | Handle system implementation |
| `lib/multimedia/buffer.cpp` | Buffer system implementation |
| `lib/multimedia/stream.cpp` | Stream system implementation |
| `lib/multimedia/event.cpp` | Event system implementation |
| `lib/multimedia/platform_*.cpp` | Platform-specific backends |
| `tests/features/multivalue_test.esk` | Multi-value tests |
| `tests/features/quasiquote_test.esk` | Quasiquote tests |
| `tests/features/match_test.esk` | Pattern matching tests |
| `tests/features/macro_test.esk` | Macro tests |
| `tests/multimedia/*.esk` | Multimedia tests |

### Files to Modify

| File | Phases |
|------|--------|
| `inc/eshkol/eshkol.h` | F1, F2, L1, L2, L3, L4, MM1-4 |
| `lib/frontend/parser.cpp` | L1, L2, L3, L4 |
| `lib/backend/llvm_codegen.cpp` | F3, L1, L2, L3, L4, M1, M2 |
| `lib/core/arena_memory.cpp` | F3 |
| All *_codegen.cpp files | M1 |

---

## 7. Testing Strategy

### 7.1 Test Progression

Each phase must:
1. Pass all existing tests
2. Pass new feature tests
3. Pass memory tests (no leaks)
4. Not regress performance by more than 5%

### 7.2 Test Commands

```bash
# After each phase
./scripts/run_tests_with_output.sh
./scripts/run_features_tests.sh

# After migration phases
./scripts/run_memory_tests.sh
./scripts/run_stdlib_tests.sh
./scripts/run_autodiff_tests.sh
./scripts/run_list_tests.sh
```

### 7.3 Regression Test Protocol

Before marking any phase complete:
1. Run full test suite
2. Verify no new failures
3. Check memory with memory tests
4. Document any intentional behavior changes

---

## 8. Checkpoints and Milestones

### Checkpoint 1: Foundation Complete
- [ ] F1: Header Infrastructure
- [ ] F2: Type Enum Refactoring
- [ ] F3: Arena Memory Updates
- [ ] All existing tests pass

### Checkpoint 2: Language Features Complete
- [ ] L1: Multiple Return Values
- [ ] L2: Quasiquotation
- [ ] L3: Pattern Matching
- [ ] L4: Basic Macros
- [ ] Feature tests pass

### Checkpoint 3: Migration Complete
- [ ] M1: Codegen Migration
- [ ] M2: Display System Updates
- [ ] M3: Testing & Validation
- [ ] All tests pass
- [ ] Old type values can be deprecated

### Checkpoint 4: Multimedia Complete
- [ ] MM1: Handle System
- [ ] MM2: Buffer System
- [ ] MM3: Stream System
- [ ] MM4: Event System
- [ ] MM5: Platform Abstraction Layer
- [ ] Multimedia tests pass

### Checkpoint 5: Release Ready
- [ ] P1: Error Messages & Polish
- [ ] Documentation updated
- [ ] Performance benchmarks pass
- [ ] All 253+ tests pass

---

## Quick Reference: Type Mapping

| Old Type | Slot | New Type | Slot | Subtype |
|----------|------|----------|------|---------|
| CONS_PTR | 3 | HEAP_PTR | 8 | HEAP_SUBTYPE_CONS (0) |
| AD_NODE_PTR | 5 | CALLABLE | 9 | CALLABLE_SUBTYPE_AD_NODE (2) |
| TENSOR_PTR | 6 | HEAP_PTR | 8 | HEAP_SUBTYPE_TENSOR (3) |
| LAMBDA_SEXPR | 7 | CALLABLE | 9 | CALLABLE_SUBTYPE_LAMBDA_SEXPR (1) |
| STRING_PTR | 8 | HEAP_PTR | 8 | HEAP_SUBTYPE_STRING (1) |
| VECTOR_PTR | 10 | HEAP_PTR | 8 | HEAP_SUBTYPE_VECTOR (2) |
| CLOSURE_PTR | 12 | CALLABLE | 9 | CALLABLE_SUBTYPE_CLOSURE (0) |
| HASH_PTR | 14 | HEAP_PTR | 8 | HEAP_SUBTYPE_HASH (5) |
| EXCEPTION | 15 | HEAP_PTR | 8 | HEAP_SUBTYPE_EXCEPTION (6) |
| (new) | - | HEAP_PTR | 8 | HEAP_SUBTYPE_MULTI_VALUE (4) |
| (new) | 16 | HANDLE | 16 | (various) |
| (new) | 17 | BUFFER | 17 | (various) |
| (new) | 18 | STREAM | 18 | (various) |
| (new) | 19 | EVENT | 19 | (various) |

---

**End of Unified Implementation Plan**
