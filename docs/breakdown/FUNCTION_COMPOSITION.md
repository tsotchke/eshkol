# Function Composition in Eshkol

## Table of Contents
- [Overview](#overview)
- [Closure Implementation](#closure-implementation)
- [Lambda Registry (Homoiconicity)](#lambda-registry-homoiconicity)
- [Composition Patterns](#composition-patterns)
- [Mutable Captures](#mutable-captures)
- [Variadic Functions](#variadic-functions)
- [Best Practices](#best-practices)
- [Performance Characteristics](#performance-characteristics)

---

## Overview

Eshkol provides first-class functions with full closure support, enabling functional programming patterns while maintaining C-level performance through LLVM compilation. Functions are values that can be passed, returned, and composed freely.

**Key properties:**
- Lexical scoping with closure capture
- Mutable captures (pointers to variables, not values)
- Homoiconicity (lambdas preserve S-expression representation)
- Variadic parameter support
- AD-aware closures (context-sensitive execution)

---

## Closure Implementation

**Implementation:** [`inc/eshkol/eshkol.h:649-658`](inc/eshkol/eshkol.h:649)

Closures are **32-byte structures** allocated in the global arena:

```c
typedef struct eshkol_closure {
    uint64_t func_ptr;           // Pointer to compiled LLVM function
    eshkol_closure_env_t* env;   // Captured variable environment (may be NULL)
    uint64_t sexpr_ptr;          // S-expression representation for homoiconicity
    uint8_t return_type;         // Return type category (CLOSURE_RETURN_*)
    uint8_t input_arity;         // Number of expected parameters (0-255)
    uint8_t flags;               // CLOSURE_FLAG_VARIADIC, etc.
    uint8_t reserved;            // Padding for alignment
    uint32_t hott_type_id;       // Packed HoTT TypeId
} eshkol_closure_t;
```

### Closure Environment

The environment stores captured variables:

```c
typedef struct eshkol_closure_env {
    size_t num_captures;                  // Packed: captures | (params << 16) | (variadic << 63)
    eshkol_tagged_value_t captures[];     // Flexible array of captured values
} eshkol_closure_env_t;
```

**Critical detail:** Captures store **pointers** (not values) to the captured variables. This enables mutable captures:

```scheme
(let ((counter 0))
  (define inc (lambda () (set! counter (+ counter 1)) counter))
  (inc)  ; Returns 1
  (inc)) ; Returns 2 - mutates the SAME counter binding
```

The environment packing encodes three values in a single `uint64_t`:
- Bits 0-15: Number of captures (up to 65,535)
- Bits 16-31: Number of fixed parameters (up to 65,535)
- Bit 63: Variadic flag (1 = accepts rest arguments)

---

## Lambda Registry (Homoiconicity)

**Implementation:** [`inc/eshkol/eshkol.h:771-791`](inc/eshkol/eshkol.h:771)

Eshkol maintains a **lambda registry** mapping function pointers to their S-expression representations:

```c
typedef struct eshkol_lambda_entry {
    uint64_t func_ptr;      // Function pointer as uint64
    uint64_t sexpr_ptr;     // Pointer to S-expression cons cell
    const char* name;       // Function name for debugging (may be NULL)
} eshkol_lambda_entry_t;

typedef struct eshkol_lambda_registry {
    eshkol_lambda_entry_t* entries;
    size_t count;
    size_t capacity;
} eshkol_lambda_registry_t;
```

**Purpose:** Enables displaying lambda source code at runtime:

```scheme
(define double (lambda (x) (* x 2)))
(display double)
;; Output: #<lambda (x) (* x 2)>

(display (list double))
;; Output: (#<lambda (x) (* x 2)>)
```

This is true homoiconicity: code is data, and data can be code.

---

## Composition Patterns

### Binary Composition

```scheme
(define (compose f g)
  (lambda (x)
    (f (g x))))

(define square (lambda (x) (* x x)))
(define add1 (lambda (x) (+ x 1)))

(define square-after-add1 (compose square add1))
(square-after-add1 4)  ; Returns 25: (4+1)² = 25
```

### N-ary Composition

```scheme
;; Compose multiple functions left-to-right
(define (pipe x . fns)
  (fold (lambda (f acc) (f acc)) x fns))

(pipe 3
      add1      ; 4
      square    ; 16
      add1)     ; 17
```

### Higher-Order Functions

```scheme
;; map is a higher-order function
(map (lambda (x) (* x x)) (list 1 2 3 4))
;; Returns: (1 4 9 16)

;; Compose map with other operations
(define (square-all lst)
  (map (lambda (x) (* x x)) lst))
```

### Currying

```scheme
;; Manual currying
(define (make-adder n)
  (lambda (x) (+ x n)))

(define add5 (make-adder 5))
(add5 10)  ; Returns 15

;; Curry utility from stdlib
(define curried-add (curry +))
((curried-add 5) 10)  ; Returns 15
```

---

## Mutable Captures

Eshkol closures capture variables **by reference** (pointers), enabling mutation:

### Example: Stateful Counter

```scheme
(define make-counter
  (lambda (initial)
    (let ((count initial))
      (lambda ()
        (set! count (+ count 1))
        count))))

(define counter (make-counter 0))
(counter)  ; Returns 1
(counter)  ; Returns 2
(counter)  ; Returns 3
```

### Example: Accumulator

```scheme
(define make-accumulator
  (lambda ()
    (let ((sum 0))
      (lambda (x)
        (set! sum (+ sum x))
        sum))))

(define acc (make-accumulator))
(acc 10)  ; Returns 10
(acc 20)  ; Returns 30
(acc 5)   ; Returns 35
```

### Memory Layout

When `(make-counter 0)` is called:

```
1. Allocate closure environment:
   ┌─────────────────────────────────┐
   │ num_captures: 1                 │
   │ captures[0]: &count (pointer)   │  ← Points to 'count' binding
   └─────────────────────────────────┘

2. Allocate closure:
   ┌─────────────────────────────────┐
   │ func_ptr: <compiled function>   │
   │ env: <pointer to environment>   │
   │ sexpr_ptr: <lambda source>      │
   │ return_type, arity, flags, ...  │
   └─────────────────────────────────┘

3. When closure is called:
   - Dereference captures[0] to get current 'count' value
   - Execute: set! count (+ count 1)
   - Return new count value
```

---

## Variadic Functions

Closures support **rest parameters** for variadic argument lists:

```scheme
;; Basic variadic function
(define (sum . numbers)
  (fold + 0 numbers))

(sum 1 2 3 4 5)  ; Returns 15

;; Mixed fixed and variadic parameters
(define (format-message prefix . args)
  (string-append prefix ": " (string-join args " ")))

(format-message "Error" "File" "not" "found")
;; Returns: "Error: File not found"
```

### Variadic Encoding

The environment's `num_captures` field packs variadic information:

```c
#define CLOSURE_ENV_IS_VARIADIC(packed) (((packed) >> 63) & 1)
#define CLOSURE_ENV_GET_FIXED_PARAMS(packed) (((packed) >> 16) & 0xFFFF)
```

When a variadic closure is created, the variadic flag is set and the rest parameters are collected into a list.

---

## Composition Patterns

### Pipeline Composition

```scheme
;; Left-to-right data flow
(define (|> x . fns)
  (fold (lambda (f acc) (f acc)) x fns))

(|> 5
    (lambda (x) (* x 2))  ; 10
    (lambda (x) (+ x 3))  ; 13
    (lambda (x) (* x x))) ; 169
```

### Function Chaining

```scheme
;; Chain operations on data structures
(define (process-data data)
  (|> data
      (filter positive?)
      (map square)
      (fold + 0)))

(process-data (list -1 2 -3 4 5))
;; Returns: 45 (2² + 4² + 5² = 4 + 16 + 25)
```

### Partial Application

```scheme
;; Create specialized functions via partial application
(define (greet greeting name)
  (string-append greeting ", " name "!"))

(define say-hello (lambda (name) (greet "Hello" name)))
(define say-goodbye (lambda (name) (greet "Goodbye" name)))

(say-hello "Alice")    ; "Hello, Alice!"
(say-goodbye "Bob")    ; "Goodbye, Bob!"
```

---

## Best Practices

### 1. Prefer Pure Functions for Composition

```scheme
;; Good: Pure function (no side effects)
(define (double x) (* x 2))
(define (add3 x) (+ x 3))
(define double-then-add3 (compose add3 double))

;; Avoid: Side effects make composition harder to reason about
(define (log-and-double x)
  (display "Doubling ") (display x) (newline)
  (* x 2))
```

### 2. Use Let for Complex Compositions

```scheme
;; Clear and readable
(define (process value)
  (let* ((doubled (* value 2))
         (squared (* doubled doubled))
         (adjusted (+ squared 1)))
    adjusted))

;; Less clear: Deeply nested composition
(define (process value)
  ((compose (lambda (x) (+ x 1))
            (lambda (x) (* x x))
            (lambda (x) (* x 2)))
   value))
```

### 3. Name Intermediate Compositions

```scheme
;; Good: Named intermediate functions
(define normalize (compose (lambda (x) (/ x 100)) abs))
(define process (compose round normalize))

;; Harder to debug: Anonymous nested composition
(define process (compose round (compose (lambda (x) (/ x 100)) abs)))
```

### 4. Leverage Currying from stdlib

```scheme
;; Available in lib/core/functional/curry.esk
(define curried-map (curry map))
(define square-all (curried-map (lambda (x) (* x x))))

(square-all (list 1 2 3))  ; Returns (1 4 9)
```

---

## Performance Characteristics

### Closure Allocation Cost

| Operation | Cost | Notes |
|-----------|------|-------|
| Create closure (no captures) | 32 bytes | Just the closure struct |
| Create closure (n captures) | 32 + 8 + 16n bytes | Closure + env header + n*16-byte tagged values |
| Call closure | 1 indirect jump | Function pointer dereference |
| Access captured variable | 1-2 pointer dereferences | env → captures[i] → value |

### Composition Overhead

```scheme
;; Direct call: 0 overhead
(define (f x) (* x 2))
(f 5)  ; Direct function call

;; Composed call: 1 extra indirect jump per composition
(define g (compose f f))
(g 5)  ; f(f(5)) - 2 indirect jumps

;; N-ary composition: N indirect jumps
(define h (compose f f f f))
(h 5)  ; 4 indirect jumps
```

**Mitigation:** LLVM's function inlining can eliminate composition overhead when functions are small and statically known.

### Memory Lifetime

Closures are arena-allocated and persist until arena reset. For long-running programs creating many closures:

```scheme
;; Closures accumulate in arena
(define (create-many-closures n)
  (map (lambda (i)
         (lambda (x) (+ x i)))  ; Each creates new closure
       (range 0 n)))

;; For very large n, consider using shared closures or function caching
```

---

## See Also

- [Type System](TYPE_SYSTEM.md) - Closure type annotations, HoTT function types
- [Memory Management](MEMORY_MANAGEMENT.md) - Arena allocation, closure lifetime
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - Closure codegen, LLVM IR
- [Scheme Compatibility](SCHEME_COMPATIBILITY.md) - R5RS/R7RS closure semantics
