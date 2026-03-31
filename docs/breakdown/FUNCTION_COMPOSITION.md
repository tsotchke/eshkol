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

Closures are **40-byte structures** allocated in the global arena:

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
| Create closure (no captures) | 40 bytes | Just the closure struct |
| Create closure (n captures) | 40 + 8 + 16n bytes | Closure + env header + n*16-byte tagged values |
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

## Closures in v1.1 Contexts

### 1. AD-Aware Closure Calls

The `codegenClosureCall` function (starting at line 4723 of `lib/backend/llvm_codegen.cpp`) is the central dispatch mechanism for calling closures at runtime. It generates a substantial amount of LLVM IR -- typically **15 or more basic blocks** -- because it must handle multiple orthogonal concerns in a single dispatch:

**Basic block breakdown of `codegenClosureCall`:**

1. **Type dispatch** (3 blocks): `call_closure` / `call_direct` / `call_merge` -- branch on whether the tagged value is a CALLABLE type or a direct function pointer.
2. **Continuation short-circuit** (3 blocks): Check the closure header subtype for `CALLABLE_SUBTYPE_CONTINUATION`. If so, load the jmp_buf, store the return value, unwind the dynamic-wind stack, and `longjmp` back. This path ends with `CreateUnreachable`.
3. **Builtin arithmetic detection** (9 blocks): For each of +, -, *, /, check if the closure wraps a builtin arithmetic function by comparing function pointers. If so, dispatch to `polymorphicAdd`/`Sub`/`Mul`/`Div` which handle type-heterogeneous operands (int+double, bignum+rational, etc.). Each arithmetic path branches to `merge_bb`.
4. **Environment null check** (3 blocks): `env_null` / `env_valid` / `env_checked` -- closures with zero captures have a null environment pointer. The null path reads arity and variadic flag directly from the closure struct layout (offset 33 for `input_arity`, offset 34 for `flags`). The valid path loads `packed_info` from the environment (format: bits 0-15 = num_captures, bits 16-31 = fixed_params, bit 63 = is_variadic).
5. **Variadic dispatch** (33+ blocks): If `is_variadic`, build a rest list by consing arguments right-to-left, then switch on `num_captures` (0 to 32), each case constructing the appropriate function call signature `(rest_list, &cap[0], ..., &cap[N-1])`.
6. **Non-variadic dispatch** (33+ blocks): Switch on capture count (0 to 32 via `MAX_CAPTURES`), building call signatures `(arg0, arg1, ..., &cap[0], &cap[1], ..., &cap[N-1])`. Includes arity mismatch padding (up to `MAX_CALL_ARGS = 4`).
7. **Direct function path** (1-2 blocks): For non-closure callable values, cast the tagged value's data to a function pointer and call directly.
8. **Merge block**: PHI node collecting results from all paths.

**Why this matters for PHI nodes:** LLVM PHI nodes require that each predecessor basic block appears exactly once and that the PHI dominates all its uses. When `codegenClosureCall` is placed inside a loop body, it inserts all of the above blocks between the loop header and the loop latch. A PHI node at the loop header that was supposed to reference `loop_body` as a predecessor now has a broken CFG because `loop_body` no longer branches back to the header -- instead, one of 30+ switch-case blocks does.

**The alloca+store/load pattern:** The proven fix (used in `string-for-each`, `string-map`, `vector-for-each`, `vector-map`, `codegenReduce`, and `mapWithClosure`) is to replace loop counter PHI nodes with stack allocations:

```
;; BROKEN: PHI pattern
loop_header:
  %i = phi i64 [0, %entry], [%i_next, %loop_body]  ;; FAILS: %loop_body is gone
  ...
  call codegenClosureCall(...)  ;; inserts 15+ blocks, %loop_body no longer exists
  %i_next = add %i, 1
  br %loop_header

;; CORRECT: alloca pattern
entry:
  %i_ptr = alloca i64
  store i64 0, %i_ptr
loop_header:
  %i = load i64, %i_ptr
  ...
  call codegenClosureCall(...)  ;; can insert any number of blocks safely
  %i_next = add %i, 1
  store i64 %i_next, %i_ptr     ;; store survives block restructuring
  br %loop_header
```

LLVM's `mem2reg` pass promotes these allocas back to SSA registers when possible, so there is no performance penalty.

**Dual number propagation through closure calls:** When `codegenDerivative` encounters a function parameter (not a known lambda), it falls through to a runtime dispatch path. The function argument is packed as a dual number tagged value (`ESHKOL_VALUE_DUAL_NUMBER`), heap-allocated via the arena (16 bytes for two doubles). This tagged value is passed through `codegenClosureCall` like any other argument. The called function, if AD-aware, checks for dual number inputs and propagates tangents through its operations. The result is unpacked with `unpackDualFromTaggedValue` and the tangent component (field 1 of the dual struct) is extracted as the derivative.

This means dual numbers flow through the standard closure calling convention without modification: they are tagged values like any other, and the closure dispatch mechanism (capture count switch, variadic rest list, etc.) is completely agnostic to whether the payload is an int64, double, dual number, or any other type. The AD awareness resides entirely in the arithmetic operations that inspect the type tag and dispatch to `dualAdd`, `dualMul`, etc.

### 2. Closures in Parallel Context

**How closures are passed to parallel workers:**

The parallel runtime (`lib/backend/parallel_codegen.cpp`) passes closures to worker threads by **value decomposition**, not by passing the closure struct directly. This is a deliberate ABI decision to avoid struct-by-value issues at the C/LLVM boundary.

For `parallel-map`, the task struct decomposes the closure into raw integer fields:

```c
struct llvm_parallel_map_task {
    uint64_t closure_ptr;   // pointer to closure struct (from fn.data.ptr_val)
    uint64_t item_type;     // item type field (i64-extended from i8)
    uint64_t item_data;     // item data field (raw_val)
    uint64_t result_ptr;    // pointer to result storage
};
```

The `closure_ptr` is the raw pointer to the `eshkol_closure_t` struct on the arena heap. This pointer is **shared** across all worker threads -- the closure itself is not copied. The LLVM-generated `__parallel_map_worker` function runs on worker threads and reconstructs tagged values from these i64 fields, then calls `__eshkol_call_unary_closure` which performs the same capture-count dispatch as `codegenClosureCall`.

**Captured environments are shared, not copied:** When multiple parallel workers receive the same `closure_ptr`, they all read from the same environment pointer (offset 8 in the closure struct), which points to the same captures array on the arena. This means:

- **Immutable captures** (the common case) are safe: all workers read the same tagged values from the same memory locations. Since captures are passed as pointers to tagged values in the environment (`&cap[0]`, `&cap[1]`, ...), and workers only load from these pointers, there are no data races.
- **Mutable captures** (via `set!` on captured variables) are **not thread-safe**. The "MUTABLE CAPTURE FIX" pattern throughout the codebase passes pointers to capture slots rather than copies of their values. If a closure captured via `set!` modifies a captured variable, and that closure runs on multiple threads via `parallel-map`, the writes to the capture slot are unprotected. There is no locking or atomic operation on capture pointer loads/stores.

**Thread safety model:** The parallel primitives assume closures are functionally pure. The runtime documentation notes that `parallel-fold` is "inherently sequential for non-associative operations" -- it uses the binary closure dispatcher sequentially rather than parallelizing the fold. `parallel-map` submits tasks to a thread pool with `thread_pool_submit` and waits for all futures before building the result list. Small lists (fewer than 4 elements) bypass the thread pool entirely and execute sequentially.

### 3. Workspace Module Closures

The consciousness engine's `ws-register!` and `ws-step!` builtins (implemented starting at line 33887 of `lib/backend/llvm_codegen.cpp`) demonstrate how closures are stored and later invoked via `codegenClosureCall` in a loop.

**`ws-register!` closure storage:** The builtin takes three arguments: workspace, name, and process-fn (a closure). All three are passed as tagged value pointers to the C runtime function `eshkol_ws_register_tagged`. The closure is stored as a raw 16-byte `eshkol_tagged_value_t` inside the `workspace_module_t` struct at offset 8 (after the 8-byte name pointer). The workspace stores up to 16 modules, each 32 bytes: `name(8) + process_fn(16) + salience(8)`.

**`ws-step!` closure invocation loop:** The `codegenWSStep` function generates a loop that iterates over all registered modules and calls each module's closure:

1. Load `num_modules` from the workspace struct (offset 0).
2. Wrap the workspace's `content` double array as a tensor tagged value via `eshkol_ws_make_content_tensor`.
3. Allocate a stack array of up to 16 result tagged values.
4. Loop using the **alloca+store/load pattern** (not PHI nodes):
   - Load `process_fn` tagged value from `modules_base + i * 32 + 8`.
   - Call `codegenClosureCall(fn_tv, {content_tv}, "ws-step-module")` -- this is a full closure dispatch with all 15+ basic blocks per iteration.
   - Store the result in the results array.
   - Increment the alloca counter.
5. After the loop, call `eshkol_ws_step_finalize` which performs softmax competition across module outputs and broadcasts the winning module's content back to the workspace.

The use of `alloca+store/load` for the loop counter (`ws_i`) is critical here because `codegenClosureCall` in the loop body creates the full set of dispatch blocks on every iteration. The alloca counter at `%ws_i` is immune to the CFG restructuring that would break a PHI-based counter.

**Interaction with AD:** The workspace closures are not AD-aware. The `content_tv` tensor passed to each module's closure is a regular tensor tagged value, not a dual number. If a workspace module's process function contains AD operations internally (e.g., computing a gradient as part of its inference step), those operations create their own tape on the thread-local tape stack. The workspace loop itself does not participate in any gradient computation.

---

## See Also

- [Type System](TYPE_SYSTEM.md) - Closure type annotations, HoTT function types
- [Memory Management](MEMORY_MANAGEMENT.md) - Arena allocation, closure lifetime
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - Closure codegen, LLVM IR
- [Scheme Compatibility](SCHEME_COMPATIBILITY.md) - R5RS/R7RS closure semantics
