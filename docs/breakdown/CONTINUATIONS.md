# First-Class Continuations in Eshkol

## Overview

Eshkol provides first-class continuations as specified by R7RS, implemented via
`setjmp`/`longjmp` at the LLVM IR level. This document describes the theoretical
foundations, the concrete implementation in the compiler backend, and the
interaction between `call/cc`, `dynamic-wind`, and the exception system
(`guard`/`raise`/`with-exception-handler`).

All continuation-related codegen resides in `lib/backend/llvm_codegen.cpp`
(34,928 lines). Runtime support functions are in `lib/core/arena_memory.cpp`.
Type definitions live in `inc/eshkol/eshkol.h`.

---

## 1. Theoretical Foundations

A continuation represents "the rest of the computation" at any given point in
program execution. In Scheme and Eshkol, continuations are first-class values:
they can be stored in variables, passed to functions, and invoked like
procedures. Invoking a continuation abandons the current computation and resumes
execution at the captured point, returning the supplied value.

Eshkol's continuations are **single-shot**: a captured continuation may be
invoked at most once. This restriction follows from the `setjmp`/`longjmp`
implementation strategy, where the call stack frame that established the
`setjmp` point must still be live at invocation time. Attempting to invoke a
continuation after its establishing frame has returned yields undefined
behavior. This is a deliberate tradeoff -- full multi-shot continuations
(as in traditional Scheme) require stack copying or CPS transformation, neither
of which is compatible with Eshkol's goals of zero-overhead native compilation.

The single-shot restriction is sufficient for the most common continuation use
cases: early exit from nested computations, exception handling, and coroutine-
style control flow where each continuation is used exactly once.

---

## 2. call/cc -- Capturing and Invoking Continuations

### 2.1 Surface Syntax

```scheme
(call/cc proc)
(call-with-current-continuation proc)
```

Both forms are equivalent. `proc` must be a one-argument procedure. It receives
the current continuation as its sole argument. If `proc` returns normally, its
return value becomes the result of the `call/cc` expression. If the
continuation is invoked with a value, that value becomes the result instead.

### 2.2 Continuation Object Representation

Continuation objects are closures with a distinguished heap subtype. The type
tag is `ESHKOL_VALUE_CALLABLE` (9), and the object header's subtype field is
set to `CALLABLE_SUBTYPE_CONTINUATION` (4). This allows the closure dispatch
logic to identify continuations at call time without additional runtime
metadata.

The continuation's state is captured in an `eshkol_continuation_state_t`
structure, defined at `inc/eshkol/eshkol.h:984`:

```c
typedef struct eshkol_continuation_state {
    void* jmp_buf_ptr;              // Points to jmp_buf on the caller's stack
    eshkol_tagged_value_t value;    // Value passed when continuation is invoked
    void* wind_mark;                // Dynamic-wind stack marker at capture time
} eshkol_continuation_state_t;
```

The three fields serve distinct purposes:

- **jmp_buf_ptr**: Points to the 200-byte `jmp_buf` allocated on the
  `call/cc` caller's stack frame. This is the target for `longjmp`.
- **value**: Storage for the value passed to the continuation upon invocation.
  The invoking code writes here before calling `longjmp`; the `setjmp` return
  path reads it.
- **wind_mark**: Snapshot of the global dynamic-wind stack pointer
  (`g_dynamic_wind_stack`) at capture time. Used during invocation to unwind
  any dynamic-wind entries established after the `call/cc`.

### 2.3 Capture Mechanism (setjmp)

The `codegenCallCC` function at `llvm_codegen.cpp` emits the following
LLVM IR sequence:

1. **Allocate jmp_buf** on the current stack frame (200 bytes, platform-safe).
2. **Create continuation state** via `eshkol_make_continuation_state(arena, jmp_buf)`
   (`arena_memory.cpp`). This arena-allocates the state struct and records
   the current `g_dynamic_wind_stack` as `wind_mark`.
3. **Create continuation closure** via `eshkol_make_continuation_closure(arena, state)`
   (`arena_memory.cpp`). This allocates a standard closure with one capture
   (the state pointer) and overrides the header subtype to
   `CALLABLE_SUBTYPE_CONTINUATION`.
4. **Package as tagged value** with type `ESHKOL_VALUE_CALLABLE`.
5. **Call setjmp**. Returns 0 on initial call; non-zero when `longjmp` fires.
6. **Branch on setjmp result**:
   - If 0 (normal path): evaluate `proc`, call it with the continuation as
     argument, merge the return value.
   - If non-zero (invoked path): load the value from `state->value` (offset 8
     in the state struct), merge it as the `call/cc` result.

The two paths merge at a `callcc_done` basic block via a PHI node. The normal
path's incoming edge is conditionally added -- if `proc` itself performs a
non-local exit (e.g., another `call/cc` or `raise`), its block may already be
terminated, and the PHI must not reference it.

### 2.4 Invocation Mechanism (longjmp)

Continuation invocation is handled in the closure dispatch path at
`llvm_codegen.cpp`. When a closure call is emitted, the codegen checks
the header subtype at `closure_ptr - 8`:

```
header_ptr = GEP(closure_ptr, -8)
subtype    = load i8 from header_ptr
is_cont    = icmp eq subtype, CALLABLE_SUBTYPE_CONTINUATION
```

If the closure is a continuation, the dispatch short-circuits to the invocation
sequence (`llvm_codegen.cpp`):

1. Load the environment pointer from the closure (offset 8).
2. Load captures[0] from the environment (offset 8 past `num_captures`),
   which holds the state pointer as a tagged `HEAP_PTR`.
3. Unpack the state pointer via `unpackInt64FromTaggedValue` + `IntToPtr`.
4. Store the invocation value at `state + 8` (the `value` field). If no
   argument is provided, `ESHKOL_VALUE_NULL` is stored.
5. Unwind the dynamic-wind stack (see Section 3).
6. Load `jmp_buf_ptr` from `state + 0`.
7. Call `longjmp(jmp_buf_ptr, 1)`.
8. Emit `unreachable` -- control never continues past `longjmp`.

Normal closure dispatch resumes in a separate basic block for non-continuation
callables.

---

## 3. dynamic-wind -- Before/After Thunks and the Wind Stack

### 3.1 Surface Syntax

```scheme
(dynamic-wind before thunk after)
```

Calls `before` (a zero-argument procedure), then `thunk`, then `after`.
Returns the result of `thunk`. The key guarantee: `after` is called even if
control exits `thunk` via a non-local jump (continuation invocation), and
`before` is called when control re-enters the dynamic extent (not applicable
for single-shot continuations, but the machinery is in place).

### 3.2 Wind Stack Architecture

The wind stack is a global singly-linked list of `eshkol_dynamic_wind_entry_t`
nodes, defined at `inc/eshkol/eshkol.h:991`:

```c
typedef struct eshkol_dynamic_wind_entry {
    eshkol_tagged_value_t before;
    eshkol_tagged_value_t after;
    struct eshkol_dynamic_wind_entry* prev;
} eshkol_dynamic_wind_entry_t;
```

The global head pointer is `g_dynamic_wind_stack` (`arena_memory.cpp`).

### 3.3 Codegen for dynamic-wind

The `codegenDynamicWind` function at `llvm_codegen.cpp` emits:

1. Evaluate all three thunk expressions (with noreturn safety checks after each).
2. Store `before` and `after` tagged values into stack allocas (placed in the
   function entry block for LLVM dominance correctness).
3. Call `before()` via `codegenClosureCall` with zero arguments.
4. Push the wind entry via `eshkol_push_dynamic_wind(arena, &before, &after)`
   (`arena_memory.cpp`). The push happens *after* `before()` succeeds,
   per R7RS semantics.
5. Call `thunk()` via `codegenClosureCall`, capturing the result.
6. Pop the wind entry via `eshkol_pop_dynamic_wind()` (`arena_memory.cpp`).
7. Call `after()` via `codegenClosureCall`.
8. Return thunk's result.

### 3.4 Unwinding on Non-Local Exit

When a continuation is invoked (Section 2.4, step 5), the codegen loads the
`wind_mark` from the continuation state (offset 24 in the struct) and calls
`eshkol_unwind_dynamic_wind(wind_mark)` (`arena_memory.cpp`):

```c
void eshkol_unwind_dynamic_wind(void* saved_wind_mark) {
    eshkol_dynamic_wind_entry_t* mark = (eshkol_dynamic_wind_entry_t*)saved_wind_mark;
    while (g_dynamic_wind_stack != NULL && g_dynamic_wind_stack != mark) {
        eshkol_dynamic_wind_entry_t* entry = g_dynamic_wind_stack;
        g_dynamic_wind_stack = entry->prev;
        call_thunk_from_tagged(&entry->after);
    }
}
```

This walks the wind stack from the current top down to (but not including) the
saved mark, calling each `after` thunk along the way. The `wind_mark` was
captured at `call/cc` time, so this unwinds exactly those `dynamic-wind` scopes
that were entered after the continuation was created.

The `call_thunk_from_tagged` helper (`arena_memory.cpp`) extracts the
closure pointer from a tagged `CALLABLE` value and calls `call_thunk_closure`
(`arena_memory.cpp`), which dispatches by capture count (0 through 4).

**Platform ABI note:** `call_thunk_closure` uses a compile-time branch on `__aarch64__` because `eshkol_tagged_value_t` (16 bytes) is returned differently on ARM64 (register pairs) vs x86/Windows (hidden return buffer). See [MEMORY_MANAGEMENT.md — Platform-Specific ABI Considerations](MEMORY_MANAGEMENT.md#platform-specific-abi-considerations).

---

## 4. Exception Handling

Eshkol's exception system is built on the same `setjmp`/`longjmp` infrastructure
as continuations, providing R7RS-compliant `guard`/`raise` and the lower-level
`with-exception-handler`.

### 4.1 guard

```scheme
(guard (var clause ...)
  body ...)
```

Evaluates `body`. If `raise` is called during evaluation, the raised value is
bound to `var` and the `clause` forms are evaluated as in a `cond` expression.
If no clause matches, the exception is re-raised.

The `codegenGuard` function at `llvm_codegen.cpp` emits:

1. **Setup block**: Allocate a 200-byte `jmp_buf` on the stack. Push an
   exception handler via `eshkol_push_exception_handler(jmp_buf)`
   (`arena_memory.cpp`), which prepends a handler entry to the global
   `g_exception_handler_stack`.
2. **Call setjmp**: Branch to the try block (result = 0) or handler block
   (result != 0).
3. **Try block**: Evaluate `body`. If it completes normally, pop the handler
   and branch to done. The body result is converted to a tagged value via
   `typedValueToTaggedValue` for PHI consistency.
4. **Handler block**: Pop the handler. Retrieve the raised value via
   `eshkol_get_raised_value` (R7RS: the original value, not the exception
   struct). Bind it to `var` in the symbol table. Evaluate clauses:
   - **else clause**: Evaluate body expressions, clear exception, branch to done.
   - **Test clause**: Evaluate test, branch to then-block or next-clause.
   - **Fallthrough**: If no clause matches, re-raise via `eshkol_raise`.
5. **Done block**: PHI node merges results from the try path and any matching
   handler clause.

Tail-call optimized guard bodies work correctly because the TCO context is
orthogonal to the `setjmp`/`longjmp` mechanism -- the `jmp_buf` lives on the
guard-establishing frame's stack, not on the tail-called function's frame.

### 4.2 raise

```scheme
(raise obj)
```

The `codegenRaise` function at `llvm_codegen.cpp` emits:

1. Store the original raised value (as a tagged value) into an alloca via
   `eshkol_set_raised_value`. This preserves the R7RS requirement that
   `guard` handlers see the original value, not a wrapped exception struct.
2. Create an `eshkol_exception_t` struct via `eshkol_make_exception_with_header`,
   with type `ESHKOL_EXCEPTION_USER_DEFINED`.
3. Call `eshkol_raise(exception)`, which is marked `noreturn`.
4. Emit `unreachable`.

The `eshkol_raise` runtime function (`arena_memory.cpp`) stores the
exception globally and performs `longjmp` to the top of the exception handler
stack. If no handler exists, it prints a diagnostic and calls `exit(1)`.

### 4.3 with-exception-handler

```scheme
(with-exception-handler handler thunk)
```

A lower-level primitive. `handler` is a one-argument procedure; `thunk` is a
zero-argument procedure. If an exception is raised during `thunk`, `handler`
is called with the raised value.

The `codegenWithExceptionHandler` function at `llvm_codegen.cpp` follows
the same `setjmp`/`longjmp` pattern as `guard`, but invokes the handler closure
with the raised value instead of evaluating `cond`-style clauses.

### 4.4 Exception Handler Stack

The exception handler stack is a global linked list separate from the
dynamic-wind stack. Each entry holds a `jmp_buf` pointer:

```c
void eshkol_push_exception_handler(void* jmp_buf_ptr);
void eshkol_pop_exception_handler(void);
```

When `eshkol_raise` fires, it jumps to `g_exception_handler_stack->jmp_buf_ptr`.
The handler pops itself before evaluating clauses, ensuring that re-raise from
within a handler propagates to the next outer handler.

---

## 5. Implementation Details

### 5.1 LLVM IR Structure

Each `call/cc` site generates four basic blocks:

| Block               | Purpose                                        |
|----------------------|------------------------------------------------|
| `callcc_setup`       | Allocate jmp_buf, create state and closure     |
| `callcc_normal`      | Evaluate proc, call with continuation          |
| `callcc_invoked`     | Load value from state after longjmp return     |
| `callcc_done`        | PHI merge of normal and invoked results        |

Each `guard` site generates a similar four-block structure:

| Block               | Purpose                                        |
|----------------------|------------------------------------------------|
| `guard_setup`        | Allocate jmp_buf, push handler, call setjmp    |
| `guard_try`          | Evaluate body expressions                      |
| `guard_handler`      | Pop handler, evaluate clauses                  |
| `guard_done`         | PHI merge of body and clause results           |

### 5.2 Continuation State Memory Layout

The `eshkol_continuation_state_t` struct has the following layout at the
machine level:

| Offset | Size | Field        | Description                          |
|--------|------|--------------|--------------------------------------|
| 0      | 8    | jmp_buf_ptr  | Pointer to jmp_buf on caller's stack |
| 8      | 16   | value        | Tagged value (type + data)           |
| 24     | 8    | wind_mark    | Saved g_dynamic_wind_stack pointer   |

Total: 32 bytes, arena-allocated with 8-byte alignment.

### 5.3 Free Variable Analysis

The `findFreeVariablesImpl` function must recurse into all continuation-related
AST node types to correctly capture variables for lambda closures:

- **ESHKOL_CALL_CC_OP** (`llvm_codegen.cpp`): Recurses into
  `call_cc_op.proc`.
- **ESHKOL_DYNAMIC_WIND_OP** (`llvm_codegen.cpp`): Recurses into
  `before`, `thunk`, and `after`.
- **ESHKOL_GUARD_OP** (`llvm_codegen.cpp`): Recurses into all body
  expressions and clause expressions.
- **ESHKOL_RAISE_OP** (`llvm_codegen.cpp`): Recurses into the exception
  expression.

Missing cases in `findFreeVariablesImpl` caused a historical bug where
`call/cc` inside `dynamic-wind` failed with "Cannot capture k from outer
function" -- the free variable analysis silently skipped the `dynamic-wind`
node, so the lambda wrapping the `call/cc` never captured the needed variable.

### 5.4 Noreturn Safety

Several codegen functions emit code after expressions that may not return
(e.g., `raise`, nested `call/cc`). The pattern throughout the continuation
codegen is:

```cpp
if (builder->GetInsertBlock()->getTerminator()) {
    return UndefValue::get(tagged_value_type);
}
```

This prevents emitting instructions into an already-terminated basic block,
which would be invalid LLVM IR. The `UndefValue` is safe because the
terminated path is unreachable.

---

## 6. Performance Characteristics

### 6.1 Capture Cost

Capturing a continuation via `call/cc` is O(1) in the common case:

- One `setjmp` call (saves registers and stack pointer, typically 50-100 ns).
- One arena allocation for the 32-byte state struct.
- One arena allocation for the closure (with header and environment).
- No stack copying, no CPS transformation, no heap allocation of frames.

The `jmp_buf` is 200 bytes on the stack. On most platforms, `setjmp` saves
only the callee-saved registers and stack pointer into this buffer.

### 6.2 Invocation Cost

Invoking a continuation has a cost proportional to the number of active
`dynamic-wind` entries that must be unwound:

- O(W) where W is the number of wind entries between the current stack and the
  saved `wind_mark`.
- Each unwind step calls an `after` thunk closure.
- The `longjmp` itself is O(1).

### 6.3 Guard/Raise Cost

Setting up a `guard` is O(1): one `setjmp` plus one handler push. Raising an
exception is O(1) for the `longjmp`, plus O(W) for any dynamic-wind unwinding
that occurs as part of handler dispatch.

### 6.4 Guidelines for Hot Loops

- Avoid `call/cc` inside tight numerical loops. The `setjmp` and arena
  allocations add overhead that is unnecessary when simple early-exit logic
  (e.g., a conditional return) suffices.
- Prefer `guard`/`raise` for structured error handling over raw `call/cc`.
  The `guard` codegen is slightly more efficient because it does not allocate
  a continuation closure.
- `dynamic-wind` entries are arena-allocated and never freed individually.
  In long-running loops with `dynamic-wind`, arena pressure may increase.

---

## 7. Code Examples

### 7.1 Early Exit from a Search

```scheme
(define (find-first pred lst)
  (call/cc
    (lambda (return)
      (for-each (lambda (x)
                  (when (pred x) (return x)))
                lst)
      #f)))

(display (find-first even? '(1 3 5 4 7)))  ; => 4
```

When `(pred x)` is true, the continuation `return` is invoked with `x`,
immediately abandoning the `for-each` and returning `x` as the result of the
`call/cc` expression.

### 7.2 Resource Cleanup with dynamic-wind

```scheme
(define (with-file filename proc)
  (let ((port (open-input-file filename)))
    (dynamic-wind
      (lambda () #t)               ; before: no-op
      (lambda () (proc port))      ; thunk: user code
      (lambda () (close-port port)))))  ; after: always close
```

The `after` thunk guarantees `close-port` executes even if `proc` performs a
non-local exit via `call/cc` or `raise`.

### 7.3 Exception Handling with guard

```scheme
(define (safe-divide a b)
  (guard (exn
          ((string? exn) (string-append "Error: " exn))
          (else "Unknown error"))
    (if (= b 0)
        (raise "division by zero")
        (/ a b))))

(display (safe-divide 10 0))   ; => "Error: division by zero"
(display (safe-divide 10 2))   ; => 5
```

The `guard` establishes a handler. When `raise` fires, execution jumps to the
handler block, which binds the raised value to `exn` and tests the clauses in
order.

### 7.4 Nested guard with Re-raise

```scheme
(guard (outer-exn
        ((number? outer-exn) (* outer-exn 2)))
  (guard (inner-exn
          ((string? inner-exn) (string-length inner-exn)))
    (raise 42)))
```

The inner `guard` catches the exception but its `string?` test fails for `42`.
No clause matches, so the inner guard re-raises (per R7RS), and the outer
guard's `number?` clause matches, returning `84`.

### 7.5 dynamic-wind with call/cc

```scheme
(let ((path '()))
  (let ((k (call/cc (lambda (c) c))))
    (dynamic-wind
      (lambda () (set! path (cons 'in path)))
      (lambda ()
        (when (null? path)
          (k k)))   ; re-enter (single-shot: only works once)
      (lambda () (set! path (cons 'out path)))))
  (reverse path))
```

This demonstrates the interaction: the `after` thunk fires when the
continuation `k` is invoked, and the `before` thunk fires on re-entry into
the `dynamic-wind` scope. With single-shot semantics, the continuation can
only be used once.

---

## 8. Comparison with Full Continuations

| Property                | Eshkol (single-shot)      | Full Scheme (multi-shot)     |
|-------------------------|---------------------------|------------------------------|
| Implementation          | setjmp/longjmp            | Stack copying or CPS         |
| Capture cost            | O(1)                      | O(stack depth)               |
| Invocation cost         | O(W) wind unwind          | O(stack depth) + O(W)        |
| Multiple invocations    | Undefined behavior        | Fully supported              |
| Coroutines              | Limited (one-shot)        | Full cooperative threading   |
| Memory overhead         | 40-byte state + closure   | Full stack copy per capture  |
| LLVM compatibility      | Native (no special passes)| Requires custom lowering     |

The single-shot design is a principled engineering choice. It enables Eshkol to
compile continuations to efficient native code using only standard C runtime
facilities (`setjmp`/`longjmp`), without requiring stack manipulation
intrinsics, CPS transformation passes, or runtime stack copying.

---

## References

- R7RS, Section 6.10: Control features (call/cc, dynamic-wind, values)
- R7RS, Section 6.11: Exceptions (guard, raise, with-exception-handler)
- `lib/backend/llvm_codegen.cpp`: codegenCallCC (§17324), codegenDynamicWind
  (§17429), codegenGuard (§16948), codegenRaise (§17221), codegenWithExceptionHandler
  (§17495), continuation invocation dispatch (§4751)
- `lib/core/arena_memory.cpp`: eshkol_make_continuation_state (§4639),
  eshkol_make_continuation_closure (§4653), wind stack runtime
  (§4773–4801), exception handler stack (§4578–4586),
  `call_thunk_closure` ABI dispatch (§4685–4763)
- `inc/eshkol/eshkol.h`: eshkol_continuation_state_t, eshkol_dynamic_wind_entry_t,
  runtime function declarations
- `lib/core/platform_runtime.cpp`: `eshkol_jmp_buf_size` (§551–553)

---

## 9. What Landed in v1.2 / v1.2.1

The v1.2 release fixed two long-standing bugs in the continuation
subsystem: the free-variable analyser was silently skipping ~30 AST
operation kinds, which caused `(call/cc ...)` inside `(dynamic-wind
...)` (and other compositions) to fail with "Cannot capture k from
outer function"; and the `jmp_buf` allocation was using a hardcoded
size that did not match every libc on every platform. This section
documents both, plus the platform-specific `setjmp` lowering on
Windows.

### 9.1 The findFreeVariablesImpl Coverage Fix

`findFreeVariablesImpl` (`lib/backend/llvm_codegen.cpp` §20919) is the
analyser used by lambda codegen to determine which outer-scope
variables a lambda body references. The result drives closure
environment capture — every referenced free variable must be packed
into the closure's environment, otherwise the JIT-emitted lambda body
loads from an undefined slot.

Pre-v1.2, the function had a large `switch` over `op->op` with a
`default: break;` fallback. This silently skipped recursion for every
operation kind not explicitly listed, including the four
continuation-related kinds (`ESHKOL_DYNAMIC_WIND_OP`,
`ESHKOL_CALL_CC_OP`, `ESHKOL_GUARD_OP`, `ESHKOL_RAISE_OP`) plus the
multi-value forms (`ESHKOL_VALUES_OP`, `ESHKOL_CALL_WITH_VALUES_OP`,
`ESHKOL_LET_VALUES_OP`, `ESHKOL_LET_STAR_VALUES_OP`), pattern-match
(`ESHKOL_MATCH_OP`), OALR forms (`ESHKOL_WITH_REGION_OP`,
`ESHKOL_BORROW_OP`, `ESHKOL_OWNED_OP`, `ESHKOL_MOVE_OP`,
`ESHKOL_SHARED_OP`, `ESHKOL_WEAK_REF_OP`), calculus operators
(`ESHKOL_DIFF_OP`, `ESHKOL_JACOBIAN_OP`, `ESHKOL_HESSIAN_OP`,
`ESHKOL_DIVERGENCE_OP`, `ESHKOL_CURL_OP`, `ESHKOL_LAPLACIAN_OP`), the
logic/consciousness ops (`ESHKOL_UNIFY_OP`, `ESHKOL_MAKE_FACT_OP`,
`ESHKOL_KB_QUERY_OP`, etc.), tensor literals (`ESHKOL_TENSOR_OP`),
case-lambda, and others — roughly thirty kinds in total.

**Symptom**: any lambda whose body transitively contained one of
these forms but whose free variables flowed only through that form's
sub-expressions would silently fail to capture the variables. A
classic repro is `(call/cc ...)` inside `(dynamic-wind ...)`:

```scheme
(define (with-cleanup body cleanup)
  (call/cc
    (lambda (return)
      (dynamic-wind
        (lambda () #t)
        (lambda () (body return))          ; <-- references `return`
        cleanup))))
```

The lambda passed as the thunk references `return`, a captured
variable from the outer lambda. The analyser walks the outer lambda's
body, hits the `dynamic-wind` node, and (pre-fix) skipped recursion
into its three sub-expressions. The thunk's free-variable set came up
empty. At codegen time, the thunk's reference to `return` triggered
"Cannot capture k from outer function" because there was no entry in
the closure environment.

**The fix** (current source, `lib/backend/llvm_codegen.cpp`
§21265–21472) adds explicit cases for every missing operation kind.
The four continuation cases are at §21265–21297:

```cpp
case ESHKOL_DYNAMIC_WIND_OP:
    if (op->dynamic_wind_op.before) findFreeVariablesImpl(op->dynamic_wind_op.before, ...);
    if (op->dynamic_wind_op.thunk)  findFreeVariablesImpl(op->dynamic_wind_op.thunk,  ...);
    if (op->dynamic_wind_op.after)  findFreeVariablesImpl(op->dynamic_wind_op.after,  ...);
    break;
case ESHKOL_CALL_CC_OP:
    if (op->call_cc_op.proc) findFreeVariablesImpl(op->call_cc_op.proc, ...);
    break;
case ESHKOL_GUARD_OP: {
    for (uint64_t i = 0; i < op->guard_op.num_body_exprs; i++)
        findFreeVariablesImpl(&op->guard_op.body[i], ...);
    for (uint64_t i = 0; i < op->guard_op.num_clauses; i++)
        findFreeVariablesImpl(&op->guard_op.clauses[i], ...);
    break;
}
case ESHKOL_RAISE_OP:
    if (op->raise_op.exception)
        findFreeVariablesImpl(op->raise_op.exception, ...);
    break;
```

The multi-value cases at §21298–21310 recurse into the producer and
consumer of `call-with-values`, the value expressions of `values`, and
(for `let-values` / `let*-values`, §21452–21494) the producers in the
outer scope plus the body in the bound-extended scope. The match,
calculus, and logic cases follow the same pattern: explicit recursion
into every sub-AST that might contain a free reference.

The lesson, recorded in the project memory: when adding a new AST op
type, always add it to `findFreeVariablesImpl` too. Grepping for
"Cannot capture" warnings in test output is the fastest way to detect
a missing case.

### 9.2 Dynamic jmp_buf Sizing

`sizeof(jmp_buf)` varies wildly across libc implementations:

- glibc on x86_64: 200 bytes (large because it stores SIMD and FPU state).
- musl on x86_64: 88 bytes.
- macOS libc on arm64: 192 bytes.
- macOS libc on x86_64: 148 bytes.
- MSVCRT on Windows x86_64: 256 bytes (the `_setjmpex` variant).

Pre-v1.2 codegen allocated a hardcoded 200-byte buffer, which was
correct on glibc x86_64 but underspecified on Windows and oversized
elsewhere. The v1.2 fix removes the hardcode and queries the libc at
runtime.

`lib/core/platform_runtime.cpp` §551–553 exports the size:

```cpp
extern "C" std::uint64_t eshkol_jmp_buf_size() {
    return sizeof(jmp_buf);
}
```

The symbol name is centralized as a compile-time constant in
`inc/eshkol/runtime_exports.h:21`:

```cpp
inline constexpr const char* jmp_buf_size_symbol = "eshkol_jmp_buf_size";
```

Codegen allocates with a runtime-sized alloca
(`lib/backend/llvm_codegen.cpp` §32149–32164):

```cpp
Value* allocaJmpBuf(const char* name) {
    FunctionType* size_type = FunctionType::get(int64_type, {}, false);
    Function* size_func = module->getFunction(eshkol::runtime::jmp_buf_size_symbol);
    if (!size_func) {
        size_func = Function::Create(size_type, Function::ExternalLinkage,
            eshkol::runtime::jmp_buf_size_symbol, module.get());
    }
    Value* size = builder->CreateCall(size_func, {}, std::string(name) + "_size");
    AllocaInst* jmp_buf = builder->CreateAlloca(builder->getInt8Ty(), size, name);
    jmp_buf->setAlignment(Align(16));
    return jmp_buf;
}
```

The 16-byte alignment is the worst-case requirement (glibc x86_64
stores AVX registers in the buffer). The IR `alloca i8, %size`
generates a dynamic alloca that LLVM lowers to a stack-pointer
subtraction by the runtime-computed size. The same IR is emitted for
both AOT and JIT; the resolution of `eshkol_jmp_buf_size` happens at
link time (AOT) or via the JIT's `DynamicLibrarySearchGenerator`
(REPL).

### 9.3 Platform-Specific setjmp Lowering

The `setjmp` symbol name and call convention differ across platforms.
`lib/backend/llvm_codegen.cpp` §32166–32183 selects per target:

```cpp
Function* getOrDeclareSetjmpFunc() {
    const Triple& target_triple = module->getTargetTriple();
    const bool is_windows_target = target_triple.isOSWindows();
    const char* setjmp_symbol = is_windows_target ? "_setjmpex" : "setjmp";
    FunctionType* setjmp_type = is_windows_target
        ? FunctionType::get(int32, {ptr, ptr}, false)    // 2 args on Windows
        : FunctionType::get(int32, {ptr},      false);   // 1 arg elsewhere
    Function* setjmp_func = module->getFunction(setjmp_symbol);
    if (!setjmp_func) {
        setjmp_func = Function::Create(setjmp_type, Function::ExternalLinkage,
            setjmp_symbol, module.get());
    }
    setjmp_func->addFnAttr(Attribute::ReturnsTwice);
    return setjmp_func;
}
```

On POSIX targets, the call is `setjmp(jmp_buf)`. On Windows targets,
the call is `_setjmpex(jmp_buf, context)` where `context` is an
architecture-specific second argument:

- **Windows ARM64**: `context = llvm.sponentry()`. The function-entry
  stack pointer. This is what Clang's lowering uses, and the runtime
  expects to find it in the second argument for SEH-aware unwinding.
- **Windows x64**: `context = llvm.frameaddress(0)`. The current
  frame address.

`makeSetjmpArgs` (`lib/backend/llvm_codegen.cpp` §32185–32209)
constructs the arguments:

```cpp
std::vector<Value*> makeSetjmpArgs(Value* jmp_buf_alloc) {
    std::vector<Value*> args = {jmp_buf_alloc};
    const Triple& target_triple = module->getTargetTriple();
    if (!target_triple.isOSWindows()) return args;

    Value* setjmp_context = nullptr;
    if (target_triple.getArch() == Triple::aarch64) {
        Function* sponentry = ESHKOL_GET_INTRINSIC(module.get(),
            Intrinsic::sponentry, {builder->getPtrTy()});
        setjmp_context = builder->CreateCall(sponentry, {}, "setjmp_sponentry");
    } else {
        Function* frameaddress = ESHKOL_GET_INTRINSIC(module.get(),
            Intrinsic::frameaddress, {builder->getPtrTy()});
        setjmp_context = builder->CreateCall(frameaddress,
            {ConstantInt::get(builder->getInt32Ty(), 0)}, "setjmp_frame");
    }
    args.push_back(setjmp_context);
    return args;
}
```

The `ReturnsTwice` attribute on the `setjmp` function declaration is
critical: it tells LLVM's optimizers that any value defined before the
`setjmp` call may have a different value after it returns from a
`longjmp`. Without this attribute, LLVM would assume the call is a
normal one and would hoist or coalesce loads/stores around it,
silently corrupting the call/cc and guard semantics.

### 9.4 ARM64 16-Byte Struct Return ABI

`eshkol_tagged_value_t` is a 16-byte struct (`{u8, u8, u16, u32, u64}`).
The two dominant ABI families handle 16-byte aggregate return
differently:

- **x86-64 (System V) and Windows x64**: The caller passes a hidden
  first argument — a pointer to the caller's return slot. The function
  writes the return value there and returns `void`.
- **AArch64 (both POSIX and Windows)**: Returns the 16-byte struct
  directly in the `x0:x1` register pair. No hidden buffer is passed.

The `call_thunk_closure` helper in `lib/core/arena_memory.cpp`
§4685–4763 dispatches by architecture at compile time:

```c
#if defined(__aarch64__) || defined(_M_ARM64)
    typedef eshkol_tagged_value_t (*fn0_t)(void);
    typedef eshkol_tagged_value_t (*fn1_t)(void*);
    // ... fn2_t through fn4_t follow the same pattern
    result = ((fn0_t)(uintptr_t)closure->func_ptr)();
#else
    typedef void (*fn0_t)(eshkol_tagged_value_t*);
    typedef void (*fn1_t)(eshkol_tagged_value_t*, void*);
    // ... fn2_t through fn4_t follow the same pattern
    ((fn0_t)(uintptr_t)closure->func_ptr)(&result);
#endif
```

This trampoline is used by `call_thunk_from_tagged` (§4766–4772),
which is called during `call/cc` continuation unwinding
(§17421 — load `wind_mark`, call `eshkol_unwind_dynamic_wind`)
and dynamic-wind before/after thunk invocation. Without the per-arch
branch, ARM64 would treat the hidden return-buffer pointer as the
first data argument, shifting all real arguments by one slot and
corrupting the return value. See MEMORY_MANAGEMENT.md
§`ARM64 Thunk Calling Convention` for the broader context on
platform ABIs and Eshkol's aggregate-value handling.

### 9.5 Continuation State and Wind-Mark Layout

The continuation state struct (`inc/eshkol/eshkol.h`, declared near
§984) is:

```c
typedef struct eshkol_continuation_state {
    void* jmp_buf_ptr;              // offset 0:  pointer to jmp_buf on caller's stack
    eshkol_tagged_value_t value;    // offset 8:  tagged value (16 bytes)
    void* wind_mark;                // offset 24: saved g_dynamic_wind_stack pointer
} eshkol_continuation_state_t;
```

Total: 32 bytes, arena-allocated. The struct is created by
`eshkol_make_continuation_state` (`lib/core/arena_memory.cpp` §4639):

```c
extern "C" eshkol_continuation_state_t* eshkol_make_continuation_state(
    void* arena_void, void* jmp_buf_ptr) {
    arena_t* arena = (arena_t*)arena_void;
    eshkol_continuation_state_t* state = (eshkol_continuation_state_t*)
        arena_allocate_aligned(arena, sizeof(*state), 8);
    state->jmp_buf_ptr = jmp_buf_ptr;
    state->value.type = ESHKOL_VALUE_NULL;
    state->value.data.raw_val = 0;
    state->wind_mark = (void*)g_dynamic_wind_stack;
    return state;
}
```

`wind_mark` snapshots `g_dynamic_wind_stack` at capture time. The
codegen for continuation invocation (`lib/backend/llvm_codegen.cpp`
§4751) emits a call to `eshkol_unwind_dynamic_wind(saved_wind_mark)`
before the `longjmp`. The unwind walks the wind stack down to the
mark, calling each `after` thunk in LIFO order (the linked list is
prepended at push, so the most recently pushed entry is at the head;
walking from the head down to but not including the mark releases the
correct subset).

The wind entry struct (`inc/eshkol/eshkol.h` §991):

```c
typedef struct eshkol_dynamic_wind_entry {
    eshkol_tagged_value_t before;
    eshkol_tagged_value_t after;
    struct eshkol_dynamic_wind_entry* prev;
} eshkol_dynamic_wind_entry_t;
```

The global head pointer `g_dynamic_wind_stack` is in
`lib/core/arena_memory.cpp` §4637, a plain C global (not thread-local)
— continuation semantics inside parallel-map are therefore *not*
isolated per worker, and this is a known limitation documented in
PARALLEL_COMPUTING.md §13.7.
