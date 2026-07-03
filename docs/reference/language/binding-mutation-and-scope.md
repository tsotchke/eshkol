# Binding, Mutation, and Scope

## Lexical scope

Eshkol is lexically scoped. A free variable in a `lambda` body refers to the
binding in the closest enclosing lexical scope at the point the `lambda` is
written, and that binding is *captured* by the resulting closure.

```scheme
(define (make-adder n) (lambda (x) (+ x n)))
(define add5 (make-adder 5))
(define add10 (make-adder 10))
(display (add5 1)) (newline)
(display (add10 1)) (newline)
```
```
6
11
```
Each call to `make-adder` produces a closure over its own `n`.

## `set!`

```
(set! var value)
```
Mutates an existing binding. `set!` does not create a binding — the variable must
already be bound in some enclosing scope. Returns an unspecified value; use it for
effect.

```scheme
(define (make-counter)
  (let ((n 0))
    (lambda () (set! n (+ n 1)) n)))
(define c1 (make-counter))
(define c2 (make-counter))
(display (c1)) (display " ") (display (c1)) (display " ") (display (c2)) (newline)
```
```
1 2 1
```
`c1` and `c2` each own their own `n`.

## Shared mutable capture

When two or more closures capture the *same* mutable variable, they share one
cell: a `set!` through one closure is visible through the others. This memory
model is guaranteed (the fix for the multi-closure shared-capture family, ESH-0074,
made this reliable).

```scheme
(define (make-pair)
  (let ((v 0))
    (list (lambda () v)          ; reader
          (lambda (x) (set! v x))))) ; writer
(define p (make-pair))
((cadr p) 42)      ; write through the writer
(display ((car p))) (newline)   ; read through the reader
```
```
42
```

The same guarantee holds for closures created in separate `letrec` activations —
each activation gets its own shared cell, isolated from other activations
(see the `counter-factory` example in
[special-forms.md](special-forms.md#per-activation-instance-isolation-fixed-esh-0075)).

## Shadowing

User bindings shadow one another normally following lexical scope. There are two
documented exceptions where shadowing does **not** work as expected:

### Known issue — `raise` cannot be shadowed (ESH-0090)

A user definition named `raise` does not intercept calls; they still reach the
builtin R7RS exception primitive.

```scheme
(define (raise x) (list 'my-raise x))
(display (raise 42)) (newline)
```
```
Unhandled exception: user exception
```
The call hit the builtin `raise` (which threw), not the user procedure. Choose a
different name (e.g. `raise*`) until this is fixed.

### Known issue — top-level names colliding with libc symbols (ESH-0092)

Top-level `define`d globals are emitted as raw C symbols. A top-level name that
matches a libc symbol corrupts that symbol. The program can appear to work and
then crash at teardown.

```scheme
(define free 0)
(set! free (+ free 1))
(display free) (newline)
```
```
1

[Eshkol] fatal signal: SIGBUS (bus error) — terminating; …
```
The displayed value `1` is correct, but the process dies with SIGBUS because
`free` aliases libc's `free`. The related **ESH-0103** covers a global named `log`
whose `set!` is silently lost on cached-JIT/AOT. Avoid libc names
(`free`, `log`, `read`, `write`, `open`, `time`, `exit`, …) for top-level globals;
locals inside a `let`/`lambda` are unaffected.
