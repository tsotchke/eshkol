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
already be bound in some enclosing scope. Returns
[the unspecified value](control-flow.md#the-unspecified-value), not the stored
value; use it for effect.

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

## Procedure bindings that change

A call through a name always reaches the procedure the name denotes **at that
moment**. The compiler calls a procedure directly when it can prove which one a
name denotes, and that proof is tied to the binding that owns the name: a
binding that is reassigned with `set!`, a top-level name that is defined again,
and a parameter or sibling `let` binding that reuses a name are all evaluated
and called through the closure ABI. The rule is the same for a direct call, for
`apply`, for `map`, `vector-map`, `reduce` and `remove`, and for the
differentiation operators
(see [the AD capture rules](../ad/operators.md#where-a-capture-is-resolved)).
The decision record is
[ADR 0015](../../design/adr/0015-static-callee-binding-identity.md).

```scheme
;; A variable bound to a procedure is called as it stands now: directly,
;; through apply, and through the list operations.
(define twice (lambda (x) (* 2 x)))
(define before (twice 5))
(set! twice (lambda (x) (* 3 x)))
(display (list before (twice 5) (apply twice (list 5)) (map twice (list 1 2 3))))
(newline)

;; A top-level redefinition replaces the procedure everywhere.
(define (step x) (+ x 1))
(define (step x) (+ x 10))
(display (map step (list 1 2)))
(newline)

;; A parameter of the same name hides the outer procedure.
(define (scale x) (* 10 x))
(define (use scale) (scale 2))
(display (use (lambda (x) (- x))))
(newline)

;; remove with a predicate computed at run time keeps each element's type.
(define (drop-if pred xs) (remove pred xs))
(display (drop-if (lambda (x) (> x 1.5)) (list 1.0 2.0 3.0)))
(newline)
```
```
(10 15 15 (3 6 9))
(11 12)
-2
(1)
```
Identical on the JIT (`-r`) and as an AOT binary. Bindings that are never
reassigned keep direct-call resolution; only a binding that can change takes the
closure path.

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

### Top-level names that look like libc symbols (ESH-0092 / ESH-0103, closed)

Top-level globals no longer collide with the C symbols of the host runtime. A
global may be named `free`, `log`, `read`, `write`, `open`, `time` or `exit`;
it is an ordinary Eshkol binding, its `set!` sticks, and the process exits
cleanly:

```scheme
(define free 0)
(set! free (+ free 1))
(display free) (newline)
```
```
1
```
Verified for each of those names on both the JIT (`-r`) and an AOT binary. The
earlier symptoms — a SIGBUS at teardown (**ESH-0092**) and a `set!` on a global
named `log` silently lost under cached-JIT/AOT (**ESH-0103**) — no longer
reproduce.

### Names shared with the C math library

A user procedure may be named after a C math library function (`exp`, `log`,
`tanh`, `pow`, ...). The compiler never finds a math routine by its bare module
name: it asks for the LLVM intrinsic (`llvm.exp.f64` and its siblings), whose
name is reserved, and where the LLVM version in use has no intrinsic it
verifies the signature of what it finds and otherwise declares a distinctly
named routine. A user's `exp` and the `exp` inside a tensor activation therefore
never meet, in either order of appearance:

```scheme
(define (exp x) (+ x 1))
(define t (tensor (list -2.0 1.0)))
(display (exp 2))
(newline)
(display (tensor-ref (elu t 2) 0))
(newline)
```
```
3
-1.7293294335267746
```
The single access point is
[`libm_codegen.h`](../../../inc/eshkol/backend/libm_codegen.h); the regressions
are `tests/ml/math_symbol_collision_test.esk` and
`tests/ml/math_symbol_user_shadow_test.esk`.
