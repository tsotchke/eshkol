# `core.operators.arithmetic` — first-class arithmetic operators

**Source**: [`lib/core/operators/arithmetic.esk`](../../../lib/core/operators/arithmetic.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.operators.arithmetic)`

Named binary wrappers around `+`, `-`, `*`, `/` so the operators can be passed as ordinary values to higher-order procedures.

## Functions

### `(add x y)`
`(+ x y)`.

### `(sub x y)`
`(- x y)`.

### `(mul x y)`
`(* x y)`.

### `(div x y)`
Intended to be `(/ x y)`. **Broken — returns `()`. See Known issues.**

```scheme
;; arithmetic.esk
(require stdlib)
(display (add 3 4)) (newline)
(display (sub 10 3)) (newline)
(display (mul 6 7)) (newline)
(display (map (lambda (p) (add (car p) (cdr p))) '((1 . 2) (3 . 4)))) (newline)
```
```
7
7
42
(3 7)
```

Edge cases: `add`/`sub`/`mul` follow the full numeric tower of the underlying operators (ints, bignums, rationals, doubles). For division, use the built-in `/` directly instead of `div`.

## Known issues

### `div` returns `()` for scalar arguments
The name `div` collides with a codegen-recognized tensor-arithmetic operation (`lib/backend/llvm_codegen.cpp:5470`; see also `lib/backend/tensor_arith_codegen.cpp`), so a scalar call is dispatched to the wrong path and yields the empty list instead of a quotient.

```scheme
(require stdlib)
(display (div 20 4)) (newline)   ; expected 5
(display (div 7 2)) (newline)    ; expected 7/2
```
```
()
()
```

For comparison, the same expression through `/` (or through a user-defined wrapper whose name is not `div`) works:

```scheme
(display (/ 20 4)) (newline)                       ; => 5
(define (mydiv a b) (/ a b)) (display (mydiv 20 4)) ; => 5
```

Workaround: call `/` directly, or wrap it under a different name. Not yet ledgered in `.swarm/tasks/` at time of writing.
