# `core.functional.curry` — currying and partial application

**Source**: [`lib/core/functional/curry.esk`](../../../lib/core/functional/curry.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.functional.curry)`

Converts multi-argument procedures into chains of single-argument procedures (`curry2`/`curry3`), reverses that (`uncurry2`), and fixes leading arguments (`partial1`/`partial2`/`partial3`/`partial`).

## Functions

### `(curry2 f)`
Curries a binary `f` into `(lambda (x) (lambda (y) (f x y)))`. Apply one argument at a time.

```scheme
;; curry.esk
(require stdlib)
(display (((curry2 +) 3) 4)) (newline)
```
```
7
```

### `(curry3 f)`
Curries a ternary `f` into a three-deep chain: `(((( curry3 f) x) y) z)`.

```scheme
(require stdlib)
(display ((((curry3 +) 1) 2) 3)) (newline)
```
```
6
```

### `(uncurry2 f)`
Inverse of `curry2`: takes a curried binary `f` and returns an ordinary two-argument procedure `(lambda (x y) ((f x) y))`.

```scheme
(require stdlib)
(display ((uncurry2 (curry2 -)) 10 3)) (newline)
```
```
7
```

### `(partial1 f x)`
Fixes the sole argument of a unary `f`, returning a **zero-argument thunk**. Call it with no arguments to force evaluation.

```scheme
(require stdlib)
(display ((partial1 + 5))) (newline)   ; thunk -> (+ 5)
```
```
5
```

### `(partial2 f x)`
Fixes the first argument of a binary `f`; returns `(lambda (y) (f x y))`.

```scheme
(require stdlib)
(display ((partial2 - 10) 3)) (newline)   ; (- 10 3)
```
```
7
```

### `(partial3 f x)`
Fixes the first argument of a ternary `f`; returns `(lambda (y z) (f x y z))`.

```scheme
(require stdlib)
(display ((partial3 + 1) 2 3)) (newline)  ; (+ 1 2 3)
```
```
6
```

### `(partial f . args)`
Generic partial application: fixes any number of leading arguments and returns a variadic procedure that appends the rest — `(lambda xs (apply f (append args xs)))`.

```scheme
(require stdlib)
(display ((partial + 1 2 3) 4 5)) (newline)   ; (+ 1 2 3 4 5)
(display ((partial * 2) 3 4)) (newline)        ; (* 2 3 4)
```
```
15
24
```

Edge cases: the fixed-arity variants (`partial1`/`partial2`/`partial3`) are also available for fixed arities.

## Known issues

None. (Historically `partial` crashed with SIGBUS: it closes over the procedure
argument and `apply`s it inside the returned lambda, and `apply` of a
closure-**captured** procedure mis-read the capture slot as a tagged value —
an invalid function pointer. The apply codegen now resolves a captured procedure
through the normal variable path, so `(apply captured-proc …)` works. Minimal
repro that used to crash:

```scheme
(define (f g) (lambda (x y) (apply g (list x y))))
(display ((f +) 3 4)) (newline)   ; => 7
```
)
