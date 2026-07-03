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

**Do not use this procedure — it crashes (SIGBUS). See Known issues below.** Prefer `partial1`/`partial2`/`partial3` for fixed arities.

Edge cases: the fixed-arity variants (`partial1`/`partial2`/`partial3`) are safe and work as shown.

## Known issues

### `partial` crashes with SIGBUS
`partial` closes over the procedure argument and `apply`s it inside a nested variadic lambda; that pattern crashes the runtime.

```scheme
(require stdlib)
(display ((partial + 1 2 3) 4 5)) (newline)
```
```
[Eshkol] fatal signal: SIGBUS (bus error) — terminating; output above is what made it to stdout before the crash
```

Minimal repro isolating the trigger — `apply` of a **captured** procedure inside a nested variadic lambda:

```scheme
(define (f g . args) (lambda xs (apply g (append args xs))))
(display ((f + 1 2) 3 4)) (newline)   ; SIGBUS
```

The same shape with a *literal* operator instead of a captured one works fine, which pinpoints the fault to closing over the procedure:

```scheme
(define (f . args) (lambda xs (apply + (append args xs))))
(display ((f 1 2) 3 4)) (newline)   ; => 10, no crash
```

Not yet ledgered in `.swarm/tasks/` at time of writing.
