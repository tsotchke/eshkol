# `core.functional.compose` — function-composition combinators

**Source**: [`lib/core/functional/compose.esk`](../../../lib/core/functional/compose.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.functional.compose)`

Small combinators for building new procedures out of existing ones: right-to-left composition plus two trivial building blocks (`identity`, `constantly`).

## Functions

### `(compose f g)`
Returns a new unary procedure `(lambda (x) (f (g x)))` — `g` runs first, then `f`. Both `f` and `g` must be unary.

```scheme
;; compose.esk
(require stdlib)
(define inc (lambda (x) (+ x 1)))
(define dbl (lambda (x) (* x 2)))
(display ((compose inc dbl) 5)) (newline)   ; inc(dbl(5))
```
```
11
```

Edge cases: strictly unary — the returned procedure takes exactly one argument. Composing procedures of other arities produces a procedure that fails when called.

### `(compose3 f g h)`
Three-way composition: `(lambda (x) (f (g (h x))))`, applied right-to-left. All three must be unary.

```scheme
(require stdlib)
(define inc (lambda (x) (+ x 1)))
(define dbl (lambda (x) (* x 2)))
(display ((compose3 inc dbl inc) 5)) (newline)  ; inc(dbl(inc(5)))
```
```
13
```

### `(identity x)`
Returns its argument unchanged. Useful as a default/no-op transform.

```scheme
(require stdlib)
(display (identity 42)) (newline)
```
```
42
```

### `(constantly x)`
Returns a unary procedure that ignores its argument and always yields `x`.

```scheme
(require stdlib)
(display ((constantly 7) 999)) (newline)
```
```
7
```

Edge cases: the returned procedure is unary; it discards whatever it is passed.
