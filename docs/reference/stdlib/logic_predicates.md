# `core.logic.predicates` — numeric predicates as first-class values

**Source**: [`lib/core/logic/predicates.esk`](../../../lib/core/logic/predicates.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.logic.predicates)`

Named, first-class wrappers around the built-in numeric tests, so they can be passed to higher-order procedures (`map`, `filter`, `all?`, `negate`, …). Each returns `#t`/`#f`.

## Functions

### `(is-zero? x)`
`(zero? x)` — true when `x` equals 0.

### `(is-positive? x)`
`(> x 0)`.

### `(is-negative? x)`
`(< x 0)`.

### `(is-even? x)`
`(even? x)`.

### `(is-odd? x)`
`(odd? x)`.

```scheme
;; predicates.esk
(require stdlib)
(display (is-zero? 0)) (newline)
(display (is-positive? 5)) (newline)
(display (is-negative? -3)) (newline)
(display (is-even? 4)) (newline)
(display (is-odd? 7)) (newline)
(display (filter is-even? '(1 2 3 4 5 6))) (newline)
```
```
#t
#t
#t
#t
#t
(2 4 6)
```

Edge cases: these delegate directly to the numeric builtins, so they inherit the builtins' behavior on non-integers (`is-even?`/`is-odd?` expect integers) and across the numeric tower.
