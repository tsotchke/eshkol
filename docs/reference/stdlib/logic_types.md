# `core.logic.types` — type-checking predicates as first-class values

**Source**: [`lib/core/logic/types.esk`](../../../lib/core/logic/types.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.logic.types)`

First-class wrappers around the built-in type tests, for use with higher-order procedures.

## Functions

### `(is-null? x)`
`(null? x)` — true when `x` is the empty list `'()`.

### `(is-pair? x)`
`(pair? x)` — true when `x` is a cons cell.

```scheme
;; types.esk
(require stdlib)
(display (is-null? '())) (newline)
(display (is-null? '(1))) (newline)
(display (is-pair? '(1 2))) (newline)
(display (is-pair? '())) (newline)
```
```
#t
#f
#t
#f
```

Edge cases: `is-null?` is true only for the empty list; `is-pair?` is false for the empty list (a list of length 0 is not a pair).
