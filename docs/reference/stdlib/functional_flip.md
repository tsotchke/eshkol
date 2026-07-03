# `core.functional.flip` — argument-swapping combinator

**Source**: [`lib/core/functional/flip.esk`](../../../lib/core/functional/flip.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.functional.flip)`

A single combinator that reverses the argument order of a binary procedure.

## Functions

### `(flip f)`
Returns `(lambda (x y) (f y x))` — a binary procedure that calls `f` with its two arguments swapped. Handy for turning a non-commutative operator around.

```scheme
;; flip.esk
(require stdlib)
(display ((flip -) 3 10)) (newline)   ; (- 10 3)
```
```
7
```

Edge cases: strictly binary. The returned procedure takes exactly two arguments.
