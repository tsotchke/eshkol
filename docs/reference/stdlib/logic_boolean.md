# `core.logic.boolean` — boolean combinators for predicates

**Source**: [`lib/core/logic/boolean.esk`](../../../lib/core/logic/boolean.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.logic.boolean)`

Combinators that build new predicates from existing ones and apply them across lists.

## Functions

### `(negate pred)`
Returns a predicate that yields the logical negation of `pred`: `(lambda (x) (not (pred x)))`.

```scheme
;; boolean.esk
(require stdlib)
(display ((negate is-even?) 3)) (newline)
```
```
#t
```

### `(all? pred lst)`
Returns `#t` if every element of `lst` satisfies `pred`, else `#f`. Recursive; short-circuits via `and`.

```scheme
(require stdlib)
(display (all? is-positive? '(1 2 3))) (newline)
(display (all? is-positive? '(1 -2 3))) (newline)
(display (all? is-positive? '())) (newline)
```
```
#t
#f
#t
```

Edge cases: vacuously `#t` on the empty list.

### `(none? pred lst)`
Returns `#t` if **no** element satisfies `pred`. Implemented as `(all? (negate pred) lst)`.

```scheme
(require stdlib)
(display (none? is-negative? '(1 2 3))) (newline)
(display (none? is-even? '(2 4))) (newline)
```
```
#t
#f
```

Edge cases: vacuously `#t` on the empty list.
