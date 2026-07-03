# `core.operators.compare` — first-class comparison operators

**Source**: [`lib/core/operators/compare.esk`](../../../lib/core/operators/compare.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.operators.compare)`

Named binary wrappers around `<`, `>`, `<=`, `>=`, `=`, so the comparison operators can be passed as ordinary procedures. All return proper `#t`/`#f` booleans, including when `apply`'d — the raw-int leakage tracked in `.swarm/tasks/ESH-0079` (status: done — "First-class / apply'd builtin equality predicates return raw int (0/1) or () instead of #f/#t") does not affect these wrappers.

## Functions

### `(lt x y)`
`(< x y)`.

### `(gt x y)`
`(> x y)`.

### `(le x y)`
`(<= x y)`.

### `(ge x y)`
`(>= x y)`.

### `(eq x y)`
Numeric equality `(= x y)` — **not** `eq?` identity. Compares numbers by value.

```scheme
;; compare.esk
(require stdlib)
(display (lt 3 5)) (newline)
(display (gt 3 5)) (newline)
(display (le 5 5)) (newline)
(display (ge 4 5)) (newline)
(display (eq 5 5)) (newline)
(display (eq 5 6)) (newline)
;; usable first-class / via apply
(display (apply eq '(5 5))) (newline)
(display (apply lt '(3 5))) (newline)
```
```
#t
#f
#t
#f
#t
#f
#t
#t
```

Edge cases: `eq` is value equality on numbers (delegates to `=`); do not confuse it with the identity predicate `eq?`. The wrappers are strictly binary.
