# `core.alist` — association-list helpers

**Source**: [`lib/core/alist.esk`](../../../lib/core/alist.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.alist)`

Structural helpers over association lists (lists of `(key . value)` pairs), keyed by `equal?` via `assoc`. Consolidates helpers that were previously redefined ad hoc across agent modules (see the source note referencing ESH-0063; no such task file is present in this worktree's `.swarm/tasks/`).

## Functions

### `(alist-ref alist key)`
Returns the value bound to `key`, or **raises an error** if the key is absent. Uses `assoc`, so keys are matched with `equal?`.

```scheme
;; alist.esk
(require stdlib)
(define a '((x . 1) (y . 2) (z . 3)))
(display (alist-ref a 'y)) (newline)
```
```
2
```

Edge cases: a missing key raises `"alist-ref: key not found"`:

```scheme
(require stdlib)
(display (alist-ref '((a . 1)) 'missing)) (newline)
```
```
Unhandled exception: alist-ref: key not found
```

### `(alist-ref-or alist key default)`
Like `alist-ref`, but returns `default` instead of raising when the key is absent.

```scheme
(require stdlib)
(define a '((x . 1) (y . 2) (z . 3)))
(display (alist-ref-or a 'q 99)) (newline)   ; absent -> default
(display (alist-ref-or a 'x 99)) (newline)   ; present
```
```
99
1
```

### `(alist-set alist key value)`
Functional update: returns a **new** alist with `key` bound to `value`, replacing an existing binding in place or appending a new one at the end. The input list is not mutated.

```scheme
(require stdlib)
(define a '((x . 1) (y . 2) (z . 3)))
(display (alist-set a 'y 20)) (newline)   ; replace existing
(display (alist-set a 'w 4)) (newline)    ; append new
(display a) (newline)                     ; original unchanged
```
```
((x . 1) (y . 20) (z . 3))
((x . 1) (y . 2) (z . 3) (w . 4))
((x . 1) (y . 2) (z . 3))
```

Edge cases: on an empty alist, `alist-set` returns a one-element list `((key . value))`. Replacement matches the first key equal to `key`.
