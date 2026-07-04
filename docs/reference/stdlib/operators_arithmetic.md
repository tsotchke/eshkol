# `core.operators.arithmetic` — first-class arithmetic operators

**Source**: [`lib/core/operators/arithmetic.esk`](../../../lib/core/operators/arithmetic.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.operators.arithmetic)`

Named binary wrappers around `+`, `-`, `*`, `/` so the operators can be passed as ordinary values to higher-order procedures. The division wrapper is named **`divide`** (not `div`) — see Known issues for why.

## Functions

### `(add x y)`
`(+ x y)`.

### `(sub x y)`
`(- x y)`.

### `(mul x y)`
`(* x y)`.

### `(divide x y)`
`(/ x y)`.

```scheme
;; arithmetic.esk
(require stdlib)
(display (add 3 4)) (newline)
(display (sub 10 3)) (newline)
(display (mul 6 7)) (newline)
(display (divide 20 4)) (newline)
(display (divide 7 2)) (newline)
(display (map (lambda (p) (add (car p) (cdr p))) '((1 . 2) (3 . 4)))) (newline)
```
```
7
7
42
5
7/2
(3 7)
```

Edge cases: `add`/`sub`/`mul`/`divide` follow the full numeric tower of the underlying operators (ints, bignums, rationals, doubles).

## Known issues

None. (The division wrapper used to be named `div` and returned `()` for scalar
arguments. Root cause: a function named `div` is emitted as a *weak external*
symbol whose bare name collides with the C standard library's `div()`; the
dynamic linker coalesced the two and bound callers to libc `div(int,int)->div_t`,
so the 16-byte tagged-value ABI was misread and the result came back as the empty
list. `add`/`sub`/`mul` have no such libc collision, which is why only `div` was
affected. The wrapper was renamed to `divide` to avoid the collision. For scalar
division you can also use the built-in `/` directly.)

Workaround: call `/` directly, or wrap it under a different name. Not yet ledgered in `.swarm/tasks/` at time of writing.
