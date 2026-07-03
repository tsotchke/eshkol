# `core.list.generate` — list constructors and number ranges

**Source**: [`lib/core/list/generate.esk`](../../../lib/core/list/generate.esk)
**Require**: `(require core.list.generate)` — auto-loaded by `(require stdlib)`.
Internally depends on `core.list.query` (for `length`, used by `make-list`).

Constructors for numeric ranges (`iota`, `range`) and simple list building
(`repeat`, `make-list`, `zip`). All produce freshly allocated proper lists.

## Functions

### `(iota count)`
Returns `(0 1 … count-1)`. Built tail-recursively from the top down.

### `(iota-from count start)`
Returns `count` consecutive integers starting at `start`:
`(start start+1 … start+count-1)`. Note arg order: **count first, start second**.

### `(iota-step count start step)`
Returns `count` numbers `(start start+step start+2·step …)`.

```scheme
;; generate.esk
(require core.list.generate)
(display (iota 5)) (newline)
(display (iota-from 5 1)) (newline)
(display (iota-step 5 0 2)) (newline)
```
```
(0 1 2 3 4)
(1 2 3 4 5)
(0 2 4 6 8)
```

Edge cases: `count <= 0` yields `'()`.

### `(repeat n x)`
Returns a list of `n` copies of `x`. `n <= 0` yields `'()`.

### `(make-list . args)`
Dual-mode constructor:
- **2 args where the first is a non-negative integer** → `(make-list n fill)`
  builds `n` copies of `fill` (delegates to `repeat`).
- **otherwise** → variadic: returns the argument list as-is.

```scheme
;; makelist.esk
(require core.list.generate)
(display (repeat 3 'x)) (newline)
(display (make-list 3 'a)) (newline)   ; mode 1: n + fill
(display (make-list 1 2 3)) (newline)  ; mode 2: variadic
```
```
(x x x)
(a a a)
(1 2 3)
```

Note: because of the dual-mode heuristic, `(make-list 2 'z)` is ambiguous with
the variadic reading but resolves to mode 1 (two copies of `z`) since `2` is a
non-negative integer and there are exactly two args. To force the literal
two-element list `(2 z)` you cannot use `make-list`; use `(list 2 'z)`.

### `(range start end)`
Returns integers from `start` up to but **excluding** `end`. `start >= end`
yields `'()`. Non-tail-recursive (see Known issues).

### `(zip lst1 lst2)`
Combines two lists element-wise into a list of two-element lists. Stops at the
shorter of the two.

```scheme
;; range-zip.esk
(require core.list.generate)
(display (range 0 5)) (newline)
(display (range 2 2)) (newline)
(display (zip '(a b c) '(1 2 3))) (newline)
(display (zip '(a b) '(1 2 3))) (newline)   ; stops at shorter
```
```
(0 1 2 3 4)
()
((a 1) (b 2) (c 3))
((a 1) (b 2))
```

### Known issues

`range` and `zip` are non-tail-recursive (they `cons` before recursing), so they
inherit the stdlib depth ceiling documented in
[`list_query.md`](list_query.md) and [`list_transform.md`](list_transform.md):
generating a multi-hundred-thousand-element range in a single call can SIGILL.
`iota` / `iota-from` / `iota-step` are tail-recursive and safe well past 1M.
