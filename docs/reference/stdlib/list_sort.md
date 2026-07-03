# `core.list.sort` — merge sort

**Source**: [`lib/core/list/sort.esk`](../../../lib/core/list/sort.esk)
**Require**: `(require core.list.sort)` — auto-loaded by `(require stdlib)`.

A single `sort` function implementing top-down merge sort with a user-supplied
comparator. Not stable-guaranteed by contract, but the merge favours the left
half on ties (`less?` strictly-less test), which preserves input order for equal
keys in practice.

## Functions

### `(sort lst less?)`
Returns a new list containing the elements of `lst` ordered by the binary
predicate `less?`. Note arg order: **list first, comparator second**.

**Comparator contract**: `less?` is called as `(less? a b)` and must return a
truthy value when `a` should come *before* `b`. Passing `<` sorts ascending;
passing `>` sorts descending.

```scheme
;; sort.esk
(require core.list.sort)
(display (sort '(3 1 4 1 5 9 2 6) <)) (newline)
(display (sort '(3 1 4 1 5 9 2 6) >)) (newline)
(display (sort '() <)) (newline)
(display (sort '(42) <)) (newline)
```
```
(1 1 2 3 4 5 6 9)
(9 6 5 4 3 2 1 1)
()
(42)
```

Edge cases: `'()` sorts to `'()`, a singleton sorts to itself.

To sort strings, supply a string comparator explicitly — `string<?` is **not**
auto-loaded into the default namespace, so `(sort strs string<?)` errors with
`Undefined variable: string<?` unless you have brought it into scope. Define your
own or require the string module that provides it.

### Known issues — recursion depth is O(n) (ESH-0098)

The recursion **control depth grows linearly with list length**, not
logarithmically, because `sort` recurses on `(take-n mid …)` / `(drop-n mid …)`
and those helpers plus `merge` are themselves non-tail-recursive. Consequences on
this machine:

| `sort` input | result |
| --- | --- |
| 50,000 | sorted (ok) |
| 100,000 | `Unhandled exception: maximum recursion depth (100000) exceeded` (rc 1) |

Repro:
```scheme
(require core.list.sort)
(require core.list.generate)
(display (car (sort (iota 100000) <))) (newline)
;; => Unhandled exception: maximum recursion depth (100000) exceeded
```

Unlike `length`/`filter` (which SIGILL silently), `sort` fails through the C
runtime list helpers, so it hits the `ESHKOL_MAX_RECURSION_DEPTH=100000` guard
and reports a clean diagnostic with exit code 1. The task note also records a
~983MB RSS peak sorting 50k. Tracked as **ESH-0098** ("stdlib sort recursion
depth is O(n): cannot sort >= ~100k elements"). Docs-only note; no code changed.
For >~50k elements, sort a vector or chunk the input.
