# `core.list.sort` — merge sort

**Source**: [`lib/core/list/sort.esk`](../../../lib/core/list/sort.esk)
**Require**: `(require core.list.sort)` — auto-loaded by `(require stdlib)`.

A single `sort` function implementing **bottom-up (iterative) merge sort** with a
user-supplied comparator. The list is converted to a vector once and merged
between that vector and one scratch vector, doubling the run width each pass, so
the workspace is O(n) and the call depth is O(log n) of tail calls — not a
recursion tree over cons cells. The merge favours the left run on ties (`less?`
is a strictly-less test), which preserves the input order of equal keys.

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

### Large inputs (ESH-0098, resolved)

There is no list-length ceiling. The earlier top-down implementation recursed on
`take-n`/`drop-n` helpers, so its control depth grew linearly with the list and
it tripped the `ESHKOL_MAX_RECURSION_DEPTH=100000` guard at 100,000 elements
(and peaked near 1 GB sorting 50,000). Those helpers no longer exist — the
current `sort` is the iterative vector merge described above.

Measured on this build, sorting a fully reversed `iota` (worst-case input) with
`eshkol-run -r`:

| `sort` input | result | peak RSS |
| --- | --- | --- |
| 50,000 | sorted, rc 0 | 139 MB |
| 100,000 | sorted, rc 0 | 147 MB |
| 250,000 | sorted, rc 0 | 139 MB |
| 1,000,000 | sorted, rc 0 | 176 MB |
| 5,000,000 | sorted, rc 0 | 848 MB |

```scheme
;; bigsort.esk
(require core.list.sort)
(require core.list.generate)
(define s (sort (reverse (iota 1000000)) <))
(display (car s)) (display " ") (display (length s)) (newline)
```
```
0 1000000
```
Identical result from an AOT binary (`eshkol-run -o bigsort bigsort.esk`). Above
a few million elements the limit is available memory, not recursion depth.
