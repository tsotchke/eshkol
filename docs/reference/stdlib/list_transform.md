# `core.list.transform` — slicing, concatenation, filtering, partitioning

**Source**: [`lib/core/list/transform.esk`](../../../lib/core/list/transform.esk)
**Require**: `(require core.list.transform)` — auto-loaded by `(require stdlib)`.

Structural list transforms: `take`/`drop` and their right-hand counterparts,
variadic `append`, tail-recursive `reverse`, `filter`, `unzip`, `partition`, and
`list-copy`. (`remove` is kept as a builtin for performance and is not defined
here.)

## Functions

### `(take lst n)`
Returns the first `n` elements. Note arg order: **list first, count second**.
`n <= 0` or a shorter list returns fewer elements (no error).

### `(drop lst n)`
Returns the list with the first `n` elements removed. `n` past the end returns
`'()`.

```scheme
;; takedrop.esk
(require core.list.transform)
(display (take '(1 2 3 4 5) 3)) (newline)
(display (drop '(1 2 3 4 5) 3)) (newline)
(display (take '(1 2) 5)) (newline)   ; n > length: clamps
(display (drop '(1 2) 5)) (newline)   ; n > length: '()
```
```
(1 2 3)
(4 5)
(1 2)
()
```

### `(take-right lst n)`
Returns the **last** `n` elements (SRFI-1), computed as `(drop lst (- (length lst) n))`.

### `(drop-right lst n)`
Returns all **but** the last `n` elements, computed as `(take lst (- (length lst) n))`.

```scheme
;; takedropright.esk
(require core.list.transform)
(display (take-right '(1 2 3 4 5) 2)) (newline)
(display (drop-right '(1 2 3 4 5) 2)) (newline)
```
```
(4 5)
(1 2 3)
```

Edge cases (verified against issue #72): both clamp gracefully because they
route through `take`/`drop`, which guard on `<= n 0`:

| call | result |
| --- | --- |
| `(take-right '(1 2 3) 5)` (n > len) | `(1 2 3)` |
| `(take-right '(1 2 3) 0)` | `()` |
| `(take-right '() 2)` | `()` |
| `(drop-right '(1 2 3) 5)` (n > len) | `()` |
| `(drop-right '(1 2 3) 0)` | `(1 2 3)` |
| `(drop-right '() 2)` | `()` |

### `(append . lists)`
Concatenates any number of lists (R7RS §6.4). `(append)` → `'()`; `(append a)`
returns `a` as-is (last arg is not copied); the last argument may be any datum
(improper tails allowed).

```scheme
;; append.esk
(require core.list.transform)
(display (append '(1 2) '(3 4) '(5 6))) (newline)
(display (append)) (newline)
(display (append '(1 2))) (newline)
```
```
(1 2 3 4 5 6)
()
(1 2)
```

### `(reverse lst)`
Reverses a list. Tail-recursive (accumulator style) — safe on very long lists.

### `(filter pred lst)`
Returns the elements satisfying `pred`, in order.

### `(unzip pairs)`
Inverse of `zip`: splits a list of two-element lists into a two-element list
`(firsts seconds)`.

### `(partition pred lst)`
Returns `(matching non-matching)` — a two-element list of the elements that
satisfy and fail `pred`, each in original order.

### `(list-copy lst)`
Returns a fresh shallow copy of `lst` (R7RS).

```scheme
;; misc.esk
(require core.list.transform)
(display (reverse '(1 2 3))) (newline)
(display (filter even? '(1 2 3 4 5 6))) (newline)
(display (unzip '((a 1) (b 2) (c 3)))) (newline)
(display (partition even? '(1 2 3 4 5 6))) (newline)
(display (list-copy '(1 2 3))) (newline)
```
```
(3 2 1)
(2 4 6)
((a b c) (1 2 3))
((2 4 6) (1 3 5))
(1 2 3)
```

Edge cases: `filter`/`unzip`/`partition`/`list-copy` all return `'()` (or
`(() ())` for the two-output ones) on empty input.

### Known issues — depth ceiling (ESH-0108)

`filter`, `take`, `list-copy`, and `append` (via `append-2`) are
non-tail-recursive: they `cons` around the recursive call, one native stack frame
per element, and SIGILL (rc 132) with **no diagnostic** on large inputs (the
`ESHKOL_MAX_RECURSION_DEPTH=100000` runtime guard does not cover stdlib user
functions). Observed on this machine:

| op | passes | SIGILLs (rc 132, no output) |
| --- | --- | --- |
| `filter` (with predicate closure) | 200,000 | 250,000 |
| `append` (single big left arg) | 500,000 | (larger) |
| `list-copy` | 300,000 | (larger) |

`reverse` is tail-recursive and completes on 1,000,000-element lists.
`take-right`/`drop-right` additionally call `length` (also non-tail — see
[`list_query.md`](list_query.md)), so their ceiling is bounded by `length`'s
(~500k). Tracked as **ESH-0108**. Docs-only note; no code changed. For very long
lists, build with tail-recursive primitives or reverse-and-cons.
