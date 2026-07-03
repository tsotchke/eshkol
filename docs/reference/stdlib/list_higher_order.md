# `core.list.higher_order` — map / fold / for-each / any / every

**Source**: [`lib/core/list/higher_order.esk`](../../../lib/core/list/higher_order.esk)
**Require**: `(require core.list.higher_order)` — auto-loaded by `(require stdlib)`.

Classic higher-order list operations. The map family is arity-suffixed
(`map1`/`map2`/`map3`) rather than variadic. The fold family provides every
common spelling (`fold`, `fold-left`, `foldl`, `fold-right`, `foldr`) as aliases
around two real implementations.

## Functions

### `(map1 proc lst)`
Maps a 1-argument `proc` over one list.

### `(map2 proc lst1 lst2)`
Maps a 2-argument `proc` over two lists element-wise; stops at the shorter list.

### `(map3 proc lst1 lst2 lst3)`
Maps a 3-argument `proc` over three lists element-wise; stops at the shortest.

```scheme
;; maps.esk
(require core.list.higher_order)
(display (map1 (lambda (x) (* x x)) '(1 2 3))) (newline)
(display (map2 + '(1 2 3) '(10 20 30))) (newline)
(display (map3 (lambda (a b c) (+ a b c)) '(1 2) '(10 20) '(100 200))) (newline)
```
```
(1 4 9)
(11 22 33)
(111 222)
```

### `(fold proc init lst)` / `(fold-left proc init lst)` / `(foldl proc init lst)`
Left fold. `proc` is called as `(proc acc elt)` — **accumulator first, element
second** — left to right. `fold-left` and `foldl` are exact synonyms of `fold`.
Tail-recursive (safe on very long lists).

### `(fold-right proc init lst)` / `(foldr proc init lst)`
Right fold. `proc` is called as `(proc elt acc)` — **element first, accumulator
second** — right to left. `foldr` is a synonym of `fold-right`. Non-tail-recursive.

```scheme
;; folds.esk
(require core.list.higher_order)
(display (fold + 0 '(1 2 3 4))) (newline)
(display (fold-left cons '() '(1 2 3))) (newline)   ; (proc acc elt)
(display (foldl - 0 '(1 2 3))) (newline)            ; ((0-1)-2)-3
(display (fold-right cons '() '(1 2 3))) (newline)  ; rebuilds list
(display (foldr - 0 '(1 2 3))) (newline)            ; 1-(2-(3-0))
```
```
10
(((() . 1) . 2) . 3)
-6
(1 2 3)
2
```

Note the differing argument order between the left and right families:
`fold`/`foldl` pass `(acc elt)`; `fold-right`/`foldr` pass `(elt acc)`.

### `(for-each proc lst)`
Applies `proc` to each element for effect, left to right. Returns `#f` on the
empty list (its return value is otherwise unspecified — do not rely on it).

```scheme
;; foreach.esk
(require core.list.higher_order)
(for-each (lambda (x) (display x) (display "-")) '(a b c)) (newline)
```
```
a-b-c-
```

### `(any pred lst)`
Returns `#t` as soon as any element satisfies `pred`, else `#f`. (Returns a plain
boolean, not the satisfying value.)

### `(every pred lst)`
Returns `#t` if every element satisfies `pred` (vacuously `#t` on `'()`), else `#f`.

### `(filter-map fn lst)`
Applies `fn` to each element and keeps only the truthy results (SRFI-1). Combines
`map` + `filter` in one pass.

```scheme
;; any-every-fm.esk
(require core.list.higher_order)
(display (any even? '(1 3 4 5))) (newline)
(display (every even? '(2 4 6))) (newline)
(display (filter-map (lambda (x) (if (odd? x) (* x x) #f)) '(1 2 3))) (newline)
```
```
#t
#t
(1 9)
```

Edge cases (verified): `(filter-map fn '())` → `'()`;
`(filter-map (lambda (x) #f) '(1 2 3))` → `'()`;
`(any pred '())` → `#f`; `(every pred '())` → `#t`.

### Known issues — depth ceiling

`map1`/`map2`/`map3`, `fold-right`/`foldr`, and `filter-map` are all
non-tail-recursive (they `cons` or apply `proc` around the recursive call), so
they hit the same silent-SIGILL ceiling as `length`/`filter`
(see [`list_query.md`](list_query.md), ESH-0108). Observed on this machine:

| op | passes | SIGILLs (rc 132, no output) |
| --- | --- | --- |
| `map1` (with closure) | ~200k | 300,000 |
| `fold-right` | — | 250,000 |

The **left folds** `fold`/`fold-left`/`foldl` are tail-recursive and complete on
1,000,000-element lists (`(fold + 0 (iota 1000000))` → `499999500000`). Prefer
them, or reverse-then-cons, when processing very long lists.
