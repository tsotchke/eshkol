# `core.list.search` — membership, association, and positional access

**Source**: [`lib/core/list/search.esk`](../../../lib/core/list/search.esk)
**Require**: `(require core.list.search)` — auto-loaded by `(require stdlib)`.

R7RS-style membership (`member`/`memq`/`memv`), association-list lookup
(`assoc`/`assq`/`assv`), and positional access (`list-ref`/`list-tail`/`list-set!`).
The `equal?`-style variants use an internal type-dispatching comparator
(`list-search-equal?`) that compares strings with `string=?`, numbers with `=`,
chars with `char=?`, symbols/booleans with `eq?`, and falls back to `equal?`.

## Functions

### `(member x lst)`
Returns the sublist beginning at the first element `equal?`-comparable to `x`,
or `#f` if not found.

### `(member? x lst)`
Boolean convenience wrapper: `#t` if `x` is present, else `#f`.

### `(memq x lst)` / `(memv x lst)`
Like `member` but compare with `eq?` (`memq`) or `eqv?` (`memv`).

```scheme
;; member.esk
(require core.list.search)
(display (member 3 '(1 2 3 4 5))) (newline)
(display (member 9 '(1 2 3))) (newline)
(display (member? 3 '(1 2 3))) (newline)
(display (memq 'c '(a b c d))) (newline)
(display (memv 2 '(1 2 3))) (newline)
```
```
(3 4 5)
#f
#t
(c d)
(2 3)
```

### `(assoc key alist)`
Returns the first `(key . …)` pair whose car is `equal?`-comparable to `key`, or
`#f`. `alist` is a list of pairs.

### `(assq key alist)` / `(assv key alist)`
Like `assoc` but compare keys with `eq?` (`assq`) or `eqv?` (`assv`).

```scheme
;; assoc.esk
(require core.list.search)
(display (assoc 'b '((a 1) (b 2) (c 3)))) (newline)
(display (assq 'c '((a 1) (b 2) (c 3)))) (newline)
(display (assv 2 '((1 one) (2 two)))) (newline)
```
```
(b 2)
(c 3)
(2 two)
```

### `(list-ref lst n)`
Returns the `n`th element (0-indexed). Note arg order: **list first, index
second**.

### `(list-tail lst n)`
Returns the sublist starting at index `n` (0-indexed). `n <= 0` returns the whole
list.

### `(list-set! lst n val)`
Mutates the `n`th pair's car to `val` in place (via `set-car!` on
`(list-tail lst n)`). Returns the result of `set-car!` (do not rely on it); the
effect is the mutation.

```scheme
;; positional.esk
(require core.list.search)
(display (list-ref '(a b c d) 2)) (newline)
(display (list-tail '(a b c d) 2)) (newline)
(define x (list 1 2 3))
(list-set! x 1 99)
(display x) (newline)
```
```
c
(c d)
(1 99 3)
```

Note: `list-set!` mutates, so its target must be a mutable list built with
`list`/`cons` (not shared with a quoted literal you also read elsewhere).

Edge cases: `member`/`memq`/`memv`/`assoc`/`assq`/`assv` return `#f` on `'()` or
when the key is absent. `list-ref`/`list-tail` are unguarded — indexing past the
end reduces to `(car '())` / `(cdr '())` (runtime error). All of these are
tail-recursive and safe on long lists.
