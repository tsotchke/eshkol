# `core.cache` — bounded LRU cache and memoizers

**Source**: [`lib/core/cache.esk`](../../../lib/core/cache.esk)
**Require**: `(require core.cache)` — **NOT** auto-loaded via `stdlib`; require it explicitly.

A bounded least-recently-used cache plus memoizers built on it. Internally a cache is a vector `#(capacity counter-box hash-table)`; each entry is `(cons value timestamp)`. Reads and writes bump a monotonic counter; eviction walks the table for the lowest timestamp (O(n) evict, constant-time hit — fine for hundreds-to-thousands of entries).

## Functions

### `(make-lru-cache capacity)`
Create a cache bounded to `capacity` (a positive integer). Errors otherwise.

### `(lru-capacity cache)`
The configured maximum.

### `(lru-size cache)`
Current entry count.

### `(lru-has? cache key)`
`#t`/`#f`. Does **not** touch the timestamp.

### `(lru-get cache key)`
Return the cached value, or `#f` on miss. Bumps the entry's timestamp on a hit. Use `lru-get/default` when `#f` is a valid stored value.

### `(lru-get/default cache key default)`
Like `lru-get` but returns `default` on miss (still touches on hit).

### `(lru-set! cache key value)`
Insert or update `key`. On a new key at capacity, evicts the least-recently-used entry first.

### `(lru-delete! cache key)`
Remove `key`; returns `#t` if it was present, `#f` otherwise.

### `(lru-clear! cache)`
Drop all entries.

```scheme
;; cache.esk
(require core.cache)
(define c (make-lru-cache 2))
(lru-set! c 'a 1)
(lru-set! c 'b 2)
(display (lru-get c 'a)) (newline)      ; 1 (touches a, so b is now oldest)
(lru-set! c 'd 4)                        ; at capacity -> evict b
(display (lru-has? c 'b)) (newline)      ; #f
(display (lru-has? c 'a)) (newline)      ; #t
(display (lru-get c 'zzz)) (newline)     ; miss -> #f
(display (lru-get/default c 'zzz -1)) (newline) ; -1
(display (lru-size c)) (newline)         ; 2
(display (lru-delete! c 'a)) (newline)   ; #t
(display (lru-delete! c 'a)) (newline)   ; #f
```
```
1
#f
#t
#f
-1
2
#t
#f
```

Edge cases: `(make-lru-cache -1)` (or `0`, or a non-integer) → `Unhandled exception: make-lru-cache: capacity must be a positive integer`.

## Memoizers

Memoizers wrap a function and cache results in an LRU cache (default capacity 1024). Because hash-table keys compare by identity, variadic memoization is not supported; use the fixed-arity helpers below. (The correctness of a memoizer's captured mutable `cache` relates to the closure-capture fix tracked in **ESH-0074**; these helpers mutate the cache vector in place, which is safe.)

### `(memoize1 fn)` / `(memoize1/cap fn capacity)`
Memoize a **1-argument** function. Key is the single argument (compared with the cache's hash-table).

### `(memoize2 fn)` / `(memoize2/cap fn capacity)`
Memoize a **2-argument** function. The pair `(x y)` is encoded into a string key (`display` of each arg joined by a NUL byte) so structurally-equal argument pairs hit. Best for scalars: numbers, symbols, strings, booleans.

### `(memoize fn)` / `(memoize/cap fn capacity)`
Aliases for `memoize1` / `memoize1/cap`.

```scheme
;; cache.esk
(require core.cache)
(define calls 0)
(define f (memoize (lambda (x) (set! calls (+ calls 1)) (* x x))))
(display (f 5)) (newline)   ; 25
(display (f 5)) (newline)   ; 25 (cached; fn not re-run)
(display calls) (newline)   ; 1
(define g (memoize2 (lambda (a b) (+ a b))))
(display (g 3 4)) (newline) ; 7
(display (g 3 4)) (newline) ; 7 (cached)
```
```
25
25
1
7
7
```

Edge cases: `memoize2`'s key uses `display`, so two values that render identically (e.g. the string `"1"` and the number `1`) would collide. For mutable/complex arguments, pre-hash to a scalar yourself.
