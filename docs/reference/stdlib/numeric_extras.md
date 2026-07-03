# `core.numeric_extras` — R7RS 6.2.6 numeric helpers

**Source**: [`lib/core/numeric_extras.esk`](../../../lib/core/numeric_extras.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.numeric_extras)`

Pure-Scheme implementations of R7RS §6.2.6 numeric procedures built on the existing numeric tower, so bignum inputs work transparently. Part of the R7RS final-mile work tracked in `.swarm/tasks/ESH-0003` ("R7RS final-mile: raise-continuable, error-object? family, exact-integer-sqrt, char-ci=? family").

## Functions

### `(exact-integer-sqrt k)`
For an exact non-negative integer `k`, returns **two values** `s` and `r` such that `k = s*s + r` and `k < (s+1)*(s+1)` — i.e. `s` is the integer square root and `r` the remainder. Uses Newton iteration; converges for fixnums and bignums alike. Because it returns multiple values, consume it with `call-with-values` (or `let-values`).

```scheme
;; numeric.esk
(require stdlib)
(define (show k)
  (call-with-values (lambda () (exact-integer-sqrt k))
    (lambda (s r)
      (display k) (display " -> s=") (display s)
      (display " r=") (display r) (newline))))
(show 0) (show 1) (show 2) (show 4) (show 5) (show 17) (show 100) (show 99)
```
```
0 -> s=0 r=0
1 -> s=1 r=0
2 -> s=1 r=1
4 -> s=2 r=0
5 -> s=2 r=1
17 -> s=4 r=1
100 -> s=10 r=0
99 -> s=9 r=18
```

Bignum input is handled exactly:

```scheme
(require stdlib)
(call-with-values (lambda () (exact-integer-sqrt (expt 10 50)))
  (lambda (s r) (display s) (display " r=") (display r) (newline)))
```
```
10000000000000000000000000 r=0
```

Edge cases: a negative argument raises an error (`"exact-integer-sqrt: argument must be non-negative"`). The result is exact; there is no rounding.
