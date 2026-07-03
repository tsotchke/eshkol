# `core.list.compound` — compound `car`/`cdr` accessors and positional selectors

**Source**: [`lib/core/list/compound.esk`](../../../lib/core/list/compound.esk)
**Require**: `(require core.list.compound)` — auto-loaded by `(require stdlib)`.

Simple compositions of `car`/`cdr`. Every two-, three-, and four-level `c[ad]+r`
combination is provided, plus the ordinal selectors `first` … `tenth`. Reading
right-to-left, each `a` means `car` and each `d` means `cdr` (e.g. `caddr` =
`(car (cdr (cdr x)))`).

## Functions

### `(caar x)` `(cadr x)` `(cdar x)` `(cddr x)`
Two-level accessors. `cadr` is "second element", `cddr` is "the list past the
first two".

### `(caaar x)` … `(cdddr x)`
All eight three-level combinations (read right-to-left):
`caaar` `(car (car (car x)))`, `caadr` `(car (car (cdr x)))`,
`cadar` `(car (cdr (car x)))`, `caddr` `(car (cdr (cdr x)))`,
`cdaar` `(cdr (car (car x)))`, `cdadr` `(cdr (car (cdr x)))`,
`cddar` `(cdr (cdr (car x)))`, `cdddr` `(cdr (cdr (cdr x)))`.

```scheme
;; three-level.esk
(require core.list.compound)
(define x '(((1 2) (3 4)) ((5 6) (7 8)) ((9 10) (11 12)) ((13 14) (15 16))))
(display (cdaar x)) (newline)
(display (caadr x)) (newline)
(display (cadar x)) (newline)
(display (cdadr x)) (newline)
(display (cddar x)) (newline)
```
```
(2)
(5 6)
(3 4)
((7 8))
()
```

### `(caaaar x)` … `(cddddr x)`
All sixteen four-level combinations:
`caaaar` `caaadr` `caadar` `caaddr` `cadaar` `cadadr` `caddar` `cadddr`
`cdaaar` `cdaadr` `cdadar` `cdaddr` `cddaar` `cddadr` `cdddar` `cddddr` —
each expands to the corresponding four-deep `car`/`cdr` chain, rightmost
letter applied first.

```scheme
;; four-level.esk
(require core.list.compound)
(define x '(((1 2) (3 4)) ((5 6) (7 8)) ((9 10) (11 12)) ((13 14) (15 16))))
(define y '((a b c d) (e f g h) (i j k l) (m n o p)))
(display (caaadr x)) (newline)
(display (cadadr x)) (newline)
(display (cdaadr x)) (newline)
(display (cddadr x)) (newline)
(display (cdadar x)) (newline)
(display (cdaddr y)) (newline)
(display (caaddr y)) (newline)
(display (cdddar y)) (newline)
(display (cddddr y)) (newline)
```
```
5
(7 8)
(6)
()
(4)
(j k l)
i
(d)
()
```

```scheme
;; compound.esk
(require core.list.compound)
(define nested '((1 2) (3 4) (5 6)))
(define deep '(((a b) (c d)) ((e f) (g h))))
(define lst '(10 20 30 40 50 60 70 80 90 100))
(display (caar nested)) (newline)     ; car of car
(display (cadr lst)) (newline)        ; second
(display (cddr nested)) (newline)     ; drop first two
(display (caddr lst)) (newline)       ; third
(display (cadddr lst)) (newline)      ; fourth
(display (caaar deep)) (newline)      ; car^3
```
```
1
20
((5 6))
30
40
a
```

### `(first x)` `(second x)` `(third x)` `(fourth x)` `(fifth x)` `(sixth x)` `(seventh x)` `(eighth x)` `(ninth x)` `(tenth x)`
Ordinal element selectors (1-based). `first` = `car`, `second` = `cadr`, …,
`tenth` returns the tenth element. Implemented as compositions, so `fifth`
onward chain through `cddddr`.

```scheme
;; ordinals.esk
(require core.list.compound)
(define lst '(10 20 30 40 50 60 70 80 90 100))
(display (first lst)) (display " ") (display (second lst)) (display " ") (display (third lst)) (newline)
(display (fourth lst)) (display " ") (display (fifth lst)) (display " ") (display (sixth lst)) (newline)
(display (seventh lst)) (display " ") (display (eighth lst)) (display " ") (display (ninth lst)) (display " ") (display (tenth lst)) (newline)
```
```
10 20 30
40 50 60
70 80 90 100
```

Edge cases: these are unguarded compositions of `car`/`cdr`. Applying an
accessor deeper than the structure allows (e.g. `(caddr '(1))` or
`(fifth '(1 2 3))`) reduces to taking `car`/`cdr` of `'()`, which is a runtime
error in the underlying `car`/`cdr` — there is no bounds checking here.

### Known issues

Taking `cdr` of a **non-pair atom** mid-chain crashes with SIGSEGV rather
than a clean runtime error. Verified repro:

```scheme
(require core.list.compound)
(define x '((1 2 3) (4 5 6)))
(display (cdaar x))   ; caar = 1 (an integer); cdr of 1 → SIGSEGV
```
```
[Eshkol] fatal signal: SIGSEGV (segmentation fault) — terminating; output above is what made it to stdout before the crash
```

This is the general car/cdr-of-non-pair behavior, not specific to this
module; deep accessors just make it easy to hit.
