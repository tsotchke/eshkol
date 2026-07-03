# `core.sexp` — S-expression rendering (display + canonical/write form)

**Source**: [`lib/core/sexp.esk`](../../../lib/core/sexp.esk)
**Require**: `(require core.sexp)` — must be required individually (not auto-loaded by `(require stdlib)`).

A correct s-expression pretty-printer that handles proper lists, improper/dotted
lists, atoms (string, symbol, number, boolean, char, null), and vectors. It exists
because the naive user walk of an s-expression crashes with `cdr: argument is not
a pair` the moment the structure contains a dotted pair (alists are everywhere in
R7RS). It provides two renderings: a human `display`-style form and a `write`-style
**canonical** form suitable for hashing / on-disk storage. Unknown objects render
as `#<object>`.

Note: `#(...)` literals in Eshkol are homogeneous double *tensors*, not
heterogeneous vectors — build a heterogeneous vector with `(vector 1 "two" 'three)`.
Both renderers handle either.

## Functions

### `(sexp->string s)`
Display-style rendering: strings appear **without** surrounding quotes, chars
without the `#\` prefix. Use for human-readable output. Returns a string.

```scheme
;; sexp.esk
(require core.sexp)
(display (sexp->string '(a b c)))            (newline)
(display (sexp->string '(a . b)))            (newline)
(display (sexp->string '(mood . 0.7)))       (newline)
(display (sexp->string "hello"))             (newline)
(display (sexp->string #\x))                 (newline)
(display (sexp->string (vector 1 "two" 'three))) (newline)
(display (sexp->string '()))                 (newline)
```
```
(a b c)
(a . b)
(mood . 0.7)
hello
x
#(1 two three)
()
```

Edge cases: `'()` renders as `"()"`. A dotted pair renders `(a . b)`. Improper
list tails render with ` . ` (e.g. `(a (b c) . d)`).

### `(sexp->canonical-string s)`
Write-style **canonical** rendering: strings are wrapped in double-quotes and chars
prefixed with `#\`. This is the form whose bytes are hashed for content-addressing
in `core.memory` / `core.memory_store`, so it must be stable. Returns a string.

```scheme
;; sexp.esk
(require core.sexp)
(display (sexp->canonical-string '(a "b" c)))         (newline)
(display (sexp->canonical-string "hello"))            (newline)
(display (sexp->canonical-string '(mood . "sad")))    (newline)
(display (sexp->canonical-string (vector 1 "two" 'three))) (newline)
(display (sexp->canonical-string #t))                 (newline)
(display (sexp->canonical-string '(a (b c) . d)))     (newline)
```
```
(a "b" c)
"hello"
(mood . "sad")
#(1 "two" three)
#t
(a (b c) . d)
```

Edge cases:
- **No escaping.** The canonical renderer does *not* escape embedded double-quotes
  or newlines inside strings. Downstream durable stores (`core.memory_store`) must
  sanitize payload strings first (see `memory-store-sanitize`) or the on-disk line
  format breaks.
- Booleans render `#t` / `#f`; numbers via `number->string`; symbols via
  `symbol->string`; anything unrecognized renders `#<object>`.
