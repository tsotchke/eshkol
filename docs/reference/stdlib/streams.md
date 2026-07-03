# `core.streams` — SRFI 41 lazy streams

**Source**: [`lib/core/streams.esk`](../../../lib/core/streams.esk)
**Require**: `(require core.streams)` — auto-loaded by `(require stdlib)`.
Internally depends on `core.list.transform`.

Lazy (possibly infinite) streams built on Eshkol's memoised `delay`/`force`
promises. A stream is either `stream-null` (the empty list) or a pair
`(head . tail-promise)`. Each tail is forced at most once, so repeated traversal
does not recompute.

**Important — no `stream-cons` macro.** Eshkol has no `define-syntax`, so SRFI 41's
implicit-delay macro form is unavailable. **The caller must wrap the tail in
`(delay …)` explicitly:**

```scheme
(define ones (stream-cons 1 (delay ones)))
(define (from n) (stream-cons n (delay (from (+ n 1)))))
```

The recursive reference is captured as a closure by `delay`'s lambda expansion,
so cyclic definitions like `ones` work.

## Constants

### `stream-null`
The empty stream. Value: `'()`.

## Predicates

### `(stream-null? s)` — `#t` iff `s` is the empty stream.
### `(stream-pair? s)` — `#t` iff `s` is a non-empty stream (a pair).
### `(stream? s)` — `#t` iff `s` is `'()` or a pair.

## Construction / access

### `(stream-cons head tail-promise)`
Builds a stream cell. `tail-promise` must already be a `(delay …)` (see above).

### `(stream-car s)` — the head element.
### `(stream-cdr s)` — forces and returns the tail stream.

## Indexing / slicing

### `(stream-take s n)`
Returns a **list** (not a stream) of the first `n` elements; terminates early on
`stream-null`.

### `(stream-drop s n)`
Returns the stream with the first `n` elements dropped (a stream).

### `(stream-ref s n)`
Returns the `n`th element (0-indexed).

```scheme
;; basics.esk
(require core.streams)
(define ones (stream-cons 1 (delay ones)))
(define (from n) (stream-cons n (delay (from (+ n 1)))))
(display (stream-take ones 5)) (newline)
(display (stream-take (from 10) 4)) (newline)
(display (stream-car (from 7))) (newline)
(display (stream-ref (stream-from 0) 5)) (newline)
(display (stream-take (stream-drop (stream-from 0) 10) 3)) (newline)
```
```
(1 1 1 1 1)
(10 11 12 13)
7
5
(10 11 12)
```

## Higher-order

### `(stream-map f s)`
Lazy map: returns a new stream of `(f x)` for each element.

### `(stream-filter pred s)`
Lazy filter: returns a new stream of the elements satisfying `pred`.

### `(stream-for-each f s)`
Strict: applies `f` to each element for effect, walking until `stream-null`.
Returns `'()`.

### `(stream-zip s1 s2)`
Lazy: stream of two-element lists `(a b)`; terminates when **either** input is
exhausted.

### `(stream-append s1 s2)`
Lazy: the elements of `s1` followed by `s2`.

```scheme
;; higher.esk
(require core.streams)
(display (stream-take (stream-map (lambda (x) (* x x)) (stream-from 1)) 5)) (newline)
(display (stream-take (stream-filter even? (stream-from 1)) 5)) (newline)
(display (stream-take (stream-zip (stream-from 0) (stream-from 100)) 3)) (newline)
(display (stream-take (stream-append (list->stream '(1 2)) (list->stream '(3 4))) 4)) (newline)
(stream-for-each (lambda (x) (display x) (display " ")) (list->stream '(a b c))) (newline)
```
```
(1 4 9 16 25)
(2 4 6 8 10)
((0 100) (1 101) (2 102))
(1 2 3 4)
a b c 
```

## Generators

### `(stream-iterate f x)`
Infinite stream `x, (f x), (f (f x)), …`.

### `(stream-from n)`
Infinite stream `n, n+1, n+2, …`.

### `(stream-take-while pred s)`
Returns a **stream** of the leading elements satisfying `pred`.

### `(stream-drop-while pred s)`
Returns the stream after dropping the leading elements satisfying `pred`.

```scheme
;; gen.esk
(require core.streams)
(display (stream-take (stream-iterate (lambda (x) (* x 2)) 1) 6)) (newline)
(display (stream->list (stream-take-while (lambda (x) (< x 5)) (stream-from 0)))) (newline)
(display (stream-take (stream-drop-while (lambda (x) (< x 5)) (stream-from 0)) 3)) (newline)
```
```
(1 2 4 8 16 32)
(0 1 2 3 4)
(5 6 7)
```

Note: `stream-take-while` returns a **stream**, so displaying it directly shows a
promise tail — `(display (stream-take-while … ))` prints `(0 . #<promise>)`. Wrap
it in `stream->list` (or `stream-take`) to realise the elements, as shown above.

## Conversions

### `(stream-length s)`
Number of elements. **Only terminates on finite streams** — calling it on an
infinite stream loops forever by design.

### `(stream->list s)`
Realises a finite stream into a list. Same caveat: infinite streams loop forever.

### `(list->stream lst)`
Converts a list into a (finite) stream.

```scheme
;; conv.esk
(require core.streams)
(display (stream->list (list->stream '(1 2 3)))) (newline)
(display (stream-length (list->stream '(a b c d)))) (newline)
```
```
(1 2 3)
4
```

Edge cases (verified): `(stream-take stream-null 3)` → `'()`;
`(stream-drop '() 3)` → `'()`; `(stream-null? stream-null)` → `#t`.

Depth note: `stream->list` and `stream-length` are non-tail-recursive
(`stream->list`) / tail-recursive (`stream-length`); realising a very long finite
stream via `stream->list` is subject to the same stdlib depth ceiling as
`length`/`filter` (see [`list_query.md`](list_query.md)). Prefer bounded
`stream-take`/`stream-ref` when the stream is large or infinite.
