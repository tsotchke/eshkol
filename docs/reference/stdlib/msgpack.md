# `core.msgpack` — MessagePack wire-format substrate

**Source**: [`lib/core/msgpack.esk`](../../../lib/core/msgpack.esk)
**Require**: `(require core.msgpack)` — auto-loaded via `(require stdlib)`.

Pure-Eshkol [MessagePack](https://msgpack.org/) encoder/decoder covering the
deterministic value subset used by the distributed/CRDT track: `null`, booleans,
exact integers across the signed/unsigned 8/16/32-bit encodings, UTF-8 strings,
binary bytevectors, list arrays, and explicit maps. Encoded output is a
**bytevector**. No bugs were observed.

Because a Scheme list already means "array", maps use a distinct wrapper value
(`msgpack-map`) so they are not confused with arrays.

## Value constructors / predicates

### `(msgpack-null)` / `(msgpack-null? value)`
The MessagePack null sentinel (the symbol `msgpack-null`) and its predicate.

```scheme
(require core.msgpack)
(display (msgpack-null)) (newline)                 ; msgpack-null
(display (msgpack-null? (msgpack-null))) (newline) ; #t
(display (msgpack-null? 5)) (newline)              ; #f
```
```
msgpack-null
#t
#f
```

### `(msgpack-map entries)` / `(msgpack-map? value)` / `(msgpack-map-entries value)`
`msgpack-map` wraps a list of `(key value)` two-element lists into a map value.
`msgpack-map?` recognises it; `msgpack-map-entries` extracts the entry list
(errors on a non-map).

```scheme
(define m (msgpack-map (list (list "a" 1) (list "b" 2))))
(display (msgpack-map? m)) (newline)               ; #t
(display (msgpack-map-entries m)) (newline)        ; ((a 1) (b 2))
```
```
#t
((a 1) (b 2))
```

## Encoding / decoding

### `(msgpack-encode value)`
Encode a value to a **bytevector**. Accepts `msgpack-null`, booleans, integers
(in the supported 32-bit signed/unsigned range), strings, bytevectors, lists
(arrays), and `msgpack-map` values. Integers outside range, or unsupported
types, raise an error.

### `(msgpack-decode bv)`
Decode a bytevector holding exactly one object; errors if trailing bytes remain.

```scheme
(display (msgpack-decode (msgpack-encode 42))) (newline)
(display (msgpack-decode (msgpack-encode -5))) (newline)
(display (msgpack-decode (msgpack-encode 100000))) (newline)
(display (msgpack-decode (msgpack-encode -100000))) (newline)
(display (msgpack-decode (msgpack-encode #t))) (newline)
(display (msgpack-decode (msgpack-encode "hello"))) (newline)
(display (msgpack-decode (msgpack-encode '(1 2 3)))) (newline)
```
```
42
-5
100000
-100000
#t
hello
(1 2 3)
```

Maps and the null sentinel round-trip too:

```scheme
(define m (msgpack-map (list (list "a" 1) (list "b" 2))))
(define dm (msgpack-decode (msgpack-encode m)))
(display (msgpack-map? dm)) (newline)                       ; #t
(display (msgpack-map-entries dm)) (newline)                ; ((a 1) (b 2))
(display (msgpack-null? (msgpack-decode (msgpack-encode (msgpack-null))))) (newline)  ; #t
```
```
#t
((a 1) (b 2))
#t
```

### `(msgpack-decode-prefix bv)`
Decode the first object from a bytevector and return `(value next-position)`
without requiring the whole buffer to be consumed — for streaming/framed input.

```scheme
(display (msgpack-decode-prefix (msgpack-encode 7))) (newline)   ; (7 1)
```
```
(7 1)
```

## Bytevector <-> byte-list conversion

### `(msgpack-bytes->bytevector bytes)` / `(msgpack-bytevector->bytes bv)`
Convert between a list of byte integers (0–255) and a bytevector. Handy for
inspecting the wire format. `msgpack-bytes->bytevector` errors on any element
outside `[0,255]`.

```scheme
(display (bytevector? (msgpack-bytes->bytevector '(1 2 3)))) (newline)       ; #t
(display (msgpack-bytevector->bytes (msgpack-encode 42))) (newline)         ; (42) -> positive fixint
(display (msgpack-bytevector->bytes (msgpack-encode "ab"))) (newline)       ; (162 97 98) -> fixstr
```
```
#t
(42)
(162 97 98)
```

## Binary round-trip

Bytevectors — including NUL (`0`) and high bytes — survive encode/decode:

```scheme
(display (msgpack-bytevector->bytes
          (msgpack-decode (msgpack-encode (msgpack-bytes->bytevector '(0 255 128)))))) (newline)
```
```
(0 255 128)
```

### Internal helpers
Not part of the public API but present in the source (wire-format primitives):
`msgpack-require-byte`, `msgpack-require-available`, `msgpack-u16-bytes`,
`msgpack-u32-bytes`, `msgpack-read-u16`, `msgpack-read-u32`, `msgpack-encode-int`,
`msgpack-encode-bin-bytes`, `msgpack-encode-string`, `msgpack-encode-array`,
`msgpack-encode-map-value`, `msgpack-encode-value`, `msgpack-decode-at`,
`msgpack-decode-bin`, `msgpack-decode-string`, `msgpack-decode-array-elements`,
`msgpack-decode-map-entries`.
