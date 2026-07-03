# `core.data.base64` — Base64 encoding/decoding

**Source**: [`lib/core/data/base64.esk`](../../../lib/core/data/base64.esk)
**Require**: `(require core.data.base64)` — auto-loaded via `(require stdlib)`.

Pure-Eshkol standard Base64 (RFC 4648 §4, `+`/`/` alphabet, `=` padding). The
byte representation is a Scheme **list of integers 0–255**; the text
representation is a string. URL-safe Base64 (`-`/`_`, no padding) lives in
[`core.url`](url.md) as `base64url-encode`/`base64url-decode`. This module is
also summarised in [`docs/STDLIB_V1_2_API.md`](../../STDLIB_V1_2_API.md).

## Functions

### `(base64-encode bytes)`
Encode a list of byte values (0–255) into a Base64 string, with `=` padding.

```scheme
(require core.data.base64)
(display (base64-encode '(77 97 110))) (newline)   ; "Man"
(display (base64-encode '(77 97)))     (newline)   ; "Ma"  -> one '='
(display (base64-encode '(77)))        (newline)   ; "M"   -> two '='
(display (base64-encode '()))          (newline)   ; empty
```
```
TWFu
TWE=
TQ==

```

### `(base64-decode str)`
Decode a Base64 string back into a list of byte values. Padding is stripped
first via `base64-remove-padding`.

```scheme
(display (base64-decode "TWFu")) (newline)
(display (base64-decode ""))     (newline)
```
```
(77 97 110)
()
```

### `(base64-encode-string str)`
Encode a string: `(base64-encode (string->bytes str))`.

```scheme
(display (base64-encode-string "Hello")) (newline)
```
```
SGVsbG8=
```

### `(base64-decode-string str)`
Decode a Base64 string to a string: `(bytes->string (base64-decode str))`.

```scheme
(display (base64-decode-string "SGVsbG8=")) (newline)
(display (base64-decode-string (base64-encode-string "The quick brown fox"))) (newline)
```
```
Hello
The quick brown fox
```

### `(string->bytes str)`
Convert a string to a list of its byte values (via `char->integer`).

```scheme
(display (string->bytes "AB")) (newline)   ; (65 66)
```
```
(65 66)
```

### `(bytes->string bytes)`
Convert a list of byte values to a string.

```scheme
(display (bytes->string '(65 66 67))) (newline)   ; "ABC"
```
```
ABC
```

### `(base64-char-at index)`
Return the alphabet character for a 6-bit value (0–63): `A`..`Z`, `a`..`z`,
`0`..`9`, `+`, `/`.

```scheme
(display (base64-char-at 0))  (newline)   ; A
(display (base64-char-at 62)) (newline)   ; +
```
```
A
+
```

### `(base64-value ch)`
Inverse of `base64-char-at`: the 6-bit value of a Base64 char. `=` and any
unrecognised char map to `0`.

```scheme
(display (base64-value #\A)) (newline)   ; 0
(display (base64-value #\/)) (newline)   ; 63
```
```
0
63
```

### `(base64-remove-padding str)`
Strip trailing `=` characters from a Base64 string.

```scheme
(display (base64-remove-padding "SGVsbG8=")) (newline)   ; "SGVsbG8"
```
```
SGVsbG8
```

### Internal helpers
These are `provide`d but are the recursive workhorses behind the public API;
you normally will not call them directly.

- `(string->bytes-helper str pos len)` — tail of `string->bytes`; collects bytes
  from `pos` to `len`. `(string->bytes-helper "Hi" 0 2)` ⇒ `(72 105)`.
- `(base64-encode-helper bytes result)` — accumulator loop behind
  `base64-encode`. `(base64-encode-helper '(77 97 110) "")` ⇒ `TWFu`.
- `(base64-decode-helper str pos len result)` — accumulator loop behind
  `base64-decode`. `(base64-decode-helper "TWFu" 0 4 '())` ⇒ `(77 97 110)`.

## Binary round-trip

Arbitrary bytes — including `0` (NUL) and the high range `253`–`255` — survive a
`base64-encode` → `base64-decode` round-trip at the byte-list level:

```scheme
(require core.data.base64)
(display (base64-decode (base64-encode '(0 1 2 253 254 255)))) (newline)
```
```
(0 1 2 253 254 255)
```
The NUL byte also survives an encode step from a byte list containing it
(`(base64-encode '(72 0 105))` ⇒ `SABp`, which decodes back to `(72 0 105)`).

No bugs observed in this module.
