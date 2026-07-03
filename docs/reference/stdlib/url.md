# `core.url` — URL percent-encoding and base64url

**Source**: [`lib/core/url.esk`](../../../lib/core/url.esk)
**Require**: `(require core.url)` — auto-loaded via `(require stdlib)`.

RFC 3986 percent-encoding for URL components, plus URL-safe Base64 (RFC 4648 §5).
Non-ASCII codepoints are encoded as their UTF-8 byte sequence. The `base64url-*`
functions build on [`core.data.base64`](data_base64.md); this module is also
summarised in [`docs/STDLIB_V1_2_API.md`](../../STDLIB_V1_2_API.md). No bugs were
observed.

## Functions

### `(url-encode str)`
Percent-encode every byte except the RFC 3986 unreserved set
`[A-Za-z0-9-_.~]`. Hex digits are uppercase. Multi-byte UTF-8 characters are
encoded byte-by-byte.

```scheme
(require core.url)
(display (url-encode "hello world")) (newline)
(display (url-encode "abcXYZ012-_.~")) (newline)     ; unreserved, unchanged
(display (url-encode "a+b&c=d/e?f")) (newline)
(display (url-encode "café")) (newline)              ; UTF-8
```
```
hello%20world
abcXYZ012-_.~
a%2Bb%26c%3Dd%2Fe%3Ff
caf%C3%A9
```
Edge case: `(url-encode "")` ⇒ `""`.

### `(url-decode str)`
Reverse of `url-encode`. Each `%XX` triple becomes the byte `0xXX`; a literal
`+` decodes to a space (legacy form-encoding); everything else passes through.
UTF-8 byte sequences are re-combined into codepoints.

```scheme
(display (url-decode "hello%20world")) (newline)
(display (url-decode "a+b")) (newline)               ; '+' -> space
(display (url-decode (url-encode "key=value&x=1 2"))) (newline)   ; round-trip
(display (url-decode (url-encode "café"))) (newline)
```
```
hello world
a b
key=value&x=1 2
café
```

### `(base64url-encode str)`
Encode a string as URL-safe Base64: standard Base64 with `+`→`-`, `/`→`_`, and
trailing `=` padding stripped. Used for JWTs, CSRF tokens, and hashes carried in
query strings or path segments.

```scheme
(display (base64url-encode "Hello")) (newline)          ; no padding
(display (base64url-encode "subjects?_d=1")) (newline)  ; note '_' for '/'
(display (base64url-encode "Ma")) (newline)             ; would be "TWE=" in std
```
```
SGVsbG8
c3ViamVjdHM_X2Q9MQ
TWE
```

### `(base64url-decode str)`
Reverse of `base64url-encode`. Restores padding and the `-`→`+`, `_`→`/`
substitutions, then delegates to `base64-decode-string`. Accepts both the
canonical unpadded form and a padded form.

```scheme
(display (base64url-decode (base64url-encode "The quick brown fox jumps"))) (newline)
```
```
The quick brown fox jumps
```

### Internal helpers
Not part of the public surface but present in the source: `byte->hex2`,
`url-unreserved?`, `encode-codepoint`, `hex-digit?`, `hex-digit->int`,
`url-read-byte`, `utf8-combine`, `b64url-strip-padding`,
`b64url-restore-padding`, `b64url-replace-encode`, `b64url-replace-decode`.
These implement the byte-level encoding/decoding and UTF-8 recombination behind
the four exported functions; they are not `provide`d for direct use.
