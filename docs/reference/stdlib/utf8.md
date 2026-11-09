# `core.utf8` — strict UTF-8 validation

**Source**: [`lib/core/utf8.esk`](../../../lib/core/utf8.esk)
**Require**: `(require core.utf8)`.

Validation of bytevector ranges as UTF-8, shared by the wire-format libraries
([`core.msgpack`](msgpack.md), [`core.protobuf`](protobuf.md)). The runtime's
`utf8->string` is permissive; a format that declares a payload to be text
validates it here first. Accepted input is exactly the shortest-form encodings
of Unicode scalar values: overlong forms, lone or missing continuation bytes,
UTF-16 surrogates (U+D800..U+DFFF) and values above U+10FFFF are refused.

### `(utf8-require-valid who bv pos len)`
Return `#t` when bytes `[pos, pos+len)` of `bv` are valid UTF-8; otherwise
raise an error whose message starts with `who` (for example
`"msgpack: invalid UTF-8 continuation"`). The range must lie inside `bv`.

### `(utf8-valid-range? bv pos len)`
`#t` or `#f` for the same check, without raising.

```scheme
(require core.utf8)
(display (utf8-valid-range? (bytevector 206 187) 0 2)) (newline)   ; λ
(display (utf8-valid-range? (bytevector 192 128) 0 2)) (newline)   ; overlong NUL
(display (utf8-valid-range? (bytevector 237 160 128) 0 3)) (newline) ; surrogate
```
```
#t
#f
#f
```
