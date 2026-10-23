# `core.protobuf` — Protocol Buffers (proto3)

**Source**: [`lib/core/protobuf.esk`](../../../lib/core/protobuf.esk)
**Require**: `(require core.protobuf)`.

Schema-driven [Protocol Buffers](https://protobuf.dev/) encoding and decoding
for Eshkol values in the binary wire format
([encoding spec](https://protobuf.dev/programming-guides/encoding/)). A schema
is plain Eshkol data, written by hand or produced from `.proto` text by the
built-in proto3 parser; `protobuf-schema` compiles it, and
`protobuf-encoder`/`protobuf-decoder` return the encoder and decoder closures
for one message type. Encoded output is a **bytevector**. The library is pure
Eshkol and behaves identically on JIT, AOT, the bytecode VM and ESKB.

The native wire primitives used by the ONNX checkpoint writer and the PJRT
compile-options encoder live in
[`inc/eshkol/core/protobuf_wire.h`](../../../inc/eshkol/core/protobuf_wire.h);
both layers are checked against the same hand-derived vectors.

## Schemas

A schema is a list of `(message NAME decl ...)` and `(enum NAME (VALUE n) ...)`
forms. Field declarations:

| Form | Meaning |
|---|---|
| `(field N name TYPE)` | singular field, implicit presence (defaults are not written) |
| `(field N name (optional TYPE))` | explicit presence: written whenever set, even to its default |
| `(field N name (repeated TYPE))` | repeated; numeric scalars and enums are packed |
| `(field N name (repeated TYPE unpacked))` | repeated, one tag per element |
| `(field N name (map KEY VALUE))` | map; KEY is an integer type, `bool` or `string` |
| `(oneof group (field ...) ...)` | at most one member set; members have explicit presence |
| `(reserved ...)` | accepted and ignored by the compiler |

`TYPE` is a scalar (`double float int32 int64 uint32 uint64 sint32 sint64
fixed32 fixed64 sfixed32 sfixed64 bool string bytes`), `(message M)` or
`(enum E)`. A singular message field always has explicit presence. Field numbers
must lie in 1..2^29-1 outside 19000..19999; enums must start with a zero value.

### `(protobuf-schema forms)` / `(protobuf-schema? value)`
Compile schema forms, checking field numbers, duplicate names and numbers, map
key types and that every referenced message and enum exists. Errors are
raised with `error`.

```scheme
(require core.protobuf)
(define schema
  (protobuf-schema
    '((message Person
        (field 1 name string)
        (field 2 id int32)
        (field 3 email (optional string))
        (field 4 scores (repeated sint32))
        (field 5 phones (repeated (message Phone)))
        (field 6 kind (enum Kind))
        (field 7 labels (map string int64))
        (oneof contact (field 8 home string) (field 9 work string)))
      (message Phone (field 1 number string))
      (enum Kind (KIND_UNSPECIFIED 0) (PERSONAL 1)))))
```

### `(protobuf-parse-proto text)` / `(protobuf-schema-from-proto text)`
Parse proto3 source text into schema forms (`protobuf-parse-proto`), or parse
and compile in one step. Accepted: `syntax = "proto3";`, `package`, `option`
(skipped), messages with nested messages and enums, scalar/message/enum fields,
`repeated`, `optional`, `map<K, V>`, `oneof`, `reserved` (fields using a
reserved number or name are refused), field options including
`[packed = false]`, and enums with options and aliases. `service` blocks are
skipped. Nested types are named `Outer.Inner`; references resolve from the
innermost scope outward, and `.package.Type` names resolve against the file's
package. `import`, proto2 files and proto2-only constructs (`required`,
`extensions`, `extend`, `group`) are refused with an error naming the
construct.

```scheme
(display (protobuf-parse-proto
  "syntax = \"proto3\"; message A { repeated int32 v = 1 [packed = false]; map<string, A> kids = 2; }"))
(newline)
```
```
((message A (field 1 v (repeated int32 unpacked)) (field 2 kids (map string (message A)))))
```

## Messages

### `(protobuf-message type entries)` / `(protobuf-message? value)`
A message value: its type name and a list of `(field-name value)` entries.
Values: exact integers for integer types (range-checked on encoding), `#t`/`#f`
for `bool`, reals for `float`/`double`, strings (valid UTF-8) for `string`,
bytevectors for `bytes`, an enum value name or its number for enums, a
`protobuf-message` for message fields, a list for repeated fields, and a list
of `(key value)` lists for maps.

### `(protobuf-message-type m)` / `(protobuf-message-entries m)` / `(protobuf-message-unknown m)`
Accessors. `protobuf-message-unknown` is the bytevector of fields the schema
does not describe, kept verbatim from decoding.

### `(protobuf-ref schema m name)` / `(protobuf-has? schema m name)`
`protobuf-ref` returns the field's value or its proto3 default (`0`, `0.0`,
`#f`, `""`, an empty bytevector, the zero enum value, `'()` for repeated and
map fields, an empty message for message fields). `protobuf-has?` is `#t` for a
set explicit-presence field, a non-default implicit field, or a non-empty
repeated or map field.

## Encoding / decoding

### `(protobuf-encode schema type m)` / `(protobuf-encoder schema type)`
Encode a message to a bytevector; `protobuf-encoder` returns the encoder
closure for `type`. Fields are written in field-number order, then the
message's unknown fields. Implicit-presence fields holding their default are
omitted (a `-0.0` float is written, since its bit pattern is non-zero); map
entries always carry both key and value. A value outside the field's range, a
value of another kind, an unknown field name, a field given twice, two
members of one oneof, or a message whose type does not match the field raises an error.

### `(protobuf-decode schema type bv)` / `(protobuf-decoder schema type)`
Decode a bytevector holding one message of `type`. The result holds every
implicit-presence, repeated and map field (defaults filled in) and the
explicit-presence fields that were present, in field-number order. Decoding
follows proto3 rules: the last value of a repeated singular scalar wins, a
repeated message field merges, repeated scalar fields accept packed and
unpacked elements in any mix, a repeated map key keeps its first position and
takes the last value, the last oneof member on the wire wins, enum numbers
without a declared name stay integers, and unknown fields (including a known
field arriving with another wire type) are kept byte-for-byte.

```scheme
(define (bytes bv)
  (let loop ((i (- (bytevector-length bv) 1)) (acc '()))
    (if (< i 0) acc (loop (- i 1) (cons (bytevector-u8-ref bv i) acc)))))
(define ada (protobuf-message 'Person '((name "Ada") (id 150) (scores (-1 2)) (kind PERSONAL))))
(define wire (protobuf-encode schema 'Person ada))
(display (bytes wire)) (newline)
(define back (protobuf-decode schema 'Person wire))
(display (protobuf-message-entries back)) (newline)
(display (protobuf-has? schema back 'email)) (newline)
```
```
(10 3 65 100 97 16 150 1 34 2 1 4 48 1)
((name Ada) (id 150) (scores (-1 2)) (phones ()) (kind PERSONAL) (labels ()))
#f
```

Unknown fields survive a decode and re-encode unchanged:

```scheme
(define kept (protobuf-decode schema 'Phone (bytevector 10 1 55 16 5)))
(display (bytes (protobuf-message-unknown kept))) (newline)
(display (bytes (protobuf-encode schema 'Phone kept))) (newline)
```
```
(16 5)
(10 1 55 16 5)
```

Generated codecs from `.proto` text:

```scheme
(define book (protobuf-schema-from-proto "
syntax = \"proto3\";
package tutorial;
message Person {
  string name = 1;
  int32 id = 2;
  message PhoneNumber { string number = 1; }
  repeated PhoneNumber phones = 4;
}
"))
(define encode-person (protobuf-encoder book 'Person))
(define decode-person (protobuf-decoder book 'Person))
(define p (protobuf-message 'Person
  (list '(name "Ada")
        (list 'phones (list (protobuf-message 'Person.PhoneNumber '((number "555"))))))))
(display (bytes (encode-person p))) (newline)
(display (protobuf-ref book (car (protobuf-ref book (decode-person (encode-person p)) 'phones)) 'number))
(newline)
```
```
(10 3 65 100 97 34 5 10 3 53 53 53)
555
```

### Malformed input
Every read is bounds-checked against the input; nothing past its end is read.
These raise a catchable error: a truncated varint, fixed-width value or
length-delimited payload; a varint longer than ten bytes or above 2^64; field
number 0 or above 2^29-1; wire types 6 and 7; an end-group without its
start-group or with another field number; invalid UTF-8 in a `string`
(overlong forms, surrogates and values above U+10FFFF included); a packed
fixed-width payload that is not a whole number of values; and nesting deeper
than 100 messages or groups.

```scheme
(display (guard (e (#t (error-object-message e)))
  (protobuf-decode schema 'Phone (bytevector 10 5 55))))
(newline)
```
```
protobuf-decode: length exceeds the input
```

### `(protobuf-enum-number schema enum name)` / `(protobuf-enum-name schema enum number)`
Translate between enum value names and numbers. `protobuf-enum-name` returns
the first name declared for a number (aliases share a number), or the number
itself when none is declared.

```scheme
(display (protobuf-enum-number schema 'Kind 'PERSONAL)) (newline)
(display (protobuf-enum-name schema 'Kind 1)) (newline)
```
```
1
PERSONAL
```

## Floating point
`double` fields carry the exact IEEE-754 binary64 bit pattern. `float` fields
round to the nearest binary32 value, ties to even, with subnormals, overflow to
infinity and signed zero handled. NaN is written as the canonical quiet NaN
(`0x7FF8000000000000` / `0x7FC00000`).

## Validation
`python3 scripts/run_wire_format_gate.py --suite protobuf --all-routes`
runs `tests/v1_2_edge_cases/protobuf_test.esk` (190 checks: the encoding
spec's worked examples, every scalar type at its boundaries, IEEE patterns,
presence, packed/unpacked repeated, maps, oneof, merging, large payloads, unknown-field
preservation, malformed input and schema refusals) and
`tests/v1_2_edge_cases/protobuf_proto_test.esk` (31 checks: `.proto` parsing,
scoping, generated codecs and parser refusals) on JIT, O1 AOT, the VM from
source and emitted ESKB. With `--native-test build/protobuf_wire_test` it also
runs the native wire-primitive golden vectors (`tests/core/protobuf_wire_test.c`).
Every expected byte is derived by hand from the encoding specification; no
`protoc` is needed.
