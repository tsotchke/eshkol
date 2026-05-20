# Eshkol v1.2-scale Standard Library — API Reference

**Version**: 1.2.1-scale (closeout 2026-05-20, base 2026-05-01)
**Audience**: implementers and library authors writing against the v1.2-scale
public stdlib surface.
**Scope**: every module added or significantly expanded between v1.1.13-accelerate
and v1.2.1-scale.

This document is the canonical, source-verified reference for the new public
surfaces shipped in v1.2-scale. Every signature, default, and edge case below
is read directly from the implementing `.esk`, `.c`, or `.cpp` file; every
example exercises only symbols listed in the corresponding `(provide …)` block
or in the codegen builtin table. Where a feature listed in
[`CHANGELOG.md`](../CHANGELOG.md) under the v1.2 "finalises the v1.2 stdlib"
line does not exist under its CHANGELOG name, the actual file and module name
are given inline.

---

## Table of contents

1. [`core.json_schema`](#1-corejson_schema) — JSON Schema Draft 7 subset
   validator
2. [`core.reflection`](#2-corereflection) — `procedure-arity`, `type-name`,
   `describe`
3. [Time API (codegen builtins)](#3-time-api-codegen-builtins) — ISO 8601
   format/parse, monotonic clocks, timezone offset
4. [`lib/agent/regex.esk`](#4-libagentregexesk) — PCRE2 regex with capture
   groups
5. [`core.cache`](#5-corecache) — LRU cache + `memoize` family
6. [PRNG: `make-prng` / `prng-random` (codegen builtins) + `lib/random/random.esk`](#6-prng-make-prng--prng-random-codegen-builtins--librandomrandomesk) —
   seedable, per-stream PRNG isolation
7. [`core.streams`](#7-corestreams) — SRFI 41 lazy streams
8. [`core.argparse`](#8-coreargparse) — CLI argument parser
9. [`core.url`](#9-coreurl) — RFC 3986 percent-encoding + `url-parse`
   (system builtin)
10. [`core.data.base64` — base64url variants](#10-coredatabase64--base64url-variants)
11. [`call-with-values` finalisation (codegen builtin)](#11-call-with-values-finalisation-codegen-builtin)
12. [Cross-references and module loading](#12-cross-references-and-module-loading)

---

## Mapping from the CHANGELOG headline to the on-disk implementation

The v1.2.0-scale release note reads:

> finalises the v1.2 stdlib (json_schema, reflection, time API, regex
> capture groups, memoization, PRNG seeding, lazy streams)

The actual layout is:

| CHANGELOG label              | Implementing file(s)                                                              | Module require line                                |
|------------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------|
| `core.json_schema` validator | [`lib/core/json_schema.esk`](../lib/core/json_schema.esk)                         | `(require core.json_schema)` (auto via `stdlib`)   |
| Reflection                   | [`lib/core/reflection.esk`](../lib/core/reflection.esk) + codegen `procedure-arity`| `(require core.reflection)` (auto via `stdlib`)    |
| Time API                     | [`lib/core/system_builtins.c`](../lib/core/system_builtins.c) (no `.esk` wrapper) | none — names are codegen builtins                  |
| Regex capture groups         | [`lib/agent/regex.esk`](../lib/agent/regex.esk) + `lib/agent/c/agent_regex.c`     | `(load "lib/agent/regex.esk")` (force-loaded in JIT)|
| Memoization / cache          | [`lib/core/cache.esk`](../lib/core/cache.esk)                                     | `(require core.cache)` — **NOT** auto-loaded       |
| PRNG seeding                 | [`lib/core/prng.cpp`](../lib/core/prng.cpp) + `lib/random/random.esk`             | codegen builtins; `(require random.random)` for high-level wrappers |
| Lazy streams                 | [`lib/core/streams.esk`](../lib/core/streams.esk)                                 | `(require core.streams)` (auto via `stdlib`)       |
| Argparse                     | [`lib/core/argparse.esk`](../lib/core/argparse.esk)                               | `(require core.argparse)` — **NOT** auto-loaded    |
| URL                          | [`lib/core/url.esk`](../lib/core/url.esk) + system builtin `url-parse`            | `(require core.url)` (auto via `stdlib`)           |
| base64 / base64url           | [`lib/core/data/base64.esk`](../lib/core/data/base64.esk) + `core.url` re-exports | `(require core.data.base64)` / `(require core.url)`|
| `call-with-values`           | codegen builtin (`ESHKOL_CALL_WITH_VALUES_OP` in `parser.cpp` and `llvm_codegen.cpp`) | none — special form                                |

Modules from the brief that have *no separate file* in `lib/core/`:

- **Time API** is **not** a `core.time` module — it ships as codegen builtins
  registered in [`lib/core/system_builtins.c`](../lib/core/system_builtins.c)
  with the C trampolines around lines 313-561 and the LLVM dispatch around
  line 12124 of `lib/backend/llvm_codegen.cpp`. Documented in section 3.
- **`core.regex`** is **not** in `lib/core/`. The capture-group-capable regex
  wrapper lives at [`lib/agent/regex.esk`](../lib/agent/regex.esk) wrapping
  `lib/agent/c/agent_regex.c` (PCRE2). Documented in section 4.
- **`core.random`** is **not** in `lib/core/`. The high-level RNG wrappers
  live at [`lib/random/random.esk`](../lib/random/random.esk); the
  per-stream isolation primitives (`make-prng`, `prng-random`,
  `prng-random-integer`, `prng?`) are codegen builtins backed by
  [`lib/core/prng.cpp`](../lib/core/prng.cpp). Documented in section 6.
- **`core.control` (`call-with-values` finalisation)**: there *is* a
  `lib/core/control/trampoline.esk`, but `call-with-values` itself is not in
  it — it is a parser-recognised special form with codegen in
  `llvm_codegen.cpp:17684`. The v1.2 work was the named-consumer
  resolution fix listed in `CHANGELOG.md:758`. Documented in section 11.

---

## 1. `core.json_schema`

**Source**: [`lib/core/json_schema.esk`](../lib/core/json_schema.esk).
**Auto-loaded**: yes, via `(require stdlib)`.
**Direct import**: `(require core.json_schema)`.

A Draft-7-subset JSON Schema validator that operates on the output of
`(json-parse …)`. The schema document is itself a parsed-JSON value
(`hash-table` for objects, list for arrays, etc.). The grammar of supported
keywords is documented at the head of the source file (lines 6-23).

### Provided symbols

```scheme
(provide json-schema-valid? json-schema-validate)
```

(Source: `lib/core/json_schema.esk:39`. Only two symbols are exported —
predicate and error-list variant.)

### Supported keywords (verbatim from source lines 6-23)

| Keyword                | Applies to                | Notes                                                                          |
|------------------------|---------------------------|--------------------------------------------------------------------------------|
| `type`                 | any                       | string or list of strings                                                      |
| `properties`           | object                    | `{name → subschema}`                                                           |
| `required`             | object                    | list of property name strings                                                  |
| `additionalProperties` | object                    | `#f` rejects extras; default is allowed                                        |
| `items`                | array                     | subschema applied to every element                                             |
| `minItems` / `maxItems`| array                     | integer bound                                                                  |
| `minLength` / `maxLength` | string                 | integer bound on `string-length`                                               |
| `minimum` / `maximum`  | number / integer          | inclusive bound                                                                |
| `exclusiveMinimum` / `exclusiveMaximum` | number / integer | strict bound                                                                |
| `enum`                 | any                       | `equal?` match against any element                                             |
| `const`                | any                       | `equal?` match                                                                 |
| `pattern`              | string                    | **substring containment** only — no full regex (deliberately so `core.json_schema` does not depend on PCRE2) |
| `oneOf`                | any                       | exactly one subschema must validate                                            |
| `anyOf`                | any                       | at least one must validate                                                     |
| `allOf`                | any                       | every subschema must validate                                                  |
| `not`                  | any                       | subschema must NOT validate                                                    |

A schema of `#t` accepts every value; a schema of `#f` rejects every value
(`json_schema.esk:116-117`). Both cases were added per Draft 6/7 boolean
shorthand.

### `(json-schema-valid? schema value)` → boolean

Predicate form. Returns `#t` iff `value` validates against `schema`. Implemented
in terms of `json-schema-validate` (line 364-365):

```scheme
(define (json-schema-valid? schema value)
  (null? (json-schema-validate schema value)))
```

**Edge cases**:

- A non-`hash-table` schema (other than `#t`/`#f`) passes through as valid
  (`json_schema.esk:118`). This is deliberate so that callers can pass `'()`
  for "no constraint".
- An empty array (`'()`) is treated as type `"array"` by `jsv-matches-type?`
  (line 65, 86) — i.e. `(json-schema-valid? '#hash(("type" . "array")) '())`
  is `#t`.

### `(json-schema-validate schema value)` → list of strings

Returns the full error list. Each error string carries a JSON-pointer-style
path joined by `/` (`json_schema.esk:103-104`):

```
"/users/0/age: expected integer, got string"
```

Empty list means valid. The path begins empty at the root and is extended via
`path-extend` at each property or array index.

**Edge cases**:

- The implementation distinguishes `hash-table` and `list` value types so a
  string is never misclassified as an array even though `(list? x)` is
  permissive on heap pointers (`json_schema.esk:55-64`, fix logged in
  `CHANGELOG.md:377-382`).
- `validate-array-items` walks the cons chain recursively (line 298). For a
  long list (>~10⁴ elements) this is `O(n)` stack frames; the validator does
  not use a trampoline.

### Worked example: a user-record schema

```scheme
(require stdlib)                               ; auto-loads core.json_schema

(define user-schema
  (json-parse
    "{\"type\": \"object\",
      \"properties\": {
        \"name\":  {\"type\": \"string\", \"minLength\": 1},
        \"age\":   {\"type\": \"integer\", \"minimum\": 0, \"maximum\": 120},
        \"email\": {\"type\": \"string\", \"pattern\": \"@\"}
      },
      \"required\": [\"name\", \"email\"]}"))

(define alice (json-parse "{\"name\":\"alice\",\"email\":\"a@x.io\",\"age\":30}"))
(define malformed (json-parse "{\"name\":\"alice\",\"email\":\"no-at-sign\"}"))

(display (json-schema-valid?    user-schema alice))      (newline)   ; #t
(display (json-schema-validate  user-schema malformed))  (newline)
;; → ("/email: string does not contain pattern \"@\"")
```

Round-trip path-bearing error reporting is exercised by
`tests/v1_2_edge_cases/json_schema_test.esk:195-201`, which checks both that
exactly one error is returned and that the error string contains both the JSON
pointer `/users/0/score` and the keyword `minimum`.

### Cross-references

- Object construction: see `(json-parse …)`, `(make-hash-table)` and
  `(hash-table-ref …)` in the main API reference.
- For pattern matching beyond substring containment, compile a PCRE2 regex with
  [`regex-compile`](#4-libagentregexesk) and check separately — the JSON Schema
  validator deliberately does not pull in `agent/regex.esk` to keep its
  dependency footprint to `core.list.*`, `core.strings`, and `core.json`
  (`json_schema.esk:33-37`).

---

## 2. `core.reflection`

**Source**: [`lib/core/reflection.esk`](../lib/core/reflection.esk).
**Auto-loaded**: yes, via `(require stdlib)`.
**Direct import**: `(require core.reflection)`.

Runtime reflection primitives. The module exposes two Scheme-level helpers;
a third primitive, `(procedure-arity proc)`, is a *codegen* builtin
implemented in `lib/backend/llvm_codegen.cpp` and not redefined here.

### Provided symbols

```scheme
(provide describe type-name)
```

(Source: `lib/core/reflection.esk:19`.)

The codegen-builtin `procedure-arity` is documented alongside since it is part
of the same v1.2 reflection surface (task #170) and is used by `describe`.

### `(type-name value)` → symbol

Returns one of: `'integer 'real 'string 'symbol 'boolean 'pair 'null
'vector 'procedure 'char 'unknown`
(source lines 30-42).

Dispatch order is significant: `null?` is checked before `pair?` (because the
empty list is a degenerate pair), and `boolean?` is checked before `integer?`
(because `#f` is *not* counted as integer 0 in Eshkol). The last fallback
`'unknown` is returned for heap subtypes the predicate chain doesn't recognise
(hash-tables, complex numbers, bignums, PRNG handles, tagged AD nodes, etc.).

```scheme
(type-name 42)              ; → 'integer
(type-name 3.14)            ; → 'real
(type-name "hello")         ; → 'string
(type-name 'foo)            ; → 'symbol
(type-name #t)              ; → 'boolean
(type-name '())             ; → 'null
(type-name (list 1 2))      ; → 'pair
(type-name (vector 1 2 3))  ; → 'vector
(type-name (lambda (x) x))  ; → 'procedure
```

### `(describe value)` → string

Returns a one-line human-readable description (source 62-91). The output format
is fixed (regression-tested by
`tests/v1_2_edge_cases/reflection_describe_test.esk`):

| Value class | Output template                           | Example                            |
|-------------|-------------------------------------------|------------------------------------|
| `null`      | `"null"`                                  | `(describe '()) ⇒ "null"`          |
| boolean     | `"boolean: #t"` / `"boolean: #f"`          | `(describe #t) ⇒ "boolean: #t"`    |
| integer     | `"integer: <n>"`                           | `(describe 42) ⇒ "integer: 42"`    |
| real        | `"real: <n>"`                              | `(describe 3.14) ⇒ "real: 3.14"`   |
| string      | `"string[<n>]: \"<s>\""`                   | `(describe "hi") ⇒ "string[2]: \"hi\""` |
| symbol      | `"symbol: <name>"`                         | `(describe 'foo) ⇒ "symbol: foo"`  |
| char        | `"char: #\\<c>"`                           | `(describe #\a) ⇒ "char: #\\a"`    |
| pair        | `"pair (length <n>)"`                      | `(describe (list 1 2 3)) ⇒ "pair (length 3)"` |
| vector      | `"vector[<n>]"`                            | `(describe (vector 1 2)) ⇒ "vector[2]"` |
| procedure   | `"procedure: arity=<n>"`                   | `(describe car) ⇒ "procedure: arity=1"` |
| anything else | `"unknown"`                              |                                    |

**Edge cases**:

- For lists, `(length value)` is invoked. On a circular list this loops
  forever; `describe` is *not* safe to call on shared mutable cons cycles.
- For procedures, `procedure-arity` returns the *fixed* parameter count.
  Variadic procedures defined with rest args return `0` (the rest portion is
  not counted) — see `tests/v1_2_edge_cases/procedure_arity_test.esk:30`.

### `(procedure-arity proc)` → integer

Codegen builtin (`lib/backend/llvm_codegen.cpp`). Returns the fixed parameter
count of any procedure value: top-level definitions, named lambdas, inline
lambdas, and stdlib builtins all answer correctly. Variadic procedures return
the number of *fixed* parameters before the rest argument (the variadic
`. rest` portion is not counted).

```scheme
(define (zero) 42)
(define (one x) x)
(define (two a b) (+ a b))
(define (variadic . rest) (length rest))

(procedure-arity zero)                    ; → 0
(procedure-arity one)                     ; → 1
(procedure-arity two)                     ; → 2
(procedure-arity variadic)                ; → 0
(procedure-arity (lambda (a b c) a))      ; → 3
(procedure-arity car)                     ; → 1
```

### `record-fields` — deferred

The CHANGELOG and reflection.esk header (`lib/core/reflection.esk:10-15`,
93-109) both note `record-fields` as deferred. The current `define-record-type`
expansion stores the record's *type tag* in slot 0 of the underlying vector,
but the field names exist only at parser-expansion time. A value-level
reflection like `(record-fields some-record)` requires a runtime registry that
has not yet been built — the file documents the implementation plan inline
(~1 day of parser + small runtime helper work).

### Worked example: a generic logger

```scheme
(require stdlib)

(define (log-arg name value)
  (display name) (display " = ") (display (describe value)) (newline))

(define (process input)
  (log-arg "input" input)
  (cond
    ((eq? (type-name input) 'pair)    (apply + input))
    ((eq? (type-name input) 'vector)  (vector-ref input 0))
    ((eq? (type-name input) 'integer) (* input 2))
    (else
      (display "unsupported type: ") (display (type-name input)) (newline)
      #f)))

(process 21)               ; input = integer: 21    →  42
(process (list 1 2 3))     ; input = pair (length 3) →  6
(process (vector 'a 'b))   ; input = vector[2]      →  'a
```

### Cross-references

- `procedure-arity` is also the foundation for `core.functional.curry` (which
  reads arity to decide when to apply).
- `describe` underpins the debugger's "show me a value" feature in the REPL
  prompt.

---

## 3. Time API (codegen builtins)

**Source**: [`lib/core/system_builtins.c`](../lib/core/system_builtins.c)
(lines 313-561 for the time block) + the LLVM dispatch in
`lib/backend/llvm_codegen.cpp` (lines 12004-12131, return-type metadata at
lines 1174-1183).
**Module require line**: none — every name below is a codegen builtin and is
available without `(require …)`. The `vm_prelude_cache.h:1064-1087` exports
them to the standalone VM as well.

There is **no** `core.time` module. The brief listed one; what actually ships
is a flat set of nine codegen-builtin time primitives. They are listed in
roughly the order they appear in `system_builtins.c`.

### Conventions

- All "epoch" times use the Unix epoch (1970-01-01T00:00:00 UTC).
- `format-iso8601` / `parse-iso8601` work in **nanoseconds since epoch**
  (`int64`), not milliseconds and not seconds. This is the granularity exposed
  by `current-time-ns`.
- `current-timestamp` returns **seconds since epoch as a `double`** (so it
  carries sub-second precision but is easier to compare against POSIX
  timestamps).
- `monotonic-time-ms` is monotonic and not adjusted by NTP / wall-clock
  changes; use it for elapsed-time measurement, never for absolute timestamps.

### `(current-time-ns)` → integer

Wall-clock nanoseconds since the Unix epoch. Codegen returns an `int64` tagged
value (`function_return_types["current-time-ns"] = BuiltinTypes::Integer`).

### `(current-timestamp)` → real

Wall-clock seconds since the Unix epoch as a `double`. The C implementation
uses `clock_gettime(CLOCK_REALTIME)` on POSIX and `GetSystemTimeAsFileTime` on
Windows (`system_builtins.c:435-450`).

```scheme
(define t (current-timestamp))
(display t) (newline)
;; → 1747700000.123456  (Wed May 20 …)
```

### `(format-iso8601 ns)` → string

Formats an `int64` nanosecond timestamp as
`"YYYY-MM-DDTHH:MM:SS.mmmZ"` (always UTC, always millisecond precision,
always `Z` suffix). Source: `system_builtins.c:326-357`.

```scheme
(format-iso8601 0)                          ; → "1970-01-01T00:00:00.000Z"
(format-iso8601 1000000000)                 ; → "1970-01-01T00:00:01.000Z"
(format-iso8601 1704067200789000000)        ; → "2024-01-01T00:00:00.789Z"
```

**Edge cases**:

- Accepts both `int64`-tagged and `double`-tagged input. If the input is a
  double (e.g. `(format-iso8601 1.7e18)`) the implementation casts back to
  `int64` (`system_builtins.c:331-336`).
- Negative ns values are accepted and represent pre-epoch dates; the formatter
  uses `gmtime_r` and is bounded by what `time_t` can hold on the host.

### `(parse-iso8601 str)` → integer or `#f`

Parses `"YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM]"` to `int64` nanoseconds since
epoch. Returns `#f` on any malformed input — so callers can disambiguate
"this string is garbage" from "this string represents epoch zero" (which
is the `int64` value `0`).

Accepted forms (`system_builtins.c:381-433`):

- `2024-01-01T00:00:00Z`                — base form, UTC
- `2024-01-01 00:00:00Z`                — space accepted as the `T` separator
- `2024-01-01T00:00:00.500Z`            — fractional seconds, up to 9 digits, internally truncated/padded to milliseconds
- `2024-01-01T05:00:00+05:00`           — positive offset (subtracted to reach UTC)
- `2024-01-01T12:00:00-05:00`           — negative offset (added to reach UTC)
- `1970-01-01T00:00:00Z`                — epoch zero parses to `0`

Rejected (returns `#f`):

- `"not-a-date"`, `"2024"`, `"2024-01-01"` (date-only — time portion required),
  and any string with trailing junk after the timezone marker.

### `(format-relative seconds-ago)` → string

Renders a non-negative `int64` of seconds into a coarse human-readable string.
Source: `system_builtins.c:452-465`.

| `seconds-ago` range | Output template      | Example                                 |
|---------------------|----------------------|-----------------------------------------|
| `[0, 60)`           | `"<n>s ago"`         | `(format-relative 30) ⇒ "30s ago"`      |
| `[60, 3600)`        | `"<n>m ago"`         | `(format-relative 600) ⇒ "10m ago"`     |
| `[3600, 86400)`     | `"<n>h ago"`         | `(format-relative 3600) ⇒ "1h ago"`     |
| `[86400, ∞)`        | `"<n>d ago"`         | `(format-relative 172800) ⇒ "2d ago"`   |

Negative input is clamped to `0` (line 454).

### `(local-timezone-offset)` → integer

Returns the local timezone offset in seconds east of UTC. Implementation uses
`localtime_r(now)` and `gmtime_r(now)` followed by `difftime(mktime(local), mktime(utc))`
(`system_builtins.c:467-483`).

On the WebAssembly build (`ESHKOL_VM_WASM`) this returns `0` unconditionally
since the browser sandbox does not expose a deterministic timezone API.

### `(monotonic-time-ms)` → integer

Monotonic milliseconds since some unspecified origin. Use it only for
*elapsed-time* measurements. On Linux/macOS it uses `clock_gettime(CLOCK_MONOTONIC)`;
on Windows it uses `GetTickCount64`. (`system_builtins.c:550-561`.)

### High-precision timing helpers (in `lib/stdlib.esk`)

`lib/stdlib.esk` defines three additional helpers on top of `current-time-ns`
(lines 96-113):

```scheme
(define (current-time-us)
  "Current Unix time in microseconds"
  (/ (current-time-ns) 1000.0))

(define (time-ns thunk)
  "Execute thunk and return elapsed time in nanoseconds"
  (let ((start (current-time-ns)))
    (thunk)
    (- (current-time-ns) start)))

(define (time-us thunk)
  "Execute thunk and return elapsed time in microseconds"
  (/ (time-ns thunk) 1000.0))

(define (time-it thunk iterations)
  "Execute thunk N times, return average time in nanoseconds"
  (let ((start (current-time-ns)))
    (let loop ((i 0))
      (when (< i iterations)
        (thunk)
        (loop (+ i 1))))
    (/ (- (current-time-ns) start) iterations)))
```

Use `time-ns` / `time-us` for one-shot measurement and `time-it` for averaged
microbenchmarks. Note that these wrap `current-time-ns` (wall-clock), not
`monotonic-time-ms`; the choice is deliberate because nanosecond-resolution
monotonic clocks vary widely across platforms, whereas `current-time-ns` is
universally implemented to nanosecond precision.

### Worked example: round-trip an ISO 8601 timestamp

```scheme
(let* ((s0 "2026-04-17T12:34:56.789Z")
       (ns (parse-iso8601 s0))
       (s1 (format-iso8601 ns)))
  (display (string=? s0 s1)) (newline))         ; → #t
```

(Source: regression-tested by `tests/v1_2_edge_cases/time_api_test.esk:50-55`.)

### Worked example: bench a function with `time-ns`

```scheme
(require stdlib)

(define (slow-sum n)
  (let loop ((i 0) (acc 0))
    (if (>= i n) acc
        (loop (+ i 1) (+ acc i)))))

(define elapsed (time-ns (lambda () (slow-sum 100000))))
(display "took ") (display elapsed) (display " ns") (newline)
```

### Cross-references

- For Wall-clock monotonicity guarantees, contrast `current-time-ns` (NTP-
  adjustable) with `monotonic-time-ms` (immune to clock jumps).
- For ISO 8601 strings produced or consumed by JSON, use `format-iso8601` /
  `parse-iso8601` directly — the JSON parser preserves them as plain strings.

---

## 4. `lib/agent/regex.esk`

**Source**: [`lib/agent/regex.esk`](../lib/agent/regex.esk) wrapping
[`lib/agent/c/agent_regex.c`](../lib/agent/c/agent_regex.c) (PCRE2).
**Module require line**: not a `(require …)` module. The agent FFI archive is
force-loaded into `eshkol-run`; the JIT picks up `eshkol_regex_*` C symbols via
`dlsym(RTLD_DEFAULT)`. In code that needs the helpers explicitly, use
`(load "lib/agent/regex.esk")` or rely on `(require stdlib)`'s force-load
side effect.

### Provided symbols

```scheme
(provide regex-compile regex-match regex-match? regex-match-all
         regex-replace regex-free
         regex-match-groups regex-group regex-named-group-number
         REGEX_CASELESS REGEX_MULTILINE REGEX_DOTALL)
```

(Source: `lib/agent/regex.esk:5-8`. The capture-group functions
(`regex-match-groups`, `regex-group`, `regex-named-group-number`) are the v1.2
additions; the earlier surface only supported match-or-no-match.)

### Flag constants

```scheme
(define REGEX_CASELESS  1)
(define REGEX_MULTILINE 2)
(define REGEX_DOTALL    4)
```

Bitwise-OR them and pass the result to `regex-compile`'s optional second
argument. They map to PCRE2's `PCRE2_CASELESS`, `PCRE2_MULTILINE`,
`PCRE2_DOTALL` (`agent_regex.c:57-60`). The compile call always sets
`PCRE2_UTF`.

### `(regex-compile pattern [flags])` → integer handle or `-1`

Compiles a PCRE2 pattern. Returns a small positive integer handle (1 ≤ h <
256, taken from a fixed-size handle pool in `agent_regex.c:21-28`) or `-1` on
compile failure. The pool is process-global; do not leak handles or you will
run out — call `(regex-free h)` when done.

```scheme
(define digits  (regex-compile "[0-9]+"))
(define ci-word (regex-compile "[a-z]+" REGEX_CASELESS))
```

### `(regex-match handle subject)` → string or `#f`

Returns the first matching substring or `#f` if no match. The internal buffer
is `4096` bytes; matches longer than that are truncated at the first NUL
(`regex.esk:32-46`).

### `(regex-match? handle subject)` → boolean

Cheap predicate form. Passes `#f` for the match buffer so no string allocation
occurs (`regex.esk:48-50`).

### `(regex-match-all handle subject [max])` → list of strings

Returns up to `max` (default 100) non-overlapping matches. The underlying
buffer is sized `max * 256` bytes; matches share the buffer and are split on
embedded NULs. Returns `'()` on no matches (`regex.esk:52-66`).

### `(regex-replace handle subject replacement)` → string

PCRE2 substitution; replaces all non-overlapping matches. The buffer is sized
`2 * (|subject| + |replacement|)`. On replace failure (negative return code)
returns the original `subject` unchanged (`regex.esk:68-72`).

### `(regex-free handle)` → none

Releases the PCRE2 handle and returns it to the pool. After `regex-free`, the
handle is invalid; later use of it returns `0` from `regex-match-raw` and
therefore `#f` from `regex-match?` / `regex-match`.

### Capture groups (v1.2 addition)

#### `(regex-match-groups handle subject)` → list of strings or `#f`

Matches `subject` and returns `(list full-match group1 group2 …)` on a match,
`#f` on no match. `group 0` is the full match; `group 1..N` are the captured
subgroups in left-to-right declaration order. An unset optional group
(e.g. the `b` in `a(b)?c` when `b` is absent) is returned as the empty
string `""` (`regex.esk:80-103`).

#### `(regex-group match-groups idx)` → string or `#f`

Accessor by integer index into the list returned by `regex-match-groups`.
Index `0` is the full match. Out-of-range indices return `#f`
(`regex.esk:105-110`).

#### `(regex-named-group-number handle name)` → integer

Returns the integer index of a PCRE2 named capture group `(?<name>...)`, or
`-1` if no such name exists in the compiled pattern (`regex.esk:112-114`).
Combine with `regex-group` to do "give me the group called 'year'":

```scheme
(define h (regex-compile "(?<year>[0-9]{4})-(?<mon>[0-9]{2})-(?<day>[0-9]{2})"))
(define m (regex-match-groups h "2026-05-20"))
(define year-idx (regex-named-group-number h "year"))
(display (regex-group m year-idx))  ; → "2026"
(regex-free h)
```

### ReDoS protection

`agent_regex.c:91-104` installs a process-global `pcre2_match_context` with
a match-limit of 10 million backtrack steps and a depth-limit of 100 000.
Pathological patterns like `(a+)+$` against a long subject of `a`s with one
trailing non-`a` will return *no match* rather than hang the calling thread.
The limit is conservatively chosen — legitimate matches finish in tens of
milliseconds even on the largest realistic inputs.

### Worked example: extract structured fields from a log line

```scheme
(define re
  (regex-compile "(?<lvl>INFO|WARN|ERROR) \\[(?<ts>[^\\]]+)\\] (?<msg>.*)"))

(define line "INFO [2026-05-20T12:34:56Z] starting compile")

(define m (regex-match-groups re line))
(when m
  (display "level: ") (display (regex-group m (regex-named-group-number re "lvl"))) (newline)
  (display "ts:    ") (display (regex-group m (regex-named-group-number re "ts")))  (newline)
  (display "msg:   ") (display (regex-group m (regex-named-group-number re "msg"))) (newline))

(regex-free re)
```

### Cross-references

- For percent-encoded URLs, use [`url-decode`](#9-coreurl) before applying a
  regex.
- The standalone VM exposes a different regex surface
  (`tests/vm/regex_surface_regression.esk`) that returns an alist instead of a
  list. The VM-side surface is documented in
  [`docs/STDLIB_EXTENSIONS.md`](STDLIB_EXTENSIONS.md). This module documents
  the LLVM/JIT/AOT path.

---

## 5. `core.cache`

**Source**: [`lib/core/cache.esk`](../lib/core/cache.esk).
**Direct import**: `(require core.cache)`. **Not auto-loaded** by
`(require stdlib)` — `lib/stdlib.esk` deliberately omits it (see comment at
`lib/stdlib.esk:64-69` regarding `core.testing`; the same approach is taken
for `core.cache` to keep the auto-load surface tight).

A bounded LRU cache and a memoizer built on top of it. The cache is a simple
hash-table-plus-timestamp scheme described at the head of the source file
(lines 16-23). Eviction walks the table to find the lowest timestamp — an
`O(n)` operation that trades implementation simplicity for strictly-bounded
memory; the design is appropriate for hundreds-to-low-thousands of entries,
which covers the typical agent-workflow cache.

### Provided symbols

```scheme
(provide make-lru-cache lru-get lru-get/default lru-set! lru-has?
         lru-delete! lru-size lru-capacity lru-clear!
         memoize memoize/cap
         memoize1 memoize1/cap memoize2 memoize2/cap)
```

(Source: `lib/core/cache.esk:29-32`.)

### Internal representation

Each cache handle is a 3-element vector `#(capacity counter-box ht)`
(`cache.esk:34-37`):

- `capacity` — the configured max entry count.
- `counter-box` — a single-cell `(list n)` used as a mutable monotonic timestamp.
- `ht` — a hash table mapping keys to `(cons value timestamp)`.

This representation matters for users who want to introspect a cache; do not
rely on it beyond what the public API gives you, but it explains why
`hash-table-keys` does not return cache keys directly (you would need
`(hash-table-keys (vector-ref cache 2))`, which is unsupported).

### `(make-lru-cache capacity)` → cache handle

Constructs an LRU cache with the given positive-integer capacity.

**Edge case**: passing `0`, a negative number, or a non-integer raises an
error via `(error …)` — exercised by
`tests/v1_2_edge_cases/cache_test.esk:107-110` (`check-raises`).

### `(lru-capacity cache)` → integer

The configured maximum, unchanged from `make-lru-cache`.

### `(lru-size cache)` → integer

Current number of entries (≤ `lru-capacity`).

### `(lru-has? cache key)` → boolean

Membership test without touching the timestamp.

### `(lru-get cache key)` → value or `#f`

Returns the cached value and bumps its timestamp on a hit. On a miss returns
`#f` (so a cache that legitimately stores `#f` is ambiguous; use
`lru-get/default` if your value space includes `#f`).

### `(lru-get/default cache key default)` → value or `default`

Like `lru-get` but returns `default` on a miss. Still bumps the timestamp on
a hit (`cache.esk:66-74`).

### `(lru-set! cache key value)` → unspecified

Stores `value` under `key`. If `key` already exists, updates the value and
touches the timestamp (does not change `lru-size`). Otherwise inserts, and if
this would exceed `capacity`, evicts the lowest-timestamp entry first
(`cache.esk:76-88`).

### `(lru-delete! cache key)` → boolean

Removes `key` from the cache. Returns `#t` if the key was present, `#f`
otherwise.

### `(lru-clear! cache)` → unspecified

Drops every entry (`cache.esk:96-97`).

### `memoize` family

`memoize`-class functions wrap a deterministic function so identical arguments
return the cached prior result. The cache is per-wrapped-function: each
invocation of `memoize` allocates a fresh `make-lru-cache`.

#### `(memoize1 fn)` → 1-ary function

Memoize a 1-argument function with default capacity 1024.

#### `(memoize1/cap fn capacity)` → 1-ary function

Memoize a 1-argument function with the given capacity.

#### `(memoize2 fn)` → 2-ary function

Memoize a 2-argument function. The cache key is the string concatenation of
`(display x port)` and `(display y port)` separated by NUL (`cache.esk:163-168`),
so structural-equality (`equal?`-style) matching is achieved without
allocating a fresh cons cell per call.

#### `(memoize2/cap fn capacity)` → 2-ary function

`memoize2` with custom capacity.

#### `(memoize fn)` and `(memoize/cap fn capacity)`

Aliases for `memoize1` and `memoize1/cap` respectively. Use them when the
arity is implicit.

**Why only 1- and 2-ary memoization?** Variadic
`(lambda args …)` capturing free variables interacts badly with the closure-
ABI codepath in compile-to-binary mode (see the explanatory comment in
`cache.esk:120-127`). Callers needing N-ary memoization should pre-curry
their arguments into a single list and memoize the 1-ary wrapper.

### Worked example: memoized Fibonacci with LRU eviction

```scheme
(require core.cache)

(define call-count 0)
(define (slow-square x)
  (set! call-count (+ call-count 1))
  (* x x))

(define square (memoize1 slow-square))

(display (square 5)) (newline)   ; 25, slow-square ran
(display (square 5)) (newline)   ; 25, cache hit (call-count unchanged)
(display (square 6)) (newline)   ; 36, new entry
(display call-count) (newline)   ; → 2 (not 3)
```

### Worked example: LRU eviction order

```scheme
(require core.cache)

(define c (make-lru-cache 3))
(lru-set! c 'a 1)
(lru-set! c 'b 2)
(lru-set! c 'c 3)
(lru-get c 'a)         ; touch 'a — 'b is now coldest
(lru-get c 'c)         ; touch 'c
(lru-set! c 'd 4)      ; capacity hit — evicts 'b (the only un-touched key)

(display (lru-has? c 'a)) (newline)   ; #t
(display (lru-has? c 'b)) (newline)   ; #f — evicted
(display (lru-has? c 'c)) (newline)   ; #t
(display (lru-has? c 'd)) (newline)   ; #t
```

(Source: regression-tested by `tests/v1_2_edge_cases/cache_test.esk:36-49`.)

### Cross-references

- For a *recursive* memoization that survives across calls, use a
  module-level `memoize1`-wrapped function rather than re-wrapping inside the
  function body — re-wrapping creates a fresh cache per call and defeats the
  purpose.
- For very large caches (>10⁶ entries) the `O(n)` eviction scan becomes
  measurable; the source-level comment recommends an external store at that
  scale (`cache.esk:21-23`).

---

## 6. PRNG: `make-prng` / `prng-random` (codegen builtins) + `lib/random/random.esk`

**Sources**:

- [`lib/core/prng.cpp`](../lib/core/prng.cpp) — C runtime backing the
  per-stream isolated PRNG state.
- `lib/backend/llvm_codegen.cpp:12478-12482` — codegen dispatch.
- `lib/backend/llvm_codegen.cpp:28028-28140` — the inline documentation block
  describing the API and the registration pattern.
- [`lib/random/random.esk`](../lib/random/random.esk) — high-level
  distribution sampling built on top of the global `(random)` function.

**Module require lines**:

- Per-stream isolation primitives (`make-prng`, `prng-random`,
  `prng-random-integer`, `prng?`, `set-random-seed!`): **none** — they are
  codegen builtins, available everywhere.
- High-level distribution wrappers: `(require random.random)` or
  `(require "random")` (path-based fallback).

There is **no** `core.random` module. The brief listed one; the actual
implementation has two halves: codegen builtins for the seed/isolation
mechanics, and a `.esk` library for the distribution-sampling helpers.

### Global PRNG (drand48-compatible)

The codegen for `(random)` calls libc's `drand48`. On Linux/macOS this is the
system implementation; on Windows it routes through the `eshkol`-shipped shim
in `platform_runtime.cpp` (`prng.cpp:32-42`).

#### `(set-random-seed! seed)` → unspecified

Seed the global PRNG. The codegen for `set-random-seed!` also flips a
`__random_seeded__` marker so the runtime's auto-seed-from-time path (which
fires on first `(random)` of a program) is suppressed (`prng.cpp:88-94`).

#### `(random)` → real in `[0.0, 1.0)`

Standard drand48-style uniform sample. Mutex-protected on the global state.

### Per-stream isolated PRNG

For parallel workloads and paper-artifact reproducibility, the global PRNG's
shared mutex is a problem: two `parallel-map` workers calling `(random)`
interleave nondeterministically. The per-stream API gives each thread its
own state and removes the lock.

#### `(make-prng seed)` → PRNG handle

Allocates a fresh PRNG state on the arena, seeded as
`(seed << 16) | 0x330E` (matching POSIX `srand48`). The handle is a heap
pointer with header subtype `HEAP_SUBTYPE_PRNG`. The same seed produces the
same sequence as a freshly-seeded global PRNG (the documentation comment at
`prng.cpp:15-17` calls this out explicitly).

#### `(prng-random p)` → real in `[0.0, 1.0)`

Advance the given PRNG state and return the next double. Lock-free.

#### `(prng-random-integer p n)` → integer in `[0, n)`

Advance the given PRNG state and return the next integer in `[0, n)`. Uses
multiply-then-floor; bias is `< 1/2^48`, well below user-visible
(`prng.cpp:80-87`). Returns `0` if `n ≤ 0` or `p` is null.

#### `(prng? x)` → boolean

Heap-subtype check: returns `#t` iff `x` is a value allocated by `make-prng`.
Other heap pointers, integers, strings, and `#f` all answer `#f`.

### Worked example: deterministic parallel sampling

```scheme
(require stdlib)

;; Each worker gets its own PRNG seeded from a deterministic root —
;; two runs with the same root produce bit-identical samples.

(define (worker root-seed worker-id n)
  (let ((p (make-prng (+ root-seed worker-id))))
    (let loop ((i 0) (acc 0.0))
      (if (>= i n) acc
          (loop (+ i 1) (+ acc (prng-random p)))))))

(define s1 (worker 42 0 1000))
(define s2 (worker 42 0 1000))
(display (= s1 s2)) (newline)              ; → #t — bit-identical

(define s3 (worker 42 1 1000))
(display (= s1 s3)) (newline)              ; → #f — different stream
```

(Source: regression-tested by
`tests/v1_2_edge_cases/prng_seeding_test.esk:36-69`.)

### Distribution sampling — `lib/random/random.esk`

The high-level wrappers in `lib/random/random.esk` build on the global
`(random)`:

#### Provided symbols

```scheme
(provide random-float random-int random-bool random-choice
         qrandom qrandom-int qrandom-bool qrandom-choice
         uniform quniform normal-pair normal exponential poisson
         bernoulli geometric binomial
         random-tensor random-normal-tensor random-vector
         random-uniform-vector random-normal-vector
         shuffle sample weighted-choice
         set-random-seed! current-time-seed randomize!)
```

(Source: `lib/random/random.esk:9-16`.)

#### Highlights

| Function                          | Signature                            | Distribution                                          |
|-----------------------------------|--------------------------------------|-------------------------------------------------------|
| `(random-float)`                  | → real                               | uniform `[0, 1)` (alias for `(random)`)               |
| `(random-int low high)`           | int int → int                        | uniform integer in `[low, high]` (inclusive)          |
| `(random-bool)`                   | → boolean                            | 50/50                                                 |
| `(random-choice lst)`             | list → element                       | uniform pick                                          |
| `(uniform low high)`              | real real → real                     | uniform `[low, high]`                                 |
| `(normal mu sigma)`               | real real → real                     | Gaussian `N(μ, σ²)` (Box-Muller, one of two samples)  |
| `(normal-pair)`                   | → (list real real)                   | both Box-Muller samples (cheaper if you need two)     |
| `(exponential lambda)`            | real → real                          | exponential, rate λ                                   |
| `(poisson lambda)`                | real → int                           | Knuth small-λ algorithm                               |
| `(bernoulli p)`                   | real → 0 or 1                        | Bernoulli, returns int                                |
| `(geometric p)`                   | real → int                           | failures before first success                         |
| `(binomial n p)`                  | int real → int                       | trial method                                          |
| `(shuffle lst)`                   | list → list                          | Fisher-Yates                                          |
| `(sample lst k)`                  | list int → list                      | without replacement (via `shuffle` + take)            |
| `(weighted-choice items weights)` | list list → element                  | weights need not sum to 1                             |
| `(qrandom)`, `(qrandom-int hi)`,  `(qrandom-bool)`, `(quniform lo hi)` | various | quantum-random variants (hardware entropy when available) |

#### `set-random-seed!`, `current-time-seed`, `randomize!`

`set-random-seed!` is *also* the codegen builtin documented above —
`lib/random/random.esk:236-238` wraps it for re-export under
`(require random.random)`. `current-time-seed` returns `(time 0)`; `randomize!`
is `(set-random-seed! (current-time-seed))`.

### Worked example: reproducible normal-distribution sampling

```scheme
(require random.random)

(set-random-seed! 12345)
(display (normal 0.0 1.0)) (newline)       ; deterministic, e.g. 0.318...
(display (normal 5.0 2.0)) (newline)

(set-random-seed! 12345)
(display (normal 0.0 1.0)) (newline)       ; same value as the first call
```

### Cross-references

- For `parallel-map` workloads where each worker needs its own randomness,
  use `make-prng` not the global `(random)`. The global form serialises on
  a mutex; the per-stream form is lock-free (`prng.cpp:19-21`).
- For the consciousness-engine workspace's stochastic broadcasts, use
  `make-prng` to give each module its own reproducible stream.

---

## 7. `core.streams`

**Source**: [`lib/core/streams.esk`](../lib/core/streams.esk).
**Auto-loaded**: yes, via `(require stdlib)`.
**Direct import**: `(require core.streams)`.

A SRFI-41-style lazy stream library built on Eshkol's existing
`delay` / `force` codegen primitives. Promises memoise — each tail is computed
at most once even if `stream-cdr` is called repeatedly.

### Provided symbols

```scheme
(provide stream-null
         stream-null?
         stream-pair?
         stream?
         stream-cons
         stream-car
         stream-cdr
         stream-take
         stream-drop
         stream-ref
         stream-map
         stream-filter
         stream-for-each
         stream-zip
         stream-append
         stream-iterate
         stream-from
         stream-take-while
         stream-drop-while
         stream-length
         stream->list
         list->stream)
```

(Source: `lib/core/streams.esk:28-49`.)

### Important: no implicit delay

Eshkol has no `define-syntax`, so SRFI 41's macro form
`(stream-cons head tail-expr)` (which would implicitly delay `tail-expr`) is
unavailable. **Callers must wrap the tail with `delay` explicitly**
(`streams.esk:9-20`):

```scheme
(define (from n) (stream-cons n (delay (from (+ n 1)))))
(define ones (stream-cons 1 (delay ones)))                  ; cyclic
```

If you forget the `delay`, the tail evaluates strictly and an infinite stream
recurses forever.

### Construction and predicates

| Function                          | Returns                              | Notes                                                  |
|-----------------------------------|--------------------------------------|--------------------------------------------------------|
| `stream-null`                     | `'()`                                | The empty stream (a value, not a function)             |
| `(stream-null? s)`                | boolean                              | `(null? s)`                                            |
| `(stream-pair? s)`                | boolean                              | `(pair? s)`                                            |
| `(stream? s)`                     | boolean                              | true for `stream-null` or any pair                     |
| `(stream-cons head tail-promise)` | new stream cell                      | `(cons head tail-promise)`. Caller must `delay` the tail|
| `(stream-car s)`                  | head value                           |                                                        |
| `(stream-cdr s)`                  | next stream cell                     | Forces and memoises the tail promise                   |

### Indexing and slicing

| Function                | Signature                         | Behaviour                                                      |
|-------------------------|-----------------------------------|----------------------------------------------------------------|
| `(stream-take s n)`     | stream int → list                 | First `n` elements as a list; truncates at `stream-null`       |
| `(stream-drop s n)`     | stream int → stream               | Skip first `n` elements; idempotent at `stream-null`           |
| `(stream-ref s n)`      | stream int → element              | `n`-th element, 0-indexed                                      |
| `(stream-length s)`     | stream → int                      | **Loops forever** on an infinite stream                        |
| `(stream->list s)`      | stream → list                     | **Loops forever** on an infinite stream                        |
| `(list->stream lst)`    | list → stream                     | Lazy traversal of an eager list                                |

### Higher-order

| Function                    | Signature                                    | Behaviour                                                    |
|-----------------------------|----------------------------------------------|--------------------------------------------------------------|
| `(stream-map f s)`          | (a → b) stream-a → stream-b                  | Lazy                                                         |
| `(stream-filter pred s)`    | (a → bool) stream-a → stream-a               | Lazy                                                         |
| `(stream-for-each f s)`     | (a → unspecified) stream-a → '()             | Strict — walks until `stream-null`                           |
| `(stream-zip s1 s2)`        | stream stream → stream                       | Stops on the shorter; element is `(list h1 h2)`              |
| `(stream-append s1 s2)`     | stream stream → stream                       | Lazy concatenation                                           |
| `(stream-iterate f x)`      | (a → a) a → stream-a                         | Infinite `x, f(x), f(f(x)), …`                               |
| `(stream-from n)`           | int → stream-int                             | Infinite `n, n+1, n+2, …`                                    |
| `(stream-take-while pred s)`| (a → bool) stream-a → stream-a               | Stop at first `pred` failure                                 |
| `(stream-drop-while pred s)`| (a → bool) stream-a → stream-a               | Skip while `pred` is true                                    |

### Memoisation guarantee

`(stream-cdr s)` calls `(force (cdr s))`. Eshkol's `force` memoises by
construction, so repeated `stream-cdr` calls on the same cell do not re-run
the tail expression. The test
`tests/v1_2_edge_cases/streams_test.esk:76-91` exercises this: a counted
generator advances exactly once per fresh `stream-cdr`, and not at all on a
repeat.

### Worked example: sieve of Eratosthenes

```scheme
(require core.streams)

(define (sieve s)
  (let ((p (stream-car s)))
    (stream-cons p
      (delay (sieve (stream-filter
                      (lambda (n) (not (= 0 (remainder n p))))
                      (stream-cdr s)))))))

(define primes (sieve (stream-from 2)))

(display (stream-take primes 10)) (newline)
;; → (2 3 5 7 11 13 17 19 23 29)
```

### Worked example: cyclic stream

```scheme
(require core.streams)

(define zeros-and-ones
  (stream-cons 0 (delay (stream-cons 1 (delay zeros-and-ones)))))

(display (stream-take zeros-and-ones 6)) (newline)
;; → (0 1 0 1 0 1)
```

(Source: regression-tested by
`tests/v1_2_edge_cases/streams_test.esk:93-102`.)

### Worked example: filter-map pipeline on an infinite stream

```scheme
(require core.streams)

(define even-squares
  (stream-filter even?
    (stream-map (lambda (x) (* x x))
      (stream-from 1))))

(display (stream-take even-squares 5)) (newline)
;; → (4 16 36 64 100)
```

### Cross-references

- For an eager list-based equivalent, see `core.list.transform` and
  `core.list.higher_order`.
- Strict `stream->list` of an infinite stream loops forever; bound it with
  `(stream-take s n)` first if the stream might be infinite.

---

## 8. `core.argparse`

**Source**: [`lib/core/argparse.esk`](../lib/core/argparse.esk).
**Direct import**: `(require core.argparse)`. **Not auto-loaded** by
`(require stdlib)` — `argparse` is a leaf module that only programs which
actually parse `command-line` need.

### Provided symbols

```scheme
(provide parse-args arg-get arg-positional arg-has? argparse-help)
```

(Source: `lib/core/argparse.esk:36`.)

### Spec format

A spec is a list of four-element lists in the form `(name type default
desc)` — purely positional because Eshkol's reader at v1.2 does not support
`#:default`-style keyword arguments in list literals
(`argparse.esk:19-21`).

```scheme
(define spec
  (list (list '--name    'string  "World" "Your name")
        (list '--count   'integer 1       "How many greetings")
        (list '--verbose 'boolean #f      "Verbose output")))
```

- `name` — symbol, must start with `--` (long-form only at v1.2)
- `type` — one of `'string 'integer 'number 'boolean`
- `default` — value used when the flag is absent from `argv`
- `desc` — string used by `argparse-help`

### Recognised flag forms

| Form                | Behaviour                                                                      |
|---------------------|--------------------------------------------------------------------------------|
| `--name value`      | Long flag with the next token as its value                                     |
| `--name=value`      | Long flag with inline value                                                    |
| `--name`            | Boolean flag (sets to `#t`)                                                    |
| `--no-name`         | Boolean flag negation (sets to `#f`)                                           |
| `--`                | End-of-options marker; everything after is positional                          |

Source: `lib/core/argparse.esk:23-30`. Unknown flags fall through into the
positional list, so callers can decide whether to reject or forward them.

### `(parse-args argv spec)` → `(list flags-alist positionals-list)`

Returns a two-element list:

- `(car result)` — the alist of `flag-symbol` → `converted-value`. Every flag
  in `spec` is present; flags not in `argv` carry their default.
- `(car (cdr result))` — the positional argument strings in declaration order.

Type coercion (`convert-value`, line 66-72):

- `'string`  — pass through
- `'integer` — `string->number`
- `'number`  — `string->number`
- `'boolean` — `#t` unless the raw string is exactly `"false"` or `"0"`

### `(arg-get parsed key)` → value

Look up a flag by symbol. Returns the default value if not explicitly set on
the command line.

### `(arg-positional parsed)` → list of strings

Returns the positional arguments preserved in left-to-right order.

### `(arg-has? parsed key)` → boolean

Returns `#t` if the flag is non-null and non-`#f`. Useful for distinguishing
"user set the value to its default" from "user did not set anything" — in
particular, a boolean flag whose default is `#f` returns `#f` here unless the
user explicitly passed `--name`.

### `(argparse-help progname spec)` → string

Renders a Usage line plus one line per spec entry. Intentionally minimal;
callers that want richer help should iterate the spec themselves.

### Worked example: a typical entry point

```scheme
(require core.argparse)

(define spec
  (list (list '--name    'string  "World" "Your name")
        (list '--count   'integer 1       "How many greetings")
        (list '--verbose 'boolean #f      "Verbose output")))

(define args (parse-args (command-line) spec))

(when (arg-get args '--verbose)
  (display "[verbose] running with args: ")
  (display args) (newline))

(let loop ((i 0))
  (when (< i (arg-get args '--count))
    (display "Hello, ") (display (arg-get args '--name)) (newline)
    (loop (+ i 1))))

(when (not (null? (arg-positional args)))
  (display "Files to process: ")
  (display (arg-positional args)) (newline))
```

### Edge cases

- `--name` with no following token: the flag is silently skipped
  (`argparse.esk:142-143`).
- Unknown flag like `--bogus`: the literal string `"--bogus"` ends up in
  `arg-positional`. There is no warning.
- `--` consumes itself and forces all subsequent tokens (including ones that
  look like flags) into positional.
- Single-dash flags (`-v`) are *not* recognised at v1.2 — they fall through
  to positional. Long-form (`--verbose`) is the only supported flag style.

### Cross-references

- `(command-line)` is the codegen builtin that returns the original `argv` as
  a list of strings.
- For programs that want richer help / subcommands, treat `parse-args` as the
  flag layer and dispatch on `(car (arg-positional args))`.

---

## 9. `core.url`

**Source**: [`lib/core/url.esk`](../lib/core/url.esk) (encoder/decoder) plus
the system-builtin `url-parse` in
[`lib/core/system_builtins.c`](../lib/core/system_builtins.c) lines 2286-2370.
**Auto-loaded**: yes, via `(require stdlib)`.
**Direct import**: `(require core.url)`.

RFC 3986 percent-encoding for URL components, plus a one-shot URL parser. The
encoder is UTF-8-aware: non-ASCII codepoints are percent-encoded as their
UTF-8 byte sequence per RFC 3986 §2.5.

### Provided symbols

```scheme
(provide url-encode url-decode base64url-encode base64url-decode)
```

(Source: `lib/core/url.esk:20`. Note that `base64url-encode` and
`base64url-decode` are re-exported here even though their underlying primitive
(`base64-encode-string`) lives in `core.data.base64`.)

`url-parse` is *not* in the `provide` list because it is a codegen builtin
(no `.esk` wrapper).

### `(url-encode str)` → string

Percent-encodes every byte of `str` except the RFC 3986 unreserved set
`[A-Za-z0-9-_.~]`. Returns the encoded string. Hex digits are upper-case per
RFC 3986 §2.1.

```scheme
(url-encode "hello world")      ; → "hello%20world"
(url-encode "k=v&q=42")         ; → "k%3Dv%26q%3D42"
(url-encode "100% off")         ; → "100%25%20off"
(url-encode "café")             ; → "caf%C3%A9"   (UTF-8 byte sequence for é)
(url-encode "-_.~")             ; → "-_.~"        (all unreserved)
```

The unreserved-set check is `lib/core/url.esk:34-41`. The UTF-8 expansion for
codepoints ≥ 128 is in `encode-codepoint` (lines 45-74) and emits 2/3/4-byte
sequences for codepoints up to `0x10FFFF`.

### `(url-decode str)` → string

Reverses `url-encode`. Each `%XX` triple decodes to the byte `0xXX`; a literal
`+` decodes to space (legacy form-encoding compatibility); UTF-8 multibyte
sequences are reassembled into the original codepoint. Malformed input
(`%` at end, `%` followed by non-hex) passes through literally.

```scheme
(url-decode "hello%20world")    ; → "hello world"
(url-decode "form+data")        ; → "form data"   (+ as space)
(url-decode "caf%C3%A9")        ; → "café"
(url-decode "100%25")           ; → "100%"
(url-decode "abc%")             ; → "abc%"        (trailing % passes through)
(url-decode "%XY")              ; → "%XY"         (non-hex after %)
```

### `(url-parse str)` → alist or `#f`  *(system builtin)*

Parses an `scheme://host[:port][/path][?query][#fragment]` URL into an alist.
Returns `#f` on input that does not have a `://` separator or that has empty
authority.

The alist always carries `"scheme"`, `"host"`, and `"path"` (the path defaults
to `"/"` if absent), and `"port"` if either explicit or the scheme has a
known default (`http` → 80, `https` → 443). It carries `"query"` and
`"fragment"` only when present.

```scheme
(url-parse "https://example.com:8443/path/to?q=1#frag")
;; → (("scheme" . "https")
;;    ("host"   . "example.com")
;;    ("port"   . 8443)
;;    ("path"   . "/path/to")
;;    ("query"  . "q=1")
;;    ("fragment" . "frag"))

(url-parse "https://example.org")
;; → (("scheme" . "https")
;;    ("host"   . "example.org")
;;    ("port"   . 443)
;;    ("path"   . "/"))

(url-parse "not a url")   ; → #f
```

Use a small alist helper to read fields:

```scheme
(define (alist-ref key alist)
  (let ((entry (assoc key alist)))
    (if entry (cdr entry) #f)))

(define u (url-parse "https://example.com:8443/api?v=2"))
(display (alist-ref "host"  u)) (newline)    ; → example.com
(display (alist-ref "port"  u)) (newline)    ; → 8443
(display (alist-ref "query" u)) (newline)    ; → v=2
```

Source: `system_builtins.c:2286-2370`; regression-tested in
`tests/vm/url_parse_surface_regression.esk`.

**Port handling**: the parser refuses ports outside `[0, 65535]` and refuses
non-numeric "port-like" suffixes; in either case the colon and following
characters stay part of the host (`system_builtins.c:2306-2318`).

### `(base64url-encode str)` and `(base64url-decode str)`

Documented in [section 10](#10-coredatabase64--base64url-variants).
Re-exported by `core.url` so a single `(require core.url)` covers all
URL-safe encoding needs.

### Worked example: encode a structured query string

```scheme
(require core.url)

(define (encode-pair p)
  (string-append (url-encode (car p)) "=" (url-encode (cdr p))))

(define (string-join xs delim)            ; tiny inline helper
  (cond
    ((null? xs) "")
    ((null? (cdr xs)) (car xs))
    (else (string-append (car xs) delim (string-join (cdr xs) delim)))))

(define qs
  (string-join
    (map encode-pair
         (list (cons "q"      "hello world")
               (cons "filter" "name=alice & status=active")
               (cons "page"   "1")))
    "&"))

(display qs) (newline)
;; → q=hello%20world&filter=name%3Dalice%20%26%20status%3Dactive&page=1
```

Use `(string-join …)` from `core.strings` instead of the inline definition in
real code.

### Cross-references

- For decoding incoming form-data with `application/x-www-form-urlencoded`,
  remember that `url-decode` treats `+` as space; this is intentional for
  form-encoding compatibility.
- For URLs that need percent-decoding *then* JSON parsing, chain
  `(json-parse (url-decode …))`.
- For URL-safe base64, see section 10.

---

## 10. `core.data.base64` — base64url variants

**Source**: [`lib/core/data/base64.esk`](../lib/core/data/base64.esk) (the
standard base64 primitives, pure Eshkol) plus the base64url variants from
[`lib/core/url.esk`](../lib/core/url.esk:185-214) built on top.
**Auto-loaded**: yes, via `(require stdlib)` (which pulls in both
`core.data.base64` and `core.url`).
**Direct import**: `(require core.data.base64)` and/or `(require core.url)`.

### Provided symbols (standard base64)

```scheme
(provide base64-encode base64-decode base64-encode-string base64-decode-string
         string->bytes bytes->string base64-char-at base64-value
         string->bytes-helper base64-encode-helper base64-decode-helper base64-remove-padding)
```

(Source: `lib/core/data/base64.esk:5-7`.)

### Provided symbols (base64url)

```scheme
;; from lib/core/url.esk
(provide url-encode url-decode base64url-encode base64url-decode)
```

### Standard base64 (RFC 4648 §4)

| Function                          | Signature                            | Behaviour                                                |
|-----------------------------------|--------------------------------------|----------------------------------------------------------|
| `(base64-encode bytes)`           | byte-list → string                   | Encodes a list of byte values to a padded base64 string  |
| `(base64-decode str)`             | string → byte-list                   | Decodes (padded or unpadded) base64 to a byte list       |
| `(base64-encode-string str)`      | string → string                      | Convenience: encodes the bytes of a string               |
| `(base64-decode-string str)`      | string → string                      | Convenience: decodes to a string                         |
| `(string->bytes str)`             | string → byte-list                   | Per-character `char->integer`                            |
| `(bytes->string bytes)`           | byte-list → string                   | Per-byte `integer->char` then `string-append`            |

The alphabet is `"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"`
(`base64.esk:14`).

`base64-encode` always emits padding (`=` / `==`); `base64-decode` accepts
both padded and unpadded forms (`base64-remove-padding` walks back from the
tail).

### Base64url (RFC 4648 §5)

The base64url variant is identical to base64 with two character substitutions
and no padding:

- `+` → `-`
- `/` → `_`
- trailing `=` stripped

Use it for JWTs, CSRF tokens, content hashes that travel through query strings
or path segments, and anything else that has to survive URL-encoding without
its own internal substitutions.

#### `(base64url-encode str)` → string

Source: `lib/core/url.esk:210-211`. Implementation:

```scheme
(define (base64url-encode str)
  (b64url-strip-padding (b64url-replace-encode (base64-encode-string str))))
```

#### `(base64url-decode str)` → string

Source: `lib/core/url.esk:213-214`. Accepts both padded and unpadded forms.
Pads back to a length divisible by 4 (`b64url-restore-padding`,
`lib/core/url.esk:193-200`).

```scheme
(define (base64url-decode str)
  (base64-decode-string (b64url-restore-padding (b64url-replace-decode str))))
```

### Worked example: encode/decode round-trips

```scheme
(require stdlib)

(define samples (list "" "f" "fo" "foo" "foob" "fooba" "foobar"
                       "Hello, World!" "??>?<"))

(for-each
  (lambda (s)
    (let ((roundtrip (base64url-decode (base64url-encode s))))
      (display "input: ")  (display s)
      (display "  enc: ")  (display (base64url-encode s))
      (display "  ok: ")   (display (string=? s roundtrip))
      (newline)))
  samples)
```

(Source: regression-tested by
`tests/v1_2_edge_cases/base64url_test.esk:27-35`.)

Sample output for `(base64url-encode "??>?<")` is `"Pz8-Pzw"` — note the `-`
where a standard-base64 encoder would emit `+`, confirming the URL-safe
substitution.

### Edge cases

- `(base64-encode '())` → `""`.
- `(base64-decode "")` → `'()` (empty byte list); `(base64-decode-string "")` → `""`.
- `(base64url-encode "")` → `""`.
- `(base64url-decode "SGVsbG8=")` → `"Hello"` — canonical padded form is
  accepted.
- `(base64url-decode "SGVsbG8")` → `"Hello"` — unpadded form is also
  accepted.
- The `=`-character in `base64-value` decodes to `0` (padding); the strip /
  restore step ensures it never reaches `base64-decode-helper` on the URL-safe
  path.

### Cross-references

- For raw-byte handling (rather than string round-trip) use
  `base64-encode` and `base64-decode` directly; they operate on byte lists.
- For non-text payloads (image bytes, signed-cookie keys), prefer
  `base64url-encode` over `base64-encode` because the result fits in a URL
  query string with no further escaping.

---

## 11. `call-with-values` finalisation (codegen builtin)

**Source**: parser dispatch at `lib/frontend/parser.cpp:981, 3309-3363`;
codegen at `lib/backend/llvm_codegen.cpp:17684-17808`.
**Module require line**: none — `call-with-values` is a parser-recognised
special form, available everywhere.

`call-with-values` is the R7RS multiple-return-value plumbing primitive. The
v1.2-scale work was *not* the introduction of the form (it has existed since
v1.1) but the **named-consumer resolution fix** logged in
[`CHANGELOG.md:758`](../CHANGELOG.md) and the **stdlib-named consumer
routing** at line 718.

### Surface

```scheme
(call-with-values producer consumer)
```

- `producer` — a thunk (zero-argument procedure) that returns either a single
  value or multiple values via `(values v1 v2 …)`.
- `consumer` — a procedure that receives the producer's values as its
  arguments.

The codegen handles three cases:

1. `producer` returns no values via `(values)`: `consumer` is called with no
   arguments (`llvm_codegen.cpp:17789`).
2. `producer` returns exactly N values via `(values v1 … vN)`: `consumer` is
   called with N arguments — there is a small switch over N
   (`llvm_codegen.cpp:17781`).
3. `producer` returns a single ordinary value: `consumer` is called with one
   argument.

### What v1.2-scale fixed

Before the named-consumer routing fix, an expression like

```scheme
(call-with-values
  (lambda () (values 1 2 3))
  list)         ; consumer is the named stdlib function `list`
```

failed because the codegen could not resolve the bare symbol `list` to its
honest first-class procedure value (the symbol resolution wandered into the
inline-expansion path used for direct calls). The fix at `llvm_codegen.cpp:8172-8188`
forces bare-symbol consumers to evaluate as honest values, so any stdlib
function that takes the right number of arguments can sit in the consumer
slot.

### `(values v1 …)` → multi-value

Companion form: `values` packages zero or more results so that a downstream
`call-with-values` consumer can deconstruct them. Outside a `call-with-values`
context, `(values x)` is equivalent to `x` and `(values)` produces an
"unspecified" / discarded value.

### Worked example: pair-returning division

```scheme
;; A producer that returns both quotient and remainder.
(define (divmod a b)
  (values (quotient a b) (remainder a b)))

;; Consumer destructures the two return values.
(call-with-values
  (lambda () (divmod 17 5))
  (lambda (q r)
    (display "quotient: ") (display q) (newline)
    (display "remainder: ") (display r) (newline)))
;; → quotient: 3
;; → remainder: 2
```

### Worked example: stdlib-named consumer (the v1.2 fix in action)

```scheme
(define triple-as-list
  (call-with-values
    (lambda () (values 1 2 3))
    list))                                ; bare-symbol stdlib consumer

(display triple-as-list) (newline)        ; → (1 2 3)
```

Prior to the v1.2 routing fix this raised a "consumer is not a procedure" or
"Unknown function" error. After the fix it returns `(1 2 3)`.

### Cross-references

- For exception-based multi-value plumbing (return-and-continuation), see
  `guard` / `raise` and `call-with-current-continuation`.
- `dynamic-wind` does *not* interact with `values` — the multi-value packet
  flows through the call frame, not through the wind protocol.

---

## 12. Cross-references and module loading

### Which modules are auto-loaded by `(require stdlib)`?

Inspect `lib/stdlib.esk` for the canonical list. As of v1.2.1-scale, the
auto-loaded set includes:

- `core.io`
- `core.operators.arithmetic`, `core.operators.compare`
- `core.logic.predicates`, `core.logic.types`, `core.logic.boolean`
- `core.functional.compose`, `core.functional.curry`, `core.functional.flip`
- `core.control.trampoline`
- `core.list.*` (compound, generate, transform, query, sort, higher_order,
  search, convert)
- `core.strings`
- `core.files`
- `core.sexp`
- `core.json`
- `core.data.csv`, `core.data.dataframe`, `core.data.base64`
- `core.plot`
- `core.reflection`
- `core.url`
- `core.streams`
- `core.json_schema`
- `signal.fft`, `signal.filters`
- `ml.optimization`

Modules that exist but **must be explicitly required** (and are not pulled in
by `stdlib`):

- `core.testing` — testing framework. The comment at `lib/stdlib.esk:64-69`
  explains: baking it into `stdlib.o` triggers symbol-renamer / external-decl
  path interactions that currently mis-handle pre-compiled modules' internal
  state.
- `core.cache` — same family of considerations; opt-in keeps the auto-load
  surface tight.
- `core.argparse` — only useful for programs that read `(command-line)`; no
  reason to pay its load cost in libraries.
- `random.random` — pulls in the distribution-sampling helpers, which most
  programs don't need.
- `lib/agent/regex.esk` — loaded via `(load …)` or by the JIT force-load
  mechanism, not by `(require …)`.

### Which surfaces are codegen builtins (no `require` needed)?

- Time API: `current-time-ns`, `current-timestamp`, `format-iso8601`,
  `parse-iso8601`, `format-relative`, `local-timezone-offset`,
  `monotonic-time-ms`.
- PRNG isolation: `make-prng`, `prng-random`, `prng-random-integer`, `prng?`,
  and the global `set-random-seed!`.
- Reflection: `procedure-arity` (the `.esk`-level `describe` and `type-name`
  in `core.reflection` use it).
- URL parser: `url-parse`.
- Multi-value plumbing: `call-with-values`, `values`.

### Self-check methodology

For each procedure cited above, the symbol was verified to be present in the
implementing file's `(provide …)` block (for `.esk` modules) or in the codegen
dispatch table (for builtins). The `function_return_types` table in
`lib/backend/llvm_codegen.cpp:1174-1183` and the `vm_prelude_cache.h:1064-1087`
exported-name lists serve as secondary cross-references for the codegen-builtin
surfaces.

Modules from the original brief that turned out not to exist under the listed
name:

| CHANGELOG label   | Actual location                                                  |
|-------------------|------------------------------------------------------------------|
| `core.time`       | Not a module. Codegen builtins in `lib/core/system_builtins.c`.  |
| `core.regex`      | Lives at `lib/agent/regex.esk` (not `lib/core/`).                |
| `core.random`     | Lives at `lib/random/random.esk` (not `lib/core/`); per-stream isolation primitives are codegen builtins backed by `lib/core/prng.cpp`. |
| `core.control` (call-with-values) | `call-with-values` is a parser-recognised special form, not a stdlib module. `lib/core/control/trampoline.esk` exists but only provides `trampoline`, `bounce`, `done`. |
| `core.memoize`    | Lives at `lib/core/cache.esk` (the CHANGELOG and brief use "cache" and "memoize" interchangeably; the actual module is `core.cache`). |

### Where to look next

- The v1.2 hardening plan (`docs/V1.2_HARDENING_PLAN.md`) lists the audit
  blockers each of these modules closes.
- The standalone VM's surface for the *same* primitives is documented in
  [`docs/STDLIB_EXTENSIONS.md`](STDLIB_EXTENSIONS.md). The shapes sometimes
  differ — for example, `regex-match-groups` returns a list in the LLVM/JIT
  surface but an alist in the VM surface — so cross-check before porting
  code between targets.
- For changes in v1.3-evolve and later, follow the per-release section of
  [`CHANGELOG.md`](../CHANGELOG.md).
