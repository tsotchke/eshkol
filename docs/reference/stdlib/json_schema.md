# `core.json_schema` — JSON Schema (Draft 7 subset) validator

**Source**: [`lib/core/json_schema.esk`](../../../lib/core/json_schema.esk)
**Require**: `(require core.json_schema)` — auto-loaded via `(require stdlib)`.

Validates values produced by [`core.json`](json.md)'s `json-parse` against a
schema that is itself expressed as parsed JSON (hash-tables / lists). This module
is also summarised in [`docs/STDLIB_V1_2_API.md`](../../STDLIB_V1_2_API.md). No
bugs were observed.

**Supported keywords**: `type` (single or list of
`object|array|string|number|integer|boolean|null`), `properties`, `required`,
`additionalProperties` (`#f` rejects extras), `items`, `minItems`/`maxItems`,
`minLength`/`maxLength`, `minimum`/`maximum`,
`exclusiveMinimum`/`exclusiveMaximum`, `enum`, `const`, `pattern` (substring
containment, **not** full regex), `oneOf`, `anyOf`, `allOf`, `not`. A boolean
schema `#t` always passes and `#f` always fails.

Error strings carry a JSON-pointer-style path; the top-level path is `""`, so
top-level errors read as `": message"`.

## Functions

### `(json-schema-valid? schema value)`
Return `#t` if `value` conforms to `schema`, else `#f`.

```scheme
(require core.json_schema)
(require core.json)
(define (sch s) (json-parse s))
(display (json-schema-valid? (sch "{\"type\":\"integer\"}") 42)) (newline)
(display (json-schema-valid? (sch "{\"type\":\"integer\"}") "x")) (newline)
(display (json-schema-valid? (sch "{\"type\":[\"integer\",\"string\"]}") "x")) (newline)
```
```
#t
#f
#t
```

### `(json-schema-validate schema value)`
Return a **list of error strings** (empty ⇒ valid). Each string is
`"<path>: <message>"`.

```scheme
(display (json-schema-validate (sch "{\"type\":\"integer\"}") "x")) (newline)
(display (json-schema-validate
   (sch "{\"type\":\"object\",\"required\":[\"age\"]}")
   (json-parse "{\"name\":\"x\"}"))) (newline)
(display (json-schema-validate
   (sch "{\"type\":\"array\",\"items\":{\"type\":\"integer\"}}")
   (json-parse "[1,\"x\",3]"))) (newline)
(display (json-schema-validate (sch "{\"minimum\":10,\"maximum\":20}") 5)) (newline)
(display (json-schema-validate (sch "{\"minLength\":3}") "ab")) (newline)
```
```
(: expected integer, got string)
(/age: missing required property)
(/1: expected integer, got string)
(: value < minimum=10)
(: string shorter than minLength=3)
```

### Keyword coverage examples

Object properties, `additionalProperties: false`, combinators, `enum`/`const`,
and `pattern` (substring match) all validate as expected:

```scheme
(display (json-schema-valid?
  (sch "{\"type\":\"object\",\"properties\":{\"age\":{\"type\":\"integer\"}},\"required\":[\"age\"]}")
  (json-parse "{\"age\":30}"))) (newline)                         ; #t
(display (json-schema-validate
  (sch "{\"type\":\"object\",\"properties\":{\"a\":{}},\"additionalProperties\":false}")
  (json-parse "{\"a\":1,\"b\":2}"))) (newline)                    ; (/b: additional property not allowed)
(display (json-schema-valid? (sch "{\"pattern\":\"@\"}") "a@b")) (newline)   ; #t (substring)
(display (json-schema-valid? (sch "{\"enum\":[1,2,3]}") 2)) (newline)         ; #t
(display (json-schema-valid? (sch "{\"const\":5}") 5)) (newline)              ; #t
(display (json-schema-valid? (sch "{\"oneOf\":[{\"type\":\"integer\"},{\"type\":\"string\"}]}") 5)) (newline)  ; #t
(display (json-schema-valid? (sch "{\"anyOf\":[{\"type\":\"integer\"},{\"type\":\"string\"}]}") "x")) (newline) ; #t
(display (json-schema-valid? (sch "{\"not\":{\"type\":\"string\"}}") 5)) (newline)  ; #t
```
```
#t
(/b: additional property not allowed)
#t
#t
#t
#t
#t
#t
```

### Boolean schemas

```scheme
(display (json-schema-valid? #t 42)) (newline)          ; #t (always valid)
(display (json-schema-validate #f 42)) (newline)        ; rejected
```
```
#t
(: schema is `false` — value rejected)
```

Edge cases: `pattern` is substring containment only — `"^v"` would require the
literal characters `^v` to appear, since no regex engine is used. The `"number"`
type accepts both integers and reals; `"integer"` is the strict subset.
