# `core.json` — JSON parsing and serialization

**Source**: [`lib/core/json.esk`](../../../lib/core/json.esk)
**Require**: `(require core.json)` — auto-loaded via `(require stdlib)`.

Pure-Eshkol JSON reader/writer. Value mapping:

| JSON | Eshkol |
|------|--------|
| object | hash-table |
| array | list |
| string | string |
| number | integer or real |
| `true` / `false` | `#t` / `#f` |
| `null` | `'()` |

Because both JSON `null` and "unparseable input" map to `'()`, use
`json-try-parse` (Result-typed) when you must distinguish them. This module is
also summarised in [`docs/STDLIB_V1_2_API.md`](../../STDLIB_V1_2_API.md). No bugs
were observed.

## Parsing

### `(json-parse str)`
Permissive parser: returns the parsed value, or a degenerate `'()` on malformed
or partial input (cannot signal failure). Kept for backwards compat.

```scheme
(require core.json)
(display (json-parse "42")) (newline)
(display (json-parse "-3.14")) (newline)
(display (json-parse "\"hi\"")) (newline)
(display (json-parse "true")) (newline)
(display (json-parse "null")) (newline)
(display (json-parse "[1,[2,3],4]")) (newline)
```
```
42
-3.14
hi
#t
()
(1 (2 3) 4)
```

### `(json-try-parse str)` — Result-typed
Returns a **Result**: `(json-ok . value)` on success, `(json-err . "reason")` on
failure. Does structural balance checking and rejects trailing garbage, so it
catches partial input that `json-parse` would silently accept.

```scheme
(define r (json-try-parse "{\"k\":1}"))
(display (json-result-ok? r)) (newline)                 ; #t
(display (json-result-error (json-try-parse "{\"k\":"))) (newline)
(display (json-result-error (json-try-parse ""))) (newline)
(display (json-result-error (json-try-parse "42 xyz"))) (newline)
```
```
#t
json-try-parse: missing closing brace
json-try-parse: empty input
json-try-parse: trailing garbage after JSON value
```

### `(json-parse-result? r)` / `(json-result-ok? r)` / `(json-result-value r)` / `(json-result-error r)`
Predicates and accessors for the Result type. `json-parse-result?` recognises
either tag; `json-result-ok?` is true only for `(json-ok . _)`.
`json-result-value` extracts the value (errors on an Err); `json-result-error`
extracts the message (errors on an Ok).

```scheme
(define r (json-try-parse "{\"k\":1}"))
(display (json-parse-result? r)) (newline)          ; #t
(display (hash-table? (json-result-value r))) (newline)  ; #t
```
```
#t
#t
```

## Accessors

### `(json-get obj key [default])`
Look up `key` in a parsed object (hash-table). Returns `#f` if absent, or the
supplied `default`.

```scheme
(define obj (json-parse "{\"a\":1,\"b\":\"x\"}"))
(display (json-get obj "a")) (newline)          ; 1
(display (json-get obj "z")) (newline)          ; #f
(display (json-get obj "z" 99)) (newline)       ; 99
```
```
1
#f
99
```

### `(json-array-ref arr idx)`
Index into a parsed array (list) — thin wrapper over `list-ref`.

```scheme
(display (json-array-ref '(10 20 30) 1)) (newline)   ; 20
```
```
20
```

### `(json-get-in obj keys)`
Follow a path of keys into nested objects/arrays. String keys index
hash-tables; integer keys index lists. Returns `#f` if any step fails.

```scheme
(display (json-get-in (json-parse "{\"a\":{\"b\":[1,2,3]}}") '("a" "b" 2))) (newline)
```
```
3
```

## Serialization

### `(json-stringify value)`
Serialize an Eshkol value to a JSON string. Handles numbers, strings, booleans,
`'()`→`null`, symbols→strings, lists→arrays, hash-tables→objects, and **alists**
(lists of `(key . value)` cons cells with string/symbol keys)→objects.

```scheme
(display (json-stringify 42)) (newline)
(display (json-stringify "hi")) (newline)
(display (json-stringify '(1 2 3))) (newline)
(display (json-stringify #t)) (newline)
(display (json-stringify '())) (newline)
(display (json-stringify '(("a" . 1) ("b" . 2)))) (newline)
```
```
42
"hi"
[1,2,3]
true
null
{"a":1,"b":2}
```

### `(alist->json alist)`
Convenience: serialize an association list as a JSON object (via a hash-table).
Note key order is hash-table order, not insertion order.

```scheme
(display (alist->json '(("name" . "test") ("value" . 42)))) (newline)
```
```
{"name":"test","value":42}
```

## Hash-table / alist conversion

### `(hash-table->alist ht)` / `(alist->hash-table alist)`
Convert between a hash-table and an alist of `(key . value)` pairs.

```scheme
(define ht (alist->hash-table '(("a" . 1) ("b" . 2))))
(display (hash-ref ht "a" #f)) (newline)            ; 1
(display (hash-table->alist ht)) (newline)          ; order is hash order
```
```
1
((b . 2) (a . 1))
```

## File / port I/O

### `(json-write value target)`
Write `value` as JSON to `target`, which may be an output port or a path string.
Returns `#t` on success, `#f` otherwise.

### `(json-write-file filename data)`
Write `data` as JSON to a file (opens/closes the file).

### `(alist-write-json filename alist)`
Write an alist as a JSON object to a file.

### `(json-read source)`
Read and parse JSON from an input port or a path string. Ports are left open;
paths are opened and closed.

### `(json-read-file filename)`
Open a file, read all lines, parse the concatenation as JSON, and close.

```scheme
(define ht (alist->hash-table '(("a" . 1) ("b" . 2))))
(json-write-file "/tmp/d.json" ht)
(display (json-get (json-read-file "/tmp/d.json") "a")) (newline)   ; 1
(json-write '(("x" . 10)) "/tmp/e.json")
(display (json-get (json-read "/tmp/e.json") "x")) (newline)        ; 10
(alist-write-json "/tmp/f.json" '(("q" . 7)))
(display (json-get (json-read-file "/tmp/f.json") "q")) (newline)   ; 7
```
```
1
10
7
```
