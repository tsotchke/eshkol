# `core.argparse` — minimal CLI argument parser

**Source**: [`lib/core/argparse.esk`](../../../lib/core/argparse.esk)
**Require**: `(require core.argparse)` — **NOT** auto-loaded via `stdlib`; require it explicitly.

A small long-flag parser. You supply a **spec** — a list of positional 4-tuples `(name type default desc)` — and an **argv** list of strings. `parse-args` returns a parsed result `(flags-alist positionals)` from which you read typed values. Type ∈ `'string 'integer 'number 'boolean`.

Flag conventions:
- `--name value` — long flag, value is the next token
- `--name=value` — long flag, inline value
- `--name` — boolean flag → `#t`
- `--no-name` — boolean negation → `#f`
- `--` — end-of-options; everything after is positional
- Unknown flags are collected as positional strings (caller decides to reject or forward).

> For docs and tests, pass a **literal argv list** rather than `(command-line)`, so output is deterministic.

## Functions

### `(parse-args argv spec)`
Parse `argv` against `spec`. Returns `(flags positionals)` where `flags` is an alist of `(name . value)` seeded with the spec defaults and `positionals` is a list of leftover strings in original order.

### `(arg-get parsed key)`
Look up a flag value by its symbol key (e.g. `'--name`). Returns the (already type-converted) value or `#f` if absent from the alist.

### `(arg-positional parsed)`
Return the list of positional arguments.

### `(arg-has? parsed key)`
`#t` if `key`'s value is neither `#f` nor `'()` — a convenience truthiness check.

### `(argparse-help progname spec)`
Return a formatted usage/help string built from the spec descriptions.

```scheme
;; argparse.esk
(require core.argparse)
(define spec
  (list (list '--name    'string  "World" "Your name")
        (list '--count   'integer 1       "How many")
        (list '--verbose 'boolean #f      "Verbose")))
(define a (parse-args
            (list "--name" "Ada" "--count=3" "--verbose" "pos1" "extra")
            spec))
(display (arg-get a '--name)) (newline)      ; Ada
(display (arg-get a '--count)) (newline)     ; 3 (integer)
(display (arg-get a '--verbose)) (newline)   ; #t
(display (arg-positional a)) (newline)       ; (pos1 extra)
(display (arg-has? a '--missing)) (newline)  ; #f
;; --no- negation and defaults
(define b (parse-args (list "--no-verbose") spec))
(display (arg-get b '--verbose)) (newline)   ; #f
(display (arg-get b '--name)) (newline)      ; World (default)
;; end-of-options
(define c (parse-args (list "--" "--name" "x") spec))
(display (arg-positional c)) (newline)       ; (--name x)
(display (argparse-help "myprog" spec))
```
```
Ada
3
#t
(pos1 extra)
#f
#f
World
(--name x)
Usage: myprog [OPTIONS] [ARGS...]

Options:
  --name <string>    Your name
  --count <integer>    How many
  --verbose <boolean>    Verbose
```

Edge cases: a `boolean` value via `=`/next-token treats `"false"` and `"0"` as `#f`, anything else as `#t`. `integer`/`number` conversion uses `string->number`, so a non-numeric value yields `#f` (no error). A long flag with no following value (e.g. trailing `--name`) leaves the default in place. Unknown flags such as `--frob` fall through to the positionals list unchanged.
