# `core.io` — first-class output wrappers

**Source**: [`lib/core/io.esk`](../../../lib/core/io.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.io)`

Named wrappers around `display` so output can be used as a first-class procedure (e.g. with `for-each`).

## Functions

### `(print x)`
`(display x)` — writes `x` to standard output with no trailing newline. Returns unspecified.

### `(println x)`
`(display x)` followed by `(newline)`.

```scheme
;; io.esk
(require stdlib)
(print "hello") (newline)
(println "world")
(for-each println '(1 2 3))
```
```
hello
world
1
2
3
```

Edge cases: `print`/`println` take exactly one argument and use `display` semantics (human-readable, strings unquoted) rather than `write` semantics.
