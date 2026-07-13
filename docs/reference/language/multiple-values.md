# Multiple Values

## `values`

```
(values obj …)
```
Produces zero or more values. A single-value `values` behaves like the plain
value. Zero and multiple values are represented by an opaque values packet and
must be received by `call-with-values`, `let-values`, `let*-values`, or
`define-values`; ordinary vectors and other heap objects remain exactly one
value.

## `call-with-values`

```
(call-with-values producer consumer)
```
Calls `producer` (a thunk) and applies `consumer` to the values it produced.

```scheme
(call-with-values
  (lambda () (values 1 2 3))
  (lambda (a b c) (display (+ a b c)) (newline)))
```
```
6
```

## `let-values`

```
(let-values (((var …) producer) …) body …)
```
Binds the values returned by each `producer` to the corresponding variables,
then evaluates `body`. All `let-values` producers are evaluated in the original
outer environment before any new binding is installed. `let*-values` evaluates
and binds clauses from left to right, so each later producer sees earlier
bindings. A value-count mismatch raises an error rather than padding or dropping
values.

```scheme
(define (divmod a b) (values (quotient a b) (remainder a b)))
(let-values (((q r) (divmod 17 5)))
  (display q) (display " ") (display r) (newline))
```
```
3 2
```

## `define-values`

```
(define-values formals producer)
```

Evaluates `producer` exactly once and defines every identifier in `formals`.
Fixed, dotted-rest, identifier-rest, and zero-value forms are supported in both
the native compiler and hosted VM.

```scheme
(define-values (x y) (values 10 20))
(display (+ x y)) (newline)
```
```
30
```

```scheme
(define-values all (values 2 3 5))
(define-values (head . tail) (values 7 11 13))
```
