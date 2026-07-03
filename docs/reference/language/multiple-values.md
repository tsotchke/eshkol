# Multiple Values

## `values`

```
(values obj …)
```
Produces zero or more values. A single-value `values` behaves like the plain
value; multiple values must be received by `call-with-values` or `let-values`.

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
Binds the values returned by each `producer` to the corresponding variables, then
evaluates `body`.

```scheme
(define (divmod a b) (values (quotient a b) (remainder a b)))
(let-values (((q r) (divmod 17 5)))
  (display q) (display " ") (display r) (newline))
```
```
3 2
```

## Known issue — `define-values` is not supported

The top-level `define-values` binding form does not work: the names it should
bind are reported as undefined.

```scheme
(define-values (x y) (values 10 20))
(display (+ x y)) (newline)
```
```
error: Undefined variable: y
error: Undefined variable: x
```
**Workaround:** use `let-values` for multiple-value destructuring, or `call-with-values`
with an explicit consumer.
