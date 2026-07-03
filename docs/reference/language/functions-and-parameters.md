# Functions and Parameters

See [special-forms.md](special-forms.md) for `define`/`lambda` basics. This page
covers parameter list features and application.

## Fixed-arity procedures

```scheme
(define (add a b) (+ a b))
(display (add 3 4)) (newline)
```
```
7
```

## Variadic (rest) parameters

A dotted tail parameter collects any extra arguments into a list.

```scheme
(define (sum . args) (apply + args))
(display (sum 1 2 3 4 5)) (newline)

(define (f a b . rest) (list a b rest))
(display (f 1 2 3 4 5)) (newline)
```
```
15
(1 2 (3 4 5))
```

A `lambda` whose entire formal list is a single symbol receives all arguments as
one list:

```scheme
(display ((lambda args args) 1 2 3)) (newline)
```
```
(1 2 3)
```

## `apply`

```
(apply proc arg … arg-list)
```
Calls `proc` with the leading args followed by the elements of the final list.

```scheme
(display (apply + 1 2 (list 3 4 5))) (newline)
```
```
15
```

## Keyword arguments (`#:name`)

Formals of the form `#:name binding` declare **keyword parameters**. Callers pass
them as `#:name value`. Keyword arguments may appear in any order and mix with
positional parameters and a rest parameter.

```scheme
(define (weighted x #:scale scale #:offset offset)
  (+ (* x scale) offset))
(display (weighted 10 #:offset 2 #:scale 4)) (newline)  ; reordered

(define kw-lambda
  (lambda (#:left left #:right right) (+ left right)))
(display (kw-lambda #:right 23 #:left 19)) (newline)

(define (mixed positional #:scale scale) (+ positional scale))
(display (mixed 19 #:scale 23)) (newline)
```
```
42
42
42
```

Keyword formals coexist with an explicit rest parameter:

```scheme
(define (explicit-rest positional #:scale scale . rest)
  (+ (* positional scale) (length rest)))
```

### Keyword-argument limitations

- **Keyword parameters are required.** There is no default-value syntax. Writing
  `(define (g #:k (k "default")) …)` fails to parse:
  `error: keyword formal requires a parameter name`. A caller that omits a
  declared keyword gets a runtime error.
- Each keyword formal is `#:kw name` — the keyword token immediately followed by
  the local parameter name that receives its value.

## Curried definition sugar is not supported

`(define ((f x) y) …)` does not parse. Use an explicit returned `lambda`:

```scheme
(define (adder x) (lambda (y) (+ x y)))
(display ((adder 3) 4)) (newline)
```
```
7
```
