# Quote and Quasiquote

## `quote`

```
(quote datum)
'datum          ; reader sugar, identical to (quote datum)
```
Returns `datum` unevaluated. **Both spellings work identically.**

```scheme
(display '(1 2 3)) (newline)
(display 'sym) (newline)
(display (quote (a b c))) (newline)
```
```
(1 2 3)
sym
(a b c)
```

## `quasiquote` / `unquote` / `unquote-splicing`

```
`template        ; quasiquote
,expr            ; unquote — evaluate expr and insert its value
,@expr           ; unquote-splicing — splice the elements of a list value
```
A quasiquoted template is like `quote`, except that `,expr` is replaced by the
value of `expr`, and `,@expr` splices a list's elements into the surrounding list.

```scheme
(define x 5)
(display `(a ,x b)) (newline)
(display `(1 ,@(list 2 3) 4)) (newline)
```
```
(a 5 b)
(1 2 3 4)
```

### Long forms

The written-out forms `(quasiquote …)`, `(unquote …)` and
`(unquote-splicing …)` mean exactly what their reader sugar means (R7RS 4.2.8),
so a program may use either spelling:

```scheme
(define x 5)
(display (quasiquote (a (unquote x) b))) (newline)
(display (quasiquote (a (unquote-splicing (list 1 2)) b))) (newline)
```
```
(a 5 b)
(a 1 2 b)
```

### Nesting

Nested quasiquotation follows the R7RS level rule: an `unquote` is evaluated
only when its nesting level matches the quasiquote it belongs to, so an inner
template comes back as *data*, still carrying its unevaluated `unquote`.

```scheme
(display `a) (newline)
(display ``a) (newline)
(display `(a `(b ,(+ 1 1)))) (newline)
```
```
a
(quasiquote a)
(a (quasiquote (b (unquote (+ 1 1)))))
```

### `'sym` inside `guard`

The `'` reader sugar behaves the same inside a `guard` form — in a clause body
and in the `raise` argument alike — as it does anywhere else:

```scheme
(display (guard (e (#t 'sym)) (raise 1))) (newline)
```
```
sym
```
See [error-handling.md](error-handling.md).
