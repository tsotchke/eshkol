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

## Known Issues

### ESH-0104 — long forms are not wired

Only the reader sugar (``` ` ```, `,`, `,@`) is interpreted. The written-out
long forms `(quasiquote …)`, `(unquote …)`, `(unquote-splicing …)` are treated as
plain data and left unsubstituted.

```scheme
(define x 5)
(display (quasiquote (a (unquote x) b))) (newline)
```
```
(a (unquote x) b)
```
Expected (R7RS 4.2.8): `(a 5 b)`. **Workaround:** use the reader sugar
``` `(a ,x b) ```, which evaluates correctly. `quote`'s long form `(quote …)` is
unaffected — it works.

### ESH-0107 — nested quasiquote collapses to `()`

Any `quasiquote` nested inside another `quasiquote` (level ≥ 2) evaluates to `()`.

```scheme
(display `a) (newline)        ; single level: correct
(display ``a) (newline)       ; nested: wrong
(display `(a `(b ,(+ 1 1)))) (newline)
```
```
a
()
(a ())
```
Single-level quasiquote/unquote/unquote-splicing are correct; only nesting is
affected.

### ESH-0106 — `'sym` sugar inside `guard` is misread

The `'` reader sugar anywhere inside a `guard` form (clause bodies **and** the
`raise` argument) is compiled as a *variable reference* rather than a quoted
symbol.

```scheme
(display (guard (e (#t 'sym)) (raise 1))) (newline)
```
```
error: Undefined variable: sym
()
```
**Workaround:** use the explicit `(quote sym)` form inside `guard`, which works
everywhere. See [error-handling.md](error-handling.md).
