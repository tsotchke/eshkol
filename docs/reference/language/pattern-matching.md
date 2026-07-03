# Pattern Matching: `match`

```
(match subject clause …)
```
where each clause is `(pattern body …)`. Clauses are tried top-to-bottom; the
body of the first matching pattern is evaluated and returned. Variables in the
pattern are bound in the body.

## Supported patterns

| Pattern | Matches |
|---------|---------|
| `_` | anything (wildcard), binds nothing |
| `var` | anything, binds `var` |
| literal (`5`, `"s"`) | that literal value |
| `(list p …)` | a list of exactly that many elements, matching each `p` |
| `(cons p1 p2)` | a pair, `p1` = car, `p2` = cdr |
| `(? pred)` | any value for which `(pred value)` is true |
| `(? pred var)` | as above, and binds `var` |

## Examples

```scheme
(define (f v) (match v ((list a b) (+ a b)) (_ -1)))
(display (f (list 10 20))) (newline)
(display (f (list 1 2 3))) (newline)
```
```
30
-1
```

Predicate patterns dispatch on type. The built-in R7RS predicates
(`number?`, `string?`, `symbol?`, `pair?`, `null?`, …) work directly:

```scheme
(define (classify v)
  (match v
    ((? number?) "number")
    ((? string?) "string")
    ((? symbol?) "symbol")
    ((? pair?)   "pair")
    (_           "other")))
(display (classify 42)) (newline)
(display (classify "x")) (newline)
(display (classify (quote foo))) (newline)
```
```
number
string
symbol
```

`cons` destructuring:

```scheme
(display (match (list 1 2 3) ((cons h t) h) (_ 0))) (newline)
```
```
1
```

## Known issue — apostrophe-quote in the subject position

Using `'datum` as the `match` **subject** hangs silently. Use the explicit
`(quote datum)` form instead:

```scheme
;; hangs:   (match 'foo ((? symbol?) "sym") (_ "other"))
;; works:
(display (match (quote foo) ((? symbol?) "sym") (_ "other"))) (newline)
```
```
sym
```
This is the same quote-dispatch family as the `guard` issue (ESH-0106). The
project's `match` predicate test suite (`(? pred)` patterns, wildcard, list/cons
destructuring, multi-clause dispatch) passes 21/21 with this convention.
