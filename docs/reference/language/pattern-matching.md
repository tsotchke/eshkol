# Pattern Matching: `match`

```
(match subject clause …)
```
The native JIT and AOT compiler try clauses in source order. A clause has
the form `(pattern body …)` and returns the value of its last expression.
A guarded clause can be written `(pattern #:when test body …)` or
`(pattern (when test) body …)`. The guard runs once after the complete
pattern matches, with the pattern variables in scope. A `#f` guard continues
with the next clause. A `when` expression with a body remains a body expression.

A match with clauses evaluates the subject once and raises an error if every
clause fails. A match with zero clauses returns the empty list and skips the
subject.

The bytecode VM in this revision does not support `match`.

## Supported patterns

| Pattern | Matches |
|---------|---------|
| `_` | anything (wildcard), binds nothing |
| `var` | anything, binds `var` |
| literal (`5`, `#\a`, `#t`, `'red`) | the same value under `eqv?` |
| string literal | a string with the same contents under `equal?` |
| `(vector p …)` or `#(p …)` | a vector with exactly that many elements, each matching its pattern |
| a backquoted datum | literal symbols and structure, with ordinary subpatterns introduced by a comma |
| `(or p …)` | the first successful alternative |
| `(list p …)` | a list of exactly that many elements, matching each `p` |
| `(cons p1 p2)` | a pair (as `pair?` decides; a string or vector is not one), `p1` = car, `p2` = cdr |
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

A quasi-pattern can contain lists, dotted tails and vectors. A comma introduces
a single pattern. For a variable-length list tail, use a dotted tail as in
the `remaining` binding below.

```scheme
(match (list 'event (vector 'start 4) 'later)
  (`(event #(start ,n) . ,remaining) #:when (> n 0)
   (cons n remaining))
  (_ 'unknown))
;; => (4 later)
```

`cons` destructuring:

```scheme
(display (match (list 1 2 3) ((cons h t) h) (_ 0))) (newline)
```
```
1
```

## Quoted subjects

Either quote spelling works in the `match` **subject** position:

```scheme
(display (match 'foo ((? symbol?) "sym") (_ "other"))) (newline)
(display (match (quote foo) ((? symbol?) "sym") (_ "other"))) (newline)
```
```
sym
sym
```
The apostrophe form used to hang silently; that was the same quote-dispatch
family as **ESH-0106**, closed with it. The project's `match` predicate test
suite (`(? pred)` patterns, wildcard, list/cons destructuring, multi-clause
dispatch) passes 21/21.
