# Booleans and Predicates

## Boolean literals

`#t` (true) and `#f` (false). In conditional contexts, **every value except `#f`
is truthy** — including `0`, `""`, and `'()`.

```scheme
(display (list #t #f)) (newline)
(display (if 0 'truthy 'falsy)) (newline)   ; 0 is truthy
```
```
(#t #f)
truthy
```

## `not`

```
(not obj)   ; => #t iff obj is #f, else #f
```
```scheme
(display (not #f)) (newline)
(display (not 0)) (newline)
```
```
#t
#f
```

## Booleans are first-class

Booleans can be stored, passed, returned, and mapped over like any other value.

```scheme
(define b #t)
(display (if b 'yes 'no)) (newline)
(display (map (lambda (v) (not v)) (list #t #f #t))) (newline)
(display (eq? #t #t)) (newline)
```
```
yes
(#f #t #f)
#t
```

## Type predicates

The core R7RS type predicates are available and each returns `#t`/`#f`. They are
LLVM-inline builtins in the native path.

| Predicate | True when the argument is… |
|-----------|----------------------------|
| `boolean?` | a boolean |
| `null?` | the empty list `'()` |
| `pair?` | a cons pair |
| `list?` | a proper list |
| `number?` | any number |
| `integer?` | an integer |
| `real?` | a real number |
| `rational?` | a rational |
| `complex?` | a complex number |
| `string?` | a string |
| `char?` | a character |
| `symbol?` | a symbol |
| `procedure?` | a procedure |
| `vector?` | a vector |

```scheme
(display (list (null? '()) (pair? '(1)) (number? 3)
               (string? "a") (symbol? 'x) (procedure? car))) (newline)
```
```
(#t #t #t #t #t #t)
```

## Equality

- `eq?` — identity / pointer equality (interned symbols, small integers, booleans).
- `eqv?` — like `eq?` but reliable across numbers and characters.
- `equal?` — deep structural equality (recurses into pairs, strings, vectors).

```scheme
(display (eq? 'a 'a)) (newline)
(display (equal? '(1 2 3) (list 1 2 3))) (newline)
(display (equal? "abc" "abc")) (newline)
```
```
#t
#t
#t
```
