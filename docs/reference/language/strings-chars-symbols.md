# Characters, Strings, and Symbols

## Characters

Character literals use the `#\` prefix.

| Literal | Character |
|---------|-----------|
| `#\a` | the letter a |
| `#\A` | the letter A |
| `#\space` | space |
| `#\newline` | newline |
| `#\tab` | tab |

```scheme
(display #\a) (newline)
(display (char->integer #\A)) (newline)
(display (integer->char 97)) (newline)
(display (char-upcase #\a)) (newline)
(display (list (char-alphabetic? #\x) (char-numeric? #\5))) (newline)
```
```
a
65
a
A
(#t #t)
```

Comparison predicates: `char=?`, `char<?`, `char>?`, `char<=?`, `char>=?`
(and case-insensitive `char-ci=?` etc.).

## Strings

Strings are sequences of characters. Common operations:

```scheme
(display (string-append "foo" "bar")) (newline)
(display (string-length "hello")) (newline)
(display (substring "hello world" 0 5)) (newline)
(display (string-ref "abc" 1)) (newline)
(display (list (string=? "a" "a") (string<? "a" "b"))) (newline)
```
```
foobar
5
hello
b
(#t #t)
```

Conversions between strings, lists, symbols, and numbers:

```scheme
(display (string->list "abc")) (newline)
(display (list->string (list #\x #\y))) (newline)
(display (string->number "42")) (newline)
(display (number->string 3.5)) (newline)
(display (string->symbol "hello")) (newline)
(display (symbol->string 'foo)) (newline)
```
```
(a b c)
xy
42
3.5
hello
foo
```

> Strings may contain embedded NUL bytes; string length and output are tracked by
> the string header's stored size (not C `strlen`), so binary payloads round-trip
> through `display`.

## String interpolation `~{ }`

Inside a string literal, `~{expr}` is replaced by the displayed value of `expr`,
evaluated in the current scope. Any expression is allowed.

```scheme
(define who "Ada")
(define n 42)
(display "Hi ~{who}, n=~{n}, 2+2=~{(+ 2 2)}") (newline)
```
```
Hi Ada, n=42, 2+2=4
```

## Symbols

Symbols are interned identifiers. `'name` (or `(quote name)`) yields a symbol.

```scheme
(display 'foo) (newline)
(display (symbol? 'foo)) (newline)
(display (symbol=? 'a 'a)) (newline)
(display (eq? 'a 'a)) (newline)   ; interned: eq? works
```
```
foo
#t
#t
#t
```

> Note the quote-in-`guard` limitation (ESH-0106): inside a `guard` form use
> `(quote name)` rather than `'name`. See
> [error-handling.md](error-handling.md) and
> [quote-and-quasiquote.md](quote-and-quasiquote.md).
