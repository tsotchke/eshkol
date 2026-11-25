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
| `#\x41` | the character with Unicode scalar value U+0041 (here `A`) |

`#\x` followed by one or more hexadecimal digits and a delimiter names a
character by its Unicode scalar value (R7RS 7.1.1), on every engine. `#\x`
alone is the letter x. A value that is a surrogate (U+D800 to U+DFFF) or above
U+10FFFF is a read error.

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

`char-alphabetic?`, `char-numeric?`, `char-whitespace?`, `char-upper-case?`,
and `char-lower-case?` classify Unicode codepoints using the R7RS Unicode
properties Alphabetic, Numeric_Digit (`Nd`), White_Space, Uppercase, and
Lowercase. The result is locale-independent, so Latin-1, non-Latin, and
multibyte UTF-8 characters have the same meaning on the native and VM engines.

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

> Both quote spellings — `'name` and `(quote name)` — behave identically
> everywhere, including inside a `guard` form (**ESH-0106**, closed). See
> [error-handling.md](error-handling.md) and
> [quote-and-quasiquote.md](quote-and-quasiquote.md).

### Reserved spellings

The compiler names the bindings it generates with spellings that no program
can write, so a generated binding never captures a program's identifier and a
program never reaches a generated one. Both readers refuse these spellings with
the diagnostic `reserved syntax identifier`, including inside a quoted datum
and inside `|vertical bars|`:

| Spelling | Used for |
|----------|----------|
| any name containing the byte U+001D | the macro expander's syntax colors |
| `_v<digits>.` followed by anything | the macro expander's fresh binder names |
| `__eshkol_macro_binding_<digits>` | macro-introduced definitions |
| `__eshkol_desugar_` followed by anything | temporaries of the forms the parser lowers (`parameterize`, `case-lambda`, keyword parameters, `define-values`) |

A symbol made at run time with one of these spellings (`string->symbol`) is an
ordinary symbol. Passing a datum that contains one to `eval` or `compile`
raises a condition naming the symbol.
