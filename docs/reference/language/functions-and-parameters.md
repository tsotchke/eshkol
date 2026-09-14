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
(display (apply + (list 1 2 3))) (newline)
```
```
15
6
```

`proc` may be **any** callable value, including a builtin reached as a
first-class value rather than in operator position:

```scheme
(display (apply vector-copy (list (vector 7 8 9)))) (newline)
(display (apply vector (list 1 2 3))) (newline)
```
```
#(7 8 9)
#(1 2 3)
```

`apply` resolves its operator through the same first-class-value route `map`
and a user higher-order call use, so a builtin gains a value representation
exactly once and every call site agrees with it by construction. A name that is
genuinely undefined fails compilation with a real diagnostic; it never
silently answers `()`.

> **Engine difference — apply's leading-args form is native-only.** The
> bytecode VM supports the plain `(apply proc arg-list)` form. It does **not**
> support leading arguments before the list, for any operator: under the VM,
> `(apply + 1 2 (list 3 4 5))` raises `arity mismatch: expected 2 arguments,
> got 4` rather than answering `15`. This is argument-list construction, a
> different code path from operator resolution. Write `(apply + (append (list
> 1 2) (list 3 4 5)))` for a form that runs on both engines.

## Builtins are first-class values

Every procedure Eshkol can call in operator position — `(name arg …)` — is
also a value like any other: it can be bound with `let`/`define`, passed to
a user procedure, stored in a data structure, and mapped or applied over.
This holds for the entire builtin surface (`vector-copy`, `string<?`,
`hash-table-ref`, `expt`, …), not only for procedures the program itself
defines.

```scheme
(define (twice f x) (f (f x)))
(display (twice abs -5)) (newline)              ; abs passed as a value
(display (map vector-copy (list (vector 1 2)))) (newline)
(display (let ((f expt)) (f 2 10))) (newline)
```
```
5
(#(1 2))
1024
```

The one exception is a **special form** — `if`, `define`, `lambda`, `quote`,
`set!`, the `let` family, and the rest of the forms the parser binds to a
dedicated AST node rather than dispatching by name at the call site (see
[special-forms.md](special-forms.md)). A special form has no runtime value:
referencing one bare, the way `abs` or `vector-copy` can be referenced above,
is a compile-time error (`Undefined variable: <name>`), not a first-class
procedure with unusual behavior.

```scheme
(display if)   ; error: Undefined variable: if
```

### A variadic builtin stays variadic as a value

A builtin whose operator-position form takes any number of arguments answers
the same way when it is reached as a value. It is not frozen at the arity of
the call site that first materialized it, and its rest list is not silently
truncated or terminated with the wrong tail.

```scheme
(display (map list (list 1 2 3))) (newline)
(display (map vector (list 1 2) (list 3 4))) (newline)
(define f string-append)
(display (f "a" "b" "c")) (newline)
(define g min)
(display (g 5 2 9)) (newline)
```
```
((1) (2) (3))
(#(1 3) #(2 4))
abc
2
```

Identical under the bytecode VM. The first-class builtin table declares which
rows are variadic and how each computes its answer from a rest list — identity
for `list` and `values`, a unary builtin for `vector` (`list->vector`) and
`string` (`list->string`), and a left fold over the binary form for
`string-append`, `min`, `max`, `gcd`, `lcm`, `vector-append` and
`bytevector-append` — so a variadic builtin has one answer rather than one per
call site.

Every builtin's value-position behavior is asserted mechanically —
generated from the language-surface manifest, not hand-picked — in
`tests/core/builtins_first_class_test_*.esk`; special forms' refusal is
pinned by `tests/core/special_form_value_refusal_test.esk`. See LE-16 in
`.icc/ledger/entries/LE-16.yaml` for how this was closed for the builtins
that were still call-position-only, including the 28 names left open there as
documented gaps rather than guessed, because resolving them safely would have
required executing an FFI/GPU/atomics side effect.

## Omitting a documented optional argument

A builtin's documented optional argument is a legal call on **both** engines.
The minimum arities the VM enforces are derived from the code that runs — the
fixed-arity macros the native dispatch expands first — rather than transcribed
into a table, so the two engines cannot drift apart on which calls are legal.

```scheme
(display (substring "hello" 1)) (newline)
(display (append)) (newline)
(display (gcd)) (newline)
(display (make-vector 3)) (newline)
(display (string-length (make-string 3))) (newline)
```
```
ello
()
0
#(0 0 0)
3
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
