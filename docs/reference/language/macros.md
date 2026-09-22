---
kind: reference
status: current
owner-area: language
since: v1.3.5
sources:
  - inc/eshkol/frontend/syntax_rules_core.h
  - inc/eshkol/frontend/syntax_color.h
  - lib/frontend/macro_expander.cpp
  - lib/backend/vm_macro.c
  - tests/vm_parity/corpus/93_macro_hygiene_matrix.esk
---
# Macros: `define-syntax`, `let-syntax`, `letrec-syntax`, `syntax-rules`

Eshkol macros are R7RS `syntax-rules` transformers (R7RS 4.3). A macro rewrites
the *syntax* of a use, as the reader produced it, into new syntax, which is then
compiled as if it had been written there. The native compiler (JIT, AOT, REPL)
and the bytecode VM (hosted, ESKB, browser) run one shared implementation of
`syntax-rules` and one renaming rule, so a macro means the same on every engine
([ADR-0026](../../design/adr/0026-syntax-rules-one-engine-one-renaming-rule.md)).

## Defining macros

```
(define-syntax keyword (syntax-rules (literal …) (pattern template) …))
(let-syntax ((keyword transformer) …) body …)
(letrec-syntax ((keyword transformer) …) body …)
```

`define-syntax` binds a keyword for the rest of the scope it appears in; at top
level the keyword is visible to the whole unit, including procedure bodies
written above the definition. `let-syntax` binds keywords for its body only, and
each transformer sees the keywords of the *enclosing* scope; `letrec-syntax`
transformers also see each other.

```scheme
(define-syntax swap!
  (syntax-rules ()
    ((_ a b) (let ((tmp a)) (set! a b) (set! b tmp)))))
(define x 1)
(define y 2)
(swap! x y)
(display (list x y)) (newline)

(display (letrec-syntax ((ev? (syntax-rules () ((_ ()) #t) ((_ (x . r)) (od? r))))
                         (od? (syntax-rules () ((_ ()) #f) ((_ (x . r)) (ev? r)))))
           (ev? (a b c d))))
(newline)
```
```
(2 1)
#t
```

## The pattern language

A rule's pattern is matched against the use; the first rule that matches is
instantiated. In a pattern:

| Pattern element | Matches |
|---|---|
| `_` (and the keyword position) | anything, binding nothing |
| a literal (listed in `(literal …)`) | an identifier of the same spelling |
| any other identifier | anything; binds a pattern variable |
| `(p … pₙ)` / `#(p …)` | a list / vector of exactly that shape |
| `(p … . tail)` | a list with at least those elements; `tail` matches the rest |
| `p ...` | zero or more elements matching `p`, anywhere in a list or vector, followed by any fixed patterns |
| a number, string, character or boolean | an equal literal |

In a template, a pattern variable is replaced by the syntax it matched; an
element followed by `...` repeats once per match (several `...` flatten nested
repetitions); `(... ...)` is a literal ellipsis. A custom ellipsis identifier
may be named before the literals list: `(syntax-rules etc () ...)`.

```scheme
(define-syntax my-let*
  (syntax-rules ()
    ((_ () body ...) (let () body ...))
    ((_ ((n v) rest ...) body ...) (let ((n v)) (my-let* (rest ...) body ...)))))
(display (my-let* ((a 1) (b (+ a 1))) (* a b))) (newline)

(define-syntax flat (syntax-rules () ((_ (a ...) ...) '(a ... ...))))
(display (flat (1 2) (3 4))) (newline)

(define-syntax rev (syntax-rules () ((_ () acc) 'acc) ((_ (x . xs) acc) (rev xs (x . acc)))))
(display (rev (1 2 3) ())) (newline)
```
```
2
(1 2 3 4)
(3 2 1)
```

A use that no rule matches is a compile-time syntax error on every engine:
`syntax error: no matching pattern for macro '…'`.

## Hygiene

Macros are hygienic in both directions (R7RS 4.3.2).

**A binding the template introduces cannot capture the caller's code, or be
captured by it.** This holds for every binding form a template can contain:
`let`, `let*`, `letrec`, named `let`, `lambda`, `do`, `let-values`, `guard`,
internal `define`.

```scheme
(define-syntax my-or
  (syntax-rules () ((_) #f) ((_ e) e) ((_ e r ...) (let ((t e)) (if t t (my-or r ...))))))
(display (let ((t 5)) (my-or #f t))) (newline)

(define-syntax sum-below
  (syntax-rules () ((_ n) (do ((i 0 (+ i 1)) (s 0 (+ s i))) ((= i n) s)))))
(display (let ((i 100) (s 200)) (sum-below 5))) (newline)
```
```
5
10
```

**An identifier the template uses freely means what it meant where the macro
was defined**, whatever the use site binds with the same spelling: a special
form (`if`, `let`, `lambda`, `cond`, `else`, ...), a builtin (`car`, `+`,
`list`), a top-level procedure or variable, a local visible at the definition
(for `let-syntax` inside a `let`), or another macro keyword.

```scheme
(define (helper x) (* x 10))
(define-syntax use-helper (syntax-rules () ((_ a) (helper a))))
(display (let ((helper (lambda (x) (- x)))) (use-helper 5))) (newline)

(define-syntax first-of (syntax-rules () ((_ l) (car l))))
(display (let ((car cdr)) (first-of '(1 2)))) (newline)

(define-syntax sign
  (syntax-rules () ((_ x) (cond ((< x 0) 'negative) ((> x 0) 'positive) (else 'zero)))))
(display (let ((else #f)) (sign 3))) (newline)

(display (let ((x 11))
           (let-syntax ((get-x (syntax-rules () ((_) x))))
             (let ((x 22)) (get-x)))))
(newline)
```
```
50
1
positive
11
```

**A lexical binding shadows a macro keyword.** Inside `(let ((m ...)) ...)`, a
form `(m ...)` calls the local procedure even if a macro `m` is defined outside.
A top-level `define` of a macro's name does not remove the macro.

```scheme
(display (let ((my-or (lambda args 'procedure))) (my-or 1 2))) (newline)
```
```
procedure
```

Two rules follow from how hygiene is implemented and are part of the language:

- A top-level definition a template introduces defines the name as written, so a
  macro can define globals: `(define-syntax def-counter (syntax-rules () ((_) (define counter 0))))`
  makes `counter` visible to the program.
- A symbol a template quotes is data and keeps its spelling: a template
  `(let ((tmp 5)) (list 'tmp tmp))` produces `(tmp 5)`.

## Notes on engines

- The bytecode VM expands each form when the compiler reaches it, in that
  form's own scope; the native compiler expands after parsing. Both read the
  macro use as syntax, so an operand only has to be syntax the template makes
  sense of: it is never evaluated or parsed as an expression before expansion.
- `syntax-case` and other procedural transformers are not provided; see
  [Scheme compatibility](../../breakdown/SCHEME_COMPATIBILITY.md#macro-system).
