# Tutorial 10: Macros and Metaprogramming

Eshkol supports R7RS hygienic macros via `syntax-rules`. Macros transform
code at compile time — they let you extend the language's syntax without
runtime cost.

---

## Part 1: Basic Macros

```scheme
;; Define a macro that swaps two variables
(define-syntax swap!
  (syntax-rules ()
    ((swap! a b)
     (let ((tmp a))
       (set! a b)
       (set! b tmp)))))

(define x 1)
(define y 2)
(swap! x y)
(display x)  ;; => 2
(display y)  ;; => 1
```

The `syntax-rules` form specifies patterns and templates. When the
compiler sees `(swap! x y)`, it expands it to the `let` expression at
compile time. No function call overhead at runtime.

---

## Part 2: Pattern Matching in Macros

```scheme
;; A 'when' macro — executes body only if condition is true
(define-syntax when
  (syntax-rules ()
    ((when condition body ...)
     (if condition (begin body ...)))))

(when (> 5 3)
  (display "five is greater")
  (newline))
;; => five is greater

;; An 'unless' macro — executes body only if condition is false
(define-syntax unless
  (syntax-rules ()
    ((unless condition body ...)
     (if (not condition) (begin body ...)))))

(unless (= 1 2)
  (display "not equal")
  (newline))
;; => not equal
```

The `...` (ellipsis) matches zero or more expressions in patterns and
repeats them in templates.

---

## Part 3: Multi-Clause Macros

```scheme
;; A 'cond' replacement with arrow syntax
(define-syntax my-cond
  (syntax-rules (else =>)
    ((my-cond (else expr ...))
     (begin expr ...))
    ((my-cond (test => proc) rest ...)
     (let ((t test))
       (if t (proc t) (my-cond rest ...))))
    ((my-cond (test expr ...) rest ...)
     (if test (begin expr ...) (my-cond rest ...)))))
```

Literal identifiers (`else`, `=>`) are listed after `syntax-rules` and
matched exactly in patterns.

---

## Part 4: Hygiene

Hygienic macros prevent accidental variable capture. The `tmp` in `swap!`
above cannot conflict with user code:

```scheme
(define tmp 42)
(define a 1)
(define b 2)
(swap! a b)
;; tmp is still 42 — the macro's tmp is distinct
(display tmp)  ;; => 42
```

Hygiene also works in the other direction: a name the template uses but
does not bind means what it meant where the macro was defined, even if the
code that uses the macro binds the same name locally. The operand, on the
other hand, is the caller's code and sees the caller's bindings:

```scheme
(define (double x) (* 2 x))
(define-syntax twice (syntax-rules () ((_ e) (double e))))

(display (let ((double (lambda (x) 0))) (twice 21)))  ;; => 42
(newline)
(display (let ((car cdr)) (twice (length (car '(5 6 7))))))  ;; => 4
```

In the second use the caller's `car` is `cdr`, so the operand is
`(length '(6 7))`, which is `2`; the template's `double` is still the
top-level procedure.

Both directions hold for every binding form a template can contain and on
every engine (JIT, AOT, bytecode VM). See the
[macro reference](../reference/language/macros.md) for the full rules.

---

## Part 5: Homoiconicity

Every Eshkol function can display its own source code because the
compiler preserves s-expression metadata:

```scheme
(define (square x) (* x x))

;; The function IS its own source code
(display square)
;; => (lambda (x) (* x x))
```

This is possible because Eshkol stores the s-expression representation
alongside the compiled code. Functions are data, data is code — the
Lisp way.

---

*All tutorials are available at [eshkol.ai/docs](https://eshkol.ai/docs)
and in the `docs/tutorials/` directory of the repository.*
