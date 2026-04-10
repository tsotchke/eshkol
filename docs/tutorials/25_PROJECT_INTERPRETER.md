# Project: Build a Calculator in Eshkol

A complete interpreter for arithmetic expressions, demonstrating
homoiconicity — in Eshkol, code IS data, so building an interpreter
is natural.

---

## The Complete Program

```scheme
;; ═══════════════════════════════════════════════════════
;; S-Expression Calculator
;; Evaluates arithmetic expressions written as Scheme lists
;; ═══════════════════════════════════════════════════════

(require core.list.higher_order)

;; --- The evaluator ---
;; Evaluate an s-expression as arithmetic
(define (calc expr)
  (cond
    ;; Numbers evaluate to themselves
    ((number? expr) expr)

    ;; Lists are function applications
    ((pair? expr)
     (let ((op (car expr))
           (args (map calc (cdr expr))))
       (cond
         ((eq? op '+) (fold-left + 0 args))
         ((eq? op '-) (if (null? (cdr args))
                         (- 0 (car args))
                         (fold-left - (car args) (cdr args))))
         ((eq? op '*) (fold-left * 1 args))
         ((eq? op '/) (fold-left / (car args) (cdr args)))
         ((eq? op 'sqrt) (sqrt (car args)))
         ((eq? op 'expt) (expt (car args) (cadr args)))
         ((eq? op 'abs) (abs (car args)))
         (else (begin (display "Unknown operator: ")
                      (display op) (newline) 0)))))

    ;; Anything else is an error
    (else (begin (display "Invalid expression: ")
                 (display expr) (newline) 0))))

;; --- Test it ---
(display "=== S-Expression Calculator ===") (newline)

(display "(+ 1 2 3) = ")
(display (calc '(+ 1 2 3)))
(newline)

(display "(* 6 7) = ")
(display (calc '(* 6 7)))
(newline)

(display "(- 100 (+ 20 30)) = ")
(display (calc '(- 100 (+ 20 30))))
(newline)

(display "(/ (* 3 4) 2) = ")
(display (calc '(/ (* 3 4) 2)))
(newline)

(display "(sqrt (+ (* 3 3) (* 4 4))) = ")
(display (calc '(sqrt (+ (* 3 3) (* 4 4)))))
(newline)

(display "(expt 2 10) = ")
(display (calc '(expt 2 10)))
(newline)

;; Homoiconicity: the calculator evaluates quoted Scheme as data.
;; The expression '(+ 1 2) is a LIST that we traverse and compute.
;; This is the same list that Eshkol's own compiler works with.
(display "Homoiconicity: code = data") (newline)
```

---

## Expected Output

```
=== S-Expression Calculator ===
(+ 1 2 3) = 6
(* 6 7) = 42
(- 100 (+ 20 30)) = 50
(/ (* 3 4) 2) = 6
(sqrt (+ (* 3 3) (* 4 4))) = 5.0
(expt 2 10) = 1024
Homoiconicity: code = data
```

---

## Key Concepts

1. **Homoiconicity** — Eshkol code and Eshkol data have the same structure
   (s-expressions). A quoted expression `'(+ 1 2)` is a list you can
   traverse with `car`/`cdr` — AND valid Eshkol code.

2. **Recursive evaluation** — `calc` calls itself on sub-expressions,
   building up results from leaves to root. This is exactly how a real
   interpreter works.

3. **Pattern dispatch** — `cond` with type predicates (`number?`, `pair?`)
   dispatches on the structure of the input.

4. **Variadic arithmetic** — `(+ 1 2 3)` handles any number of arguments
   via `fold-left`.
