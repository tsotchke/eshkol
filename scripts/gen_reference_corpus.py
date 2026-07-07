#!/usr/bin/env python3
"""gen_reference_corpus.py — R7RS-small portable corpus generator for the
reference-implementation differential oracle (adversarial testing pillar P7a).

Unlike scripts/gen_differential.py (which diffs Eshkol against ITSELF across
its own execution axes — self-consistency only), this corpus is diffed against
an EXTERNAL reference R7RS Scheme (chibi-scheme). Any divergence on a portable
program is an Eshkol conformance bug measured against ground truth, not just an
internal inconsistency.

Design constraints (every program MUST satisfy all of these):
  * Uses ONLY R7RS-small identifiers/forms that BOTH Eshkol and the reference
    accept. Non-portable extensions (fold-left, list-sort, sort, filter,
    reduce, hash tables) and ALL Eshkol-only features (automatic
    differentiation, tensors, logic vars ?x, agent/FFI) are excluded — they
    would produce BOTH-ERROR noise or are simply not R7RS.
  * Prints fully deterministic output (no time, no randomness, no addresses).
  * Runs cleanly (exit 0, no error) on a conformant R7RS implementation.
    chibi-scheme is the ground truth; if chibi errors on a program that is a
    corpus-authoring bug, not a finding.
  * Contains NO `(import ...)` line. The reference runner prepends a fixed
    import prologue (see scripts/run_reference_differential.sh); Eshkol needs
    no imports. Thus the program *body* is byte-identical across both engines.
  * Never `display`s a raw multiple-values object (implementation-defined
    rendering) — always funnel `values` through call-with-values / let-values.

Some programs deliberately exercise portable R7RS features that are known or
suspected to be mis-handled by Eshkol (apply with leading args, multi-vector
vector-map, `cond`/`case` `=>` clauses). These are legitimate ground-truth
probes: a divergence there is the treasure this pillar exists to find.

Usage:
  python3 scripts/gen_reference_corpus.py [--out DIR]
      DIR defaults to tests/reference-diff/corpus/

The generator is fully deterministic: same inputs -> byte-identical corpus.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Corpus. Each entry is (category, name, body). `body` is the exact program
# text run on BOTH engines. Keep bodies self-contained and deterministic.
# ---------------------------------------------------------------------------

PROGRAMS = []

def add(category, name, body):
    PROGRAMS.append((category, name, body.strip() + "\n"))

# ---- core arithmetic / numeric tower --------------------------------------
add("numeric", "int_arith", r"""
(display (+ 1 2 3 4 5))(newline)
(display (- 100 20 5))(newline)
(display (* 2 3 4))(newline)
(display (quotient 17 5))(newline)
(display (remainder 17 5))(newline)
(display (modulo -7 3))(newline)
(display (remainder -7 3))(newline)
(display (abs -42))(newline)
(display (gcd 12 18 24))(newline)
(display (lcm 4 6))(newline)
""")

add("numeric", "rationals", r"""
(display (/ 1 2))(newline)
(display (/ 6 3))(newline)
(display (/ 10 4))(newline)
(display (+ 1/3 1/6))(newline)
(display (* 2/3 3/4))(newline)
(display (- 1/2 1/3))(newline)
(display (< 1/3 1/2))(newline)
(display (= 2/4 1/2))(newline)
""")

add("numeric", "exact_inexact", r"""
(display (exact 3.0))(newline)
(display (inexact 1/4))(newline)
(display (inexact 1/2))(newline)
(display (exact 2.0))(newline)
(display (exact? 3))(newline)
(display (inexact? 3.0))(newline)
(display (* 1.0 3))(newline)
(display (+ 0.5 0.25))(newline)
""")

add("numeric", "float_ops", r"""
(display (sqrt 2))(newline)
(display (sqrt 4))(newline)
(display (expt 2 10))(newline)
(display (expt 2.0 0.5))(newline)
(display (floor 3.7))(newline)
(display (ceiling 3.2))(newline)
(display (truncate -3.7))(newline)
(display (round 2.5))(newline)
(display (round 3.5))(newline)
(display (min 3 1.0 2))(newline)
(display (max 3 1 2))(newline)
""")

add("numeric", "bignum", r"""
(display (expt 2 64))(newline)
(display (expt 10 30))(newline)
(display (* 99999999999 99999999999))(newline)
(display (- (expt 2 100) 1))(newline)
(display (+ 100000000000000000000 1))(newline)
""")

add("numeric", "predicates", r"""
(display (zero? 0))(newline)
(display (positive? 3))(newline)
(display (negative? -3))(newline)
(display (even? 10))(newline)
(display (odd? 7))(newline)
(display (integer? 3.0))(newline)
(display (exact-integer? 5))(newline)
(display (number? 1/2))(newline)
(display (rational? 1/2))(newline)
""")

add("numeric", "number_string", r"""
(display (number->string 255))(newline)
(display (number->string 255 16))(newline)
(display (number->string 255 2))(newline)
(display (number->string 3.14))(newline)
(display (string->number "42"))(newline)
(display (string->number "1/4"))(newline)
(display (string->number "ff" 16))(newline)
(display (string->number "3.5"))(newline)
(display (string->number "not-a-number"))(newline)
""")

# ---- lists ----------------------------------------------------------------
add("list", "basic", r"""
(display (cons 1 2))(newline)
(display (list 1 2 3))(newline)
(display (car '(1 2 3)))(newline)
(display (cdr '(1 2 3)))(newline)
(display (cadr '(1 2 3)))(newline)
(display (caddr '(1 2 3)))(newline)
(display (length '(a b c d)))(newline)
(display (list-ref '(a b c) 2))(newline)
(display (list-tail '(1 2 3 4 5) 2))(newline)
(display (reverse '(1 2 3 4)))(newline)
(display (append '(1 2) '(3 4) '(5)))(newline)
""")

add("list", "membership", r"""
(display (member 2 '(1 2 3)))(newline)
(display (memq 'b '(a b c)))(newline)
(display (memv 2 '(1 2 3)))(newline)
(display (assoc 2 '((1 . "a") (2 . "b"))))(newline)
(display (assq 'b '((a 1) (b 2))))(newline)
(display (assv 2 '((1 . x) (2 . y))))(newline)
(display (list-copy '(1 2 3)))(newline)
""")

add("list", "higher_order", r"""
(display (map (lambda (x) (* x x)) '(1 2 3 4)))(newline)
(display (map + '(1 2 3) '(10 20 30)))(newline)
(display (map + '(1 2 3) '(10 20 30) '(100 200 300)))(newline)
(for-each display '(1 2 3))(newline)
(display (apply + '(1 2 3 4)))(newline)
(display (apply max '(3 1 4 1 5)))(newline)
""")

# apply with leading fixed args before the final list (R7RS core).
add("list", "apply_leading_args", r"""
(display (apply + 1 2 '(3 4 5)))(newline)
(display (apply + 1 (list 3 4 5)))(newline)
(display (apply max 1 '(9 2)))(newline)
(display (apply list 'a 'b '(c d)))(newline)
""")

add("list", "quasiquote", r"""
(display `(1 ,(+ 1 1) ,@(list 3 4) 5))(newline)
(display `(a (b ,(* 2 3)) c))(newline)
(display `#(1 ,(+ 2 2) 3))(newline)
(let ((x 10)) (display `(x is ,x)))(newline)
""")

# ---- vectors --------------------------------------------------------------
add("vector", "basic", r"""
(display (vector 1 2 3))(newline)
(display #(4 5 6))(newline)
(display (vector-ref #(10 20 30) 1))(newline)
(display (vector-length #(1 2 3 4)))(newline)
(display (vector->list #(1 2 3)))(newline)
(display (list->vector '(7 8 9)))(newline)
(display (make-vector 3 0))(newline)
(let ((v (vector 1 2 3))) (vector-set! v 1 99) (display v))(newline)
(display (vector-copy #(1 2 3 4 5) 1 4))(newline)
""")

add("vector", "vector_map_single", r"""
(display (vector-map (lambda (x) (* x x)) #(1 2 3 4)))(newline)
(vector-for-each display #(1 2 3))(newline)
""")

# multi-argument vector-map / vector-for-each (R7RS core).
add("vector", "vector_map_multi", r"""
(display (vector-map + #(1 2 3) #(10 20 30)))(newline)
(display (vector-map * #(1 2 3) #(4 5 6)))(newline)
(display (vector-map + #(1 2 3) #(10 20 30) #(100 200 300)))(newline)
(vector-for-each (lambda (a b) (display (+ a b)) (display " ")) #(1 2 3) #(10 20 30))(newline)
""")

# ---- strings --------------------------------------------------------------
add("string", "basic", r"""
(display (string-append "foo" "bar" "baz"))(newline)
(display (string-length "hello"))(newline)
(display (substring "hello world" 0 5))(newline)
(display (string-copy "hello" 1 4))(newline)
(display (string-ref "abc" 1))(newline)
(display (string-upcase "Hello"))(newline)
(display (string-downcase "Hello"))(newline)
(display (make-string 4 #\*))(newline)
(display (string->list "abc"))(newline)
(display (list->string (list #\x #\y #\z)))(newline)
""")

add("string", "comparison", r"""
(display (string=? "abc" "abc"))(newline)
(display (string<? "abc" "abd"))(newline)
(display (string>? "b" "a"))(newline)
(display (string->symbol "foo"))(newline)
(display (symbol->string 'bar))(newline)
""")

add("string", "map_foreach", r"""
(display (string-map char-upcase "hello"))(newline)
(string-for-each (lambda (c) (display c) (display ".")) "abc")(newline)
""")

# characters displayed inside a compound (display should render glyphs).
add("string", "char_in_list", r"""
(display (reverse (string->list "abc")))(newline)
(display (list #\a #\b #\c))(newline)
""")

# ---- characters -----------------------------------------------------------
add("char", "basic", r"""
(display (char->integer #\A))(newline)
(display (integer->char 97))(newline)
(display (char-upcase #\a))(newline)
(display (char-downcase #\Z))(newline)
(display (char<? #\a #\b))(newline)
(display (char-alphabetic? #\x))(newline)
(display (char-numeric? #\7))(newline)
(display (char-whitespace? #\a))(newline)
""")

# ---- let family -----------------------------------------------------------
add("binding", "let_family", r"""
(display (let ((x 1) (y 2)) (+ x y)))(newline)
(display (let* ((x 1) (y (* x 2)) (z (* y 3))) z))(newline)
(display (letrec ((ev? (lambda (n) (if (= n 0) #t (od? (- n 1)))))
                  (od? (lambda (n) (if (= n 0) #f (ev? (- n 1))))))
          (ev? 10)))(newline)
(display (let loop ((i 0) (acc 0)) (if (= i 5) acc (loop (+ i 1) (+ acc i)))))(newline)
(display (let-values (((q r) (floor/ 17 5))) (list q r)))(newline)
(display (let-values (((q r) (truncate/ -17 5))) (list q r)))(newline)
""")

add("binding", "internal_defines", r"""
(define (f x)
  (define y (* x 2))
  (define z (+ y 1))
  (+ y z))
(display (f 10))(newline)
(define (g n)
  (define (helper a b) (if (= a 0) b (helper (- a 1) (+ b a))))
  (helper n 0))
(display (g 100))(newline)
""")

add("binding", "set_local", r"""
(define (counter)
  (let ((n 0))
    (lambda () (set! n (+ n 1)) n)))
(define c (counter))
(display (c))(display (c))(display (c))(newline)
(let ((x 10)) (set! x (* x x)) (display x))(newline)
""")

# ---- control flow ---------------------------------------------------------
add("control", "cond_case", r"""
(display (cond ((= 1 2) 'a) ((= 1 1) 'b) (else 'c)))(newline)
(display (case 3 ((1 2) 'lo) ((3 4) 'hi) (else 'other)))(newline)
(display (case 'x ((a b) 1) ((x y) 2) (else 3)))(newline)
(display (if (> 3 2) 'yes 'no))(newline)
(display (when (> 3 2) 'when-fires))(newline)
(display (and 1 2 3))(newline)
(display (or #f #f 7))(newline)
(display (not #f))(newline)
""")

# cond / case with `=>` (R7RS core).
add("control", "arrow_clauses", r"""
(display (cond ((assv 2 '((1 . "a") (2 . "b"))) => cdr) (else "none")))(newline)
(display (cond ((memv 3 '(1 2 3 4)) => car) (else 'no)))(newline)
(display (case 5 ((1 2 3) 'lo) ((4 5 6) => (lambda (x) (* x 10))) (else 'no)))(newline)
""")

add("control", "do_loop", r"""
(display (do ((i 0 (+ i 1)) (s 0 (+ s i))) ((= i 5) s)))(newline)
(display (do ((v (make-vector 5))
              (i 0 (+ i 1)))
             ((= i 5) v)
          (vector-set! v i (* i i))))(newline)
""")

add("control", "tail_recursion", r"""
(define (sum-to n acc) (if (= n 0) acc (sum-to (- n 1) (+ acc n))))
(display (sum-to 1000000 0))(newline)
(define (count-down n) (if (= n 0) 'done (count-down (- n 1))))
(display (count-down 1000000))(newline)
""")

add("control", "callcc", r"""
(display (call-with-current-continuation (lambda (k) (+ 1 (k 42)))))(newline)
(display (+ 1 (call-with-current-continuation (lambda (k) 10))))(newline)
(display (call/cc (lambda (k) (* 2 (+ 3 (k 99))))))(newline)
""")

add("control", "dynamic_wind", r"""
(dynamic-wind
  (lambda () (display "in "))
  (lambda () (display "body "))
  (lambda () (display "out ")))
(newline)
(display (call/cc (lambda (k)
  (dynamic-wind
    (lambda () (display "["))
    (lambda () (k 'escaped))
    (lambda () (display "]"))))))
(newline)
""")

add("control", "values", r"""
(call-with-values (lambda () (values 1 2 3)) (lambda (a b c) (display (+ a b c)) (newline)))
(display (call-with-values (lambda () (values 10 20)) +))(newline)
(display (let-values (((a b) (values 3 4)) ((c) (values 5))) (+ a b c)))(newline)
""")

add("control", "guard_raise", r"""
(display (guard (e (#t (list 'caught e))) (raise 'boom)))(newline)
(display (guard (e ((symbol? e) 'was-symbol) (else 'other)) (raise 'sym)))(newline)
(display (guard (e ((error-object? e) (error-object-message e)))
          (error "something failed")))(newline)
(display (guard (e (#t 'recovered)) (car '())))(newline)
""")

# ---- equality / predicates ------------------------------------------------
add("equality", "eq_family", r"""
(display (eq? 'a 'a))(newline)
(display (eqv? 2 2))(newline)
(display (eqv? 2 2.0))(newline)
(display (equal? '(1 2 (3 4)) '(1 2 (3 4))))(newline)
(display (equal? #(1 2 3) #(1 2 3)))(newline)
(display (equal? "abc" "abc"))(newline)
(display (boolean=? #t #t))(newline)
(display (eq? '() '()))(newline)
""")

add("equality", "type_predicates", r"""
(display (null? '()))(newline)
(display (pair? '(1)))(newline)
(display (list? '(1 2 3)))(newline)
(display (symbol? 'x))(newline)
(display (string? "s"))(newline)
(display (char? #\a))(newline)
(display (vector? #(1)))(newline)
(display (procedure? car))(newline)
(display (boolean? #f))(newline)
""")

# ---- write vs display -----------------------------------------------------
add("io", "write_forms", r"""
(write "hello\nworld")(newline)
(write #\a)(newline)
(write '(1 "two" #\3 sym))(newline)
(write 'symbol)(newline)
(display "hello\nworld")(newline)
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output corpus directory")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(repo_root, "tests", "reference-diff", "corpus")
    os.makedirs(out_dir, exist_ok=True)

    # Clear any stale generated corpus (deterministic regeneration).
    for f in os.listdir(out_dir):
        if f.endswith(".scm"):
            os.remove(os.path.join(out_dir, f))

    idx = 0
    for category, name, body in PROGRAMS:
        idx += 1
        fname = "%04d_%s_%s.scm" % (idx, category, name)
        header = ";; %s / %s  (R7RS-small portable; reference-differential corpus)\n" % (
            category, name)
        with open(os.path.join(out_dir, fname), "w") as fh:
            fh.write(header)
            fh.write(body)

    print("Wrote %d programs to %s" % (idx, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
