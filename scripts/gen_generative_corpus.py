#!/usr/bin/env python3
r"""gen_generative_corpus.py — GENERATIVE R7RS-small program generator for the
multi-oracle differential harness (adversarial testing pillar P7c).

Motivation (from the maintainer): "if our system does not constantly expose
every single hidden bug then it has no coverage." The hand-written
reference-differential corpus (scripts/gen_reference_corpus.py, ~34 programs)
is far too small to keep exposing miscompiles. This module GENERATES a large,
deterministic family of closed, printable R7RS-small programs so the harness
in scripts/run_generative_differential.py can cross-check every one across as
many execution oracles as are installed (chibi-scheme, Eshkol JIT, Eshkol AOT
at -O0 and -O2, and the Eshkol bytecode VM). Any pairwise disagreement is an
exposed bug.

Two program families are produced:

  * DIFFERENTIAL programs (family "diff"): a sequence of typed probe
    expressions, each wrapped in `(display EXPR)(newline)`. Every probe is
    TOTAL (well-defined on a conformant R7RS-small implementation — no
    division by zero, no out-of-range index, no car of '()) and DETERMINISTIC
    (no time / randomness / addresses). chibi-scheme is ground truth: if chibi
    errors on a generated program that is a generator bug, not a finding.

  * METAMORPHIC programs (family "meta"): self-checking property assertions
    that need NO external reference. Each line evaluates a property that must
    hold under R7RS semantics and displays `#t`. Any oracle that prints `#f`
    (or errors, or disagrees with another oracle) has exposed a bug. Encoded
    properties:
      - (f x)            == (apply f (list x))            [apply equivalence]
      - (map f xs)       == hand-rolled left-to-right map [map ordering]
      - (+ a b)          == (+ b a) ; (* a b) == (* b a)  [commutativity]
      - (reverse (reverse xs)) == xs                       [involution]
      - (length (append a b)) == (+ (length a)(length b)) [homomorphism]
      - let/let* re-association where independent
      - fold-based sum   == (apply + xs)                   [fold equivalence]

Determinism: the corpus is a pure function of (seed, count). Same inputs ->
byte-identical programs, so CI is reproducible. The generator NEVER embeds a
timestamp or absolute path.

Portability: every program body uses ONLY R7RS-small identifiers that BOTH
chibi and Eshkol accept. Higher-order helpers that are NOT in (scheme base)
(filter, fold) are DEFINED inline under `gd-`/`gm-` prefixes in a shared
prelude, so no `(import (scheme list))` is needed and the program body is
byte-identical across engines. The reference runner prepends chibi's import
prologue; Eshkol needs no imports.

Usage:
  python3 scripts/gen_generative_corpus.py --out DIR [--seed N] [--count K]
      Writes K programs per family to DIR (default tests/generative-diff/corpus).

  The harness (run_generative_differential.py) imports generate_programs()
  directly and does not need the on-disk corpus; the CLI exists for
  inspection, reproducibility and manual minimisation of a divergence.
"""

import argparse
import os
import random
import sys

# ---------------------------------------------------------------------------
# Shared portable prelude. R7RS-base only. Prefixed helpers so they can never
# shadow (or be shadowed by) an Eshkol or chibi builtin. Included verbatim and
# identically in EVERY generated program body (both engines see the same text).
# ---------------------------------------------------------------------------
PRELUDE = r"""
(define (gd-filter p xs)
  (if (null? xs) '()
      (if (p (car xs))
          (cons (car xs) (gd-filter p (cdr xs)))
          (gd-filter p (cdr xs)))))
(define (gd-foldl f acc xs)
  (if (null? xs) acc (gd-foldl f (f acc (car xs)) (cdr xs))))
(define (gd-map1 f xs)
  (if (null? xs) '() (cons (f (car xs)) (gd-map1 f (cdr xs)))))
(define (gd-sum xs) (gd-foldl + 0 xs))
(define (gd-iota n) (let loop ((i 0) (acc '()))
  (if (= i n) (reverse acc) (loop (+ i 1) (cons i acc)))))
""".strip() + "\n"


# ---------------------------------------------------------------------------
# Manifest-driven deterministic surface probes.
#
# Random generation is valuable for combinations, but it must not be the only
# reason a construct is covered: changing a seed or production weight could
# silently make a builtin disappear from the corpus.  These self-checking R7RS
# programs pin high-risk portable constructs on every run.  The declared head
# set is checked against tests/coverage/language_surface.json by
# language_coverage.py, so misspelling or removing a language construct is a
# hard tracker failure rather than fake coverage.
# ---------------------------------------------------------------------------
SURFACE_PROBES = (
    {
        "name": "surface_numeric_exact_complex",
        "heads": (
            "exact?", "inexact?", "exact->inexact", "inexact->exact",
            "numerator", "denominator", "rational?", "complex?",
            "make-rectangular", "real-part", "imag-part", "magnitude",
            "conjugate", "square",
        ),
        "checks": (
            "(exact? 42)",
            "(inexact? (exact->inexact 42))",
            "(= (inexact->exact 3.0) 3)",
            "(= (numerator (/ 6 8)) 3)",
            "(= (denominator (/ 6 8)) 4)",
            "(rational? (/ 6 8))",
            "(complex? (make-rectangular 3 4))",
            "(= (real-part (make-rectangular 3 4)) 3)",
            "(= (imag-part (make-rectangular 3 4)) 4)",
            "(= (magnitude (make-rectangular 3 4)) 5)",
            "(= (real-part (conjugate (make-rectangular 3 4))) 3)",
            "(= (imag-part (conjugate (make-rectangular 3 4))) -4)",
            "(= (square 9) 81)",
        ),
    },
    {
        "name": "surface_numeric_transcendental",
        "heads": (
            "asin", "acos", "atan", "tan", "sinh", "cosh",
            "rationalize", "number->string", "string->number",
            "make-polar", "angle",
        ),
        "checks": (
            "(< (abs (- (sin (asin 0.5)) 0.5)) 1e-12)",
            "(< (abs (- (cos (acos 0.5)) 0.5)) 1e-12)",
            "(< (abs (- (tan (atan 0.25)) 0.25)) 1e-12)",
            "(< (abs (- (sinh 0.0) 0.0)) 1e-12)",
            "(< (abs (- (cosh 0.0) 1.0)) 1e-12)",
            "(= (rationalize 0.3 0.01) 3/10)",
            "(string=? (number->string 255 16) \"ff\")",
            "(= (string->number \"255\") 255)",
            "(< (abs (- (magnitude (make-polar 2.0 0.5)) 2.0)) 1e-12)",
            "(< (abs (- (angle (make-polar 2.0 0.5)) 0.5)) 1e-12)",
        ),
    },
    {
        "name": "surface_control_values",
        "heads": (
            "case", "when", "unless", "do", "values", "call-with-values",
            "let-values", "let*-values", "delay", "force", "call/cc",
            "dynamic-wind",
        ),
        "checks": (
            "(case 2 ((1) #f) ((2 3) #t) (else #f))",
            "(let ((x 0)) (when #t (set! x 3)) (= x 3))",
            "(let ((x 0)) (unless #f (set! x 4)) (= x 4))",
            "(do ((i 0 (+ i 1)) (s 0 (+ s i))) ((= i 5) (= s 10)))",
            "(call-with-values (lambda () (values 2 3)) (lambda (a b) (= (+ a b) 5)))",
            "(let-values (((a b) (values 2 3))) (= (+ a b) 5))",
            "(let*-values (((a b) (values 2 3)) ((c) (+ a b))) (= c 5))",
            "(let ((p (delay (+ 2 3)))) (= (force p) 5))",
            "(= (call/cc (lambda (escape) (escape 7) 99)) 7)",
            "(let ((x 0)) (dynamic-wind (lambda () (set! x (+ x 1))) "
            "(lambda () (set! x (+ x 2))) (lambda () (set! x (+ x 4)))) "
            "(= x 7))",
        ),
    },
    {
        "name": "surface_control_exceptions_promises",
        "heads": (
            "call-with-current-continuation", "delay-force",
            "with-exception-handler", "raise",
        ),
        "checks": (
            "(= (call-with-current-continuation (lambda (escape) (escape 11) 99)) 11)",
            "(letrec ((p (delay (+ 4 5))) (q (delay-force p))) (= (force q) 9))",
            "(= (with-exception-handler (lambda (x) (+ x 1)) (lambda () (raise 6))) 7)",
        ),
    },
    {
        "name": "surface_macro_hygiene",
        "heads": (
            "define-syntax", "let-syntax", "letrec-syntax",
            "quasiquote", "unquote", "unquote-splicing",
        ),
        "forms": (
            "(define-syntax gd-swap! "
            "(syntax-rules () ((_ a b) (let ((tmp a)) (set! a b) (set! b tmp)))))",
        ),
        "checks": (
            "(let ((tmp 99) (a 1) (b 2)) (gd-swap! a b) "
            "(and (= tmp 99) (= a 2) (= b 1)))",
            "(let-syntax ((twice (syntax-rules () ((_ x) (+ x x))))) (= (twice 6) 12))",
            "(letrec-syntax ((identity (syntax-rules () ((_ x) x)))) "
            "(equal? (identity '(1 2)) '(1 2)))",
            "(let ((x 2) (xs '(3 4))) (equal? `(1 ,x ,@xs 5) '(1 2 3 4 5)))",
        ),
    },
    {
        "name": "surface_lists_cxr",
        "heads": (
            "caar", "caaar", "caadr", "cadar", "cdar", "cdaar", "cdadr",
            "cddr", "cddar", "cdddr", "cadddr", "list-tail",
            "member", "memq", "assoc",
        ),
        "checks": (
            "(= (caar '((1 2) (3 4))) 1)",
            "(= (caaar '(((1 2)) ((3 4)))) 1)",
            "(= (caadr '((1 2) (3 4))) 3)",
            "(= (cadar '((1 2) (3 4))) 2)",
            "(equal? (cdar '((1 2) (3 4))) '(2))",
            "(equal? (cdaar '(((1 2 3)))) '(2 3))",
            "(equal? (cdadr '((0) (1 2 3))) '(2 3))",
            "(equal? (cddr '(1 2 3 4)) '(3 4))",
            "(equal? (cddar '((1 2 3 4))) '(3 4))",
            "(equal? (cdddr '(1 2 3 4)) '(4))",
            "(= (cadddr '(1 2 3 4)) 4)",
            "(equal? (list-tail '(1 2 3 4) 2) '(3 4))",
            "(equal? (member 3 '(1 2 3 4)) '(3 4))",
            "(let ((xs '(a b c))) (eq? (memq 'b xs) (cdr xs)))",
            "(equal? (assoc 'b '((a . 1) (b . 2))) '(b . 2))",
        ),
    },
    {
        "name": "surface_string_vector_predicates",
        "heads": (
            "char->integer", "char-alphabetic?", "char-numeric?",
            "char-whitespace?", "char-upper-case?", "char-lower-case?",
            "char=?", "char<?", "char>?", "string", "string-length",
            "string-ref", "string->list", "string->symbol", "symbol->string",
            "string=?", "string<?", "string>?", "vector->list",
            "vector-fill!", "vector-copy!", "vector?", "string?", "char?",
            "symbol?", "number?", "integer?", "real?", "procedure?",
        ),
        "checks": (
            "(= (char->integer #\\A) 65)",
            "(char-alphabetic? #\\A)",
            "(char-numeric? #\\7)",
            "(char-whitespace? #\\space)",
            "(char-upper-case? #\\A)",
            "(char-lower-case? #\\a)",
            "(char=? #\\a #\\a)",
            "(char<? #\\a #\\b)",
            "(char>? #\\b #\\a)",
            "(equal? (string #\\a #\\b #\\c) \"abc\")",
            "(= (string-length \"abc\") 3)",
            "(char=? (string-ref \"abc\" 1) #\\b)",
            "(equal? (string->list \"abc\") '(#\\a #\\b #\\c))",
            "(eq? (string->symbol \"abc\") 'abc)",
            "(string=? (symbol->string 'abc) \"abc\")",
            "(string=? \"abc\" \"abc\")",
            "(string<? \"abc\" \"abd\")",
            "(string>? \"abd\" \"abc\")",
            "(equal? (vector->list #(1 2 3)) '(1 2 3))",
            "(let ((v (vector 1 2 3))) (vector-fill! v 7) (equal? v #(7 7 7)))",
            "(let ((v (vector 1 2 3))) (vector-copy! v 1 #(8 9)) (equal? v #(1 8 9)))",
            "(and (vector? #(1)) (string? \"x\") (char? #\\x) (symbol? 'x) "
            "(number? 1) (integer? 1) (real? 1) (procedure? (lambda (x) x)))",
        ),
    },
)


def surface_probe_heads():
    """Return the constructs deliberately pinned by deterministic probes."""
    return {head for spec in SURFACE_PROBES for head in spec["heads"]}


def build_surface_program(spec):
    lines = [PRELUDE]
    lines.extend(spec.get("forms", ()))
    for check in spec["checks"]:
        lines.append("(display %s)(newline)" % check)
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)


# ---------------------------------------------------------------------------
# Typed expression generator. Each gen_<type> returns a string that evaluates,
# on any conformant R7RS-small implementation, to a value of that type WITHOUT
# raising an error. Totality is enforced structurally: divisors/moduli are
# always nonzero literals, indices are always in range for a list/string/vector
# whose length is known at generation time, expt exponents are non-negative.
# ---------------------------------------------------------------------------

NONZERO = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -2, -3, -4, -7]


class Gen:
    def __init__(self, rng):
        self.rng = rng
        self._vc = 0

    def fresh(self, tag="v"):
        self._vc += 1
        return "g%s%d" % (tag, self._vc)

    # ---- integers (exact) -------------------------------------------------
    def gen_int(self, depth):
        r = self.rng
        if depth <= 0:
            return str(r.randint(-50, 50))
        pick = r.randrange(14)
        if pick == 0:
            return str(r.randint(-50, 50))
        if pick == 1:
            op = r.choice(["+", "-", "*"])
            return "(%s %s %s)" % (op, self.gen_int(depth - 1), self.gen_int(depth - 1))
        if pick == 2:
            op = r.choice(["quotient", "remainder", "modulo"])
            return "(%s %s %s)" % (op, self.gen_int(depth - 1), str(r.choice(NONZERO)))
        if pick == 3:
            op = r.choice(["abs", "-"])
            return "(%s %s)" % (op, self.gen_int(depth - 1))
        if pick == 4:
            op = r.choice(["min", "max"])
            return "(%s %s %s)" % (op, self.gen_int(depth - 1), self.gen_int(depth - 1))
        if pick == 5:  # bignum via bounded expt
            return "(expt %d %d)" % (r.choice([2, 3, 5, 7, 10]), r.randint(1, 45))
        if pick == 6:
            op = r.choice(["gcd", "lcm"])
            a = r.randint(1, 48)
            b = r.randint(1, 48)
            return "(%s %d %d)" % (op, a, b)
        if pick == 7:  # if
            return "(if %s %s %s)" % (self.gen_bool(depth - 1),
                                      self.gen_int(depth - 1), self.gen_int(depth - 1))
        if pick == 8:  # let binding + use
            v = self.fresh()
            val = self.gen_int(depth - 1)
            body = "(%s %s %s)" % (r.choice(["+", "-", "*"]), v, self.gen_int(depth - 1))
            return "(let ((%s %s)) %s)" % (v, val, body)
        if pick == 9:  # fold / apply over a list literal
            lst = self.int_list_literal()
            return r.choice([
                "(apply + (list %s))" % lst,
                "(apply * (list %s))" % (self.small_int_list_literal()),
                "(gd-sum (list %s))" % lst,
                "(gd-foldl - 0 (list %s))" % lst,
            ])
        if pick == 10:  # car/cadr/caddr of known-length list
            n = r.randint(3, 5)
            elems = " ".join(self.gen_int(0) for _ in range(n))
            acc = r.choice(["car", "cadr", "caddr"])
            return "(%s (list %s))" % (acc, elems)
        if pick == 11:  # length / list-ref within range
            n = r.randint(1, 5)
            elems = " ".join(self.gen_int(0) for _ in range(n))
            if r.random() < 0.5:
                return "(length (list %s))" % elems
            return "(list-ref (list %s) %d)" % (elems, r.randrange(n))
        if pick == 12:  # cond
            return ("(cond (%s %s) (%s %s) (else %s))"
                    % (self.gen_bool(depth - 1), self.gen_int(depth - 1),
                       self.gen_bool(depth - 1), self.gen_int(depth - 1),
                       self.gen_int(depth - 1)))
        # pick == 13: named-let accumulator loop (tail recursion)
        n = r.randint(0, 12)
        return ("(let gloop ((gi 0) (gacc 0)) "
                "(if (= gi %d) gacc (gloop (+ gi 1) (+ gacc gi))))" % n)

    def int_list_literal(self):
        n = self.rng.randint(2, 6)
        return " ".join(str(self.rng.randint(-30, 30)) for _ in range(n))

    def small_int_list_literal(self):
        n = self.rng.randint(2, 4)
        return " ".join(str(self.rng.randint(-4, 4)) for _ in range(n))

    # ---- exact rationals --------------------------------------------------
    def gen_ratio(self, depth):
        r = self.rng
        if depth <= 0 or r.random() < 0.4:
            return "(/ %d %d)" % (r.randint(-20, 20), r.choice(NONZERO))
        op = r.choice(["+", "-", "*"])
        return "(%s %s %s)" % (op, self.gen_ratio(depth - 1), self.gen_ratio(depth - 1))

    # ---- reals (inexact) --------------------------------------------------
    def gen_real(self, depth):
        r = self.rng
        if depth <= 0:
            return "%.3f" % r.uniform(-20, 20)
        pick = r.randrange(7)
        if pick == 0:
            return "%.3f" % r.uniform(-20, 20)
        if pick == 1:
            op = r.choice(["+", "-", "*"])
            return "(%s %s %s)" % (op, self.gen_real(depth - 1), self.gen_real(depth - 1))
        if pick == 2:
            op = r.choice(["floor", "ceiling", "truncate", "round"])
            return "(%s %s)" % (op, self.gen_real(depth - 1))
        if pick == 3:
            return "(sqrt %s)" % ("%.3f" % r.uniform(0.1, 40.0))
        if pick == 4:
            return "(abs %s)" % self.gen_real(depth - 1)
        if pick == 5:
            return "(* 1.0 %s)" % self.gen_int(depth - 1)   # exact->inexact via *1.0
        return "(min %s %s)" % (self.gen_real(depth - 1), self.gen_real(depth - 1))

    # ---- booleans ---------------------------------------------------------
    def gen_bool(self, depth):
        r = self.rng
        if depth <= 0:
            return r.choice(["#t", "#f"])
        pick = r.randrange(9)
        if pick == 0:
            return r.choice(["#t", "#f"])
        if pick in (1, 2):
            op = r.choice(["<", "<=", ">", ">=", "=", "eqv?"])
            return "(%s %s %s)" % (op, self.gen_int(depth - 1), self.gen_int(depth - 1))
        if pick == 3:
            op = r.choice(["and", "or"])
            return "(%s %s %s)" % (op, self.gen_bool(depth - 1), self.gen_bool(depth - 1))
        if pick == 4:
            return "(not %s)" % self.gen_bool(depth - 1)
        if pick == 5:
            op = r.choice(["even?", "odd?", "zero?", "positive?", "negative?"])
            return "(%s %s)" % (op, self.gen_int(depth - 1))
        if pick == 6:
            op = r.choice(["null?", "pair?", "list?"])
            n = r.randint(0, 4)
            elems = " ".join(self.gen_int(0) for _ in range(n))
            return "(%s (list %s))" % (op, elems)
        if pick == 7:
            return "(equal? %s %s)" % (self.gen_listint(depth - 1), self.gen_listint(depth - 1))
        return "(if %s %s %s)" % (self.gen_bool(depth - 1),
                                  self.gen_bool(depth - 1), self.gen_bool(depth - 1))

    # ---- lists of integers ------------------------------------------------
    def gen_listint(self, depth):
        r = self.rng
        if depth <= 0:
            n = r.randint(0, 4)
            return "(list %s)" % " ".join(str(r.randint(-9, 9)) for _ in range(n))
        pick = r.randrange(10)
        if pick == 0:
            n = r.randint(0, 5)
            return "(list %s)" % " ".join(str(r.randint(-9, 9)) for _ in range(n))
        if pick == 1:
            return "(cons %s %s)" % (self.gen_int(depth - 1), self.gen_listint(depth - 1))
        if pick == 2:
            return "(append %s %s)" % (self.gen_listint(depth - 1), self.gen_listint(depth - 1))
        if pick == 3:
            return "(reverse %s)" % self.gen_listint(depth - 1)
        if pick == 4:
            return "(cdr (cons 0 %s))" % self.gen_listint(depth - 1)
        if pick == 5:
            v = self.fresh("x")
            return "(map (lambda (%s) (* %s %s)) %s)" % (v, v, v, self.gen_listint(depth - 1))
        if pick == 6:
            v = self.fresh("x")
            return "(map (lambda (%s) (+ %s %d)) %s)" % (v, v, r.randint(-5, 5),
                                                         self.gen_listint(depth - 1))
        if pick == 7:
            v = self.fresh("x")
            pred = r.choice(["even?", "odd?", "positive?", "negative?"])
            return "(gd-filter %s %s)" % (pred, self.gen_listint(depth - 1))
        if pick == 8:
            return "(gd-iota %d)" % r.randint(0, 8)
        # cons two ints then existing
        return "(list-copy %s)" % self.gen_listint(depth - 1)

    # ---- characters -------------------------------------------------------
    def gen_char(self, depth):
        r = self.rng
        c = r.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789"))
        pick = r.randrange(4)
        if pick == 0:
            return "#\\%s" % c
        if pick == 1:
            return "(char-upcase #\\%s)" % c
        if pick == 2:
            return "(char-downcase #\\%s)" % c
        return "(integer->char %d)" % r.randint(33, 126)

    # ---- strings ----------------------------------------------------------
    ALPHA = "abcdefghijklmnopqrstuvwxyz"

    def gen_string(self, depth):
        r = self.rng
        base = "".join(r.choice(self.ALPHA) for _ in range(r.randint(3, 9)))
        if depth <= 0:
            return '"%s"' % base
        pick = r.randrange(7)
        if pick == 0:
            return '"%s"' % base
        if pick == 1:
            return "(string-append %s %s)" % (self.gen_string(depth - 1), self.gen_string(depth - 1))
        if pick == 2:
            n = len(base)
            a = r.randint(0, n)
            b = r.randint(a, n)
            return '(substring "%s" %d %d)' % (base, a, b)
        if pick == 3:
            return "(string-upcase %s)" % self.gen_string(depth - 1)
        if pick == 4:
            return "(string-downcase %s)" % self.gen_string(depth - 1)
        if pick == 5:
            return "(list->string (list %s))" % " ".join(
                "#\\%s" % r.choice(self.ALPHA) for _ in range(r.randint(1, 5)))
        return "(make-string %d #\\%s)" % (r.randint(0, 5), r.choice(self.ALPHA))

    # ---- vectors ----------------------------------------------------------
    def gen_vector(self, depth):
        r = self.rng
        n = r.randint(0, 5)
        elems = " ".join(str(r.randint(-9, 9)) for _ in range(n))
        pick = r.randrange(4)
        if pick == 0 or n == 0:
            return "(vector %s)" % elems
        if pick == 1:
            v = self.fresh("x")
            return "(vector-map (lambda (%s) (* %s %s)) (vector %s))" % (v, v, v, elems)
        if pick == 2:
            return "(list->vector (list %s))" % elems
        return "(vector-copy (vector %s) %d)" % (elems, r.randrange(n))

    # ---- dispatch a printable probe of a random type ----------------------
    PRINTABLE = ["int", "ratio", "real", "bool", "listint", "char", "string", "vector"]

    def gen_printable(self, depth):
        t = self.rng.choice(self.PRINTABLE)
        return {
            "int": self.gen_int,
            "ratio": self.gen_ratio,
            "real": self.gen_real,
            "bool": self.gen_bool,
            "listint": self.gen_listint,
            "char": self.gen_char,
            "string": self.gen_string,
            "vector": self.gen_vector,
        }[t](depth)


# ---------------------------------------------------------------------------
# Program builders.
# ---------------------------------------------------------------------------

def build_diff_program(seed, n_probes=8, depth=4):
    rng = random.Random(seed)
    g = Gen(rng)
    lines = [PRELUDE]
    for _ in range(n_probes):
        lines.append("(display %s)(newline)" % g.gen_printable(depth))
    return "".join(l if l.endswith("\n") else l + "\n" for l in lines)


def build_meta_program(seed, n_props=8, depth=3):
    """Every displayed value MUST be #t on a correct implementation. A #f (or an
    error, or cross-oracle disagreement) exposes a bug — no reference needed."""
    rng = random.Random(seed)
    g = Gen(rng)
    lines = [PRELUDE]

    def prop(expr_bool):
        return "(display %s)(newline)" % expr_bool

    for _ in range(n_props):
        kind = rng.randrange(7)
        if kind == 0:  # apply equivalence: (f x) == (apply f (list x))
            v = g.fresh("x")
            body = g.gen_int(depth)
            arg = g.gen_int(1)
            f = "(lambda (%s) %s)" % (v, body)
            lines.append(prop("(let ((gf %s) (ga %s)) (equal? (gf ga) (apply gf (list ga))))"
                              % (f, arg)))
        elif kind == 1:  # map ordering: (map f xs) == gd-map1 f xs
            v = g.fresh("x")
            f = "(lambda (%s) (* %s %s))" % (v, v, v)
            xs = g.gen_listint(depth)
            lines.append(prop("(let ((gf %s) (gl %s)) (equal? (map gf gl) (gd-map1 gf gl)))"
                              % (f, xs)))
        elif kind == 2:  # commutativity of + and *
            a = g.gen_int(depth)
            b = g.gen_int(depth)
            lines.append(prop("(let ((ga %s) (gb %s)) "
                              "(and (= (+ ga gb) (+ gb ga)) (= (* ga gb) (* gb ga))))"
                              % (a, b)))
        elif kind == 3:  # reverse involution
            xs = g.gen_listint(depth)
            lines.append(prop("(let ((gl %s)) (equal? (reverse (reverse gl)) gl))" % xs))
        elif kind == 4:  # length homomorphism over append
            a = g.gen_listint(depth)
            b = g.gen_listint(depth)
            lines.append(prop("(let ((ga %s) (gb %s)) "
                              "(= (length (append ga gb)) (+ (length ga) (length gb))))"
                              % (a, b)))
        elif kind == 5:  # fold equivalence: gd-sum == apply +
            xs = g.gen_listint(depth)
            lines.append(prop("(let ((gl %s)) (= (gd-sum gl) (apply + gl)))" % xs))
        else:  # let re-association where independent
            a = g.gen_int(depth)
            b = g.gen_int(depth)
            lines.append(prop("(= (let ((ga %s) (gb %s)) (- ga gb)) "
                              "(let* ((gb %s) (ga %s)) (- ga gb)))"
                              % (a, b, b, a)))
    return "".join(l if l.endswith("\n") else l + "\n" for l in lines)


def generate_programs(seed=1234, count=60, diff_probes=8, meta_props=8):
    """Return a deterministic list of (family, name, body). Pure function of
    the arguments — the harness and the CLI both call this."""
    progs = [
        ("meta", spec["name"], build_surface_program(spec))
        for spec in SURFACE_PROBES
    ]
    for i in range(count):
        s = seed * 1_000_003 + i
        progs.append(("diff", "diff_%05d" % i, build_diff_program(s, diff_probes)))
    for i in range(count):
        s = seed * 2_000_003 + i
        progs.append(("meta", "meta_%05d" % i, build_meta_program(s, meta_props)))
    return progs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output corpus directory")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--count", type=int, default=60,
                    help="programs PER family (diff + meta) — total is 2*count")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(repo_root, "tests", "generative-diff", "corpus")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".scm"):
            os.remove(os.path.join(out_dir, f))

    progs = generate_programs(seed=args.seed, count=args.count)
    idx = 0
    for family, name, body in progs:
        idx += 1
        fname = "%04d_%s.scm" % (idx, name)
        header = ";; %s  (generated; seed=%d; R7RS-small portable)\n" % (name, args.seed)
        with open(os.path.join(out_dir, fname), "w") as fh:
            fh.write(header)
            fh.write(body)
    print("Wrote %d programs (seed=%d, %d per random family + %d surface probes) to %s"
          % (idx, args.seed, args.count, len(SURFACE_PROBES), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
