/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * vm_prelude_source.h — single canonical definition of the bytecode VM's
 * Scheme-level prelude. Three different sites used to carry their own
 * (slightly drifting) copies of this string:
 *
 *   1. compile_and_run()         — eshkol_vm.c, batch / file mode
 *   2. repl_session_create()     — eshkol_vm.c, interactive REPL
 *   3. vm_prelude_cache.c        — bytecode cache generator
 *
 * Keeping three copies in sync is what let `(map f a b)` silently drop the
 * second list (returning the wrong value for things like
 * `(fold-left + 0 (map * a b))`) for so long. This header is now the ONE
 * place that defines the prelude — all three sites #include it and use the
 * same `ESHKOL_VM_PRELUDE_SOURCE` constant.
 *
 * Notes for editors
 * -----------------
 *  • The prelude is plain Scheme source compiled at startup; no preprocessor
 *    interpolation is needed beyond C string concatenation.
 *  • The variadic `map` handles 1, 2, or 3 input lists. Higher arities
 *    require `apply`, which the bytecode VM does not yet implement, so the
 *    fall-through arm raises a clear error rather than silently dropping
 *    arguments.
 *  • After editing this file, the bytecode cache (`vm_prelude_cache.h`)
 *    must be regenerated:
 *        gcc -O2 -DGENERATE_PRELUDE_CACHE -Iinc -Ilib \
 *            lib/backend/vm_prelude_cache.c -o /tmp/gen_prelude -lm
 *        /tmp/gen_prelude > lib/backend/vm_prelude_cache.h
 */

#ifndef ESHKOL_VM_PRELUDE_SOURCE_H
#define ESHKOL_VM_PRELUDE_SOURCE_H

static const char* const ESHKOL_VM_PRELUDE_SOURCE =
    /* ── Higher-order list operations ─────────────────────────────────── */
    /* Variadic R7RS map. Handles 1, 2, or 3 input lists. The previous
     * single-list definition silently dropped extra arguments, so
     * (map * '(1 2 3) '(4 5 6)) returned (1 2 3) instead of (4 10 18). */
    "(define (map f . lsts)\n"
    "  (let ((n (length lsts)))\n"
    "    (cond\n"
    "      ((= n 1)\n"
    "       (let loop ((l (car lsts)) (acc '()))\n"
    "         (if (null? l) (reverse acc)\n"
    "             (loop (cdr l) (cons (f (car l)) acc)))))\n"
    "      ((= n 2)\n"
    "       (let loop ((a (car lsts)) (b (cadr lsts)) (acc '()))\n"
    "         (if (if (null? a) #t (null? b)) (reverse acc)\n"
    "             (loop (cdr a) (cdr b)\n"
    "                   (cons (f (car a) (car b)) acc)))))\n"
    "      ((= n 3)\n"
    "       (let loop ((a (car lsts)) (b (cadr lsts)) (c (caddr lsts)) (acc '()))\n"
    "         (if (if (null? a) #t (if (null? b) #t (null? c))) (reverse acc)\n"
    "             (loop (cdr a) (cdr b) (cdr c)\n"
    "                   (cons (f (car a) (car b) (car c)) acc)))))\n"
    "      (else (error \"map: only 1-3 input lists supported in VM REPL\")))))\n"
    "(define (filter pred lst)\n"
    "  (let loop ((l lst) (acc (list)))\n"
    "    (if (null? l) (reverse acc)\n"
    "      (if (pred (car l)) (loop (cdr l) (cons (car l) acc))\n"
    "        (loop (cdr l) acc)))))\n"
    "(define (fold-left f init lst)\n"
    "  (let loop ((l lst) (acc init))\n"
    "    (if (null? l) acc\n"
    "      (loop (cdr l) (f acc (car l))))))\n"
    /* fold and foldl are R6RS / Racket synonyms for fold-left so user code
     * written against either spelling resolves correctly. */
    "(define (fold f init lst) (fold-left f init lst))\n"
    "(define (foldl f init lst) (fold-left f init lst))\n"
    "(define (fold-right f init lst) (if (null? lst) init (f (car lst) (fold-right f init (cdr lst)))))\n"
    "(define (foldr f init lst) (fold-right f init lst))\n"
    "(define (for-each f lst) (if (null? lst) 0 (begin (f (car lst)) (for-each f (cdr lst)))))\n"
    "(define (any pred lst) (if (null? lst) #f (if (pred (car lst)) #t (any pred (cdr lst)))))\n"
    "(define (every pred lst) (if (null? lst) #t (if (pred (car lst)) (every pred (cdr lst)) #f)))\n"
    "(define (find pred lst) (if (null? lst) #f (if (pred (car lst)) (car lst) (find pred (cdr lst)))))\n"
    "(define (take n lst) (if (= n 0) (list) (if (null? lst) (list) (cons (car lst) (take (- n 1) (cdr lst))))))\n"
    "(define (drop n lst) (if (= n 0) lst (if (null? lst) (list) (drop (- n 1) (cdr lst)))))\n"
    "(define (reduce f init lst) (fold-left f init lst))\n"
    "(define (merge compare a b)\n"
    "  (cond ((null? a) b) ((null? b) a)\n"
    "    ((compare (car a) (car b)) (cons (car a) (merge compare (cdr a) b)))\n"
    "    (else (cons (car b) (merge compare a (cdr b))))))\n"
    "(define (sort compare lst)\n"
    "  (if (or (null? lst) (null? (cdr lst))) lst\n"
    "    (let ((half (quotient (length lst) 2)))\n"
    "      (merge compare (sort compare (take half lst)) (sort compare (drop half lst))))))\n"
    /* ── Variadic numeric operators ───────────────────────────────────── */
    "(define + (lambda args (fold-left add2 0 args)))\n"
    "(define * (lambda args (fold-left mul2 1 args)))\n"
    "(define (- . args) (if (null? (cdr args)) (sub2 0 (car args)) (fold-left sub2 (car args) (cdr args))))\n"
    "(define (/ . args) (if (null? (cdr args)) (div2 1 (car args)) (fold-left div2 (car args) (cdr args))))\n"
    /* ── Variadic wrappers around 2-arg builtins ──────────────────────── */
    "(define _append-2 append)\n"
    "(define (append . lists) (fold-right _append-2 '() lists))\n"
    "(define (number->string n . args) (_number->string-2 n (if (null? args) 10 (car args))))\n"
    "(define (atan x . rest) (if (null? rest) (_atan1 x) (_atan2 x (car rest))))\n"
    "(define (max a . rest) (fold-left _max2 a rest))\n"
    "(define (min a . rest) (fold-left _min2 a rest))\n"
    "(define (string-append . args) (fold-left _string-append-2 \"\" args))\n"
    "(define (make-list n val) (let loop ((i 0) (acc (list))) (if (= i n) acc (loop (+ i 1) (cons val acc)))))\n"
    "(define (make-factor-graph n . rest) (if (null? rest) (_make-fg2 n (make-list n 2)) (_make-fg2 n (car rest))))\n"
    /* ── Tensor reduction wrappers ────────────────────────────────────── */
    "(define (tensor-sum t . args) (if (null? args) (_tensor-reduce-sum t -1) (_tensor-reduce-sum t (car args))))\n"
    "(define (tensor-mean t . args) (if (null? args) (_tensor-reduce-mean t -1) (_tensor-reduce-mean t (car args))))\n"
    "(define (tensor-max t . args) (if (null? args) (_tensor-reduce-max t -1) (_tensor-reduce-max t (car args))))\n"
    "(define (tensor-min t . args) (if (null? args) (_tensor-reduce-min t -1) (_tensor-reduce-min t (car args))))\n";

#endif /* ESHKOL_VM_PRELUDE_SOURCE_H */
