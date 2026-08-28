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
 *  • After editing this file — OR after adding, removing or reordering an
 *    entry in eshkol_vm.c's BUILTINS[] table, which emit_builtin_preamble()
 *    turns into one prelude local apiece — the bytecode cache
 *    (`vm_prelude_cache.h`) must be regenerated:
 *
 *        scripts/regenerate_vm_prelude_cache.sh
 *
 *    SW-49 (closed): a hand-copied shell recipe used to be recorded here in
 *    its place (`cc -DGENERATE_PRELUDE_CACHE ... build/libeshkol-runtime.a
 *    -lm -lc++ -framework ...`), and its predecessor
 *    (`gcc -DGENERATE_PRELUDE_CACHE eshkol_vm.c -o gen_prelude -lm`, still
 *    visible as the stale top-of-file comment history in
 *    vm_prelude_cache.c) had stopped linking outright once this
 *    translation unit's #include of eshkol_vm.c grew a transitive
 *    dependency on the rest of the runtime (arena/bignum/tensor/image-io/
 *    GPU) that `-lm` alone cannot satisfy. Either way it was a frozen,
 *    platform-specific guess at eshkol-static's link requirements rather
 *    than something the build system derived, nobody had ever actually run
 *    it end to end, and this generator's only consumer is the
 *    Emscripten-built WASM REPL (vm_wasm_repl.c, the one place that defines
 *    ESHKOL_VM_NO_DISASM as a macro) — so no native lane or ctest noticed
 *    when the committed cache drifted 30 builtins stale (missing
 *    `string-length`, `string-ref`, `integer?`, `gensym`,
 *    `ad-note-finite-difference!` and the whole c[ad]{3,4}r family).
 *
 *    The script above builds the generator through the real CMake target
 *    (`eshkol-vm-prelude-cache-gen`, CMakeLists.txt, right after
 *    `eshkol-vm-standalone-test`) so it always links against whatever
 *    `eshkol-static` currently requires — on any platform, under any
 *    BLAS/GPU/quantum/tensorcore configuration — instead of a comment that
 *    can silently fall behind. Two gates now hold this file to that source
 *    of truth on every PR: the build-free
 *    `scripts/check_vm_prelude_cache_builtins.py` (diffs BUILTINS[] against
 *    the committed name list as text, so it runs even on docs-only PRs)
 *    and the ctest `vm_prelude_cache_is_current` (builds the real generator
 *    and byte-diffs its output, so it also catches a stale bytecode BODY
 *    behind an unchanged name list).
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
    /* SRFI-1 order: (take lst n) / (drop lst n) — this used to be reversed
     * ((take n lst)), diverging from the SRFI-1 definition every other
     * engine (core/list/transform.esk, the native/AOT compiler's stdlib
     * path) uses. A call written against the documented order silently
     * took the wrong branch on the VM (e.g. `(take '(1 2 3) 2)` treated
     * the list as the count and 2 as the list). Converged on SRFI-1 order
     * so the VM's always-available embedded prelude agrees with the
     * on-disk stdlib module — see tests/vm_parity/corpus/54_take_drop_srfi1_order.esk. */
    "(define (take lst n) (if (= n 0) (list) (if (null? lst) (list) (cons (car lst) (take (cdr lst) (- n 1))))))\n"
    "(define (drop lst n) (if (= n 0) lst (if (null? lst) (list) (drop (cdr lst) (- n 1)))))\n"
    /* SRFI-1 iota: (iota count [start [step]]). Mirrors
     * lib/core/list/generate.esk exactly so the VM's always-available
     * embedded prelude and the on-disk stdlib module agree; the VM used to
     * leave `iota` bound to a dead BUILTINS-table entry (native id 141, no
     * dispatcher case) whenever a program said `(require stdlib)` instead of
     * explicitly `(require core.list.generate)`, so it silently returned ()
     * for any arity (filed: tests/vm_parity/found/iota_returns_empty.esk). */
    "(define (iota count . rest)\n"
    "  (let ((start (if (pair? rest) (car rest) 0))\n"
    "        (step (if (and (pair? rest) (pair? (cdr rest))) (cadr rest) 1)))\n"
    "    (let loop ((n (- count 1)) (acc (list)))\n"
    "      (if (< n 0) acc (loop (- n 1) (cons (+ start (* n step)) acc))))))\n"
    "(define (reduce f init lst) (fold-left f init lst))\n"
    "(define (merge compare a b)\n"
    "  (cond ((null? a) b) ((null? b) a)\n"
    "    ((compare (car a) (car b)) (cons (car a) (merge compare (cdr a) b)))\n"
    "    (else (cons (car b) (merge compare a (cdr b))))))\n"
    "(define (sort compare lst)\n"
    "  (if (or (null? lst) (null? (cdr lst))) lst\n"
    "    (let ((half (quotient (length lst) 2)))\n"
    "      (merge compare (sort compare (take lst half)) (sort compare (drop lst half))))))\n"
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
    "(define (format fmt . args) (_format-list fmt args))\n"
    /* User-reachable region handles (#341). The variadic surface is folded onto
     * the fixed-arity natives 2210/2211; #f stands for an omitted argument, and
     * the natives apply the same "lone numeric argument is the size hint" rule
     * the native backend does, so every arity agrees across substrates. */
    "(define (region-open . a)\n"
    "  (cond ((null? a) (_region-open #f #f))\n"
    "        ((null? (cdr a)) (_region-open (car a) #f))\n"
    "        (else (_region-open (car a) (car (cdr a))))))\n"
    "(define (region-close h . keeps) (_region-close-list h keeps))\n"
    "(define (emit! emitter event . args) (_emit-event emitter event args))\n"
    "(define (read . args)\n"
    "  (cond ((null? args) (_read0))\n"
    "        ((null? (cdr args)) (_read1 (car args)))\n"
    "        (else (error \"read: expected zero or one port argument\"))))\n"
    "(define (write value . args)\n"
    "  (cond ((null? args) (_write1 value))\n"
    "        ((null? (cdr args)) (_write2 value (car args)))\n"
    "        (else (error \"write: expected one value and at most one port\"))))\n"
    "(define (make-list n val) (let loop ((i 0) (acc (list))) (if (= i n) acc (loop (+ i 1) (cons val acc)))))\n"
    "(define (make-fact . args) (_make-fact1 (if (and (not (null? args)) (null? (cdr args)) (pair? (car args))) (car args) args)))\n"
    "(define (make-factor-graph n . rest) (if (null? rest) (_make-fg2 n (make-list n 2)) (_make-fg2 n (car rest))))\n"
    /* ── Tensor reduction wrappers ────────────────────────────────────── */
    "(define (tensor-sum t . args) (if (null? args) (_tensor-reduce-sum t -1) (_tensor-reduce-sum t (car args))))\n"
    "(define (tensor-mean t . args) (if (null? args) (_tensor-reduce-mean t -1) (_tensor-reduce-mean t (car args))))\n"
    "(define (tensor-max t . args) (if (null? args) (_tensor-reduce-max t -1) (_tensor-reduce-max t (car args))))\n"
    "(define (tensor-min t . args) (if (null? args) (_tensor-reduce-min t -1) (_tensor-reduce-min t (car args))))\n";

#endif /* ESHKOL_VM_PRELUDE_SOURCE_H */
