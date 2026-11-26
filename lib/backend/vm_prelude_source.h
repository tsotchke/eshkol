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
 * The generated vm_prelude_cache.h extends this prelude with the canonical
 * lib/stdlib.esk dependency closure. That header is the complete bootstrap
 * image used by filesystem-free WASM; this source remains the cache generator
 * and desktop VM prelude input.
 *
 * Notes for editors
 * -----------------
 *  • The prelude is plain Scheme source compiled at startup; no preprocessor
 *    interpolation is needed beyond C string concatenation.
 *  • The variadic `map` and `for-each` use explicit lockstep arms through four
 *    input sequences, and the vector wrappers collect their sequences before
 *    delegating to those arms. This keeps the requested multi-list closure
 *    calls on one VM-safe path and stops at the shortest input as required by
 *    R7RS.
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
    /* These helpers walk the cursor list directly. They deliberately do not
     * call map recursively: doing so nests closure frames before the outer
     * cursor advances and can exhaust the VM frame budget. */
    ";; Algebraic effects (ADR-0045): the handler context. Native keeps it in\n"
    ";; the runtime (runtime_continuations.cpp); the VM in one global.\n"
    "(define __eshkol-effect-context (quote ()))\n"
    "(define (%effect-context) __eshkol-effect-context)\n"
    "(define (%effect-context-set! v) (set! __eshkol-effect-context v) v)\n"
    "(define (__eshkol-any-null? xs)\n"
    "  (if (null? xs) #f\n"
    "      (if (null? (car xs)) #t (__eshkol-any-null? (cdr xs)))))\n"
    "(define (__eshkol-cursor-heads xs acc)\n"
    "  (if (null? xs) (reverse acc)\n"
    "      (__eshkol-cursor-heads (cdr xs)\n"
    "                             (cons (car (car xs)) acc))))\n"
    "(define (__eshkol-cursor-cdrs xs acc)\n"
    "  (if (null? xs) (reverse acc)\n"
    "      (__eshkol-cursor-cdrs (cdr xs)\n"
    "                            (cons (cdr (car xs)) acc))))\n"
    "(define (__eshkol-map1 f xs)\n"
    "  (if (null? xs) (list)\n"
    "      (cons (f (car xs)) (__eshkol-map1 f (cdr xs)))))\n"
    "(define (__eshkol-map2 f xs ys)\n"
    "  (if (or (null? xs) (null? ys)) (list)\n"
    "      (cons (f (car xs) (car ys))\n"
    "            (__eshkol-map2 f (cdr xs) (cdr ys)))))\n"
    "(define (__eshkol-map3 f xs ys zs)\n"
    "  (if (or (null? xs) (null? ys) (null? zs)) (list)\n"
    "      (cons (f (car xs) (car ys) (car zs))\n"
    "            (__eshkol-map3 f (cdr xs) (cdr ys) (cdr zs)))))\n"
    "(define (__eshkol-map4 f xs ys zs ws)\n"
    "  (if (or (null? xs) (null? ys) (null? zs) (null? ws)) (list)\n"
    "      (cons (f (car xs) (car ys) (car zs) (car ws))\n"
    "            (__eshkol-map4 f (cdr xs) (cdr ys) (cdr zs) (cdr ws)))))\n"
    "(define (map f . lsts)\n"
    "  (if (null? lsts) (error \"map: requires at least one input list\")\n"
    "      (if (null? (cdr lsts))\n"
    "          (__eshkol-map1 f (car lsts))\n"
    "          (if (null? (cdr (cdr lsts)))\n"
    "              (__eshkol-map2 f (car lsts) (car (cdr lsts)))\n"
    "              (if (null? (cdr (cdr (cdr lsts))))\n"
    "                  (__eshkol-map3 f (car lsts) (car (cdr lsts))\n"
    "                                  (car (cdr (cdr lsts))))\n"
    "                  (__eshkol-map4 f (car lsts) (car (cdr lsts))\n"
    "                                  (car (cdr (cdr lsts)))\n"
    "                                  (car (cdr (cdr (cdr lsts))))))))))\n"
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
    "(define (__eshkol-for-each1 f xs)\n"
    "  (if (null? xs) (if #f #f)\n"
    "      (begin (f (car xs)) (__eshkol-for-each1 f (cdr xs)))))\n"
    "(define (__eshkol-for-each2 f xs ys)\n"
    "  (if (or (null? xs) (null? ys)) (if #f #f)\n"
    "      (begin (f (car xs) (car ys))\n"
    "             (__eshkol-for-each2 f (cdr xs) (cdr ys)))))\n"
    "(define (__eshkol-for-each3 f xs ys zs)\n"
    "  (if (or (null? xs) (null? ys) (null? zs)) 0\n"
    "      (begin (f (car xs) (car ys) (car zs))\n"
    "             (__eshkol-for-each3 f (cdr xs) (cdr ys) (cdr zs)))))\n"
    "(define (__eshkol-for-each4 f xs ys zs ws)\n"
    "  (if (or (null? xs) (null? ys) (null? zs) (null? ws)) 0\n"
    "      (begin (f (car xs) (car ys) (car zs) (car ws))\n"
    "             (__eshkol-for-each4 f (cdr xs) (cdr ys) (cdr zs) (cdr ws)))))\n"
    "(define (for-each f . lsts)\n"
    "  (if (null? lsts) (error \"for-each: requires at least one input list\")\n"
    "      (if (null? (cdr lsts))\n"
    "          (__eshkol-for-each1 f (car lsts))\n"
    "          (if (null? (cdr (cdr lsts)))\n"
    "              (__eshkol-for-each2 f (car lsts) (car (cdr lsts)))\n"
    "              (if (null? (cdr (cdr (cdr lsts))))\n"
    "                  (__eshkol-for-each3 f (car lsts) (car (cdr lsts))\n"
    "                                      (car (cdr (cdr lsts))))\n"
    "                  (__eshkol-for-each4 f (car lsts) (car (cdr lsts))\n"
    "                                      (car (cdr (cdr lsts)))\n"
    "                                      (car (cdr (cdr (cdr lsts))))))))))\n"
    "(define (vector-map f . vecs)\n"
    "  (list->vector (apply map (cons f (map vector->list vecs)))))\n"
    "(define (vector-for-each f . vecs)\n"
    "  (apply for-each (cons f (map vector->list vecs))))\n"
    /* LE-16: vector-copy / vector-copy! / vector-append had NO representation
     * at all on the VM — not a value-position gap but a call-position one:
     * `(vector-copy (vector 1 2 3))` raised "undefined variable 'vector-copy'"
     * and then crashed the VM ("calling non-function"). Native has dedicated
     * IR for these (collection_codegen.cpp); the VM gets the R7RS semantics
     * for free by defining them in Scheme over the vector-ref/vector-set!/
     * vector-length primitives the VM already has as opcodes. Being ordinary
     * `define`s, they are first-class values by construction like every
     * other prelude procedure (vector-map/vector-for-each above) — this is
     * the SAME fix shape LE-01's notes point at for the VM side. */
    "(define (vector-copy v . rest)\n"
    "  (let* ((len (vector-length v))\n"
    "         (start (if (pair? rest) (car rest) 0))\n"
    "         (end (if (and (pair? rest) (pair? (cdr rest))) (cadr rest) len))\n"
    "         (result (make-vector (- end start) 0)))\n"
    "    (let loop ((i start))\n"
    "      (if (< i end)\n"
    "          (begin (vector-set! result (- i start) (vector-ref v i))\n"
    "                 (loop (+ i 1)))\n"
    "          result))))\n"
    "(define (vector-copy! to at from . rest)\n"
    "  (let* ((flen (vector-length from))\n"
    "         (start (if (pair? rest) (car rest) 0))\n"
    "         (end (if (and (pair? rest) (pair? (cdr rest))) (cadr rest) flen)))\n"
    "    (let loop ((i start) (j at))\n"
    "      (if (< i end)\n"
    "          (begin (vector-set! to j (vector-ref from i))\n"
    "                 (loop (+ i 1) (+ j 1)))\n"
    "          to))))\n"
    /* Written directly over vector-ref/vector-set!/vector-length rather than
     * as (list->vector (apply append (map vector->list vecs))): `append` is
     * defined LATER in this same prelude string, and while that is legal
     * Scheme (the reference inside a lambda body is only resolved when the
     * lambda is CALLED, by which point every top-level define here has
     * already run), the VM's prelude compiler resolved it eagerly and bound
     * it to whatever `append` meant at vector-append's OWN define point —
     * silently dropping every vector past the first two
     * ((vector-append #(1) #(2) #(3)) -> #(1 2), not #(1 2 3)). Avoiding the
     * forward reference sidesteps that rather than depending on prelude
     * definition order, which is fragile to get right and easy to get wrong
     * again on the next edit. */
    "(define (vector-append . vecs)\n"
    "  (let* ((total (let __loop ((vs vecs) (n 0))\n"
    "                  (if (null? vs) n\n"
    "                      (__loop (cdr vs) (+ n (vector-length (car vs)))))))\n"
    "         (result (make-vector total 0)))\n"
    "    (let __outer ((vs vecs) (offset 0))\n"
    "      (if (null? vs)\n"
    "          result\n"
    "          (let ((v (car vs)))\n"
    "            (let __inner ((i 0))\n"
    "              (if (< i (vector-length v))\n"
    "                  (begin (vector-set! result (+ offset i) (vector-ref v i))\n"
    "                         (__inner (+ i 1)))\n"
    "                  (__outer (cdr vs) (+ offset (vector-length v))))))))))\n"
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
    "(define (sort lst compare)\n"
    "  (if (not (list? lst)) (error \"sort: first argument must be a list\")\n"
    "    (if (or (null? lst) (null? (cdr lst))) lst\n"
    "      (let ((half (quotient (length lst) 2)))\n"
    "        (merge compare (sort (take lst half) compare) (sort (drop lst half) compare))))))\n"
    /* ── Variadic numeric operators ───────────────────────────────────── */
    "(define + (lambda args (fold-left add2 0 args)))\n"
    "(define * (lambda args (fold-left mul2 1 args)))\n"
    "(define (- . args) (if (null? (cdr args)) (sub2 0 (car args)) (fold-left sub2 (car args) (cdr args))))\n"
    "(define (/ . args) (if (null? (cdr args)) (div2 1 (car args)) (fold-left div2 (car args) (cdr args))))\n"
    /* Keep the raw opcode available while exposing the public optional
     * default as a procedure value, with an explicit upper arity. */
    "(define _hash-ref-raw hash-ref)\n"
    "(define (hash-ref table key . rest)\n"
    "  (cond ((null? rest) (_hash-ref-raw table key #f))\n"
    "        ((null? (cdr rest)) (_hash-ref-raw table key (car rest)))\n"
    "        (else (error \"Arity mismatch: hash-ref expects 2 or 3 arguments\"))))\n"
    /* ── Variadic wrappers around 2-arg builtins ──────────────────────── */
    "(define _append-2 append)\n"
    "(define (append . lists) (fold-right _append-2 '() lists))\n"
    "(define (number->string n . args) (_number->string-2 n (if (null? args) 10 (car args))))\n"
    "(define (atan x . rest) (if (null? rest) (_atan1 x) (_atan2 x (car rest))))\n"
    "(define (max a . rest) (fold-left _max2 a rest))\n"
    "(define (min a . rest) (fold-left _min2 a rest))\n"
    "(define (string-append . args) (fold-left _string-append-2 \"\" args))\n"
    /* SW-241: the R7RS comparison chains as VALUES take two or more
     * arguments; the call position keeps its own lowering. */
    "(define (_chain2 f) (lambda (a b . rest) (let loop ((a a) (b b) (rest rest)) (if (f a b) (if (null? rest) #t (loop b (car rest) (cdr rest))) #f))))\n"
    "(define _cmp2_0 <) (define < (_chain2 _cmp2_0))\n"
    "(define _cmp2_1 >) (define > (_chain2 _cmp2_1))\n"
    "(define _cmp2_2 <=) (define <= (_chain2 _cmp2_2))\n"
    "(define _cmp2_3 >=) (define >= (_chain2 _cmp2_3))\n"
    "(define _cmp2_4 =) (define = (_chain2 _cmp2_4))\n"
    "(define _cmp2_5 string=?) (define string=? (_chain2 _cmp2_5))\n"
    "(define _cmp2_6 string<?) (define string<? (_chain2 _cmp2_6))\n"
    "(define _cmp2_10 string-ci=?) (define string-ci=? (_chain2 _cmp2_10))\n"
    "(define _cmp2_15 char=?) (define char=? (_chain2 _cmp2_15))\n"
    "(define _cmp2_16 char<?) (define char<? (_chain2 _cmp2_16))\n"
    "(define _cmp2_17 char>?) (define char>? (_chain2 _cmp2_17))\n"
    /* Derive the inclusive predicates from the native strict comparisons.
     * Reuse their binary comparison results while exposing the existing
     * _chain2 two-or-more argument procedure shape. */
    "(define _char-le-2 (lambda (a b) (not (char>? a b))))\n"
    "(define char<=? (_chain2 _char-le-2))\n"
    "(define _char-ge-2 (lambda (a b) (not (char<? a b))))\n"
    "(define char>=? (_chain2 _char-ge-2))\n"
    /* SW-173: `list`, `vector` and `string` are compiled by head symbol in
     * CALL position (vm_compiler.c lowers `(list a b)` to a cons chain and
     * `(vector …)` to OP_VEC_CREATE before it ever looks a binding up), so
     * the names themselves had no VALUE — `(map list xs)` died with
     * "undefined variable 'list'" while `(map (lambda (x) (list x)) xs)`
     * worked. These give the bare names the honest variadic procedure the
     * call position already implements; the head-symbol fast paths are
     * unaffected because they are matched before any variable lookup. */
    "(define (list . args) args)\n"
    "(define (vector . args) (list->vector args))\n"
    "(define (string . chars) (list->string chars))\n"
    "(define (format fmt . args) (_format-list fmt args))\n"
    /* Keep the documented seed spelling available in the VM's always-loaded
     * prelude; it delegates to the same fixed-arity srand48 builtin used by
     * the on-disk random library. */
    "(define (set-random-seed! seed) (srand48 seed))\n"
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
    /* R7RS 6.13.1: the standard ports are parameter objects, so
     * `parameterize` rebinds them and a write or read that names no port
     * uses the current one (vm_current_port in vm_native.c). */
    "(define current-input-port (_std-port-parameter 0))\n"
    "(define current-output-port (_std-port-parameter 1))\n"
    "(define current-error-port (_std-port-parameter 2))\n"
    /* (swish x [beta]) = x * sigmoid(beta * x), a number or a tensor like x,
     * as the native activation family lowers it (codegenActivationFamily):
     * the unit-beta tensor case is the SiLU kernel, another beta scales the
     * tensor by a one-element tensor, and a number goes through sigmoid so
     * derivatives follow. */
    "(define (swish x . b)\n"
    "  (let ((beta (if (null? b) 1.0 (car b))))\n"
    "    (if (or (vector? x) (tensor? x))\n"
    "        (if (null? b) (_swish-tensor x)\n"
    "            (tensor-mul x (sigmoid (tensor-mul x (make-tensor (list 1) beta)))))\n"
    "        (* x (sigmoid (* beta x))))))\n"
    "(define _display1 display)\n"
    "(define (display value . args)\n"
    "  (cond ((null? args) (_display1 value))\n"
    "        ((null? (cdr args)) (_display2 value (car args)))\n"
    "        (else (error \"display: expected one value and at most one port\"))))\n"
    "(define _newline0 newline)\n"
    "(define (newline . args)\n"
    "  (cond ((null? args) (_newline0))\n"
    "        ((null? (cdr args)) (_newline1 (car args)))\n"
    "        (else (error \"newline: expected at most one port\"))))\n"
    "(define (make-list n val) (let loop ((i 0) (acc (list))) (if (= i n) acc (loop (+ i 1) (cons val acc)))))\n"
    "(define (make-fact . args) (_make-fact1 (if (and (not (null? args)) (null? (cdr args)) (pair? (car args))) (car args) args)))\n"
    "(define (make-factor-graph n . rest) (if (null? rest) (_make-fg2 n (make-list n 2)) (_make-fg2 n (car rest))))\n"
    /* ── Tensor reduction wrappers ────────────────────────────────────── */
    /* The reduction natives take the optional-argument list itself: '() is
     * "no axis" (a full reduction) and (axis) carries the axis a program
     * wrote, whatever its value, so no operand a program can write is read
     * as "no axis". */
    "(define (tensor-sum t . args) (_tensor-reduce-sum t args))\n"
    "(define (tensor-mean t . args) (_tensor-reduce-mean t args))\n"
    "(define (tensor-max t . args) (_tensor-reduce-max t args))\n"
    "(define (tensor-min t . args) (_tensor-reduce-min t args))\n"
    "(define (bytevector . bytes)\n"
    "  (let ((result (_bytevector-from-list bytes))) result))\n"
    /* ── ADR-0064 Wave 1A: value bindings for names the native LLVM backend
     * lowers in call position only. Everything below is expressed over
     * primitives the VM already carries (opcodes, BUILTINS[] rows and the
     * compiler's head-symbol special forms), so the desktop VM and the
     * generated WASM prelude image gain the same surface at once.
     *
     * Naming convention: a name the vm_compiler.c head-symbol dispatch
     * already lowers (floor-quotient, make-promise, exact-rational, …) is
     * reached through a differently-spelled helper. The helper's body is the
     * special form; binding the public name to the helper then gives the
     * name a first-class VALUE without depending on whether the dispatcher
     * consults its special-form table before or after local resolution. */
    /* ── SRFI-1 list operations ───────────────────────────────────────── */
    /* (acons key value alist) = ((key . value) . alist), the pair shape
     * codegenAcons builds. */
    "(define (acons key value alist) (cons (cons key value) alist))\n"
    /* (list* a b c d) = (a b c . d); one argument is returned unchanged,
     * matching codegenListStar's single-argument arm. */
    "(define (list* first . rest)\n"
    "  (if (null? rest)\n"
    "      first\n"
    "      (let __ls ((head first) (tail rest))\n"
    "        (if (null? (cdr tail))\n"
    "            (cons head (car tail))\n"
    "            (cons head (__ls (car tail) (cdr tail)))))))\n"
    /* codegenLast answers the empty list with '() rather than raising. */
    "(define (last lst)\n"
    "  (if (null? lst) (list)\n"
    "      (if (null? (cdr lst)) (car lst) (last (cdr lst)))))\n"
    /* codegenRemove's two modes: a procedure operand is CALLED and the
     * elements it accepts are dropped; any other operand is COMPARED, under
     * equal? (remove), eqv? (remv) or eq? (remq). */
    "(define (__eshkol-remove-with same p lst)\n"
    "  (if (procedure? p)\n"
    "      (filter (lambda (x) (not (p x))) lst)\n"
    "      (filter (lambda (x) (not (same x p))) lst)))\n"
    "(define (remove p lst) (__eshkol-remove-with equal? p lst))\n"
    "(define (remv p lst) (__eshkol-remove-with eqv? p lst))\n"
    "(define (remq p lst) (__eshkol-remove-with eq? p lst))\n"
    /* codegenSplitAt returns ONE pair (prefix . suffix), not two values, and
     * stops at the end of a list shorter than the index — exactly the
     * take/drop pair defined earlier in this prelude. */
    "(define (split-at lst k) (cons (take lst k) (drop lst k)))\n"
    /* ── Vectors ──────────────────────────────────────────────────────── */
    /* R7RS (vector-fill! v fill [start [end]]) over the vector-set! /
     * vector-length primitives the VM already has, returning unspecified.
     * CollectionCodegen::vectorFill admits the two-argument form only, so
     * the optional range is a VM-side superset rather than a divergence on
     * any call the native backend accepts. */
    "(define (vector-fill! v fill . rest)\n"
    "  (let* ((len (vector-length v))\n"
    "         (start (if (pair? rest) (car rest) 0))\n"
    "         (end (if (and (pair? rest) (pair? (cdr rest))) (cadr rest) len)))\n"
    "    (let __vf ((i start))\n"
    "      (if (< i end)\n"
    "          (begin (vector-set! v i fill) (__vf (+ i 1)))\n"
    "          (if #f #f)))))\n"
    /* ── Sum types (Either) ───────────────────────────────────────────── */
    /* codegenSumInject / codegenSumTag / codegenSumValue / codegenSumPredicate
     * carry no dedicated runtime representation: a sum value IS an ordinary
     * cons cell (tag . value) with an exact-integer tag, 0 for left and 1 for
     * right, and the accessors are car / cdr. */
    "(define (inject-left value) (cons 0 value))\n"
    "(define (inject-right value) (cons 1 value))\n"
    "(define (sum-tag s) (car s))\n"
    "(define (sum-value s) (cdr s))\n"
    "(define (left? s) (= (car s) 0))\n"
    "(define (right? s) (= (car s) 1))\n"
    /* ── Symbols ──────────────────────────────────────────────────────── */
    /* R7RS 6.5: symbol=? takes two or more arguments. Symbols are interned,
     * so identity IS equality — the same comparison codegenSymbolEq emits —
     * and the chain shape comes from the _chain2 combinator above. */
    "(define _symbol-eq-2 (lambda (a b) (eq? a b)))\n"
    "(define symbol=? (_chain2 _symbol-eq-2))\n"
    /* ── Exactness and the rational tower ─────────────────────────────── */
    /* R7RS (exact z) IS (inexact->exact z). */
    "(define (exact x) (inexact->exact x))\n"
    /* Native answers #f for a non-number rather than raising, so both
     * predicates keep a number? guard ahead of the exactness test. An exact
     * value is a fixnum, a bignum or a reduced rational, which is precisely
     * the exact half of the rational tower. */
    "(define (exact-integer? x) (and (number? x) (exact? x) (integer? x)))\n"
    "(define (exact-rational? x) (and (number? x) (exact? x) (rational? x)))\n"
    /* R7RS 6.2.6 integer division. The VM already carries a fully typed
     * implementation of each of the four (fixnum / bignum / i128 / flonum
     * arms), reached through the compiler's head-symbol lowering; these give
     * the same four names a first-class value. */
    "(define (__eshkol-floor-quotient-2 n d) (floor-quotient n d))\n"
    "(define floor-quotient __eshkol-floor-quotient-2)\n"
    "(define (__eshkol-floor-remainder-2 n d) (floor-remainder n d))\n"
    "(define floor-remainder __eshkol-floor-remainder-2)\n"
    "(define (__eshkol-truncate-quotient-2 n d) (truncate-quotient n d))\n"
    "(define truncate-quotient __eshkol-truncate-quotient-2)\n"
    "(define (__eshkol-truncate-remainder-2 n d) (truncate-remainder n d))\n"
    "(define truncate-remainder __eshkol-truncate-remainder-2)\n"
    /* floor/ and truncate/ return TWO values, as the native backend's
     * multi-value packing does; call-with-values / let-values unpack them. */
    "(define (floor/ n d)\n"
    "  (values (__eshkol-floor-quotient-2 n d) (__eshkol-floor-remainder-2 n d)))\n"
    "(define (truncate/ n d)\n"
    "  (values (__eshkol-truncate-quotient-2 n d) (__eshkol-truncate-remainder-2 n d)))\n"
    /* The short spellings the native backend accepts as aliases: `mod` is
     * `modulo`, `ceil` is `ceiling`, `trunc` is `truncate`. `fabs` is NOT
     * `abs` — codegenAbs preserves an exact integer argument's type while
     * codegenMathFunction("fabs") always answers a flonum, so the inexact
     * coercion is part of the name's meaning. */
    "(define (mod n d) (modulo n d))\n"
    "(define (ceil x) (ceiling x))\n"
    "(define (trunc x) (truncate x))\n"
    "(define (fabs x) (exact->inexact (abs x)))\n"
    /* make-rational / /rational build an EXACT rational. The VM's exact
     * constructor is the `exact-rational` head-symbol lowering the reader
     * already desugars `1/3` into, so both names reach the one normalising
     * constructor (reduced, positive denominator, denominator 1 collapsing
     * to a plain integer) rather than a second copy of it. */
    "(define (__eshkol-make-rational-2 n d) (exact-rational n d))\n"
    "(define make-rational __eshkol-make-rational-2)\n"
    "(define /rational __eshkol-make-rational-2)\n"
    /* ── Promises ─────────────────────────────────────────────────────── */
    /* (make-promise v) wraps an already-available value in a promise, as the
     * native allocation of a forced promise does. The operand is evaluated
     * before `delay` ever sees it, so the thunk is a constant one and `force`
     * answers v. Native wraps unconditionally — it does not return an operand
     * that is already a promise — and this follows it. */
    "(define (__eshkol-make-promise-1 v) (delay v))\n"
    "(define make-promise __eshkol-make-promise-1)\n"
    /* ── Strings over user closures ───────────────────────────────────── */
    /* string-map / string-for-each walk ONE string and call the closure with
     * each character, like the native lowering; string-for-each answers '()
     * and string-map answers the mapped string. Built on the lockstep map /
     * for-each arms defined at the top of this prelude so the closure calls
     * stay on the same VM-safe path vector-map and vector-for-each use. */
    "(define (string-map f s) (list->string (__eshkol-map1 f (string->list s))))\n"
    "(define (string-for-each f s)\n"
    "  (begin (__eshkol-for-each1 f (string->list s)) (list)))\n"
    /* ── Tensors ──────────────────────────────────────────────────────── */
    /* (trace m) is the LINEAR-ALGEBRA trace: the sum of the diagonal of a
     * rank-2 tensor, over min(rows, cols) terms, as a flonum. A tensor of any
     * other rank answers 0.0, which is what codegenTrace's non-2-D arm does
     * rather than raising. */
    "(define (trace m)\n"
    "  (let ((s (tensor-shape m)))\n"
    "    (if (not (= (length s) 2))\n"
    "        0.0\n"
    "        (let* ((rows (car s))\n"
    "               (cols (cadr s))\n"
    "               (n (if (< rows cols) rows cols)))\n"
    "          (let __tr ((i 0) (acc 0.0))\n"
    "            (if (< i n)\n"
    "                (__tr (+ i 1) (+ acc (tensor-ref m (* i (+ cols 1)))))\n"
    "                acc))))))\n"
    /* (concatenate axis t1 t2 ...) joins rank-n tensors along AXIS, the
     * stride-aware copy TensorCodegen::concatenate emits: the result keeps
     * the first operand's shape with dimension AXIS replaced by the sum of
     * the operands' AXIS dimensions, and each operand contributes a block of
     * (its axis extent * inner stride) elements per outer position. Native
     * requires an axis and at least two tensors. */
    "(define (__eshkol-shape-product dims) (fold-left * 1 dims))\n"
    "(define (concatenate axis . tensors)\n"
    "  (if (or (null? tensors) (null? (cdr tensors)))\n"
    "      (error \"concatenate: requires an axis and at least two tensors\")\n"
    "      (let* ((shape (tensor-shape (car tensors)))\n"
    "             (prefix (take shape axis))\n"
    "             (suffix (drop shape (+ axis 1)))\n"
    "             (outer (__eshkol-shape-product prefix))\n"
    "             (inner (__eshkol-shape-product suffix))\n"
    "             (extents (__eshkol-map1 (lambda (t)\n"
    "                                       (list-ref (tensor-shape t) axis))\n"
    "                                     tensors))\n"
    "             (total (fold-left + 0 extents))\n"
    "             (out (make-tensor (append prefix (cons total suffix)) 0.0))\n"
    "             (dest-block (* total inner)))\n"
    "        (let __cc ((ts tensors) (offset 0))\n"
    "          (if (null? ts)\n"
    "              out\n"
    "              (let* ((t (car ts))\n"
    "                     (extent (list-ref (tensor-shape t) axis))\n"
    "                     (src-block (* extent inner)))\n"
    "                (let __co ((o 0))\n"
    "                  (if (< o outer)\n"
    "                      (begin\n"
    "                        (let __ci ((j 0))\n"
    "                          (if (< j src-block)\n"
    "                              (begin\n"
    "                                (tensor-set! out\n"
    "                                             (+ (* o dest-block)\n"
    "                                                (* offset inner) j)\n"
    "                                             (tensor-ref t\n"
    "                                                         (+ (* o src-block) j)))\n"
    "                                (__ci (+ j 1)))\n"
    "                              (if #f #f)))\n"
    "                        (__co (+ o 1)))\n"
    "                      (if #f #f)))\n"
    "                (__cc (cdr ts) (+ offset extent))))))))\n"
    /* ── Runtime ──────────────────────────────────────────────────────── */
    /* R7RS (emergency-exit [obj]) terminates with the given status, zero when
     * omitted. The native lowering reaches _exit directly; the VM has one
     * process-termination builtin, which this folds the optional argument
     * onto. A VM native that bypasses the runtime's shutdown hooks the way
     * _exit does is recorded as a follow-up for the Wave 1 integrator. */
    "(define (emergency-exit . args)\n"
    "  (exit (if (null? args) 0 (car args))))\n";

#endif /* ESHKOL_VM_PRELUDE_SOURCE_H */
