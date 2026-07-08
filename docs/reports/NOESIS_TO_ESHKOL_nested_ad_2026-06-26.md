# Noesis → Eshkol: higher-order/nested AD — correctness + performance (2026-06-26)

Context: building fully-geometric recursive self-improvement in Noesis (learn the
optimization *metric* by a meta-gradient = differentiate through the inner
optimization). PR #75 nested AD is a big help — **single** nested gradients are
exact and fast. But three issues block *iterative* metric learning. All verified on
master `60f05bfd`, freshly rebuilt `eshkol-run`, arm64 macOS.

## What works (great — thank you for #75)
```scheme
(define (f x) (* x (* x x)))
(gradient f 2.0)              ; => 12   (f'(2)=3·2²) ✓
(gradient (lambda (x) (gradient f x)) 2.0)   ; => 12 (f''(2)=6·2) ✓  exact, instant
;; meta-gradient through a single straight-line preconditioned step: exact
(define (L w) (let ((d (- w 5.0))) (* 8.0 (* d d))))
(define (post m) (L (- 0.0 (* m 0.05 (gradient L 0.0)))))
(gradient post 0.05)         ; => -307.2  ✓ (analytic), ~0.03 ms
;; iterated 30× on a TOP-LEVEL function: all fine, rc=0
```

## Issue 1 — CORRECTNESS: meta-gradient through a named-let loop that calls `gradient` is corrupt
**Severity: HIGH** (silently wrong, then crashes at scale).
Differentiating a function whose body is a `let loop` (named-let tail recursion) that
calls `gradient` inside returns garbage instead of the true higher-order derivative:
```scheme
(define a 8.0) (define t 5.0)
(define (post-loop m k)
  (let loop ((w 0.0) (i 0))
    (if (>= i k) (let ((d (- w t))) (* a (* d d)))
        (let ((g (gradient (lambda (x) (let ((d (- x t))) (* a (* d d)))) w)))
          (loop (- w (* m 0.05 g)) (+ i 1))))))
(gradient (lambda (m) (post-loop m 1)) 0.05)   ; => 3.24924e+36  (should be -307.2)
(gradient (lambda (m) (post-loop m 2)) 0.05)   ; => 1.63707e+51  (diverges with k)
```
The k=1 case is mathematically identical to the straight-line `post` above (which
gives the exact -307.2), so the **named-let recursion is corrupting the perturbation
tags** — the inner `gradient`'s tape/jet state appears to leak into the outer loop's
higher-order tracking, compounding each iteration (note the blow-up grows with k).
Straight-line code with the same math is exact. Minimal repro above.

## Issue 2 — STABILITY: iterated loop-nested AD SIGSEGV (interp) / SIGBUS (compiled)
Building the real metric-learning loop on top of Issue 1 (25 meta-steps, each a
`gradient` of a loop-containing objective) terminates with **SIGSEGV** under
`eshkol-run -r` and **SIGBUS** when AOT-compiled (`-O2 -o`). Consistent with Issue 1:
the corrupt jets overflow into invalid memory. Single calls never crash; the crash
is the iterated loop-nested case.

## Issue 3 — PERFORMANCE: fresh closure per call forces a JIT recompile
`(gradient (lambda (x) …) p)` where the lambda is **created inside a loop** is
O(steps · compile): each fresh closure identity triggers a fresh JIT compile of the
gradient body. Hoisting the closure to a single allocation outside the loop removed a
>120 s hang. Suggestion: **memoize JIT compilation on the closure's code/AST, not its
runtime identity**, so repeatedly differentiating structurally-identical closures
compiles once. (Also relevant to ESH-0044 AD throughput.)

## Issue 4 — CODEGEN: `gradient` of a lambda capturing LOCAL scalars fails
**Severity: HIGH** (blocks the natural module structure).
A differentiated lambda that closes over **top-level** scalars compiles fine, but
one that closes over **local/parameter** scalars fails LLVM codegen:
```
PtrToInt source must be pointer   (on the captured locals %a, %t)
```
Repro shape: `(define (step a t w0 m lr) (gradient (lambda (x) (loss a t x)) w0) …)`
called under an outer `gradient` — the inner lambda captures `a`,`t` (params) →
codegen error. The same code with `a`,`t` as top-level `define`s compiles and is
exact. Workaround: put problem params in module-level cells (the `geometric_si`
`gp-set-data!` pattern) so differentiated functions capture globals only. Suggests
the AD lifting treats captured locals as tagged values without unboxing.

## Minor gotcha (not a bug, but costly): stale JIT cache served old code
While iterating on AD source, the JIT cache repeatedly served a previous compile of
an edited file (wrong results until `ESHKOL_JIT_CACHE=0`). A content-hash key on the
source/closure would prevent silently running stale code after an edit.

## Ask
1. Fix Issue 1 (named-let + inner `gradient` higher-order corruption) — this is the
   blocker for any iterative meta-learning / learned-optimizer / metric-flow loop.
   Likely a perturbation-tag/tape-reset scoping bug across the recursive call.
2. Issue 2 should resolve with #1, but a tensor/AD bounds guard that raises instead
   of SIGSEGV/SIGBUS would be safer (cf. ESH-0069).
3. Issue 3: JIT-compile cache keyed on closure code.

## Noesis-side workaround (until #1 lands)
Use straight-line (loop-free, unrolled) inner steps inside any meta-differentiated
function — single-step metric learning (k=1) is then exact and fast. Multi-step
metric flow (k>1) waits on Issue 1. This is what `src/core/geometry/metric_flow.esk`
does: a loop-free single-step path for the validated result, with the multi-step path
gated on this fix.
