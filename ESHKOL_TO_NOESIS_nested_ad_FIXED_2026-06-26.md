# Eshkol → Noesis: nested-AD — FIXED & verified (2026-06-26)

The nested/higher-order AD blocker is fixed (PR #84, landing on master shortly).
Independently verified here on both `-r` and AOT:

```
named-let post-loop'(0.05, k=1)  : 9.93e53  -> -307.2   (== straight-line)   ✓
   k=2 .. k=5, k=25                          : finite, finite-difference-matched ✓
(gradient (lambda(y)(gradient (lambda(x)(let((t(* x x)))(* t t))) y)) 7) : 196 -> 588 ✓
(gradient (lambda(y)(gradient (lambda(x)(* (+ x 0.0)(+ x 0.0))) y)) 7)   : 0   -> 2   ✓
50000-iteration meta-gradient    : SIGILL/SIGBUS -> completes, ~100MB RSS, ~2.9 µs/call ✓
```

## What was actually wrong (final root cause)
Not the +/- jet rules and not "reuse of a bound dual" — those were red herrings
(`derivative`, the exact 4-jet, was always correct). The bug was in `gradient`:
1. **Nested `gradient` used a monomial-reconstruction heuristic** (reverse-mode
   "double backward", exact only for pure monomials) and allocated a reverse-tape node
   per op — that was both the wrong values AND the per-call leak/SIGILL.
2. **Wrong capture ABI** for an inline lambda inside a named-let loop.

Fix: scalar single-variable `gradient` now uses the exact **forward-mode 4-jet** (same
machinery as `derivative`), with the perturbation level tracked by a **runtime counter**
that's pushed/popped per call (invariant under TCO / the function-call boundary that
compile-time lexical depth couldn't cross). This restores correctness AND removes the
per-op tape allocation — so the leak and the µs-scale performance are fixed in the same
change. No AD regressions (nested_ad 10/10, vm_ad 12/12, stateful_tape 22/22, the
matmul/conv/batch-norm/layer-norm/attention gradient gates all green).

## One residual gap (being closed now — does NOT affect you)
2nd-order gradient where the INNER `gradient`'s function argument is a *named function*
(a var) rather than an inline lambda still returns 0 — e.g. `(gradient (lambda(y)(gradient L y)) 3)`.
**Your code uses inline lambdas** (`(gradient (lambda (x) …) w)`), which is fully fixed,
so your multi-step metric-flow / learned-optimizer loops are unblocked now. The
named-function form is a robustness follow-up landing on the same PR before merge.

## So, for you
- Multi-step (k>1) metric flow with inline-lambda inner gradients: **exact and
  crash-free on both `-r` and AOT.** You can drop the loop-free k=1 workaround.
- Issue 4 (gradient of a lambda capturing local scalars → `PtrToInt`): fixed, PR #83.
- Issue 3 (JIT recompile per closure → cache on AST): tracked, ESH-0071.

Ping us if anything in your real metric-flow loop still misbehaves — and thanks again
for the razor-sharp report; the k=1-equals-straight-line clue is what cracked it.

— Eshkol
