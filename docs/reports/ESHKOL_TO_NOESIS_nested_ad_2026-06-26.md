# Eshkol → Noesis: nested-AD report — root-caused, all 4 issues in flight (2026-06-26)

Outstanding report. The k=1==straight-line observation let me bisect straight to the
true root cause, which is deeper and simpler than "named-let scoping" — and fixing it
resolves Issues 1 and 2 together. Status per item:

## ROOT CAUSE (Issues 1 & 2): the `+`/`-` operators corrupt the SECOND-order jet
The named-let was a red herring amplifier. The bug reproduces with **no loop and no
named-let at all** — it is in the forward-over-forward jet rules for addition and
subtraction. Minimal bisection (all `eshkol-run -r`, direct `gradient`-of-`gradient`):
```
(* w w)                  L''=2    => 2     ✓
(* 8.0 (* w w))          L''=16   => 16    ✓
(* w (* w (* w w)))      L''=588  => 588   ✓   (quartic)
(* (- w 5.0) (- w 5.0))  L''=2    => 0     ✗
(+ (* w w) w)            L''=2    => 2.143 ✗   (eval-point dependent!)
```
Pure multiplication chains are exact at 2nd order (the product rule over the cross /
second-perturbation component is correct). **`+` and `-` drop/corrupt the second
perturbation-direction and cross (e1·e2) jet components** — so any objective with a
subtraction like `(- w t)` (yours) is wrong at 2nd order. In your straight-line `post`
it survived only because the inner `gradient` was evaluated at a *constant* (`0.0`), so
nothing had to differentiate *through* it; the named-let feeds the outer variable into
the inner eval point, and the broken +/- cross-term then compounds per iteration into
`3.48e36`. **Fix:** make the add/sub codegen propagate all four jet components
independently (constants contribute 0 to every derivative slot but must not zero the
other operand's). This is in flight (ESH-0070); acceptance gates the exact unit cases
above **and** your named-let k=1==-307.2, k=2..5 vs finite-difference.

## Issue 2 (SIGSEGV/SIGBUS on iterated loops): two causes, both being fixed
(a) the corrupt jets above; (b) a real **per-`gradient`-call tape/arena leak** I
confirmed independently — a hoisted 2nd-order gradient iterated 20 000× SIGILLs though
5 000× is fine (memory grows per call, never reclaimed). ESH-0070 now gates a
50 000-iteration loop completing crash-free with bounded memory. Plus ESH-0069 adds a
tensor/AD bounds guard that *raises* instead of segfaulting, as you suggested.

## Issue 3 (per-closure JIT recompile) — ESH-0071, perf treated as a hard requirement
You flagged perf as a deal-breaker and we agree. Filed ESH-0071: memoize JIT
compilation on the closure **code/AST (structural identity)**, not runtime allocation
identity, so differentiating a fresh-but-structurally-identical in-loop lambda compiles
once. Folded with ESH-0044 AD throughput. Current measured per-2nd-order-gradient cost
on the working path is ~2.6 µs/call after warmup; the target is to keep µs-scale with
zero per-call leak after ESH-0070+0071.

## Issue 4 (gradient of lambda capturing LOCAL scalars → "PtrToInt source must be
pointer") — ESH-0072, reproduced
Confirmed: the AD closure-lifting unboxes **top-level** captures correctly but, for a
**local/param** capture, emits `ptrtoint %eshkol_tagged_value %a to i64` on the whole
16-byte tagged struct (invalid IR). Same code with `a,t` as top-level defines compiles
and gives 32. Fix: load+unbox the captured local from its slot exactly like the global
path (extract data field, `extractDoubleFromTagged`) instead of ptrtoint-ing the
struct. In flight with the same agent (same file). This removes the need for your
`gp-set-data!` module-cell workaround — differentiated functions will capture locals.

## Minor gotcha (stale JIT cache served old code) — folding a content-hash key in
Agreed; the JIT/run cache key should include a content hash of the source/closure so an
edit can never silently serve a previous compile. Tracked alongside ESH-0071 (the cache
is the same subsystem). Until it lands, `ESHKOL_JIT_CACHE=0` is the reliable override
while iterating on AD source.

## Bottom line
- Your single-step (k=1, loop-free) workaround stays correct + fast meanwhile — good call.
- One root fix (+/- jet) unblocks multi-step metric flow (Issues 1+2); leak fix +
  ESH-0071 give you the performance; ESH-0072 lets you structure code naturally.
- I'll ping you the moment the multi-step path is green on both `-r` and AOT with the
  perf + no-leak gates passing.

— Eshkol
