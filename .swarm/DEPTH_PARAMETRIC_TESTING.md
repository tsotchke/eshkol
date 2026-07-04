# Depth-parametric adversarial testing (Pillar P6) — 2026-07-03

Meta-lesson from ESH-0117: our five harnesses tested FIXED, SHALLOW shapes
(feature pairs; AD nesting depth <=2; a few recursion depths). Depth-dependent
bugs — a composition correct at depth 1 but broken at depth 2+ — slipped through
(reverse-over-single-forward worked; reverse-over-nested-forward returned 0).

Principle: for EVERY composable construct, do not test one or two depths — generate
the composition PARAMETRICALLY at depth d = 1,2,3,...,N, verify each against a
ground-truth oracle that scales with depth, and record the maximum depth at which
each construct remains correct (and whether it FAILS = wrong value, or DEGRADES =
clean error/limit). A silent wrong answer at any depth is a bug; a clean depth
limit is a documented capability boundary.

## Ground-truth oracles (no hand computation)
- AD: central finite differences (nth-derivative stencils) + analytic for polynomials
  (where high derivatives are exactly known / zero).
- General execution: differential agreement across JIT/AOT/O0/O2/VM axes.
- Recursion/data: closed-form expected (sum/length/depth counters).

## The grid — "arbitrary depth, for everything"
P6a AD depth: derivative^n, gradient∘derivative^n (reverse-over-nested-forward),
   gradient^n, jacobian/hessian nesting, mixed forward/reverse, scalar+vector+tensor
   points, all binding/capture forms — each swept d=1..8, vs finite-diff/analytic.
P6b Recursion/control depth: per KIND — self-tail, mutual-tail (2- and 3-cycle),
   non-tail, CPS/continuation chains, recursion-through-map, recursion-through-eval
   (metacircular), dynamic-wind nesting, call/cc nesting — swept to find + gate the
   max safe depth of each (guard against silent SIGILL/SIGBUS; require clean limit).
P6c Syntax/data nesting depth: nested quote/quasiquote/unquote, nested let-family,
   nested lambda/closure capture chains, nested vectors/lists, nested guard/raise,
   nested define scopes, string/number extremes — swept d=1..N, differential-verified.

## Operating rule
Every harness generator gains a --max-depth parameter and emits per-depth PASS/FAIL/
LIMIT with ICC trace events. Findings -> ESH tasks with the exact breaking depth.
The depth sweeps run in the nightly adversarial lane. A construct ships as "correct
to depth D_max" — documented, not assumed-arbitrary.
