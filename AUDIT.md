# eshkol Audit — Index

This fork (`waefrebeorn/eshkol_audit`) is a **cross-validation audit** of
`tsotchke/eshkol`. Two independent, runnable work products back every claim:

## 1. `cross-validation/REPORT.md` — the math audit (primary)
A standalone C11 port of eshkol's Riemannian-geometry library
(`lib/core/manifold.esk`) plus numeric cross-checks against `WuBuMath`.
**Empirically run** (`cross-validation/crossval` → "4/5 checks as expected").
Headline findings:
- **F1 (fixed):** `manifold-exp-map` violated the geodesic invariant off the
  origin (`dist(p,exp_p(v))/|v|` drifted 2.83→4.96 for `p≠0`); corrected to the
  `lam`-free `tanh(√c|v|/2)` form, now constant. Verified by `./crossval`.
- **F3/F4 (verified):** analytic Christoffel symbols + curvature
  (K=−1/+1/0, scalar R=K·n(n−1), Ricci) are the standard formulas and pass
  symbolically; the numeric geodesic cross-check is **open** (documented).
- WuBuMath's origin convention is internally consistent; the corrected eshkol
  `exp_p` matches WuBuMath's `exp_0` extended to arbitrary base.

Run it: `cd cross-validation && gcc -std=c11 -O3 -I. wubu_poincare_geom.c test_crossval.c -o crossval -lm && ./crossval`

## 2. `slerm/` — clean-room reimplementation (backing the AD contract)
An independent from-scratch C11 Scheme + forward-mode AD interpreter
(`slermes-eshkol`) that verifies eshkol's **AD headline claims** by running them
and comparing to the original. Verified parity: `derivative x³ 2`=12,
`derivative-n x⁵ 2 3`=240, `taylor eˣ` exact, `sin'`=1 / `cos'`=0, **nested
derivatives = 160**, gradient descent → 2. See `slerm/AUDIT.md` and
`slerm/README.md`.

## What this fork is NOT
- It does **not** rebuild eshkol's LLVM 21 frontend/codegen/VM (those require the
  full toolchain); the AD contract is verified via the independent `slerm/` and
  the geometry via `cross-validation/`.
- It keeps upstream files **verbatim** (the buggy `manifold-exp-map` is retained
  as evidence) and documents defects in `cross-validation/REPORT.md`. Fixes go
  upstream via PR; the fork is the trail.

## TL;DR verdict
eshkol's *mathematics* is largely correct (Christoffel/curvature/standard
formulas), with one real **convention-mismatch bug** in the exp-map (F1, now
fixed in the fork's port and recommended upstream), and an **open** numeric
geodesic cross-check (F3). Its *AD claims* hold (verified by `slerm/`). The
headline risk is documentation/claim over-reach, not wholesale broken math.
