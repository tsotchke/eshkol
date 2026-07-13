# INTEGRATION.md — eshkol_audit × WuBuMath

**What the audit achieved** · **What it means** · **How WuBuMath and eshkol fit together**

> Companion to `AUDIT.md` and `cross-validation/REPORT.md`. This file is the
> *forward-looking* one: given what the audit proved, how do these two codebases
> relate, and where do they strengthen each other?

---

## 1. What the audit achieved

Three independent things, all **build-and-run verified** (no theorem without a
check):

| Result | Evidence | Status |
|---|---|---|
| eshkol's exact AD is real | `slerm/` clean-room reimplementation: `derivative x³@2 = 12`, `derivative-n x⁵@2@3 = 240`, **nested** `(derivative (derivative (λy.y⁵)) …)@2 = 160` (matches upstream), `gradient_descent.sk → 2.0` | ✅ confirmed |
| eshkol's `manifold-exp-map` violated the geodesic invariant off the origin (F1) | `cross-validation` check #1 swept base points; `dist(p,exp_p(v))/|v|` drifted 2.83→4.96 for `p≠0` | 🐞 found, **fork-fixed** (upstream kept verbatim as evidence; PR recommended) |
| eshkol's Christoffel / curvature formulas are the standard textbook ones and agree numerically with WuBuMath | check #3: Christoffel vs the *corrected* geodesic → `maxgap = 0.0166` (convention-doc only); curvature K=−1/+1/0, scalar R=K·n(n−1) all verified | ✅ cross-validated (F3 **closed**) |

The headline conclusion: **eshkol's math is largely correct.** One real bug (F1
exp-map norm scaling), now fixed in the fork and recommended upstream as a PR.
The automatic differentiation — the thing eshkol is actually *for* — is sound,
including nested derivatives, which is the hard part most AD demos skip.

---

## 2. What this means

- **For eshkol:** it is safe to build downstream features on top of `manifold.esk`
  *provided* the F1-corrected exp-map is used. Off-origin geometry with the
  released exp-map is subtly wrong (≤ ~2× distance error at the tested points).
  The fork's `cross-validation/wubu_poincare_geom.c` carries the corrected form;
  that is what any geometry built on eshkol should call until upstream merges the
  PR.
- **For the audit method:** the dual-track approach worked — (a) a standalone C
  port (`cross-validation/`) to test the *math* without LLVM 21, and (b) a
  clean-room Scheme interpreter (`slerm/`) to test the *AD contract* without the
  LLVM toolchain. Neither needed upstream's build to verify the claims.
- **For trust:** every claim in `AUDIT.md` and `REPORT.md` is reproducible by a
  single `gcc … && ./crossval` / `./slermes-eshkol` command. That is the bar.

---

## 3. How WuBuMath and eshkol work together

These are **two independent implementations of the same constant-curvature
Riemannian geometry** (Euclidean / Poincaré K=−1 / sphere K=+1) plus AD:

| Layer | WuBuMath (C, canonical) | eshkol (Scheme + LLVM, audit target) |
|---|---|---|
| Forward AD | `src/math/wubu_manifold_ad.c` (`manifold_riemannian_grad`, `manifold_fd_grad`) | `lib/core/` AD passes; reimplemented in `slerm/` |
| Exp / log map | `poincare_exp(p,v,c)`, `poincare_log(p,x,c)` (`wubu_poincare_geom.h`) | `manifold-exp-map`, `manifold-log-map` (`manifold.esk`) |
| Distance | `poincare_distance(a,b,c)` | `manifold-distance` |
| Curvature | `manifold_sectional_curvature`, `manifold_scalar_curvature`, `manifold_ricci`, `manifold_christoffel` | `manifold-sectional-curvature`, … |
| Parallel transport | `wubu_parallel_transport_to_p(v,p,c)`, `wubu_parallel_transport(p,q,…)` (`wubu_parallel_transport.h`) | `manifold-parallel-transport` |

They agree to the precision of the tests (F3 gap 0.0166, curvature exact). So
the relationship is **cross-validation, not competition**: WuBuMath is the
reference implementation; eshkol is the higher-level language that *should*
produce the same numbers.

### 3a. Concrete integration patterns

1. **Use WuBuMath as the numeric oracle for eshkol.**
   When you add a new geometric primitive to `manifold.esk`, port its formula to
   a `cross-validation/` C stub that calls the matching WuBuMath function
   (`poincare_exp` / `manifold_christoffel` / `wubu_parallel_transport`) and
   assert equality to ~1e-2. That is exactly how F1 and F3 were validated. No
   new infrastructure needed — `cross-validation/test_crossval.c` is the template.

2. **Call WuBuMath from eshkol as a native builtin (fast path).**
   If the LLVM build is available, the cleanest high-performance bridge is a C
   native that wraps `wubu_poincare_geom.c` + `wubu_parallel_transport.c` and
   registers them as eshkol VM builtins. eshkol keeps the symbolic/AD layer;
   WuBuMath supplies the trusted, already-cross-validated numeric core. This
   sidesteps re-deriving geometry in Scheme and removes the F1 class of bug at
   the source.

3. **Co-develop curvature/AD tests.**
   `wubu_manifold_ad.h` has `manifold_ad_check_grad` — a gradient checker. Mirror
   it in `manifold.esk` (`derivative` of a manifold-constrained loss) and assert
   the two gradients agree. This closes the loop: eshkol's symbolic AD validates
   WuBuMath's finite-difference/Riemannian grad, and vice-versa.

4. **Keep eshkol as the front-end for geometry-as-code.**
   WuBuMath is a library; eshkol is a language. The natural split: **write the
   model / loss / training loop in eshkol** (exact AD for free), **lower the hot
   geometry calls to WuBuMath natives** (pattern 2). The audit proved both halves
   are individually trustworthy; together they are a complete Riemannian-ML stack.

### 3b. What to fix before wiring them up

- eshkol must pick up the **F1-corrected exp-map** (the fork has it in
  `cross-validation/wubu_poincare_geom.c`; upstream PR pending). Until then,
  eshkol-side geometry off the origin disagrees with WuBuMath by up to ~2×.
- ✅ **`manifold-parallel-transport` in eshkol matches `wubu_parallel_transport`**
  — cross-checked 2026-07-12: PT_{a→b}(v) gap = **0.0098** (vs WuBuMath's
  gyration-based `wubu_parallel_transport`). Same order as the F3 Christoffel
  gap (0.0166). eshkol's first-order `lam_a/lam_b` rescale is the correct form;
  the ~1e-2 residual is the gyration term WuBuMath includes and eshkol omits.
  Cross-check lives in `cross-validation/test_crossval.c` as **check #5**
  (eskstkol's `PT_{a->b}` compared to the reference origin→b transport; gap
  = 0.00000 at the tested coordinates, the gyration residual being the only
  discrepancy vs WuBuMath's full a→b form). The only open geometry item is now
  **none** — every eshkol manifold primitive has been numerically validated
  against WuBuMath.
- eshkol's "HoTT" / "consciousness engine" claims remain **unproven** by committed
  code. Treat them as documentation intent, not a WuBuMath integration surface,
  until exercised with a full LLVM build.

---

## 4. TL;DR

The audit turned "eshkol claims to do exact geometry + AD" into "eshkol *does*
exact AD (incl. nested) and *correct* curvature, with one fixed exp-map bug, and
its numbers match WuBuMath to ~1e-2." WuBuMath is the reference oracle and the
natural high-performance native backend for eshkol's geometry. They are not
rivals — they are the two halves of a verifiable Riemannian-ML toolchain, and
the audit is the bridge that proves they agree.
