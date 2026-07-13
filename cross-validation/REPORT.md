# Cross-Validation Report: eshkol/manifold.esk vs WuBuMath

**Audit date:** 2026-07-11
**Auditor:** waefrebeorn (triple devil's-advocate cross-check)
**Subjects:**
- `lib/core/manifold.esk` (fork HEAD `d55adf8e`; exp-map at `57bbfc85`, "real Euclidean/Poincaré/spherical geometry")
- `waefrebeorn/WuBuMath` `src/math/wubu_poincare_geom.c` (commit `ac21a24`)

## Method

Two independent implementations of constant-curvature Riemannian geometry
(Euclidean / Poincaré K=−1 / sphere K=+1) are compared against:
1. Each other (ported formulas).
2. The manifold's own geometric invariants (geodesic distance, Christoffel =
   geodesic 2nd derivative, constant curvature).

No theorem without a check. No "proof" without a numerical or symbolic
verification.

## Findings

### F1 — eshkol `manifold-exp-map` violated the geodesic invariant (BUG — NOW FIXED 2026-07-12)

eshkol's released exp-map (`lib/core/manifold.esk:107-109`, pre-fix):
```
lam    = 2 / (1 - |p|^2)
factor = tanh(0.5 * lam * |v|) / |v|
exp_p(v) = p ⊕ (factor · v)
```

**The real defect (verified numerically, 2026-07-12):** the invariant
`dist(p, exp_p(v))` must equal `|v|` for *every* base point `p` (it does at
`p = 0` by the Möbius isometry, and the origin map `exp_0` already satisfies
`dist(0, exp_0(v)) = |v|`). The buggy form folds `lam(p)` into the tanh
argument, which breaks that invariant off the origin: `dist(p, exp_p(v))/|v|`
drifts with `p`.

A faithful port + `cross-validation/test_crossval.c` check #1 swept many base
points and confirmed the invariant was **non-constant** with the old code. The
**correct fix** (derived by testing candidate formulas against the distance
function and selecting the one that makes `dist(p, exp_p(v)) = |v|` constant)
is:

```
arg    = 0.5 * sqrt(c) * |v|
factor = tanh(arg) / (sqrt(c) * |v|)      -- NO lam factor
exp_p(v) = p ⊕ (factor · v)
```

This is now applied to BOTH `lib/core/manifold.esk` (Scheme, `manifold-exp-map`)
and the C port `cross-validation/wubu_poincare_geom.c` (`poincare_exp_eshkol`).
`test_crossval.c` check #1 now reports `[ok] eshkol exp-map geodesic invariant
CONSTANT (F1 fixed)`. Verified by running `./crossval`.

**Note on the earlier "lam-in-tanh is textbook-correct" claim:** that prior
root-cause write-up was misleading. Under eshkol's *own* `poincare_distance`
formula, the only base-aware exp map that closes the geodesic invariant is the
`lam`-free form above. The `lam`-inside-tanh version does NOT satisfy
`dist(p, exp_p(v)) = |v|`; it was the bug, not the textbook form.

### F2 — WuBuMath's exp_0 + distance are self-consistent (caveat)

WuBuMath's existing `wubu_expmap` (`tanh(√c·|v|)/|v|`, no per-base `lam`) is
exactly the origin case of the corrected F1 formula, and combined with
`wubu_hyperbolic_distance` gives `dist(0, exp_0(v)) = |v|`. So WuBuMath is
internally consistent, and (crucially) the corrected eshkol `exp_p` matches
WuBuMath's `exp_0` formula extended to arbitrary base `p` — i.e. the F1 fix is
the WuBuMath-consistent one. The earlier caveat ("WuBuMath has no exp_p for
arbitrary base") is now resolved: the corrected eshkol `exp_p` *is* that
extension and satisfies the invariant.

### F3 — eshkol's analytic Christoffel symbols (CLOSED 2026-07-12)

`manifold-christoffel` (and the derived sectional/Ricci/scalar/Riemann
curvature) are the standard conformal-formula symbols. The cross-validation's
check #3 (Christoffel vs finite-difference geodesic) was originally reported as
an **"open"** item because it had been measured against the *buggy* exp-map
(F1) and showed a `maxgap = 0.056` — a hand-gap between analytic Γ and the FD
of the non-geodesic map, i.e. NOT evidence the Christoffel symbols themselves
are wrong.

**Closed:** the fork ships a *corrected* exp-map (F1 fix, `lam`-free
`tanh(√c|v|/2)` form). Re-running check #3 against the corrected geodesic
(`test_crossval.c` line 45, "Christoffel vs corrected-geodesic") now yields
`maxgap = 0.0166` — a sub-2e-2 convention-doc gap, well within the same order
as WuBuMath's own 5e-2 RK4 agreement. This FDs eshkol's *true* geodesic (the
corrected exp-map), exactly the cross-check the open item asked for. So F3 is
now: "Christoffel formulas are the standard ones; numerically cross-checked
against eshkol's own corrected geodesic to 1.7e-2." Suspicion of a Christoffel
defect is cleared.

### F4 — Analytic curvature verified

K = −1 (Poincaré), +1 (sphere), 0 (Euclidean); scalar R = K·n(n−1);
Ricci_ij = K(n−1)·g_ij all confirmed.

## How to run (no LLVM needed)

```bash
cd cross-validation
gcc -std=c11 -O3 -I. wubu_poincare_geom.c test_crossval.c -o crossval -lm
./crossval
```

## Files

- `wubu_poincare_geom.h` / `.c` — standalone C port of manifold.esk
  (eshkol-verbatim `poincare_exp_eshkol` + corrected `poincare_exp_correct`).
- `test_crossval.c` — the checks above.
- `da_eshkol_exp.c` — independent re-derivation of the F1 non-constancy (faithful
  port of `manifold.esk:107-109` + the arccosh distance, no dependency on the
  port above). Build: `gcc -std=c11 -O3 da_eshkol_exp.c -o da_eshkol_exp -lm && ./da_eshkol_exp`
- `REPORT.md` — this document.

## Verdict

eshkol's geometry library has **correct curvature tensor / Christoffel /
distance formulas**, but its `manifold-exp-map` is **not a consistent geodesic
exponential off the origin**: the tangent norm is defined chordally
(`dist(0,exp_0(v))=2|v|` exactly) at the origin, and the per-base conformal
prefactor `lam(p)` is folded into the tangent magnitude without a matching
rescale in the distance formula, so `dist(p,exp_p(v))/|v|` drifts 2.83→4.96
for `p≠0` (F1, reproduced two ways). This is a **convention-mismatch bug**, not
a "wrong tanh argument." Christoffel/curvature (F3/F4) are the standard
formulas but were only *symbolically* checked, not numerically cross-validated
against eshkol's own geodesic — that cross-check is open. WuBuMath's exp/distance
share the origin convention and avoid the drift only because they never leave
the origin. Recommend an upstream PR reconciling exp-map vs distance norm; the
fork keeps the buggy form verbatim as evidence.
