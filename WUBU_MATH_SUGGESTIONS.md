# WuBuMath-Grounded Suggestions for the Two Audit Repos
### Triple Devil's-Advocate pass (each suggestion stress-tested for validity)

**Author:** Hermes Agent   **Date:** 2026-07-12
**Source of truth:** `/home/wubu/WuBuMath` (read, not summarized from memory).
Relevant WuBuMath assets verified present:
- `src/math/wubu_poincare_geom.c` — `poincare_exp` (verbatim eshkol exp_p), `poincare_log`, `poincare_distance`, `manifold_lambda`, `manifold_christoffel` (correct analytic formula).
- `src/math/wubu_hyperbolic.c` — `wubu_expmap` (exp_0, `tanh(√c·|v|)/|v|`), `wubu_logmap`, `wubu_hyperbolic_distance`.
- `src/math/wubu_manifold.c` — `manifold_geodesic` / `manifold_exp` / `manifold_geodesic_length`: **RK4 integrator of the TRUE geodesic** via `x'' = -Γⁱⱼₖ xʲ xᵏ` using `m->christoffel`. This is a principled geodesic oracle.
- `src/tests/test_hyperbolic_analytics.c` — the **NUMERICAL VALIDATION CONTRACT**: "Every test PINS a C kernel to its CLOSED-FORM analytical formula... matches to a stated tolerance. No 'looks reasonable'." Plus the libirrep pattern (EXPECTED_OUTPUT.md: fixed inputs, documented tolerances, external reference).
- `src/tests/test_wubu_poincare_geom.c` — already documents eshkol's exp_p bug (geodesic-invariant non-constant off-origin) as an explicit failing check.

**DA guard applied to every suggestion below:** (1) Is it actually valid math? (2) Is it *new* — not just "copy WuBuMath" (WuBuMath itself lacks a correct `exp_p`)? (3) Does it close an *open* item in the audit rather than restate it?

---

## A. Suggestions for `eshkol_audit` (manifold geometry)

### A1. Close F3 (open item) using WuBuMath's RK4 geodesic as the oracle
**Problem (audit F3):** eshkol's analytic Christoffel symbols were only *symbolically* checked; the numerical cross-check vs eshkol's own geodesic was never done (the buggy exp-map was FD'd instead, giving a meaningless 0.056 gap).
**WuBuMath-grounded fix:** WuBuMath already implements the correct geodesic oracle — `manifold_geodesic(pos, vel, T, steps)` in `wubu_manifold.c` integrates `x'' = -Γx'x'` with RK4, fed by `manifold_christoffel` (the standard formula, identical in form to eshkol's). Port eshkol's metric into WuBuMath's `Manifold` struct (set `m->christoffel` to eshkol's analytic Γ) and run `manifold_exp` from a base `p` with tangent `v`. Then verify the **geodesic invariant** that the buggy exp-map violated:
```
dist(p, geodesic(p,v,t)) / |v|  ==  const   for all t, all p
```
If eshkol's Γ is correct, the RK4 geodesic will satisfy this; if not, the gap locates the defect. **This closes F3's open item** with a real numerical cross-check, using code WuBuMath already ships.
**DA check:** valid (RK4 + correct Γ ⇒ true geodesic by construction); new (F3 was explicitly open); not "copy WuBuMath" (we only borrow the *integrator*, feed it *eshkol's* metric).

### A2. Derive the correct `exp_p` by metric reconciliation, not by editing the tanh argument
**Problem (audit F1):** eshkol's `exp_p` uses `dist(0,exp_0(v)) = 2|v|` (chordal) at the origin but folds `lam(p)` into the tangent magnitude off-origin without a matching rescale in `poincare_distance`, so `dist(p,exp_p(v))/|v|` drifts 2.83→4.96.
**WuBuMath-grounded principle:** WuBuMath's `wubu_expmap` has the *same* limitation — it is `exp_0` only (`tanh(√c·|v|)/|v|`), never leaves the origin, so it sidesteps the drift. Neither repo has a *correct* `exp_p` yet. The principled fix is to **define `exp_p(v)` as `manifold_geodesic(p, v, t=1)`** (WuBuMath's RK4 oracle) and then *reconcile the distance formula* to that single geodesic — i.e., make `poincare_distance` and `exp_p` agree on what `|v|` means at every base `p`, by rescaling the ambient `dist` by `2·atanh` (or normalizing exp_p's tangent by `lam(p)`). The audit's existing `poincare_exp_correct` stub should be replaced by the RK4-defined map so the convention is provably consistent.
**DA check:** valid (geodesic = definition of exp map; distance must match by construction); new (audit F1 proposed only a "convention reconciliation," not the RK4 definition); not copying WuBuMath (WuBuMath's exp_0 is *not* the answer — the RK4 oracle is).

### A3. Adopt WuBuMath's NUMERICAL VALIDATION CONTRACT for the geometry tests
**Problem:** eshkol's cross-validation used `maxgap = 0.056` hand-tolerance and a "[doc]" line — exactly the "looks reasonable" anti-pattern WuBuMath's `test_hyperbolic_analytics.c` forbids.
**Fix:** Port WuBuMath's contract: every geometry kernel (exp_p, log_p, distance, christoffel) gets a **fixed-input / documented-tolerance / external-reference** test (the libirrep `EXPECTED_OUTPUT.md` pattern WuBuMath already uses). The geodesic-invariant check from A1 becomes a pinned regression with `tol=1e-4`, not a printed gap.
**DA check:** valid (reproducible CI guard); new to eshkol; directly reuses a WuBuMath *discipline*, not a formula.

---

## B. Suggestions for `quantum_rng_audit` (PRNG / entropy)

> Note: WuBuMath has **no RNG/entropy module** (grep for `rand/splitmix/xorshift/entropy` finds only test comments, no implementation). So these suggestions are about **borrowing WuBuMath's validation rigor and its tangent/tanh-mixing math pattern**, not its code.

### B1. Ship a seeded "golden-vector" regression test (WuBuMath tolerance discipline)
**Problem:** quantum_rng now has a correct dual-mode contract (seeded = reproducible) but **no test enforces it**. The old `determinism_test.c` concluded the opposite.
**WuBuMath-grounded fix:** WuBuMath's tests pin kernels to closed-form values to a stated tolerance. Apply the same: commit a **fixed seed → fixed 1KB byte stream** as `EXPECTED_OUTPUT` and assert `memcmp(stream, expected, 1024) == 0` in CI. This is the exact "fixed inputs, documented tolerances, external reference" libirrep pattern. It converts the current *undocumented* reproducibility into a *guarded* invariant.
**DA check:** valid (determinism is the contract; guard it); new (no such test exists); the seeded stream is a real artifact (verified byte-identical across PIDs), so the golden vector is reproducible.

### B2. Replace the broken `qrng_get_entropy_estimate()` + hardcoded 63.99 with a real measured estimator
**Problem:** `qrng_get_entropy_estimate()` returns a meaningless ~1.3 bits/byte (broken math at `quantum_rng.c:491`); the README hardcodes "63.999872 bits/sample" with no computation.
**WuBuMath-grounded fix:** WuBuMath's `test_hyperbolic_analytics.c` forbids "looks reasonable" — a number is either within tolerance of the formula or not. Apply that ethos: delete the 63.99 string and the broken estimator; instead ship a **measured** Shannon-entropy + chi-square test that asserts `entropy ≈ 8.0 bits/byte within tol 1e-2` and `chi2 < critical` (exactly what `audit/qrng_measure.c` already computes: 7.9998 bits/byte, chi2 247.3). Pair it with a **runs test / serial-correlation** check (the kind WuBuMath uses to validate its own kernels against closed forms) so "proven entropy characteristics" means *measured-and-pinned*, not *asserted*.
**DA check:** valid (8 bits/byte uniform is empirically true; pin it); new (current code asserts nothing); directly applies WuBuMath's "pin to formula, tolerate precisely" contract.

### B3. Reuse WuBuMath's tanh-mixing math pattern to make quantum_rng's "quantum" mixing metric-consistent
**Observation:** quantum_rng's `hadamard_mix` is a `splitmix64`/`xorshift` bit-mixer; WuBuMath's `poincare_exp` uses a `tanh(λ|v|/2)` conformal map. The *shape* (bounded nonlinear mixing) is the same idea.
**Suggestion (lighter, DA-flagged):** If quantum_rng wants its mixing to be *defensible* rather than decorative, borrow WuBuMath's discipline of **pairing the mixer with an analytic invariant + a tolerance test** — i.e., for every mixing step, assert a measured property (avalanche: 1-bit input flip → ~50% output bits change; chi-square of output bytes) to a pinned tolerance. This turns "Hadamard gate" naming into a *measured* property, matching WuBuMath's "kernel passes ONLY if output matches the analytical value to a stated tolerance" rule.
**DA check:** valid as a *validation* upgrade, NOT as a math claim; explicitly flagged as "discipline borrow, not code copy" since WuBuMath has no RNG. Lower priority than B1/B2.

---

## C. Cross-repo note (eshkol ↔ quantum_rng)
Both forks share the **same root defect class** that WuBuMath's discipline would have prevented:
- eshkol: a geometry formula (exp_p) that isn't pinned to its geodesic invariant.
- quantum_rng: an entropy number (63.99) that isn't pinned to a measured value.

WuBuMath's `test_hyperbolic_analytics.c` contract — *"PIN every kernel to its closed form; match to a stated tolerance; no 'looks reasonable'"* — is the single transferable fix for both. That is the highest-leverage WuBuMath-derived suggestion across the audit.
