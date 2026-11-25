# ADR-0063: Computable Hodge Theory, Diffeomorphic Algebraic Groups, and Exact Integrable Systems

- **Status:** Accepted — implemented in `core.hodge`, `core.alg_group`, and `core.integrable`
- **Date:** 2026-11-25
- **Depends on:** `ADR-0034` (`core.ring`), `ADR-0047` (`core.certificate`), `ADR-0058` (`core.exact_linalg`, `core.exact_psd`)

## 1. Context & Operational Correspondence Mandate

Modern algebraic geometry, Lie theory, and mathematical physics share a unified core connecting:
1. **Hodge theory and algebraic cycles (`core.hodge`):** polarized pure and mixed Hodge structures, Riemann–Hodge bilinear relations, Hard Lefschetz $\mathfrak{sl}_2$-triples, Chow rings, cycle class maps $\mathrm{cl}: A^p(X)_{\mathbb{Q}} \to \mathrm{Hdg}^p(X)$, Mumford–Tate Lie algebras, Siegel period matrices, Griffiths transversality, and Picard–Fuchs differential operators.
2. **Algebraic groups and diffeomorphic group equivalences (`core.alg_group`):** functor-of-points algebraic group schemes over arbitrary `core.ring` carriers, Zariski tangent Lie algebras via dual-number jets, Bruhat and Cartan decompositions, rational Cayley diffeomorphisms, discrete $\pi_1$ covering obstructions, exact rational Hilbert–Mumford GIT stability and Kempf–Ness moment maps, and truncated volume-preserving/symplectic diffeomorphism Lie algebras.
3. **Exact classical, algebraic-geometric, and quantum integrable systems (`core.integrable`):** Lax pairs with explicit isospectral invariant verification, exact Classical and Quantum Yang–Baxter equations over $\mathbb{Q}[i]$, exact-field Hitchin spectral curves $\det(\eta I - L(\lambda)) = 0$ bridging directly to Jacobians and period matrices, exact transfer-matrix commutation $[T(u), T(v)] = 0$, Hirota bilinear $\tau$-functions, and the Heisenberg XXX Algebraic Bethe Ansatz.

## 2. Rigor Rules (Zero Overclaiming)

1. **The Hodge Conjecture is open; the gap is the primary output:** `hodge-cycle-class-certify` computes exact rational bases of both $\mathrm{Hdg}^p(X)$ and $\mathrm{im}(\mathrm{cl})$ and explicitly returns the **unaccounted-Hodge-class gap basis** $\mathrm{Hdg}^p(X) / \mathrm{im}(\mathrm{cl})$. Extraordinary Hodge classes (such as Weil classes on CM abelian fourfolds where $\mathrm{Hdg}^2$ strictly exceeds $\mathrm{Sym}^2 \mathrm{Hdg}^1$) are surfaced as first-class non-empty gap bases rather than errors.
2. **Strict certification vs. hypothesis separation:** Only exact `core.exact_psd` certifications (Riemann–Hodge second bilinear relation, Siegel $\mathrm{Im}\,\tau \succ 0$) and explicit Chow-ring cycle expressions verified over $\mathbb{Q}$ carry `'certified-algebraic` / `core.certificate` status. Numerical period relations without an explicit algebraic cycle witness are labeled `'hypothesis-only`.
3. **Smooth diffeomorphism $\neq$ algebraic group isomorphism:** `core.alg_group.diffeo` separates algebraic isomorphism, real Lie isomorphism, local diffeomorphism, global diffeomorphism, and discrete $\pi_1$ covering obstructions as independent typed fields.
4. **Exact fields for GIT, spectral curves, and Yang–Baxter:** Kempf–Ness moment-map zero sets, Hitchin spectral curves $\det(\eta I - L(\lambda)) = 0$, CYBE, QYBE, and transfer-matrix commutation $[T(u), T(v)] = 0$ are verified over exact rational/Gaussian-rational fields (`Q`, `Q[i]`, `Q[lambda, eta]`), returning exact counterexample witnesses on violation.

## 3. Modules & Executable Verification

- `lib/core/hodge.esk`, `lib/core/hodge/{structure,cycles,periods}.esk`
- `lib/core/alg_group.esk`, `lib/core/alg_group/{reductive,diffeo}.esk`
- `lib/core/integrable.esk`, `lib/core/integrable/{lax,hitchin,quantum}.esk`
- Executable JIT & AOT verification suites:
  - `tests/stdlib/hodge_cycles_periods_test.esk`
  - `tests/stdlib/alg_group_diffeo_git_test.esk`
  - `tests/stdlib/integrable_lax_hitchin_qybe_test.esk`
