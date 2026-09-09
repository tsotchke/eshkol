# Mechanizing the finite-time Navier-Stokes blowup construction in Eshkol

**Status:** design document. The capability ledger in Section 3 is the authority
on what exists today; every gap below is a BUILD ITEM with a version and a gate,
never a hedge.
**Author:** tsotchke
**Scope:** one published paper, and what it would take to obtain its
construction inside Eshkol at every single step.
**Companion:** the `examples/mathematics_navier_stokes_*.esk` family and
[docs/AI_MATHEMATICS_EXAMPLES.md](../AI_MATHEMATICS_EXAMPLES.md).

---

## 1. Purpose and the theorem statement

This document is a step-by-step map from a specific published proof to Eshkol
primitives. For every step of the construction it names the mathematical
operation, the Eshkol primitive that performs it today (verified against the
tree), or the BUILD ITEM that will, and the gate that certifies the step. It is
a blueprint for a capability, not a claim about a result.

The source is *Finite Time Blowup for Navier-Stokes*, OpenAI, 2026
(<https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf>).
All theorem, proposition, lemma and equation numbers below refer to that paper.
Its main result, quoted:

> **Theorem 1.1.** For every `ν > 0` there exist a force `f ∈ Cc∞(R³ × (0, ∞); R³)`,
> a compact set `K ⊂ R³`, and smooth velocity and pressure fields `u, p` on
> `R³ × [0, 1)` satisfying
>
> ```text
> ∂t u + (u · ∇)u − ν Δu + ∇p = f ,    ∇ · u = 0 ,    u(·, 0) = 0        (1.1)
> ```
>
> such that `supp u(·, t) ∪ supp p(·, t) ⊂ K` for every `0 ≤ t < 1`,
>
> ```text
> sup_{0 ≤ t < 1} ‖u(t)‖_{L²(R³)} < ∞ ,      limsup_{t ↑ 1} ‖u(t)‖_{L∞(R³)} = ∞ .
> ```
>
> Consequently, there is no smooth solution `(u, P)` on `R³ × [0, ∞)` with the
> same force and initial datum whose kinetic energy is uniformly bounded
> `sup_{t ≥ 0} ½ ∫_{R³} |u(x, t)|² dx < ∞`.

The paper's Corollary 10.6 transports the same construction to the torus
`T³ = R³/Z³`.

The mechanization target is not "reproduce the estimates." It is the
construction: the paper's proof is a *pipeline of explicit algebraic
constructions* — similarity ansatz, coefficient recursions, finite moment
systems, a two-family stress solve, a residual-exponent ladder — punctuated by
analytic estimates. The algebraic spine is exactly what a compiler with exact
arithmetic, exact-coefficient Taylor towers and automatic differentiation is
built to carry. Section 2 tables that spine step by step. Section 3 says which
parts of it Eshkol runs today. Section 4 states what "mechanized" would mean
operationally, and how to run what exists now.

---

## 2. The proof as a pipeline

The paper's own structure is followed: the Section 3 outline first, then
Sections 4-10 and Appendices A-C in the order the construction consumes them.
Every row names a concrete operation. `Version` is the release stage from
[ROADMAP.md](../../ROADMAP.md) at which the row is expected to be executable
end-to-end; rows marked SHIPPED are executable now. `Gate` names either an
existing ICC completion-oracle criterion in
[.icc/completion-oracles.yaml](../../.icc/completion-oracles.yaml) or a criterion
proposed in Section 4.2 under the new `navier-stokes-mechanization` oracle.

### 2.1 Setup, similarity coordinates and the leading field (Section 3.1, Section 4.1)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 1 | Section 3.1; Section 3.6 | Fix the parameters `τ = 1 − t`, `A = ½ + h`, `D = ½ − h` and check the standing constraints `0 < h < 1/100` and (for finite energy, Section 3.5) `h < 1/6` | Exact rational literals and comparison over the numeric tower (`lib/core/rational.cpp`, `lib/core/bignum.cpp`; `docs/breakdown/EXACT_ARITHMETIC.md` Sections 3 and 5) | SHIPPED | v1.3.0-evolve | `ns_similarity_exponents_solved` |
| 2 | (3.2); Lemma 4.1 | Given `τ = q(1 − η²)`, `z = q^D η`, `X = r²/(2q)`, eliminate `η` and verify `∂q(q − z² q^{2h}) = 1 − 2hη² ≥ 1 − 2h > 0` on `\|η\| < 1`, so `q = q(z, τ)` is single-valued | `derivative` on the closed form, plus exact rational comparison of `1 − 2h` | SHIPPED | v1.3.0-evolve | `ns_similarity_exponents_solved` |
| 3 | Section 3.1; (4.3) | Substitute the ansatz `u_θ⁽⁰⁾ = q^{−A}E`, `u_z⁽⁰⁾ = q^{−A}U`, `r u_r⁽⁰⁾ = V₀`, `p⁽⁰⁾ = q^{−2A}Π` into the residual `R(u,p) = ∂t u + (u·∇)u − Δu + ∇p` and collect powers of `q` | BUILD ITEM: **polynomial-valued duals** — a symbolic ansatz differentiated by the existing AD operators, with the residual returned as a value in a graded ring rather than a number | PLANNED | v1.4.0-connection | `ns_leading_profile_balance` |
| 4 | (3.1); (4.12) | Evaluate the residual operator itself at a numeric point: `∂t u + (u·∇)u − νΔu + ∇p` | `derivative`, `gradient`, `jacobian`, `divergence`, `curl`, `laplacian` (compiler builtins; `docs/guide/AUTOMATIC_DIFFERENTIATION.md` Section 1) | SHIPPED | v1.3.0-evolve | `ns_viscosity_scaling_exact` |
| 5 | (4.7); (5.2) | Solve `∂X V₀ = −Z_{−A} U` with `V₀(0, η) = 0` — incompressibility plus axis regularity fixes the radial profile from the axial one | `taylor` coefficient recursion for the series solution of a first-order linear ODE in `X`; `taylor-ode-solve` (`lib/core/ad/taylor_numerics.esk`) for the numeric check | SHIPPED | v1.3.0-evolve | `ns_leading_profile_balance` |
| 6 | Section 3.1; (4.31) | Impose the leading radial balance `∂r p⁽⁰⁾ = (u_θ⁽⁰⁾)²/r` and normalize at radial infinity: `Π(X, η) = −∫_X^∞ E(x, η)²/(2x) dx` | `(integrate f a b n)` (Simpson, `lib/math.esk`) for the numeric value; exact tail handling requires the BUILD ITEM in row 22 | SHIPPED (finite range) | v1.3.0-evolve | `ns_leading_profile_balance` |
| 7 | Section 3.1 | Derive the core extents `ℓr ≍ τ^{1/2}`, `ℓz ≍ τ^{1/2−h}`, `ℓz/ℓr ≍ τ^{−h}` and the growth `‖u_θ⁽⁰⁾‖_{L∞(Cτ)} ≍ τ^{−1/2−h}` as an exactly solved exponent system | Explicit exact rational elimination written in Eshkol over the scalar exact tower (Scheme vectors carry exact rationals; the shipped `lib/math.esk` `solve`/`det`/`inv` are written for double data and seed inexact constants, and Eshkol tensors are f64-backed) | SHIPPED | v1.3.0-evolve | `ns_similarity_exponents_solved` |
| 8 | Section 3.5 | Derive `E_core ≍ τ^{3/2−h} τ^{−1−2h} = τ^{1/2−3h}` and `D_core ≍ τ^{−1/2−3h}`, then verify `∫₀^{τ₀} τ^{−1/2−3h} dτ = τ₀^{1/2−3h}/(½ − 3h) < ∞` exactly for `h < 1/6` | Exact rational exponent arithmetic; the inequality is decided exactly on rationals, so no enclosure is needed. `interval-lo`/`interval-hi` (`lib/core/ad/interval.esk`) bound the constants | SHIPPED | v1.3.0-evolve | `ns_core_energy_exponents` |
| 9 | (10.22)-(10.23) | Verify the viscosity-scaling identity: `uν(x,t) = √ν u(x/√ν, t)`, `pν = ν p(x/√ν, t)`, `fν = √ν f(x/√ν, t)` reproduces (1.1) at viscosity `ν`, and `‖uν(t)‖₂² = ν^{5/2}‖u(t)‖₂²` | The AD residual operator applied to the rescaled fields; the chain-rule factors are produced by `gradient`/`laplacian`, and the identity is checked term by term | SHIPPED | v1.3.0-evolve | `ns_viscosity_scaling_exact` |

### 2.2 The background stress and the cumulative radial integrals (Sections 3.2, 4.1-4.2)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 10 | Proposition 4.2; (4.12) | Show the tangential residuals `R_θ⁽⁰⁾`, `R_z⁽⁰⁾` are minus the cylindrical divergences `(∂r + 2/r)` and `(∂r + 1/r)` of a stress pair | BUILD ITEM (row 3): the residual as a symbolic value, then a divergence-form factorization checked coefficient by coefficient | PLANNED | v1.4.0-connection | `ns_leading_profile_balance` |
| 11 | Section 3.2 | Integrate radially with the axis-regularity constant: `T_{rθ}(r) = −r^{−2}∫₀^r s² R_θ⁽⁰⁾(s) ds`, `T_{rz}(r) = −r^{−1}∫₀^r s R_z⁽⁰⁾(s) ds` | `(integrate f a b n)` for values; `taylor` for the series of the integrand near `X = 0`, integrated coefficientwise | SHIPPED | v1.3.0-evolve | `ns_cumulative_moment_identity` |
| 12 | Section 3.2; Lemma A.8 | Impose the total-moment identities `∫₀^∞ r² R_θ⁽⁰⁾ dr = 0` and `∫₀^∞ r R_z⁽⁰⁾ dr = 0`, which force the stress to vanish beyond the exterior radius as well as near the axis | Cumulative-integral evaluation plus an exact zero test on the resulting rational; a zero over an unbounded range needs the rigorous tail bound of row 19 | SHIPPED (compact range) | v1.3.0-evolve | `ns_cumulative_moment_identity` |
| 13 | (4.15) | Form the five cumulative radial integrals `M = ∫₀^X U dx`, `I = ∫₀^X H dx`, `J = ∫₀^X UH dx`, `S = ∫₀^X (U² − E²/2) dx`, `C_p = ∫₀^X E²/(2x) dx`, with `Π = Π(0,η) + C_p` | `(integrate f a b n)`; near the axis the same five as exact Taylor coefficients of the integrands under `taylor` seeded at an exact point | SHIPPED | v1.3.0-evolve | `ns_cumulative_moment_identity` |
| 14 | Lemma 4.3; (4.16) | Reduce `Q_s`, `N_s` to closed expressions in `(M, I, J, S, Π)` and their `η`-derivatives — integration by parts in `Π_X = E²/(2X)` | `derivative` in `η`, plus the BUILD ITEM of row 3 for the by-parts identity as a symbolic rewrite | PLANNED | v1.4.0-connection | `ns_cumulative_moment_identity` |
| 15 | Lemma 4.4(i) | Certify the joining rule: if `(U₁,E₁) = (U₂,E₂)` for `X ≥ X_h` and `Δm(X_h, η) = 0`, then all of `Π, V₀, Q_s, N_s, p_s, a, b_s, T₀` agree for `X ≥ X_h` | Exact equality of five rational-valued functions at the joining radius; `exact?` predicate plus rational comparison | SHIPPED | v1.3.0-evolve | `ns_cumulative_moment_identity` |

### 2.3 The admissible stress cone (Section 4.3, Lemma 4.5)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 16 | (4.20) | From the radial shear `(a, −b_s)` and the integrated inviscid contribution `p_s`, form `t_s = −b_s/a`, `v_s = a(1 + t_s²)`, `P_c = p_{s,1} + t_s p_{s,2}`, `J_c = p_{s,2} − t_s p_{s,1}` | Exact rational arithmetic; the shear components come from `derivative` applied to the profile | SHIPPED | v1.3.0-evolve | `ns_cone_condition_equivalence` |
| 17 | (4.21) | Evaluate `U(P_c, J_c) = P_c + J_c²/4 − \|J_c\|·√((P_c − 2)/2 + J_c²/16)` and test the relaxed cone condition `P_c > 2`, `v_s < U(P_c, J_c)` | Exact rational arithmetic; the square root is enclosed by `taylor-model` (`lib/core/ad/taylor_models.esk`) and compared through `interval-contains?`. Note the enclosure is validated, not yet rigorous — see Section 3.5 | SHIPPED | v1.3.0-evolve | `ns_cone_condition_equivalence` |
| 18 | Lemma 4.5; (4.22) | Certify the equivalence: for `v_s > 2`, the admissible cone condition holds iff `P_c > v_s` and `(v_s − 2)J_c² < 2(P_c − v_s)²` — a root computation for the quadratic `2(P_c − v)² − (v − 2)J_c²` in `v` | Exact quadratic-root algebra over the scalar exact tower; `taylor-root` (`lib/core/ad/taylor_numerics.esk`) as the numeric cross-check | SHIPPED | v1.3.0-evolve | `ns_cone_condition_equivalence` |
| 19 | Lemma 4.5, second assertion | Produce the uniform threshold `P_K = max{(B_K + 2)/c_K, 8B_K/γ_K}` on a compact parameter set `K`, from the compactness bounds `c ≥ c_K`, `G ≥ γ_K`, `v ≤ B_K`, `cv ≤ B_K` | BUILD ITEM: **rigorous compact-set bounds** — directed-rounding interval arithmetic plus a proved (not sampled) Taylor-model remainder, with adaptive subdivision over the parameter box | PLANNED | v1.5.0-intelligence | `ns_cone_condition_equivalence` |
| 20 | (4.23) | Rewrite the cone in stress coordinates: `T_{0,θ} + t_s T_{0,z} > 0` and `(v_s − 2)(T_{0,z} − t_s T_{0,θ})² < 2(T_{0,θ} + t_s T_{0,z})²`, homogeneous in `T₀`, so a strict margin survives to the annular edges | Exact rational inequality evaluation on the normalized stress direction | SHIPPED | v1.3.0-evolve | `ns_cone_condition_equivalence` |
| 21 | Theorem 4.6 | Assemble the leading-profile theorem: existence of `h`, `λ`, `C`, `0 < X_a < X_b` with the regular axis, the heat exterior, the moment identities and the flat-at-the-edge stress with a cone margin | The composite of rows 10-20 and 22-33; no single primitive | PLANNED | v1.5.0-intelligence | `ns_leading_profile_balance` |

### 2.4 Heat exterior and moment matching (Appendix A)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 22 | Lemma A.1 | Certify that the moment matrix `B_ij = ∫₀^∞ x^{α_i} β_j(x) dx` is invertible for distinct real exponents `α_i` and ordered bump supports `I₁ < ⋯ < I_m` — a sign argument on `det[x_j^{α_i}]` plus multilinearity | Exact rational determinant written in Eshkol over the scalar exact tower (`lib/math.esk`'s `det` is double-seeded, so it is a cross-check, not the exact witness); the generic sign argument is the BUILD ITEM of row 3 | SHIPPED (instances) / PLANNED (generic) | v1.3.0-evolve / v1.4.0-connection | `ns_moment_matrix_invertible` |
| 23 | (A.1) | Quantify the loss when exponents collide: for translated bumps, `det B = b(s₁)b(s₂)(e^{s₂Δ} − e^{s₁Δ})`, giving `‖B^{−1}‖ = O(λ^{−1})` at separation `λ` | Exact rational algebra plus an `exp` enclosure via `taylor-model` (validated, not yet rigorous — Section 3.5) | SHIPPED | v1.3.0-evolve | `ns_moment_matrix_invertible` |
| 24 | Lemma A.2; (A.2)-(A.3) | Solve the quadratic moment system `F_η(c) = B(η)c + Q_η(c,c) = d(η)` by the contraction `c ↦ B^{−1}(d − Q(c,c))`, under `8β₀²κ₀d₀ ≤ 1`, and bound `‖c‖_{C^k} ≤ 2β_k‖d‖_{C^k}` | Exact rational elimination for the linear part, written in Eshkol; the fixed-point iteration in plain Eshkol; the contraction hypothesis `8β₀²κ₀d₀ ≤ 1` decided exactly once the three norms are bounded | SHIPPED | v1.3.0-evolve | `ns_moment_quadratic_solve` |
| 25 | Corollary A.3; (A.4) | Normalize the five moments by `(ρ, ρ^{3/2}, ρ^{3/2}, ρ, 1)`, split into a `U` block with powers `(1, x^{α+1/2})` and an `E` block with powers `(x^{1/2}, x^{α}, x^{α−1})`, and check `α ∉ {−1/2, 1/2, 3/2}` so both lists have distinct powers | Exact rational exponent arithmetic plus set-membership; then row 22 on each block | SHIPPED | v1.3.0-evolve | `ns_moment_matrix_invertible` |
| 26 | Proposition A.4 | Construct the reference outer profile with a purely azimuthal tail and read `Π(0, η)` off its radial pressure integral (4.31) | `integrate` over the profile; parameter selection in ordinary Eshkol code | SHIPPED | v1.3.0-evolve | `ns_heat_exterior_recurrence` |
| 27 | Lemma A.6; (A.33)-(A.38) | Build the heat exterior: `K(r,t) = c_∞ s^{−A} H(2τ/s)`, `s = r²/2`; verify `Z²H'' + (1 + 2a_K Z)H' + a_K(a_K − 1)H = 0`, the derivative values `H^{(m)}(0) = (−1)^m (h)_m (1 + h)_m`, and the expansion `H(Z) = 1 − h(1 + h)Z + O_h(Z²)` | `taylor` seeded at an exact point: the rising-factorial coefficients `(h)_m(1+h)_m` are exact rationals in `h`, and substituting the truncated series into the ODE must annihilate every coefficient | SHIPPED | v1.3.0-evolve | `ns_heat_exterior_recurrence` |
| 28 | Lemma A.6, monotonicity | Certify `K_r < 0` and `0 ≤ −ZH'/H < h`, i.e. `rK_r/(2K) = −A − ZH'/H < −½` | Enclosure of the ratio `ZH'/H` over the domain via `taylor-model` and `interval-contains?` | SHIPPED | v1.3.0-evolve | `ns_heat_exterior_recurrence` |
| 29 | Proposition A.7 | After replacing the power tail by the exact heat flow, restore the pressure integral at smaller radii while preserving the axis pressure | The moment-correction solve of row 24 applied to the pressure row | SHIPPED | v1.3.0-evolve | `ns_moment_quadratic_solve` |
| 30 | Lemma A.8; Proposition A.10 | Under the exact moment conditions `M(∞) = J(∞) = S(∞) = 0`, `∫₀^∞ (H − H_pow) dX = 0` and the normalized `Π`, conclude the leading tangential stresses vanish for `X ≥ X_b`, with the backward stress formula (A.46) valid (no boundary term, by the tail bounds (A.44)) | Exact zero tests on the four moments; the convergence of the subtracted angular moment needs the rigorous tail-bound BUILD ITEM of row 19 | SHIPPED (moments) / PLANNED (tail) | v1.3.0-evolve / v1.5.0-intelligence | `ns_cumulative_moment_identity` |

### 2.5 Analytic profiles near the axis (Appendix B)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 31 | Lemma B.1 | Establish that multiplication is bounded on the analytic space `B_ρ`, and that the radial inverses `J_ν` solving `Y G_YY + ν G_Y = F` are bounded there | BUILD ITEM: **series values with a norm** — a first-class truncated-series type carrying a coefficient norm, so the algebra of `B_ρ` is expressible; the underlying arithmetic is the shipped tower | PLANNED | v1.4.0-connection | `ns_axis_series_contraction` |
| 32 | Proposition B.2; (B.12)-(B.15) | Solve the axis system `2(YΦ_YY + 2Φ_Y) = −χΦ + Λ^{−1}R₁`, `2(Yu_YY + u_Y) = −Z_*/L + Λ^{−1}R₂` by contraction about the explicit comparison profile, obtaining `φ`, `U`, `Π` analytic on `0 ≤ Y = ΛX ≤ 4.1` | `taylor` recursion for the coefficients of `Φ` and `u`, with the `Λ^{−1}` remainders driving a fixed-point iteration on series values; exact coefficients where the sources are rational | SHIPPED (coefficient recursion) / PLANNED (contraction on series values) | v1.3.0-evolve / v1.4.0-connection | `ns_axis_series_contraction` |
| 33 | Proposition B.3; Lemma B.4; Proposition B.5 | Continue the analytic axis profile to `X_i` with `E > 0`, keeping the shear inside the relaxed cone and bounding the parameter response | The continuation is repeated local expansion — the shipped `taylor`/`taylor-ode-solve` continuation loop — with the cone check of rows 16-18 at every step | SHIPPED | v1.3.0-evolve | `ns_axis_series_contraction` |
| 34 | Lemma B.7; Proposition B.8; Corollary B.10 | Match the five cumulative radial integrals of the inner profile to the reference outer profile at the joining radius, using the uniformly invertible Jacobian of Corollary A.3, then join | Row 24's quadratic solve with the five-moment Jacobian of row 25, then row 15's exact equality test at the joining radius | SHIPPED | v1.3.0-evolve | `ns_moment_quadratic_solve` |

### 2.6 Realizing the admissible stress cone (Appendix C)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 35 | Lemma C.1; (C.1)-(C.5) | Construct a period-one loop `(a_L, −b_L)(X, η, φ)` with prescribed mean `∫₀¹ (a_L, −b_L) dφ = (a, −b_s)`, cone gaps `Ψ_j ≥ κ_L > 0`, and equality with the original shear near the radial boundaries; the family uses `M_e(z) = ⟨e^{z sin θ'}⟩_{θ'}` and `t(θ'; µ) = t_s + d₀(e^{µ p_{s,2} sin θ'}/M_e(µ p_{s,2}) − 1)/p_{s,2}` | BUILD ITEM: **periodic (torus) averaging as a builtin** — `⟨·⟩` over `T¹`/`T²` with exact treatment of trigonometric moments; today the modified-Bessel moments `M_e` are reachable as `taylor` coefficients but the averaging operator is not a primitive | PLANNED | v1.5.0-intelligence | `ns_shear_loop_mean` |
| 36 | Lemma C.1, removable singularity | Certify that `G(p)/p = ∫₀¹ G'(up) du` removes the apparent singularity at `p_{s,2} = 0`, giving joint smoothness and the value `t = t_s + d₀ µ sin θ'` | Exact series division: `taylor` seeded at an exact point divides out the common factor with zero floating error | SHIPPED | v1.3.0-evolve | `ns_shear_loop_mean` |
| 37 | Proposition C.2; (C.11)-(C.13) | Take the zero-mean periodic antiderivatives `∂φ A = −½(a_L − a)`, `∂φ B = ½E(b_L − b_s)`, set `E_N = E exp(A/N)`, `U_N = U + B/N` at phase `N log X`, and verify the exact shears `a_N = a_L − 2D_X A/N`, `b_N = e^{−A/N}(b_L + 2D_X B/(NE))` using `X∂X = D_X + N∂φ` | The `X∂X = D_X + N∂φ` split is a chain-rule identity the AD operators produce directly; the antiderivative needs the torus BUILD ITEM of row 35 | PLANNED | v1.5.0-intelligence | `ns_high_frequency_shear_moments` |
| 38 | Proposition C.2; (C.14) | Bound the profile and moment perturbation by `O(N^{−1})` in every fixed number of `η`-derivatives, while allowing `X∂X(E_N − E)` to be order one — the whole point of the modulation | `O(N^{−1})` bounds via `taylor-model` enclosures on the fixed compact radial range | PLANNED | v1.5.0-intelligence | `ns_high_frequency_shear_moments` |
| 39 | Proposition C.2, restoration; Proposition C.3 | Restore all five radial moments exactly on the reserved correction patch, so the exterior fields and the inner solution are unchanged, and conclude the admissible cone throughout `(X_a, X_b)` | Row 24's quadratic moment solve on the reserved patch, then rows 16-20 for the cone check | SHIPPED (solve) / PLANNED (composition) | v1.3.0-evolve / v1.5.0-intelligence | `ns_moment_quadratic_solve` |

### 2.7 Correction of the base flow to every order (Section 5)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 40 | (5.1) | Set `λ_n = 2nh` and write the physical coefficients `u_{θ,n} = q^{−A+λ_n}E_n`, `u_{z,n} = q^{−A+λ_n}U_n`, `r u_{r,n} = q^{λ_n}V_n`, `p_n = q^{−2A+λ_n}Π_n`, with `E_n = √(2X) φ_n / C` | The `2h` spacing is an exact rational exponent arithmetic fact (`1 − 2D = 2A − 1 = 2h`); the graded expansion itself is the BUILD ITEM of row 3 | PLANNED | v1.4.0-connection | `ns_order_n_coefficient_solve` |
| 41 | (5.2)-(5.6) | Form the order-`n` coefficient system: the azimuthal equation `2(Xφ_n'' + 2φ_n') = T_{b,n}φ_n + Σ_{i+j=n}{V_i(φ_j' + φ_j/X) + U_i Z_{b,j}φ_j} − Z_{b,n−1}^{[2]}φ_{n−1}`, the axial equation (5.4), the pressure equation `Π_n' = C^{−2}Σ_{i+j=n} φ_iφ_j − Ω_{n−1}/(2X)`, and `Ω_k` from (5.6) | Convolution over the expansion index is exactly the Cauchy convolution the Taylor tower already performs; the coefficients `φ_i`, `U_i` become tower entries. Realizing the *system* needs polynomial-valued duals (row 3) | PLANNED | v1.4.0-connection | `ns_order_n_coefficient_solve` |
| 42 | Section 5.1, splitting | Certify the system is *linear* at each positive order: splitting the transport sums into `(i,j) = (0,n), (n,0)` and `1 ≤ i,j < n` leaves only the current unknown linearly, e.g. `Π_n' = 2C^{−2}φ₀φ_n + C^{−2}Σ_{i=1}^{n−1}φ_iφ_{n−i} − Ω_{n−1}/(2X)` | Index-set manipulation over the shipped convolution; verified by comparing the assembled source against the split form coefficientwise | PLANNED | v1.4.0-connection | `ns_order_n_coefficient_solve` |
| 43 | Section 5.1, regularity | Verify `Ω_k` is divisible by `X` — since `V_j = X v_j` with smooth `v_j`, e.g. `V_i(∂X V_j − V_j/(2X)) = X v_i(v_j/2 + X ∂X v_j)` — so `Ω_k / X` is regular at the axis including all fixed parameter derivatives | Exact series division by `X` with a zero-remainder assertion (`taylor` seeded at an exact point) | SHIPPED | v1.3.0-evolve | `ns_order_n_coefficient_solve` |
| 44 | Lemma 5.1 | Solve, at every positive order, uniquely on `0 ≤ ξ ≤ a`, `ξ = √X`, with `φ_n(0,η) = U_n(0,η) = Π_n(0,η) = 0` | Series solution of the linear ODE system per order — the shipped `taylor` recursion plus row 41's system assembly | PLANNED | v1.4.0-connection | `ns_order_n_coefficient_solve` |
| 45 | Lemma 5.2 | Extend each order radially over `X_a < X_− < X_+ < X_b` and impose the five integral conditions that keep its stress supported in the prescribed annulus | Row 24's moment solve, per order | SHIPPED (per instance) | v1.3.0-evolve | `ns_moment_quadratic_solve` |
| 46 | Proposition 5.3 | Run the coefficient induction: complete order `n` before its coefficients become the sources of order `n + 1`, and produce a sequence of positive-order profiles with a finite residual | BUILD ITEM: **incremental knowledge-base evaluation** — `core.dbsp` (`lib/core/dbsp.esk`, first slice SHIPPED v1.3.3-evolve; GA at v1.5.0) drives the recursion so completing order `n + 1` re-evaluates only what changed | IN PROGRESS | v1.5.0-intelligence | `ns_coefficient_induction_closes` |
| 47 | Lemma 5.4; Section 5.4 | Sum the coefficients with cutoffs `χ` on their vector potentials and pressures, with `a_{j+1} ≥ 2a_j`, taking curls *after* multiplication so incompressibility is preserved exactly | `curl` (compiler builtin) applied after the cutoff product; the divergence-free property is then an identity, not an estimate | SHIPPED | v1.3.0-evolve | `ns_summation_flatness` |
| 48 | Proposition 5.5; (5.41) | Conclude the smooth axisymmetric background `(u_B, p_B)` with `R(u_B, p_B) = −(∂r + 2/r)T_{phys,θ} e_θ − (∂r + 1/r)T_{phys,z} e_z + E_B`, `E_B` flat as `q ↓ 0` | The composite of rows 40-47 | PLANNED | v1.5.0-intelligence | `ns_coefficient_induction_closes` |

### 2.8 Auxiliary torus and separation of oscillatory supports (Section 6)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 49 | (6.1) | Set the dyadic chart: `Q = 2^{−ℓ}`, `ε = Q^h`, `S_* = ℓ²`, `R = r/√Q`, `Z = z/Q^D`, `T = τ/Q`, with chart representatives obtained by multiplying by `Q^A`, `Q^{2A}`, `Q^{2A+1/2}` | Exact rational exponent bookkeeping over the numeric tower | SHIPPED | v1.3.0-evolve | `ns_torus_support_disjointness` |
| 50 | (6.3); (6.6) | Introduce the auxiliary variable `Y ∈ T² = R²/Z²`, the fixed phase map `Y(r,t)`, and derive the derivative operators *after* phase evaluation, including every chain-rule term | BUILD ITEM (row 35): torus-valued fields and the evaluation map as first-class values; the chain-rule terms themselves are produced by the shipped AD operators once the value type exists | PLANNED | v1.5.0-intelligence | `ns_torus_support_disjointness` |
| 51 | Lemma 6.1; (6.13)-(6.15) | Choose color centers `c_ν ∈ T²` and a radius `r₀` so that overlapping slow supports get disjoint auxiliary rectangles: greedy finite coloring of a bounded-degree graph, subject to `c_µ ≠ J_g^Δ c_ν (mod Z²)` for `0 ≤ Δ ≤ Δ_max`, using invertibility of `J_g^Δ − I` | Exact integer-matrix algebra (bignum) for `J_g^Δ − I` and its inverse; the coloring is ordinary Eshkol code over the shipped data structures | SHIPPED | v1.3.0-evolve | `ns_torus_support_disjointness` |
| 52 | Lemma 6.1, consequence | Certify `F_γ F_{γ'} = 0` for `γ ≠ γ'` pointwise, including after evaluation at `Y(r,t)` and for derivatives — the identity that kills cross-label quadratic interactions | Support bookkeeping on the torus value type of row 50, then an exact product-is-zero assertion | PLANNED | v1.5.0-intelligence | `ns_torus_support_disjointness` |
| 53 | (6.14) | Bound the covering-index spread `\|i(ℓ) − i(ℓ')\| ≤ 1 + (4(1+h)log 2 + 8/ℓ₀)/log T_g ≤ Δ_max` | Logarithm enclosure via `taylor-model`, then an exact integer ceiling (rigorous once the Section 3.5 build item lands) | SHIPPED | v1.3.0-evolve | `ns_torus_support_disjointness` |
| 54 | Definitions 6.4-6.5; Proposition 6.6 | Define the mean/moment coefficient classes and the labelled wave classes, and certify each is closed under finite sums at a fixed exponent, under products, and under differentiation | BUILD ITEM: a graded coefficient-class type whose product and derivative rules are checked by construction (the exponent algebra is exact rational arithmetic) | PLANNED | v1.6.0-reasoning | `ns_torus_support_disjointness` |

### 2.9 Oscillatory realization and the two-family stress solve (Section 7)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 55 | Lemma 7.1; (7.2)-(7.3) | Construct the phase and a frame `B` orthogonal to its gradient, on neighborhoods of diameter `O(S_*^{−3})` | `cross`, `dot`, `normalize` (`lib/math.esk`, loaded with `(require math)`) for the frame; the phase is a closed expression differentiated by `gradient` | SHIPPED | v1.3.0-evolve | `ns_pulse_amplitude_envelope` |
| 56 | Proposition 7.2; Corollary 7.3; (7.5) | Solve the projected amplitude equation `L_m(t_m, π_m) = −f_m` for each harmonic `0 < \|m\| ≤ M`, and record the solution operator `f_m ↦ (t_m, π_m)` | `taylor-ode-solve` for the amplitude ODE along the pulse path; `solve` for the projection | SHIPPED | v1.3.0-evolve | `ns_pulse_amplitude_envelope` |
| 57 | Lemma 7.4; (7.22) | Integrate the homogeneous pulse `(th_σ)' = A_Φ th_σ − d th_σ` and certify the envelope: shear amplification followed by viscous decay, with the temporal cutoff acting only in exponentially small tails | `taylor-ode-solve` with step control from `taylor-model` remainders; monotonicity of the envelope tested through `interval-contains?` | SHIPPED | v1.3.0-evolve | `ns_pulse_amplitude_envelope` |
| 58 | Proposition 7.5; (7.27)-(7.28) | Average the quadratic product over the auxiliary torus and the angle: the angular average of `cos²(kΦ_σ)` is exactly `1/2`, and the covariance column is `H_σ = h_σ(−A_c N − σ u_* K + e_σ)` with `\|e_σ\| ≤ C S_*^{−1/2}` | BUILD ITEM (row 35): torus averaging as a builtin. The exact `1/2` for a nonzero integer angular frequency is an exact rational fact the tower already produces | PLANNED | v1.5.0-intelligence | `ns_covariance_two_family_solve` |
| 59 | Proposition 7.5; (7.24), (7.29) | Solve the 2x2 system for the two pulse-family squared amplitudes: `y = H^{−1}T_{0,*}`, `a_σ = √(y_σ)`, i.e. `h_± y_± = ½(−T_N/A_c ∓ T_K/u_*)`; certify `y_σ > 0`, `\|det H\| ≥ c/S_*`, `‖H^{−1}‖ ≤ C√S_*`, and hence `C(W₀) = ε T_{0,*}` | A 2x2 exact rational solve written directly in Eshkol (Cramer over the scalar exact tower); positivity and the determinant lower bound are exact sign tests. This is the arithmetic core of the momentum-transport step | SHIPPED | v1.3.0-evolve | `ns_covariance_two_family_solve` |
| 60 | Proposition 7.5, Step 3 | Bound every fixed derivative of `a_σ` through the Faà di Bruno sum `D^I a_σ = Σ_n Σ_{I₁+⋯+I_n = I} c_{I₁,…,I_n} y_σ^{1/2 − n} Π_ν D^{I_ν} y_σ`, so the zero extension at the shell edges is smooth | The multi-index chain rule is exactly what the Taylor tower computes; `mixed-partial` (`lib/core/ad/guw.esk`) supplies the multivariate case | SHIPPED | v1.3.0-evolve | `ns_covariance_two_family_solve` |
| 61 | (7.30) | Sum slow boxes and bands to the leading physical stress `Σ_β Q^{−2A} η_β² ε T_{0,*} = q^{−A−1/2}T₀`, checking the exponent identity `Q^{−A+h+1/2} q^{−A−1/2}T₀` with `A = ½ + h` and `Σ_β η_β² = 1` | Exact rational exponent arithmetic; cross-label products vanish by row 52 | SHIPPED (exponents) / PLANNED (composition) | v1.3.0-evolve / v1.5.0-intelligence | `ns_covariance_two_family_solve` |
| 62 | Proposition 7.6; (7.31) | Form the signed amplitude increments `dΣ = H^{−1}(Σ/ε)`, `δa_σ = (dΣ)_σ/(2a_σ)`, `L_Σ = √ε Σ_σ δa_σ b_σ` — the first-order variation of the positive representation, giving stress increments of either sign | Differentiating the row-59 solve: `jacobian` of `y ↦ H^{−1}T` composed with `derivative` of the square root, both shipped | SHIPPED | v1.3.0-evolve | `ns_signed_amplitude_increment` |
| 63 | Lemma 7.7; Corollary 7.8; (7.35) | Take exact curls of the localized vector potentials evaluated at `Y(r,t)`, retain the tails and cutoff errors, and bound the covariance remainder | `curl` gives `∇·w = 0` exactly; the retained tails are extra AD chain-rule terms, produced rather than estimated away | SHIPPED (curl) / PLANNED (torus evaluation) | v1.3.0-evolve / v1.5.0-intelligence | `ns_covariance_two_family_solve` |

### 2.10 Compactly supported mean corrections (Section 8)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 64 | Proposition 8.1; (8.1)-(8.3) | Write the conservative momentum equations for the divergence-free decomposition with slow base `(b, V, G)` and read off the integral constraints | The decomposition is expressed with the shipped tensor and AD surface (`make-tensor`, `tensor-ref`, `divergence`, `curl`) | SHIPPED | v1.3.0-evolve | `ns_five_moment_vandermonde` |
| 65 | Lemma 8.2; Proposition 8.3; (8.12) | Build a compactly supported primitive for the cylindrical weights, and reconstruct pressure from the updated radial equation after each correction | `integrate` for the primitive; the reconstruction is a linear map applied per step | SHIPPED | v1.3.0-evolve | `ns_five_moment_vandermonde` |
| 66 | Proposition 8.4; Corollary 8.5; (8.15) | Identify the three compatibility defects `(P, J_θ, J_z)` — the radial integral defects for pressure and tangential momentum — as covariance targets | Cumulative-integral evaluation plus exact subtraction | SHIPPED | v1.3.0-evolve | `ns_five_moment_vandermonde` |
| 67 | Lemma 8.6; (8.20) | Invert the fast auxiliary-time derivative `N = v_t · ∂y` on functions of zero Haar mean on `T²` | BUILD ITEM (row 35): the torus type, its Haar mean, and the inverse of the directional derivative on the zero-mean subspace | PLANNED | v1.5.0-intelligence | `ns_fast_time_inverse` |
| 68 | Lemma 8.7; (8.24)-(8.25) | Solve the five radial moment equations with three azimuthal and two axial bumps: with base `V_q = a(η)x^{−1−2λ}`, the angular block has powers `2, −2−2λ, −2λ` and the axial block `1, 1−2λ`; the geometric copies `η_j(x) = a_j^{−1}η₀(x/a_j)`, `a_j = e^{jd}` give power moments `µ_p e^{jdp}`, so after dividing by `µ_p` the blocks are Vandermonde in the distinct numbers `e^{dp}` | Exact rational Vandermonde determinants for `A_θ` (3x3) and `A_z` (2x2), then exact elimination for `(u₀,u₁,u₂)` and `(s₀,s₁)` — written in Eshkol over the scalar exact tower, with `λ > 0` fixed | SHIPPED | v1.3.0-evolve | `ns_five_moment_vandermonde` |
| 69 | Lemma 8.7, Step 3 | Pass from `q`-normalized rows to a fixed `Q` chart, multiplying each row and target by the same powers of `q/Q`, and certify the `S_α → M_α` bound | Exact rational power bookkeeping plus derivative bounds from the shipped tower | SHIPPED | v1.3.0-evolve | `ns_five_moment_vandermonde` |
| 70 | Lemma 8.8; (8.27); Section 8.7 | Record the exact difference between the linearized correction and the updated defects, certify the improved decay of the nonlinear remainder, and check chart-to-chart compatibility of the recomputation | Exact-difference arithmetic; the decay improvement is an exact inequality on rational exponents | SHIPPED (algebra) / PLANNED (composition) | v1.3.0-evolve / v1.6.0-reasoning | `ns_residual_exponent_ladder` |

### 2.11 Residual improvement and the local field (Section 9)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 71 | Section 3.3, exact increment identity | Use `R(u_B + w, p_B + π) = R(u_B, p_B) + L_{u_B}(w, π) + ∇·(w ⊗ w)` with `L_v(w,π) = ∂t w + (v·∇)w + (w·∇)v − Δw + ∇π` — the identity that separates the cancellation from the linear evolution | The AD residual operator applied to the sum, then to each piece; the identity is *checked*, not assumed | SHIPPED | v1.3.0-evolve | `ns_viscosity_scaling_exact` |
| 72 | Proposition 9.1; Lemma 9.2 | Bound the residual of velocities constructed by curls, and the wave-mean interaction terms, per harmonic class | Enclosures through `taylor-model` over the chart box; rigorous once the Section 3.5 build item lands | PLANNED | v1.5.0-intelligence | `ns_residual_exponent_ladder` |
| 73 | Proposition 9.3; (9.3) | Recompute the *full* residual after every operation, retaining oscillatory interactions, curl and cutoff corrections, and every other error term | The AD residual operator recomputed on the updated fields — recomputation is cheap and exact, which is precisely the mechanization advantage | SHIPPED | v1.3.0-evolve | `ns_residual_exponent_ladder` |
| 74 | Definition 9.4; Proposition 9.5; (9.7)-(9.10) | Initialize: perform the temporal mean update (8.20) and the five-equation correction (8.25), reconstructing pressure before and after, and certify `B₀ = 0.7`, `C₀* = 1.2`, with the two preserved integrals `∫₀^∞ R²⟨v⟩_Y dR = 0`, `∫₀^∞ R⟨γ⟩_Y dR = 0` | Rows 67-68 plus exact zero tests on the two preserved moments | PLANNED | v1.5.0-intelligence | `ns_residual_exponent_ladder` |
| 75 | Proposition 9.6, steps (i)-(iv) | Run one full correction cycle: (i) solve the inhomogeneous amplitude equation per supported nonzero harmonic; (ii) change wave amplitudes to correct the auxiliary-averaged tangential residual; (iii) apply the temporal mean inverse to the zero-auxiliary-average part; (iv) apply the five-equation correction to the three current defects — reconstructing pressure after each | Rows 56, 62, 67, 68 composed, with `core.dbsp` (row 46) driving the recompute so only the changed terms are re-evaluated | IN PROGRESS | v1.5.0-intelligence | `ns_residual_exponent_ladder` |
| 76 | (9.8); (9.18) | Certify the residual-exponent ladder `σ₀ = 1/5`, `σ_{j+1} = σ_j + 1/10`, `σ_j = 1/5 + j/10 → ∞`, with `B_j = ½ + σ_j`, `C_j* = 1 + σ_j`, and the derivative bound carrying `q^{hσ_j − K_m}` | Exact rational recurrence on the exponent, with an exact monotonicity and divergence test | SHIPPED | v1.3.0-evolve | `ns_residual_exponent_ladder` |
| 77 | Lemma 9.7; Lemma 9.8 | Certify a common domain `q < q_big` independent of stage and derivative order, and physical derivative estimates with stage-independent exponent loss | Enclosures over the common domain box; rigorous once the Section 3.5 build item lands | PLANNED | v1.5.0-intelligence | `ns_summation_flatness` |
| 78 | Proposition 9.9; (3.4); (9.20) | Sum the corrections with shrinking cutoffs (each equal to one near `q = 0`), take curls, and certify flatness: `\|∂ˣ_α ∂ᵇ_t R(u,p)\| ≤ C_{α,b,N,X₁} q^N` for `0 ≤ X ≤ X₁` as `q ↓ 0`, while preserving the leading velocity growth (3.6) | `curl` after cutoff multiplication keeps `∇·u = 0` exactly; the flatness bound is an enclosure for each `(α, b, N)` up to a stated order | SHIPPED (curl, growth) / PLANNED (flatness certificate) | v1.3.0-evolve / v1.6.0-reasoning | `ns_summation_flatness` |

### 2.12 Compact forcing, whole-space breakdown, and the torus (Section 10)

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 79 | Section 3.5; Proposition 10.1; (3.3), (10.4) | Localize: with `c = χ_x χ_t` equal to one near `(0,1)`, set `u = curl(cA) + cB e_θ`, `p = c p_loc`, so `u = c u_loc + ∇c × A` and both terms are separately divergence-free; extend by zero outside the cutoff support | `curl` applied after multiplication; the extra term `∇c × A` is produced by `cross` and `gradient` rather than estimated | SHIPPED | v1.3.0-evolve | `ns_localization_smooth_extension` |
| 80 | Lemma 10.2; Lemma 10.3; (10.5), (10.8)-(10.9) | Certify that `∂ˣ_α ∂ᵗ_j f` converges uniformly on `R³` as `t ↑ 1`, using the flatness bound near the origin, the identically-zero residual for `X ≥ X_ext`, and the heat bounds plus the normalized pressure integral (3.5) in the remaining region; then realize the limits from `t > 1` with `f ∈ Cc∞(R³ × (0,∞); R³)` supported in `K × [0,2]` | Enclosures per derivative order (row 78) plus the heat-exterior bounds of rows 27-28 | PLANNED | v1.6.0-reasoning | `ns_localization_smooth_extension` |
| 81 | Lemma 10.4; (10.13) | Certify the global energy `F(1) < ∞` and the integrated dissipation bound directly from the equation for the localized fields | Cumulative-integral inequality on a compact interval: `(integrate f a b n)` (Simpson, `lib/math.esk`) for the value, `taylor-model` for the tail, `interval-contains?` for the verdict | SHIPPED (compact range) / PLANNED (tail) | v1.3.0-evolve / v1.5.0-intelligence | `ns_energy_dissipation_bounds` |
| 82 | Lemma 10.5; (10.20)-(10.21) | Along `x_τ = (√(2X_in τ), 0, 0)`, `t = 1 − τ`, verify `u_θ(x_τ, 1 − τ) = τ^{−A}(e₀ + O(τ^{2h})) → +∞`, exclude a classical `H³` continuation through time one via `H³(R³) ↪ L∞(R³)`, and run the difference-energy plus Gronwall comparison | The growth path is evaluated directly; the comparison argument is the cumulative-integral inequality pattern of row 81 | SHIPPED (growth path) / PLANNED (comparison certificate) | v1.3.0-evolve / v1.6.0-reasoning | `ns_energy_dissipation_bounds` |
| 83 | (10.22)-(10.23) | Rescale to arbitrary `ν > 0`: verify term by term that `∂t uν + (uν·∇)uν − νΔuν + ∇pν = √ν[∂t u + (u·∇)u − Δu + ∇p](y,t) = fν`, `∇·uν = 0`, `∂ˣ_α ∂ᵐ_t fν = ν^{(1−\|α\|)/2}(∂ʸ_α ∂ᵐ_t f)(x/√ν, t)`, and `‖uν(t)‖₂² = ν^{5/2}‖u(t)‖₂²` | The AD residual operator applied to the rescaled fields, with the exact `√ν` and `ν^{5/2}` factors carried through the exact numeric tower — the identity closes to zero, not to a tolerance | SHIPPED | v1.3.0-evolve | `ns_viscosity_scaling_exact` |
| 84 | Corollary 10.6 | Transport to `T³ = R³/Z³`: choose `λ > 1` with `λ^{−1}K_ν ⋐ Q₀`, set `ũ(x,t) = λu(λx, λ²(t − t₀))`, `p̃ = λ²p(…)`, `f̃ = λ³f(…)` with `t₀ = 1 − λ^{−2}`, extend by zero for `t < t₀`, and check every momentum term is `λ³` times its original value so the viscosity is unchanged | Exact rational scaling arithmetic plus the residual operator re-evaluated on the rescaled fields; disjointness of integer translates is an exact support test | SHIPPED | v1.3.0-evolve | `ns_torus_corollary_scaling` |

---

## 3. Capability ledger

### 3.1 SHIPPED

Verified against the tree. Version column is the release that shipped the item.

| Capability | Eshkol surface | Where | Version | Gate |
|---|---|---|---|---|
| Exact rational and bignum arithmetic | `numerator`, `denominator`, `exact?`, `exact->inexact`, `rationalize`, `expt`, full contagion rules | `lib/core/rational.cpp`, `lib/core/bignum.cpp`; `docs/breakdown/EXACT_ARITHMETIC.md` Sections 3-5 | v1.1-accelerate | `numeric-depth` oracle |
| Exact-coefficient Taylor towers | `taylor`, `derivative-n` (compiler builtins) | `lib/frontend/parser.cpp:1852-1853` (special forms, not `BUILTINS[]` rows), `lib/backend/autodiff_codegen.cpp` (`taylorSeries`, `derivativeN`, `taylorApiCore`), `lib/core/runtime_taylor.c` (`alloc_exact_series`, `taylor_pow_exact`) | v1.3.0-evolve | `ad_taylor_p6_exact_coefficients` |
| Forward and reverse AD | `derivative`, `gradient`, `jacobian`, `hessian`, `divergence`, `curl`, `laplacian`, `directional-derivative` | `lib/backend/autodiff_codegen.cpp`; `docs/guide/AUTOMATIC_DIFFERENTIATION.md` Section 1 | v1.0-foundation onward | `ad-oracle`, `ad-depth` |
| Multivariate mixed partials | `mixed-partial` (`core.ad.guw`), with an internal exact Gaussian solve | `lib/core/ad/guw.esk` (`guw-gauss-solve`) | v1.3.0-evolve | `ad_taylor_p4_guw_multivariate` |
| Tensor-valued towers | `core.ad.tensor_tower` | `lib/core/ad/tensor_tower.esk` | v1.3.0-evolve | `ad_taylor_p7_tensor_towers` |
| Interval arithmetic and validated Taylor models | `taylor-model` (signature `(taylor-model f x0 r k)`), `tm-range`, `tm-eval`, `tm-remainder`, `interval-lo`, `interval-hi`, `interval-contains?`, `interval-widen` | `lib/core/ad/taylor_models.esk`, `lib/core/ad/interval.esk` | v1.3.0-evolve | `scripts/run_ad_validated_bounds_gate.sh` |
| Sparse high-order tensors | `sparse-hessian` (`core.ad.sparse_guw`) | `lib/core/ad/sparse_guw.esk` | v1.3.0-evolve | `ad-taylor-campaign` P12 criterion |
| Tower numerics | `taylor-ode-solve`, `taylor-root`, `taylor-inverse-series` | `lib/core/ad/taylor_numerics.esk` | v1.3.0-evolve | `ad-taylor-campaign` P11 criterion |
| Tensors | `make-tensor`, `tensor-ref`, `tensor-set!`, `tensor-copy`, `tensor-solve` | `docs/API_REFERENCE.md`, tensor codegen | v1.1-accelerate | `tensor-collection-depth` |
| Numeric linear algebra and quadrature (double data) | `(det M n)`, `(inv M n)`, `(solve A b n)` (LU with partial pivoting), `(integrate f a b n)` (Simpson), `(newton f df x0 tol iters)`, `cross`, `dot`, `normalize` — loaded with `(require math)`, not auto-loaded | `lib/math.esk`; `docs/reference/stdlib/math.md` | v1.1-accelerate | `stdlib-ready` |
| Symbolic differentiation | `(diff expr var)` / `(differentiate expr var)` — compile-time, returns a quoted S-expression, with a simplifier over `+ - * / expt sin cos tan asin acos exp log sqrt` | `lib/frontend/parser.cpp:1847-1848`; `lib/backend/llvm_codegen.cpp` (`codegenDiff`, `buildSymbolicDerivative`, `simplifySymbolicAST`) | v1.1-accelerate | `metaprog-depth` |
| Incremental dataflow, first slice | `core.dbsp` Z-sets and stream operators | `lib/core/dbsp.esk`; `docs/reference/stdlib/dbsp.md` | v1.3.3-evolve | `v1.3-evolve` oracle |

**Engine qualifier.** Of the surface above, `taylor`, `derivative-n`, `diff` and
the classic AD operators are compiler special forms; the `core.ad.*` modules and
`lib/math.esk` are Eshkol library code, loaded with `(require …)` rather than
auto-loaded. The bytecode VM does not carry the Taylor-tower surface — that is
the existing BUILD ITEM recorded in [ROADMAP.md](../../ROADMAP.md) (VM
Taylor-tower builtins, target v1.4.1) — and on the VM `diff` is an alias of the
numeric `derivative` rather than the symbolic form. Everything in Section 2 that
names `taylor` or `derivative-n` runs today on the LLVM backend, AOT and JIT.

### 3.2 The four example programs

The `examples/mathematics_navier_stokes_*.esk` family lands with this document
and exercises the four steps of Section 2 that are fully executable today. It
follows the flat `examples/*.esk` convention, so
[`scripts/run_examples_tests.sh`](../../scripts/run_examples_tests.sh) discovers
the files without further wiring, exactly as the existing AI-mathematics family
does (`docs/AI_MATHEMATICS_EXAMPLES.md`).

| Program | Proof steps | What it computes | Status | Version | Gate |
|---|---|---|---|---|---|
| `mathematics_navier_stokes_viscosity_scaling.esk` | 9, 71, 83 | The exact viscosity-scaling identity through the AD residual operator: the rescaled fields are pushed through `∂t u + (u·∇)u − νΔu + ∇p` and the residual difference closes to exact zero, not to a tolerance | IN PROGRESS | v1.3.5 | `ns_viscosity_scaling_exact` |
| `mathematics_navier_stokes_similarity_scales.esk` | 1, 2, 7, 8 | The similarity scale exponents `(A, D, ℓr, ℓz, E_core, D_core)` derived as a linear system solved exactly over rationals on the scalar exact tower, with the finite-energy condition `h < 1/6` verified exactly | IN PROGRESS | v1.3.5 | `ns_similarity_exponents_solved` |
| `mathematics_navier_stokes_first_principles.esk` | 3, 5, 6, 43 | The leading-order profile balance obtained by collecting Taylor coefficients of the residual of the similarity ansatz, with a negative control: a deliberately wrong exponent must leave a nonzero coefficient | IN PROGRESS | v1.3.5 | `ns_leading_profile_balance` |
| `mathematics_navier_stokes_pulse_stress.esk` | 58, 59, 62 | The pulse momentum-flux averages and the two-family stress solve: the exact angular average `1/2`, the 2x2 covariance system `y = H^{−1}T`, positivity of both squared amplitudes, and the first-order signed increment | IN PROGRESS | v1.3.5 | `ns_covariance_two_family_solve` |

### 3.3 IN PROGRESS

| BUILD ITEM | What it unlocks | Version | Gate |
|---|---|---|---|
| `core.dbsp` GA — incremental evaluation over the closed world | Steps 46, 75: the coefficient induction and the correction cycle re-evaluate only what changed, so order `n+1` and stage `j+1` are incremental rather than full recomputations; this is also the substrate for search over ansatz families | v1.5.0-intelligence | `ns_coefficient_induction_closes` |
| The four example programs of Section 3.2 | Steps 1, 2, 5-9, 43, 58, 59, 62, 71, 83 executable and gated in CI | v1.3.5 | `./scripts/run_examples_tests.sh` |

### 3.4 PLANNED

| BUILD ITEM | What it unlocks | Version | Gate |
|---|---|---|---|
| **Symbolic multivariate polynomial and series values.** A sparse multivariate polynomial ring over `Q` and `GF(p)`, and a truncated multivariate power-series value with a coefficient norm. Verified absent today: `groebner`, `resultant`, `discriminant` and `poly-add` have zero definitions in the tree (ICC `find-symbol`); the only polynomial algebra present is SICP exercise code in `tests/sicp/ch2_polynomial_arithmetic.esk` | Steps 22 (generic), 31, 40-42, 54 | v1.4.0-connection | `ns_order_n_coefficient_solve` |
| **Polynomial-valued duals.** The residual of a symbolic ansatz to every order: AD operators applied to a symbolic field so that `R(u, p)` returns a graded value whose coefficients can be collected and solved, rather than a number. Rests on the item above plus the shipped tower | Steps 3, 10, 14, 40-42, 44 | v1.4.0-connection | `ns_leading_profile_balance` |
| **Torus averaging and oscillatory stress realization as builtins.** `T¹`/`T²`-valued fields, the Haar mean `⟨·⟩_Y`, the angular mean `⟨·⟩_θ`, evaluation at a phase map `Y(r,t)` with full chain-rule propagation, support-disjointness bookkeeping, and the inverse of a directional derivative on the zero-mean subspace | Steps 35, 37, 38, 50, 52, 58, 61, 63, 67 | v1.5.0-intelligence | `ns_torus_support_disjointness`, `ns_fast_time_inverse` |
| **Exact linear algebra.** Rational and bignum Gaussian elimination, determinant and inverse over the exact scalar tower, and an exact tensor element type. Verified absent today: `lib/math.esk`'s `det`/`inv` seed inexact constants, `lib/core/linear_solve.cpp` and the BLAS entry points are f64-only, and Eshkol tensors are f64-backed (`(tensor 1/3)` prints `#(0)`). Until it lands, every exact system in Section 2 is written out directly over the scalar exact tower on Scheme vectors, which do carry exact rationals | Steps 7, 22, 24, 25, 34, 59, 68 | v1.4.0-connection | `ns_moment_matrix_invertible` |
| **Rigorous enclosures.** Directed-rounding interval arithmetic (Eshkol exposes no `nextafter`/`fesetround` today, so `lib/core/ad/interval.esk` widens by a relative epsilon instead) and a proved Taylor-model remainder in place of the current 65-point sampled derivative bound times a safety factor. This is the ROADMAP Formal Verification item that stages Lean-certifying the validated-AD Taylor models | Every row in Section 2 whose gate is an enclosure | v1.5.0-intelligence (directed rounding), v2.0-starlight (proved remainder) | `ns_certificate_external_check` |
| **Rigorous compact-set bounds.** Supremum/infimum over a parameter box with adaptive subdivision on top of the two items above, so a compactness constant is produced rather than asserted | Steps 19, 30 (tail), 72, 77, 81 (tail) | v1.5.0-intelligence | `ns_cone_condition_equivalence` |
| **Interval and Taylor-model tensor element types with AD.** Tensor element types whose entries are intervals or Taylor models, differentiable by the same operators — the composition of the shipped tensor towers with the shipped Taylor models, which today are separate | Steps 72, 77, 78, 80 | v1.6.0-reasoning | `ns_summation_flatness` |
| **Graded coefficient classes.** A value type carrying an exponent grade whose product and derivative rules are checked by construction, so the paper's coefficient classes (Definitions 6.4-6.5, Proposition 6.6) are types rather than side conditions | Steps 54, 70 | v1.6.0-reasoning | `ns_residual_exponent_ladder` |
| **Proof-object emission and an independent checker.** Every certified step emits a machine-readable certificate (the exact rational witnesses, the interval endpoints and their rounding direction, the derivation of each inequality), and a checker outside Eshkol re-verifies the certificate without trusting the compiler. This is the assurance-workstream (W2) machine-checked-invariants track applied to a construction rather than to the compiler | The whole of Section 2; without it "mechanized" means "Eshkol says so" | v1.7.0-synthesis | `ns_certificate_external_check` |
| **Incremental knowledge-base evaluation for search over ansatz families.** The DBSP spine plus the knowledge-base and inference surface, so an ansatz family is a query whose answer updates incrementally as constraints are added — the search half of the construction, as opposed to the verification half | Steps 21, 46, 75 as a search rather than a replay | v1.6.0-reasoning | `ns_ansatz_family_search` |

### 3.5 Known exactness and rigor boundaries

A blueprint is only useful if it is honest about where the floor is. Four
boundaries are load-bearing for Section 2, all of them recorded in the tree
itself rather than inferred:

- **Exactness is scalar.** The bignum and rational tower is scalar-only. Every
  tensor is f64-backed (`lib/core/arena_memory.h`: storage is always f64 bit
  patterns, the dtype records the logical precision), so `(tensor 1/3)` prints
  `#(0)`. Scheme vectors and lists do carry exact rationals; exact work in
  Section 2 uses those.
- **AD is exact at exact scalar points, f64 at vector points.** `derivative`,
  `gradient` and `hessian` return exact rationals at an exact scalar point and
  run the same tower pass; vector points intentionally stay on the inexact
  carrier. The exact tier also demotes silently on the first transcendental, on
  a division with a bignum operand, and when the monomorphized no-heap tier or
  a live reverse tape is in play. Section 2 therefore states, per row, whether
  the exactness it claims is scalar.
- **Enclosures are validated, not yet rigorous.** `lib/core/ad/interval.esk`
  widens outward by a relative epsilon rather than using directed rounding, and
  `lib/core/ad/taylor_models.esk` bounds the remainder by a 65-point sample of
  the next derivative times a safety factor — both files say so in their own
  headers. Every "enclosure" in Section 2 is sound in practice and gated in CI
  (`scripts/run_ad_validated_bounds_gate.sh`), and becomes rigorous with the
  build item above. No row in Section 2 should be read as a proof today.
- **Symbolic algebra means symbolic differentiation only.** `(diff expr var)`
  is real and returns a quoted S-expression, but there is no polynomial ring,
  no resultants, no series *values* — that is the v1.4.0-connection build item.
  Quoted input to `diff` raises rather than differentiating.

---

## 4. The end-to-end verification contract

### 4.1 What "the construction is mechanized" means

Three conditions, in increasing strength. None of them is a claim about the
theorem; they are claims about what a machine has checked.

1. **The residual closes to order `N` with exact coefficients.** For the
   leading field and for each background order `n`, the residual of the
   similarity ansatz, expanded in the graded parameter and collected
   coefficientwise, is *exactly* zero up to the stated order — an exact rational
   zero, not a small float. Steps 3, 40-44 and 48. Negative control is
   mandatory: a deliberately perturbed exponent must produce a nonzero
   coefficient, or the test proves nothing (the pattern the example family of
   Section 3.2 already uses).

2. **Every appendix lemma is certified.** Each numbered statement in
   Appendices A, B and C that the construction consumes has an Eshkol-checkable
   form: an exact algebraic identity, an exactly solved finite linear system, or
   an inequality with explicit interval endpoints and a stated rounding
   direction. Steps 22-39. Two things this condition forbids: a lemma whose
   Eshkol form is a bare floating-point comparison is not certified and must be
   recorded as an open BUILD ITEM rather than a pass; and an enclosure produced
   by the epsilon-widened intervals or the sampled Taylor-model remainder of
   Section 3.5 is validated, not proved, and must be labelled as such until the
   rigorous-enclosure build item lands.

3. **Certificates are checkable outside Eshkol.** Each certified step emits a
   proof object — the witnesses, the endpoints, the rounding mode, the
   derivation chain — and an independent checker re-verifies it without running
   Eshkol. Until this exists, condition 2 rests on trusting the compiler, and
   the trust is not transferable. This is the `ns_certificate_external_check`
   BUILD ITEM at v1.7.0-synthesis.

The order matters. Condition 1 is arithmetic and is reachable with what is
shipped plus the three v1.4.0-connection items (symbolic values, polynomial-valued
duals, exact linear algebra). Condition 2 needs the torus, rigorous-enclosure and
compact-set items at v1.5.0-intelligence, and the proved Taylor-model remainder
that ROADMAP.md already stages under Formal Verification. Condition 3 is the
assurance-track deliverable and is what makes the result citable by someone who
does not run this compiler.

**One property of the harness is part of the contract.** The examples suite is a
compile-and-run smoke gate: it fails a program that crashes, not one that prints
a failure and exits zero. Every program in Section 3.2 must therefore exit
nonzero when a check fails, and must carry a negative control, or its green run
proves nothing. Note also that Scheme fences inside `docs/design/` are outside
the documentation-example extractor's scope, so the commands in Section 4.3 are
the only executable contract this document has.

### 4.2 The proposed oracle

The gates named throughout Section 2 belong to one new completion oracle,
`navier-stokes-mechanization`, added to
[.icc/completion-oracles.yaml](../../.icc/completion-oracles.yaml) alongside the
existing `ad-taylor-campaign` and `sdnc-paper-reproducibility` targets, which it
follows in shape: one `runtime_event` criterion per step group, each with an
explicit id, an `action` naming the script that produces the trace, and
`severity: high` for every numerical-correctness claim.

Criterion ids, in construction order:
`ns_similarity_exponents_solved`, `ns_viscosity_scaling_exact`,
`ns_core_energy_exponents`, `ns_leading_profile_balance`,
`ns_cumulative_moment_identity`, `ns_cone_condition_equivalence`,
`ns_moment_matrix_invertible`, `ns_moment_quadratic_solve`,
`ns_heat_exterior_recurrence`, `ns_axis_series_contraction`,
`ns_shear_loop_mean`, `ns_high_frequency_shear_moments`,
`ns_order_n_coefficient_solve`, `ns_coefficient_induction_closes`,
`ns_torus_support_disjointness`, `ns_pulse_amplitude_envelope`,
`ns_covariance_two_family_solve`, `ns_signed_amplitude_increment`,
`ns_five_moment_vandermonde`, `ns_fast_time_inverse`,
`ns_residual_exponent_ladder`, `ns_summation_flatness`,
`ns_localization_smooth_extension`, `ns_energy_dissipation_bounds`,
`ns_torus_corollary_scaling`, `ns_certificate_external_check`,
`ns_ansatz_family_search`.

The oracle is registered when its first criteria have traces to grade; an oracle
with no evidence grades red by design, which is the correct reading of a
construction that is not yet mechanized. Each criterion must ship with an
`action:` that names a script which actually exists — the `ad-taylor-campaign`
target already carries criteria whose actions resolve to no script in the tree
even though the underlying capability is real and CTest-gated, and this oracle
must not repeat that pattern.

### 4.3 Running what exists today

Build the compiler, then run the example family. From a clean checkout:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./scripts/run_examples_tests.sh
```

To run a single program in each of the three execution modes, use the same
binary the AI-mathematics family documents:

```bash
./build/eshkol-run -r examples/mathematics_navier_stokes_viscosity_scaling.esk -L build
./build/eshkol-run examples/mathematics_navier_stokes_similarity_scales.esk -L build
```

The AD surface the four programs rest on has its own suites, which are the
upstream gates for every SHIPPED row in Section 2:

```bash
./scripts/run_ad_depth.sh
./scripts/run_ad_validated_bounds_gate.sh
```

To grade a target with ICC, regenerate traces first, then read the verdict:

```bash
./scripts/run_icc_smoke.sh
icc readiness --repo eshkol --target ad-taylor-campaign --trace-dir scripts/icc_traces
```

A readiness verdict taken without regenerating traces is not evidence.

---

## 5. Release note text

Ready to paste; the release cut owns `RELEASE_NOTES.md`.

> Eshkol now documents, step by step, how a published finite-time
> Navier-Stokes blowup construction would be obtained inside the language. The
> new design note walks the paper's own structure — similarity coordinates and
> the leading field, the cumulative radial moments, the admissible stress cone,
> the heat exterior and the analytic axis profiles, the order-by-order
> background correction, the auxiliary torus, the two-family stress solve, the
> residual-improvement ladder, and the localization to a compactly supported
> force — and for each step names the Eshkol primitive that performs it or the
> build item that will, together with the gate that certifies it. Four new
> example programs run the steps that are executable today: the viscosity
> scaling identity through the AD residual operator, the similarity exponents as
> an exactly solved rational system, the leading-order profile balance by
> Taylor-coefficient collection with a negative control, and the pulse
> momentum-flux averages with the two-family stress solve. What makes this
> tractable is the combination the language already ships: exact rational and
> bignum arithmetic, Taylor towers whose coefficients stay exact, forward and
> reverse differentiation, and validated enclosures — so an identity that is
> supposed to cancel closes to exact zero rather than to a tolerance.
</content>
</invoke>
