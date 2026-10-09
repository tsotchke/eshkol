# Navier-Stokes examples: residuals, scaling laws and similarity profiles

<!-- example-catalogue:ns-summary:start -->
The authored matrix contains **12 programs, 14 criteria and 28 CTest entries**, each with native JIT and AOT variants, under `ESHKOL_BUILD_TESTS AND TARGET eshkol-run`. Localization supplies three criteria. Every one of these 12 programs was run at release SHA `60f345def` (Eshkol Compiler v1.3.6-evolve, Release build, macOS arm64, measured 2026-10-09) in native JIT and native AOT. **12** completed in both modes; **12** of those printed their own verdict `RESULT: ALL PASS` with `Failed: 0` in both modes. Wall times measured with 6 programs running concurrently on a shared machine; CPU user/sys from wait4.
<!-- example-catalogue:ns-summary:end -->

Source: an external reference construction for the three-dimensional
Navier-Stokes equations, OpenAI, "Finite time blowup for Navier–Stokes" (2026),
<https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf>.
Its Theorem 1.1 states that for every viscosity `nu > 0` there are a smooth force
compactly supported in space and time, a compact set `K` and smooth velocity and
pressure fields on `R^3 x [0, 1)`, starting from rest and supported in `K`,
solving the forced incompressible equations with uniformly bounded kinetic
energy while `||u(t)||_inf` is unbounded as `t -> 1`. The PDF carries no date
line; the year is the one the program headers give.
Section, equation and lemma numbers cited below (e.g. "Lemma 4.5", "(4.20)")
refer to that source.

These programs do not reproduce the source's full argument. They compute the
*leading structure* of the construction mechanically, where a reader would otherwise differentiate,
rescale and balance powers by hand: the residual operator is assembled from AD
partials, the similarity coordinates are differentiated through their own
implicit definition, the scale exponents are solved as a linear system in exact
rational arithmetic, and the leading profile series is derived from the leading
balance operator rather than quoted. Every number printed is computed by the
program. The only literal constants that appear in a comparison are the exact
values the paper states, and each such comparison is either an exact rational
equality or carries a stated tolerance.

Each program prints one `PASS:`/`FAIL:` line per check, a `Passed:`/`Failed:`
count, a `RESULT: ALL PASS` or `RESULT: FAILURES DETECTED` verdict, and exits
nonzero if any check failed.

## What each program covers

<!-- example-catalogue:ns-table:start -->
| Program | Paper sections | Implemented calculation | CTest criteria | Measured JIT / AOT |
|---|---|---|---|---|
| [mathematics_navier_stokes_first_principles.esk](../examples/mathematics_navier_stokes_first_principles.esk#L1) | 2.1, 3.1, 4.1 ((4.9), (4.12), (4.13)) | Solve an exact exponent linear system with free h; interpolate the leading operator, solve finite profile recurrences and compare leading physical residuals. | `ns_leading_profile_balance` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-first-principles) |
| [mathematics_navier_stokes_heat_exterior.esk](../examples/mathematics_navier_stokes_heat_exterior.esk#L1) | 2.3, Appendix A, Lemma A.1 | Solve curvature-corrected radial heat coefficients by triangular recursion, invert a distinct-power moment matrix and inspect near-terminal-time behavior. | `ns_heat_exterior_exact` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-heat-exterior) |
| [mathematics_navier_stokes_localization.esk](../examples/mathematics_navier_stokes_localization.esk#L1) | 10, Proposition 10.1, Lemmas 10.2-10.4, Corollary 10.6 | Cancel a temporal pole with a cutoff polynomial, guard spatial bumps, periodize within one cell and differentiate a discrete energy model. | `ns_localization_smooth_extension`, `ns_energy_dissipation_bounds`, `ns_torus_corollary_scaling` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-localization) |
| [mathematics_navier_stokes_mean_corrections.esk](../examples/mathematics_navier_stokes_mean_corrections.esk#L1) | 8, Lemma 8.2 and Lemma 8.7 | Build supported divergence-free bump fields, compute a five-moment generalized Vandermonde matrix and solve amplitudes with exact_linalg. | `ns_five_moment_vandermonde` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-mean-corrections) |
| [mathematics_navier_stokes_oscillatory_realization.esk](../examples/mathematics_navier_stokes_oscillatory_realization.esk#L1) | 6, 7, 2.2 | Construct trigonometric polynomials on a four-point auxiliary torus, extract zero-mode fluxes and solve a two-component stress matrix. | `ns_oscillatory_zero_mode` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-oscillatory-realization) |
| [mathematics_navier_stokes_pulse_growth.esk](../examples/mathematics_navier_stokes_pulse_growth.esk#L1) | Introduction ([9]), 2.2 | Derive the affine-background wavevector law and log-amplitude rate, bisect the crossover and integrate the amplitude by RK4. | `ns_pulse_growth_crossover` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-pulse-growth) |
| [mathematics_navier_stokes_pulse_stress.esk](../examples/mathematics_navier_stokes_pulse_stress.esk#L1) | 2.2, 3.2 (Figure 4), 3.3 and 7 | Average angular products, solve a rational two-family covariance system and integrate one transverse mode on an affine background. | `ns_covariance_two_family_solve` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-pulse-stress) |
| [mathematics_navier_stokes_residual_ladder.esk](../examples/mathematics_navier_stokes_residual_ladder.esk#L1) | 9, (9.8), (9.18), Propositions 9.3 and 9.6 | Compute the rational exponent ladder and cancel three Taylor coefficients of the scalar residual with linear term 2w and quadratic w². | `ns_residual_exponent_ladder` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-residual-ladder) |
| [mathematics_navier_stokes_residual_order_n.esk](../examples/mathematics_navier_stokes_residual_order_n.esk#L1) | 5, 5.1, equations (5.1)-(5.6) | Apply a fixed linear correction operator to known lower-order forcing; compare Taylor coefficients with derivative-n/factorial extraction. | `ns_residual_order_n_vanishes` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-residual-order-n) |
| [mathematics_navier_stokes_similarity_scales.esk](../examples/mathematics_navier_stokes_similarity_scales.esk#L1) | 2.1, 3.1, 4.1 (Lemma 4.1, (3.2), (4.3), (4.7)) | Build the self-similar axisymmetric ansatz, differentiate q’s implicit solve, fit scale exponents and evaluate core energy/physical residual scaling. | `ns_similarity_exponents_solved` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-similarity-scales) |
| [mathematics_navier_stokes_stress_cone.esk](../examples/mathematics_navier_stokes_stress_cone.esk#L1) | 4.3, Lemma 4.5, Appendix C | Use rational quadratic sign tests, integrate the radial profile equation and reuse an established threshold P_K on constructed stresses. | `ns_cone_condition_equivalence` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-stress-cone) |
| [mathematics_navier_stokes_viscosity_scaling.esk](../examples/mathematics_navier_stokes_viscosity_scaling.esk#L1) | 3 (outline), 10.4 equations (10.22)-(10.23) | Assemble time/advection/Laplacian/pressure derivatives by AD for u=(1+t)(y²z,z²x,x²y), p=xyz+tx²; rescale fields and grids. | `ns_viscosity_scaling_exact` | [PASS / PASS](MATHEMATICS_EXAMPLES.md#mathematics-navier-stokes-viscosity-scaling) |
<!-- example-catalogue:ns-table:end -->

## What the verdicts certify

### `mathematics_navier_stokes_viscosity_scaling.esk`

The test field is `u = (1+t)(y^2 z, z^2 x, x^2 y)`, `p = xyz + t x^2`, chosen so
that `d_x u1 = d_y u2 = d_z u3 = 0` makes it divergence free while advection and
the Laplacian are both nonzero. Every derivative is an AD call; the residual is
never hand-differentiated.

- Divergence of the reference field and of each rescaled field is an exact zero.
- The identity of (10.22), `R_nu(u_nu,p_nu)(x,t) = sqrt(nu) R_1(u,p)(x/sqrt(nu),t)`,
  holds as an exact rational equality at `nu = 1/4, 1, 9/16, 25/9, 4`, whose
  square roots are rational. Two negative controls confirm the check is not
  vacuous: omitting the `sqrt(nu)` on the velocity, or the `nu` on the pressure,
  makes the gap nonzero.
- At `nu = 2`, whose square root is irrational, the same identity holds to `1e-9`.
- The discrete energy and dissipation of (10.23) scale by exactly `nu^{5/2}` on
  the correspondingly scaled grid. These are Riemann sums, not integrals: the
  `nu` grid is the reference grid scaled by `sqrt(nu)`, so the identity is exact
  term by term, and that is what is certified.
- The time variable is untouched by the rescaling, so the time coordinate
  itself is unchanged.

### `mathematics_navier_stokes_similarity_scales.esk`

Concrete smooth profiles are installed that satisfy incompressibility (4.7), the
centrifugal pressure balance and the axis regularity (4.4), with the slightly
asymmetric axial profile of Section 2.1. They are **not** the profiles of
Theorem 4.6: nothing here makes the leading residual vanish on the inner region
or matches the heat exterior, so the residual is unbounded throughout the core
rather than only in the annulus.

- The divergence-free identity of (4.7) and the pressure balance `Pi_X = E^2/(2X)`
  hold at every sample point to `1e-9`, with a negative control on a perturbed
  `V0`.
- `q - z^2 q^{2h} = tau` is solved to `1e-14` by a fixed-point iteration whose
  contraction rate is `2 h eta^2`, and the five coordinate derivatives of
  Lemma 4.1 agree with AD through that solve to `1e-7`.
- The scale exponents come out as pure power laws and are fitted to `1e-9`:
  `l_r ~ tau^{1/2}`, `l_z ~ tau^{1/2-h}`, `l_r/l_z ~ tau^h`,
  `sup|u_theta|, sup|u_z| ~ tau^{-1/2-h}`, `sup|u_r| ~ tau^{-1/2}`,
  `Re_theta ~ tau^{-h}` (unbounded), `Re_r ~ tau^0`, and the axial-to-radial
  diffusion ratio `~ tau^{2h}` (vanishing).
- The core kinetic energy separates into a tangential part scaling as exactly
  `tau^{1/2-3h}` (fitted to `1e-9`) and a radial part scaling as `tau^{1/2-h}`.
  The total therefore carries a relative correction of size `tau^{2h}`, which
  biases the fitted slope of the total upward by at most `2h`. The verdict on the
  total energy uses that `2h` bound and additionally requires the fitted slope to
  decrease monotonically toward `1/2-3h`; the sharp statement is the tangential
  one. With the paper's `h < 1/100` the correction decays too slowly for a naive
  log-log fit of the total to converge, and the program says so by construction
  rather than by widening a tolerance without a reason.
- The leading angular and axial residuals, with axial viscosity removed as in
  (4.12), grow like exactly `tau^{-(1+A)}`: `q^{A+1} R_theta^{(0)}` is
  independent of `tau` to a relative `1e-6` over six decades, which is the
  self-similarity of the residual itself. The full residual, axial viscosity
  included, is within `2h` of the same exponent. The magnitude at `tau = 1e-8`
  exceeds `1e6`, so no smooth force can sustain this background alone.

### `mathematics_navier_stokes_first_principles.esk`

**Part A.** The exponents in `l_r ~ tau^a`, `l_z ~ tau^b`, `u_theta ~ tau^-c`,
`u_z ~ tau^-d`, `u_r ~ tau^-e` are unknowns. Five requirements from Sections 2.1
and 3.1 — radial diffusion and radial and axial transport all entering the
leading balance at rate `tau^-1`, the angular Reynolds number growing like
`tau^-h`, and the axial-to-radial diffusion ratio being `tau^{2h}` — form a
linear system whose right-hand sides are affine in `h`. It is solved exactly over
the rationals, twice, for the constant part and the `h` coefficient. The program
derives

```text
a = 1/2,   b = 1/2 - h,   c = 1/2 + h,   d = 1/2 + h,   e = 1/2
```

so `c = d = A` and `b = D`, and the compound exponents follow: core energy
`1/2 - 3h`, `Re_theta ~ tau^{-h}`, `Re_r ~ tau^0`, diffusion ratio `tau^{2h}`.
Incompressibility, `u_r/l_r ~ u_z/l_z`, is **not** imposed; it is checked
afterwards and is satisfied, which is what makes the system consistent.

The admissible range of `h` is then derived, not stated: each requirement is
"this exponent is strictly positive", an affine condition `p + qh > 0` that
gives `h > -p/q` when `q > 0` and `h < -p/q` when `q < 0`. Intersecting them
gives `0 < h < 1/6`. The program prints which requirement binds each end: the
lower end by the core becoming slender (`l_r/l_z -> 0`) and the upper end by the
core kinetic energy vanishing, `1/2 - 3h > 0`. The paper's `0 < h < 1/100` lies
strictly inside.

**Part B.** For the leading tangential residual (4.12) the identities

```text
q^{A+1} R_theta^{(0)} = (E/L) B_theta,     q^{A+1} R_z^{(0)} = B_z / L
```

hold exactly, with `B_theta` and `B_z` the profile operators of (4.13) built from
the sources (4.9). Both `E * B_theta` and `B_z` are linear in the profile they
act on, so the program evaluates each operator on the monomial basis `X^j` with
AD, recovers each image polynomial by exact rational interpolation (an exact
Vandermonde solve), and then solves, again exactly, for the coefficients that
annihilate every image coefficient below the truncation order. That derives the
profile series from the operator.

- The derived series annihilate their operators below the truncation exactly, and
  the truncation residual is an exact rational.
- The derived coefficient ratios are printed and match the Kummer recurrences of
  the two operators, identifying the profiles as `1F1(1+h; 2; X/2)` and
  `1F1(A; 1; X/2)`. The recurrences are a consequence of the derivation, not an
  input to it.
- Negative control: perturbing one interior coefficient of either series leaves a
  nonzero coefficient below the truncation.
- Pushed back through the physical residual, assembled by AD in `(r,z,t)` through
  the implicit solve for `q`, the derived profiles make `q^{A+1} R_theta^{(0)}`
  vanish on `z = 0` to `1e-7`, while the perturbed profile leaves a nonzero
  `tau`-independent value and a physical residual that diverges like
  `tau^{-(1+A)}` with the fitted exponent matching to `1e-4`.

All of Part B is evaluated at `eta = 0` with profiles independent of `eta`. The
`eta`-derivatives that appear in (4.9) are still computed rather than dropped;
they evaluate to zero there, and that is what makes the reduced balance a pair of
Kummer equations.

### `mathematics_navier_stokes_pulse_stress.esk`

- Both pulse families have zero angular mean in every cylindrical component
  (`< 1e-12` by discrete angular averaging over a full ring), while the averaged
  products `<w_r w_theta>` and `<w_r w_z>` are nonzero and agree with the exact
  rational values `a_i a_j / 2` to `1e-12`. Reversing both components of a flux
  leaves the flux unchanged, which is the mechanism of Section 2.2.
- The cylindrical divergence of each pulse, computed by AD, reduces to the `O(1)`
  envelope term `w_r/r`: the term of the order of the wavenumber cancels exactly
  under the transversality `a . k = 0`. A negative control detunes `k_z` and
  leaves an `O(wavenumber)` divergence.
- The two covariance vectors are linearly independent — an exact rational
  determinant `19/80` — so the 2x2 system spans both required stress components.
  A target inside the cone has strictly positive weights and is reproduced
  exactly by the positive combination; a target outside it has a negative weight.
  All of this is written out over the scalar exact tower on Scheme lists of
  rationals; it does not go through the tensor `solve`/`det` path, which is
  f64-backed.
- The Craik-Criminale mode on an affine background: the closed-form wavevector
  satisfies `k' = -S^T k` as an exact rational identity under AD, the AD gradient
  of the production quadratic form equals `-(S + S^T) a`, the wave stays
  transverse along the integration, and the amplitude grows by a factor above
  `1.5`, peaks at an interior time near the minimum of `|k|^2`, then decays below
  both the maximum and its initial value as the shear shortens the radial
  wavelength and viscous damping overtakes the amplification.

### `mathematics_navier_stokes_stress_cone.esk`

The stress T0 is constructed, not assumed: with `eta = 0` and profiles
independent of `eta`, (4.9) collapses to `S_q = -l - h` and `S_n = -X U' - A U`,
the radial equations (4.10) are integrated in exact rational arithmetic, and
Q_s is checked against its own radial equation (4.9). Lemma 4.5's cone
equivalence and threshold are certified without ever evaluating a square root:
the relaxed and admissible cone tests are exact rational sign tests on the
quadratic `Pq(v) = 2(P_c-v)^2 - (v-2)J_c^2`, whose root structure (leading
coefficient positive, `Pq(2) > 0`, `Pq(P_c) <= 0`) is itself certified rather
than assumed; the floating-point value of `U(P_c,J_c)` is computed only to be
compared against the exact decision, never to gate a verdict.

Part D's threshold `P_K` is established once, over a parameter sample K, and
Part E reuses that SAME derivation for the physical `(a, -b_s)` of the
constructed base flow (added to K with `w = 0`) rather than re-deriving a
second, disconnected argument: with `p_s = (P_K * factor, 0)`, the constructed stress lands inside
the admissible cone at every sampled radius, the cone decision stays exact
rational throughout, is homogeneous under rescaling, and a separate radius with
`v_s <= 2` demonstrates the relaxed cone (the connecting interval) is genuinely
weaker than the admissible one. Negative controls: a rescaled `Q_s` breaks its
own radial equation; a stress with `T_theta + t_s T_z <= 0` or a
transverse-heavy stress is refused; a target far below `P_K` is refused.

### `mathematics_navier_stokes_residual_order_n.esk`

Section 5.1's equations (5.1)-(5.6) are a linear recursion: at each order `n`
the same fixed operator is solved against a forcing built from already-
constructed lower orders. This is mechanized on a scalar model of that
skeleton — a fixed operator `c(X)` and a known forcing sequence `f_n(X)` — with
`y_n(X) = f_n(X)/c(X)` solving `L[y_n] = f_n` exactly at each of `N = 4`
orders. Substituting the truncated expansion `y(tau,X) = sum_{n<N} y_n(X)
tau^n` into the model residual `R = c*y - F` and reading off the coefficient
of each power of `tau` with `taylor`/`derivative-n` at the exact point `tau =
0` gives orders `0..N-1` vanishing exactly and order `N` equal to `-f_N(X)`
exactly (the background residual beyond leading order, nonzero in general —
why Section 5 keeps going to the next order). Negative controls: perturbing
the order-`k` correction leaves a nonzero coefficient at EXACTLY order `k`,
every other order (including the order-`N` tail) untouched, for `k = 0, 2, 3`;
the unperturbed (`delta = 0`) case recovers the all-orders-vanish result.

### `mathematics_navier_stokes_heat_exterior.esk`

For a purely azimuthal, height-independent field, the momentum equation
reduces to the curvature-corrected radial heat equation `d_t v = nu(d_r^2 v +
(1/r) d_r v - v/r^2)`. Its operator satisfies the closed form `L[r^n] =
(n^2-1) r^{n-2}`, so an odd polynomial in r with time-dependent coefficients
solves it exactly whenever those coefficients satisfy the resulting triangular
linear recursion in t — verified by AD at four `(r,t)` samples, with a
negative control that perturbs one coefficient without its coupled partners.
Appendix A's Lemma A.1 (distinct power weights give an invertible moment
matrix) is certified exactly for a 2x2 case (two polynomial bumps, weights
`x^0` and `x^2`): the moment matrix is exactly invertible, solving it exactly
reproduces a prescribed discrepancy, and colliding weights give a singular
matrix. The one moment that needs a transcendental profile piece is computed
to a stated numerical tolerance (mesh refinement, not `=`), never asserted
exact — the exactness boundary Appendix A itself does not need to cross for
polynomial pieces. The exterior stays finite as `t -> 1` at fixed radius,
contrasted against a deliberately wrong candidate carrying the inner profile's
own `(1-t)^{-1}`-type divergent factor, which grows without bound.

### `mathematics_navier_stokes_oscillatory_realization.esk`

Two pulse families are built as trigonometric polynomials on a 4-point
auxiliary torus, where `cos`/`sin` of the sample angles `k pi/2` are the exact
integers `{1,0,-1,0}`/`{0,1,0,-1}` (a lookup table — no transcendental call
anywhere in this file). Each family's radial, azimuthal and axial components
are in-phase with distinct amplitude ratios, matching Section 2.2's "different
ratios of radial angular-momentum flux to radial axial-momentum flux": each
family's velocity has exactly zero angular mean (`torus-average`), and the
quadratic momentum-flux products' zero mode — extracted BY `torus-average`,
not asserted — matches `A*B/2` and `A*C/2` exactly. The stacked 2x2 flux
matrix is exactly invertible, and `core.exact_linalg`'s `exact-solve` finds
POSITIVE weights realizing a prescribed target stress (both families
contributing constructively, i.e. inside the cone). Negative controls: one
family alone has `exact-rank` 1, and being non-square cannot realize a general
two-component target (`exact-solve` raises rather than fabricating a value).

### `mathematics_navier_stokes_pulse_growth.esk`

For a simple-shear background `U = (Sy, 0)`, the Craik-Criminale wavevector
equation `dk/dt = -(grad U)^T k` gives `k_theta` constant and `k_r(t) =
k_r(0) - S k_theta t` exactly affine in t — verified against the CL ODE by AD
at five sampled t, with a negative control on a perturbed law. A
representative growth-rate model `rate(t) = G0 - lambda t - nu|k(t)|^2`
(amplification proportional to shear, weakened linearly by the changing
orientation, opposed by viscous dissipation growing with `|k(t)|^2`) is
positive initially and eventually negative, monotonically decreasing, with the
crossover bisected to an exact rational bracket under `1e-9` width — no square
root, no `exp`. The log-amplitude `P(t)`, an exact rational cubic, is verified
to be the exact AD antiderivative of `rate(t)`, and `exp(P(t))` is
cross-checked against an independently RK4-integrated amplitude curve to
`1e-3` relative, with the numerically-integrated curve's peak matching the
exact crossover to within one integration step. The tail decays (`P(t)` very
negative at large t). Negative control: zero shear collapses the model's
amplification terms and gives pure viscous decay with no crossover.

### `mathematics_navier_stokes_mean_corrections.esk`

Section 8 builds compactly supported, divergence-free mean corrections and
solves five radial moment conditions (Lemma 8.7). Two facts are mechanized and
then tied together.

- **Divergence-free by construction, checked rather than trusted.** A
  compactly supported, piecewise-polynomial vector potential is built from
  guarded bumps, its "curl" is written out, and `div` of the resulting field
  is an exact zero by `derivative-n` at rational interior points of each
  azimuthal and axial box, and far from a box. The support cutoff itself is
  checked: the field is exactly zero outside every box and nonzero inside.
- **The five-moment solve.** With geometric bump copies
  `eta_j(x) = a_j^{-1} eta_0(x/a_j)`, `a_j = r^j`, whose power moments are
  exactly `a_j^p`, the three azimuthal and two axial moment conditions at
  `lambda = 1` (exponents `(2, -4, -2)` and `(1, -1)`) form generalized
  Vandermonde systems. `exact-det` certifies them nonsingular and `exact-solve`
  (`core.exact_linalg`) returns the amplitudes in exact rationals; at
  `60f345def` they are `(u0 u1 u2) = (1949/945, -3260/189, 3776/945)` and
  `(s0 s1) = (23/8, -21/8)`, and `M u + T = 0` holds exactly for both blocks.
- The solved amplitudes weight the five bumps of the first part, so the
  realized correction field is the one that cancels the defect; it is again
  exactly divergence free by AD and still compactly supported. Negative
  controls: a correction placed on the wrong support leaves the nonzero
  residuals `(-3776/135, 295/8748, 236/1215)` and `(21/8, -7/16)`; an oversized
  cutoff is nonzero where the correct one vanishes.

### `mathematics_navier_stokes_residual_ladder.esk`

Section 9 improves the residual stage by stage, with residual exponents on the
ladder of (9.8) and (9.18).

- **The exponent ladder** `sigma_0 = 1/5`, `sigma_{j+1} = sigma_j + 1/10`,
  `B_j = 1/2 + sigma_j`, `C*_j = 1 + sigma_j` is checked in exact arithmetic for
  `j = 0..5`, together with the closed form and strict, unbounded growth
  (`sigma_20 > 2`); the run prints `sigma_0..sigma_3 = (1/5, 3/10, 2/5, 1/2)` and
  `B_0..B_3 = (7/10, 4/5, 9/10, 1)`.
- **The coefficient ladder** is a scalar model of the exact increment identity
  `R(u_B + w, pi_B + p) = R(u_B, pi_B) + L_{u_B}(w, p) + div(w (x) w)`: a linear
  operator `2w` and a quadratic remainder `w^2` play the roles of `L_{u_B}` and
  `div(w (x) w)` against a base residual with nonzero coefficients through order
  4. Each rung probes the map from the new correction coefficient to the
  order-`j` Taylor coefficient at two points (it is affine, by the splitting
  argument), interpolates the root exactly, and re-extracts the coefficient with
  `taylor` to confirm it is the exact rational zero. The solved coefficients are
  `c1 = 3/2`, `c2 = -1/8`, `c3 = 43/16`; after three rungs orders 1–3 vanish and
  the order-4 coefficient `59/64` remains, as it should.
- Negative controls skip each rung in turn and recover exactly the
  pre-correction coefficient at that order.

### `mathematics_navier_stokes_localization.esk`

Section 10 localizes the construction with smooth cutoffs, proves the energy
identity, and transports it to the torus (Corollary 10.6). Four independent
facts, each with a negative control, supply three CTest criteria
(`ns_localization_smooth_extension`, `ns_energy_dissipation_bounds`,
`ns_torus_corollary_scaling`).

- **A cutoff cancels a temporal pole.** The uncut field `V(r,t) = r^2/(1-t)^2`
  has an order-2 pole at `t = 1`; the cutoff `chi(t) = (1-t)^3 (w^2-(t-1)^2)^2`
  vanishes to order 3, so the product is a polynomial and its Taylor
  coefficients about `t = 1` at `r = 3` are exact rationals, `(0, -9/16, 0, 9/2, 0)`.
  The uncut field grows as `900, 90000, 9000000` at `t = 9/10, 99/100, 999/1000`;
  an under-cancelling cutoff still leaks unbounded growth.
- **Compact support.** The spatial part of the cutoff force is exactly zero
  outside its box at every sampled time, nonzero inside, and zero at `t = 1`;
  an oversized cutoff leaks outside the box.
- **Periodization onto `T^3 = R^3/Z^3`.** For support radius `1/5 < 1/2` the
  only nonzero translate inside the fundamental cell is the untranslated one,
  checked exactly, and `torus-average-2d` (`core.exact_linalg`) gives the same
  value, `1/3906250000`, for the periodized and the single-cell field; an
  oversized radius lets aliased translates leak in.
- **The energy identity.** For a discrete model with the force defined, as in
  Proposition 10.1, to be the residual `f_k = d_t u_k + nu u_k`, the identity
  `d/dt (1/2 sum u_k^2) = <f, u> - nu ||u||^2` holds exactly by AD in `t`:
  both sides are `26/9` at `t0 = 1/3`; dropping the viscous term gives `338/135`.

## Running them

<!-- example-catalogue:ns-commands:start -->
Prerequisites and scope are stated in the [complete mathematics catalogue](MATHEMATICS_EXAMPLES.md). Run from the repository root after building the compiler and stdlib.

```bash
mkdir -p .scratch/example-manual
./build/eshkol-run -r examples/mathematics_navier_stokes_first_principles.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_first_principles.esk -o .scratch/example-manual/mathematics_navier_stokes_first_principles && .scratch/example-manual/mathematics_navier_stokes_first_principles

./build/eshkol-run -r examples/mathematics_navier_stokes_heat_exterior.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_heat_exterior.esk -o .scratch/example-manual/mathematics_navier_stokes_heat_exterior && .scratch/example-manual/mathematics_navier_stokes_heat_exterior

./build/eshkol-run -r examples/mathematics_navier_stokes_localization.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_localization.esk -o .scratch/example-manual/mathematics_navier_stokes_localization && .scratch/example-manual/mathematics_navier_stokes_localization

./build/eshkol-run -r examples/mathematics_navier_stokes_mean_corrections.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_mean_corrections.esk -o .scratch/example-manual/mathematics_navier_stokes_mean_corrections && .scratch/example-manual/mathematics_navier_stokes_mean_corrections

./build/eshkol-run -r examples/mathematics_navier_stokes_oscillatory_realization.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_oscillatory_realization.esk -o .scratch/example-manual/mathematics_navier_stokes_oscillatory_realization && .scratch/example-manual/mathematics_navier_stokes_oscillatory_realization

./build/eshkol-run -r examples/mathematics_navier_stokes_pulse_growth.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_pulse_growth.esk -o .scratch/example-manual/mathematics_navier_stokes_pulse_growth && .scratch/example-manual/mathematics_navier_stokes_pulse_growth

./build/eshkol-run -r examples/mathematics_navier_stokes_pulse_stress.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_pulse_stress.esk -o .scratch/example-manual/mathematics_navier_stokes_pulse_stress && .scratch/example-manual/mathematics_navier_stokes_pulse_stress

./build/eshkol-run -r examples/mathematics_navier_stokes_residual_ladder.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_residual_ladder.esk -o .scratch/example-manual/mathematics_navier_stokes_residual_ladder && .scratch/example-manual/mathematics_navier_stokes_residual_ladder

./build/eshkol-run -r examples/mathematics_navier_stokes_residual_order_n.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_residual_order_n.esk -o .scratch/example-manual/mathematics_navier_stokes_residual_order_n && .scratch/example-manual/mathematics_navier_stokes_residual_order_n

./build/eshkol-run -r examples/mathematics_navier_stokes_similarity_scales.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_similarity_scales.esk -o .scratch/example-manual/mathematics_navier_stokes_similarity_scales && .scratch/example-manual/mathematics_navier_stokes_similarity_scales

./build/eshkol-run -r examples/mathematics_navier_stokes_stress_cone.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_stress_cone.esk -o .scratch/example-manual/mathematics_navier_stokes_stress_cone && .scratch/example-manual/mathematics_navier_stokes_stress_cone

./build/eshkol-run -r examples/mathematics_navier_stokes_viscosity_scaling.esk
./build/eshkol-run -L./build examples/mathematics_navier_stokes_viscosity_scaling.esk -o .scratch/example-manual/mathematics_navier_stokes_viscosity_scaling && .scratch/example-manual/mathematics_navier_stokes_viscosity_scaling

ctest --test-dir build --output-on-failure -R '^ns_'
```

The criterion table above lists the `_jit` and `_aot` variants; the measured outcome of each, at `60f345def`, is in the table's last column and in each program's entry in the [mathematics catalogue](MATHEMATICS_EXAMPLES.md).
<!-- example-catalogue:ns-commands:end -->

## Exactness boundaries

- Exact arithmetic here is mostly **scalar**. Most exact systems in this
  family — the exponent solve, the Vandermonde interpolation, the
  profile-coefficient solve, the stress-cone threshold and identities — are
  written out over the scalar exact tower on Scheme lists of rationals.
  The oscillatory-realization, mean-correction and localization programs also
  use `core.exact_linalg` for their exact solves or torus averages. Its
  `exact-solve`/`exact-rank`/`torus-average` operations are scalar-exact —
  Eshkol tensors remain f64-backed, so `exact_linalg`'s matrices are vectors
  of row-vectors, never tensors.
- Derivatives are exact at exact scalar points. Earlier rounds of this family
  found four additional exactness leaks in `derivative`/`derivative-n`/`taylor`
  beyond the three closed SW-148/149/150 defects: a top-level `define`d
  constant (vs. an inline literal) in the differentiand, a loop-derived point
  argument, a several-deep composed differentiand, and a function-call
  expression passed directly as the differentiand alongside another
  `derivative`/`taylor` call site in the same file causing a hard compile
  failure. All four are now fixed and no longer reproduce; the affected
  programs' AD surface notes record this in the past tense. Where the field carries irrational
  data (`q^{2h}`, `sqrt(2X)`, the trigonometric pulses, the implicit solve for
  `q`, the RK4-integrated amplitude curve), the arithmetic is double precision
  and the verdicts state a tolerance.
- This family does not invoke `core.ad.interval` or `core.ad.taylor_models`.
  Read each verdict according to the construction it checks: an exact
  algebraic identity, an explicitly constructed finite object, a truncated
  series identity, or a numerical comparison with a stated tolerance. The
  numerical comparisons do not establish a uniform interval bound.
- All derivatives are automatic, not symbolic. `(diff expr var)` exists and
  returns a quoted S-expression, but it is not used in this family; every
  derivative printed or compared comes from `derivative`, `derivative-n`,
  `gradient` or `taylor`.

## Compiler defects found while building this family

All defects this section used to describe in detail are closed: SW-148
(an exact operand stealing the scalar dispatch from a live jet), SW-149 (a
vanishing tangent returning an inexact zero at an exact rational seed) and
SW-150 (a captured outer variable lost by an inner differentiation pass) are
pinned by `tests/ad/exact_rational_derivative_test.esk` and
`tests/ad/nested_operator_matrix_test.esk`, and the four programs from the
first round of this family now call `derivative`/`derivative-n` directly at
their own call sites instead of routing around them. The four additional
exactness leaks a later round of this family found — see "Exactness
boundaries" above — are also fixed and no longer reproduce.

## Not mechanized yet

These are capability gaps, stated as such.

- **Exact-element tensors.** Eshkol tensors are f64-backed
  (`(tensor 1/3)` prints `#(0)`), so every exact system in this family is
  written out over the scalar exact tower on Scheme vectors/lists, or (for
  `mathematics_navier_stokes_oscillatory_realization.esk`) through
  `core.exact_linalg`'s vector-of-row-vectors matrices — never through the
  tensor linear-algebra path. An exact tensor element type would let these be
  one call each.
- **Certified enclosures with directed rounding.** Every tolerance in this
  family is a numerical agreement at sampled points, not a bound over a
  region, so no statement here is a certified bound. The capability itself is
  no longer missing: v1.3.5 ships the directed-rounding builtins `fl-next-up` /
  `fl-next-down` on both engines, outward-rounded interval arithmetic
  (`core.ad.rigorous_interval`) and Makino-Berz rigorous Taylor models whose
  remainder is **derived** rather than sampled
  (`core.ad.rigorous_taylor_models`, with `tm-bound` / `tm-prove-bound` /
  `tm-prove-nonzero`) — see
  [reference/stdlib/certified-enclosures.md](reference/stdlib/certified-enclosures.md).
  What remains is routing these programs' numerical tolerances through that
  layer. The examples currently use direct exact-equality checks or sampled
  numerical comparisons; they do not call either `core.ad.interval` or
  `core.ad.taylor_models`.
- **Symbolic series values for the residual to every order.** Only a scalar
  model of the order-by-order recursion is closed here
  (`mathematics_navier_stokes_residual_order_n.esk`), not the paper's actual
  second-order elliptic system near the axis at every order. That needs a
  residual that returns a graded value whose coefficients can be collected
  and solved — symbolic multivariate polynomial and series *values* and
  polynomial-valued duals, neither of which exists today.
- **Proof-object export.** Nothing here emits a certificate that could be
  checked without running Eshkol. Every verdict above rests on trusting this
  compiler, and that trust is not transferable.
