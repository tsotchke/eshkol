# Navier-Stokes finite-time blowup: mechanizing the leading structure

Status: verified on macOS arm64 with native JIT (`-r`) and native AOT. The four
programs are discovered by `scripts/run_examples_tests.sh`, which runs on the
lite CI lanes, and each is additionally pinned by a JIT and an AOT CTest entry.

Source: "Finite Time Blowup for Navier-Stokes" (OpenAI, 2026),
<https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf>

These programs do not reprove the theorem. They compute the *leading structure*
of the construction mechanically, where a reader would otherwise differentiate,
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

| Program | Paper sections | What it computes | Checks |
|---|---|---|---|
| [`mathematics_navier_stokes_viscosity_scaling.esk`](../examples/mathematics_navier_stokes_viscosity_scaling.esk) | 3 (outline), 10.4 equations (10.22)-(10.23) | The residual `R(u,p) = d_t u + (u.grad)u - nu Lap u + grad p` assembled from AD partials of an explicit smooth divergence-free polynomial field, and the viscosity rescaling identity verified as an exact rational equality at four rational viscosities, with the energy and dissipation identities | 9 |
| [`mathematics_navier_stokes_similarity_scales.esk`](../examples/mathematics_navier_stokes_similarity_scales.esk) | 2.1, 3.1, 4.1 (Lemma 4.1, (3.2), (4.3), (4.7)) | The self-similar axisymmetric ansatz in similarity coordinates: incompressibility and the centrifugal pressure balance as AD identities, the coordinate derivatives of Lemma 4.1 differentiated through the implicit solve for `q`, the nine scale-law exponents fitted from computed quantities, the core kinetic energy exponent, and the unbounded background residual | 23 |
| [`mathematics_navier_stokes_first_principles.esk`](../examples/mathematics_navier_stokes_first_principles.esk) | 2.1, 3.1, 4.1 ((4.9), (4.12), (4.13)) | The scale exponents derived, not assumed: a linear system in the five unknown exponents with one free parameter `h`, solved exactly over the rationals, followed by the admissible range of `h` derived from the positivity requirements; then the leading profile series derived from the leading balance operator by exact interpolation and an exact linear solve, with a negative control | 21 |
| [`mathematics_navier_stokes_pulse_stress.esk`](../examples/mathematics_navier_stokes_pulse_stress.esk) | 2.2, 3.2 (Figure 4), 3.3 and 7 | The oscillatory ring pulses: zero angular means, nonzero averaged momentum fluxes, the leading transversality that makes them divergence free, the two-family covariance system solved in exact rational arithmetic with positivity, and the Craik-Criminale growth-then-decay of a single mode on an affine background | 20 |

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
- The time variable is untouched by the rescaling, so the singular time is
  unchanged.

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

## Running them

```bash
cmake -S . -B build -G Ninja -DESHKOL_BUILD_TESTS=ON
cmake --build build -j8

# JIT
./build/eshkol-run -r examples/mathematics_navier_stokes_viscosity_scaling.esk
./build/eshkol-run -r examples/mathematics_navier_stokes_similarity_scales.esk
./build/eshkol-run -r examples/mathematics_navier_stokes_first_principles.esk
./build/eshkol-run -r examples/mathematics_navier_stokes_pulse_stress.esk

# AOT
./build/eshkol-run -o build/ns_vs examples/mathematics_navier_stokes_viscosity_scaling.esk
./build/ns_vs
```

The examples suite discovers the four files automatically:

```bash
./scripts/run_examples_tests.sh
```

The CTest entries, four JIT and four AOT, are named for the criterion ids used by
the mechanization design document:

```bash
ctest --test-dir build --output-on-failure -R '^ns_'
```

`ns_viscosity_scaling_exact`, `ns_similarity_exponents_solved`,
`ns_leading_profile_balance` and `ns_covariance_two_family_solve`, each with a
`_jit` and an `_aot` variant, pin the verdict line in both execution modes.

## Exactness boundaries

- Exact arithmetic here is **scalar**. Eshkol tensors are f64-backed, so every
  exact system in this family — the exponent solve, the Vandermonde
  interpolation, the profile-coefficient solve, the 2x2 covariance solve — is
  written out over the scalar exact tower on Scheme lists of rationals, not
  through the tensor linear-algebra path.
- Derivatives are exact only at exact scalar points and only through
  `derivative-n`; see the defect list below. Where the field carries irrational
  data (`q^{2h}`, `sqrt(2X)`, the trigonometric pulses, the implicit solve for
  `q`), the arithmetic is double precision and the verdicts state a tolerance.
- No enclosure or interval bound is used anywhere in this family, and nothing
  here should be read as a certified bound. `core.ad.interval` widens by a
  relative epsilon rather than using directed rounding, and
  `core.ad.taylor_models` bounds its remainder by a sampled derivative; neither
  is rigorous today, and neither is invoked here.
- All derivatives are automatic, not symbolic. `(diff expr var)` exists and
  returns a quoted S-expression, but it is not used in this family; every
  derivative printed or compared comes from `derivative`, `derivative-n`,
  `gradient` or `taylor`.

## Compiler defects found while building this family

Each is reproduced by the one-liner given with it. All three are silent: a wrong
value is returned and nothing is raised.

1. **`derivative` returns 0 when a non-integer exact rational reaches the
   differentiand** — as a literal, as a captured argument, or as the seed.
   `(derivative (lambda (s) (* 1/2 s s)) 0.4)` gives `0` where `0.4` is expected;
   `(derivative-n ... 1)` and `(taylor ...)` on the same differentiand at the
   same point are correct. This contradicts the identity
   `(derivative f x) == (derivative-n f x 1)` documented in
   `docs/guide/AUTOMATIC_DIFFERENTIATION.md`. Repro:
   `(derivative (lambda (s) (* 1/2 s s)) 0.4)` returns `0`.
2. **A derivative whose differentiand does not depend on the seed variable
   returns an inexact zero at an exact rational seed**, which demotes any exact
   sum it enters — a divergence, a Laplacian, a residual. A derivative that
   merely happens to vanish is exact, so the trigger is a constant differentiand,
   not a zero value.
   Repro: `(exact? (derivative-n (lambda (s) 1/3) 1/2 1))` is `#f`.
3. **Nested differentiation returns 0 when the outer pass is `derivative-n`, or
   when the inner pass is `taylor`.** `derivative` nests correctly, including
   three levels deep. `docs/guide/AUTOMATIC_DIFFERENTIATION.md` section 11 states
   that nested differentiation is safe. Repro:
   `(derivative-n (lambda (a) (derivative-n (lambda (b) (* a b)) 1.0 1)) 2.0 1)`
   returns `0` where `1` is expected.

Defects 1 and 3 together leave no single operator that is correct both with exact
rationals present and as the outer pass of a nested differentiation. That
constraint shaped these programs: exact rational work uses `derivative-n` and
never nests, while the physical residuals are pure double arithmetic and use
`derivative`. Each program carries a defect note at the top saying which route it
takes and why.

## Not mechanized yet

These are capability gaps, stated as such.

- **The residual to every order.** Only the leading order is closed here. The
  paper's Section 5 corrections in powers of `q^{2nh}` need a residual that
  returns a graded value whose coefficients can be collected and solved, which in
  turn needs symbolic multivariate polynomial and series *values* and
  polynomial-valued duals. Neither exists today; the leading balance above is
  reached instead by evaluating a linear operator on a monomial basis and
  recovering its image by exact interpolation.
- **Certified derivative bounds through `t = 1`.** Every tolerance in this family
  is a numerical agreement at sampled points, not a bound over a region. Bounding
  a Cartesian space-time derivative uniformly on a compact set needs
  directed-rounding interval arithmetic and a proved Taylor-model remainder;
  Eshkol's interval arithmetic widens by a relative epsilon and its Taylor-model
  remainder is sampled, so no statement here is a certified bound.
- **Torus averaging as a builtin.** The angular mean is a discrete sum over a
  ring in `mathematics_navier_stokes_pulse_stress.esk`. The construction needs
  `T^1`/`T^2`-valued fields, a Haar mean with full chain-rule propagation,
  evaluation at a phase map, and support-disjointness bookkeeping.
- **Exact linear algebra as a library.** The Gaussian elimination, the
  Vandermonde solve and the 2x2 covariance solve are written out in Scheme over
  the exact scalar tower because the tensor path is f64-backed. An exact tensor
  element type would let these be one call.
- **Proof-object export.** Nothing here emits a certificate that could be checked
  without running Eshkol. Every verdict above rests on trusting this compiler,
  and that trust is not transferable.
