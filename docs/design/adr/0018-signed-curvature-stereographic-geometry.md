# ADR 0018: Signed-curvature stereographic geometry and the K = 0 execution contract

- **Status:** Accepted (maintainer ruling 2026-08-30); implemented in v1.4.0-connection (see Implementation)
- **Date:** 2026-08-30
- **Renumbered:** was ADR 0012 until 2026-09-17; renumbered to give every record a unique id.
- **Decision owners:** geometry core (`riemannian_core.h`), VM geometry, qLLM bridge, AD, Moonlab integration maintainers
- **Depends on:** the shared geometry core established by the VM-geometry and squared-distance work (SW-73 lineage); ADR-0002 (external oracle) for reference grids
- **Scope:** the curvature parameter and chart used by every constant-curvature primitive (distance, exp/log maps, parallel transport, conformal factor, curvature derivatives, geodesic attention), the behaviour at and through `K = 0`, and the serialization/migration contract

## Context

Eshkol's constant-curvature primitives grew up with a radius parameterization: the negative-curvature branch tends to the metric `4I` as `K -> 0-`, exact zero executes the Euclidean metric `I`, and the positive branch uses ambient-sphere coordinates rather than the same chart. Curvature first and second derivatives, and geodesic attention, are exposed to automatic differentiation, so the discontinuity at zero is an AD contract defect, not a cosmetic one: adjoints near zero are not the derivative of a single executed function, and the negative-curvature attention scale contains `1/sqrt(-K)`, which diverges as `K -> 0-` and is replaced by `1` at exactly zero. qLLM persists coordinates, curvature schedules and product-space factors that depend on the convention. The independent verification rounds on the geometry PRs established the exponent-range and forward/reverse failure classes that any convention must pass. An analysis (Selene, `research/k0/K0-CONVENTION-SURVEY.md`) derived the exact formulas, regularity and numerical behaviour of three options and recommended the first; the maintainer ruled for it.

## Decision

1. `K` always means **signed sectional curvature**. Negative is hyperbolic, positive is spherical, zero is Euclidean. The legacy positive magnitude `c` is not used in public APIs.
2. Every public constant-curvature point API uses **one signed-curvature stereographic chart** for all signs (the κ-stereographic model of Bachmann, Bécigneul and Ganea), so the family is real-analytic through `K = 0` at fixed chart coordinates.
3. The normalization is fixed everywhere to the **canonical-flat gauge `alpha = 1`, `rho = K/4`**:

   ```
   g_K(x) = I / (1 + K ||x||^2 / 4)^2
   ```

   Consequences: every existing exact `K = 0` Euclidean result is preserved (distance `||x - y||`, unscaled Riemannian gradients); the hyperbolic ball radius becomes `2 / sqrt(-K)`; the spherical chart has an explicit pole and cut locus.
4. `K = 0` is an **analytic branch evaluated by curvature jets** (value, first and second `K` derivatives from the series with a bounded switch, see Numerics), not a special unrelated metric. Left and right `K` jets agree at zero.
5. **Raw distance has no endpoint backward at coincidence**; it refuses with a diagnostic. Geodesic attention scores use **squared distance** `-d^2 / (2 t)` with a finite temperature in squared-distance units — no `1 / sqrt(|K|)` scale anywhere.
6. Positive cut loci and negative ball boundaries are **explicit domain errors**, never clamped.
7. All serialized geometry (checkpoints, curvature schedules, product factors, attention temperatures) carries a **convention/version id**; an unversioned or mismatched checkpoint is refused, never guessed.
8. Native, VM and bridge forwards call the **same core**; every reverse rule differentiates the executed branch.

### Numerics

Near zero the generalized trigonometric ratios (`tan_K`, `artan_K`, and the conformal factor) are evaluated in a dimensionless variable `q = K ||x||^2 / 4`; for `|q| <= 2^-8` a degree-10 series branch is used with the survey's explicit error bound, otherwise the closed form. Products and denominators use the exponent-separated, compensated forms already required by the geometry gates, so every finite `K` binade for both signs, all subnormal binades, `DBL_TRUE_MIN`, the largest finite `K`, and exactly zero are representable and differentiable.

## Alternatives rejected

- **Refuse `K = 0`** (radius parameterization kept): coherent, but incompatible with learnable sign crossing and the universal geometric attention family; requires a separate Euclidean type.
- **Keep the discontinuity**: preserves published values but leaves curvature AD undefined at the limit — unacceptable for a capability advertised as differentiable in curvature.
- **Unified model with `alpha = 2`** (keeps radius `1 / sqrt(-K)`): coherent, but exact `K = 0` distance becomes `2 ||x - y||`, Riemannian gradients divide by four, and attention temperatures change even for exact-zero users.
- **Clamp `|K|` to epsilon**: differentiates a clamped function and creates a dead zone.

## Migration (v1.4)

1. Introduce `GEOMETRY_CONVENTION_STEREOGRAPHIC_V2_ALPHA1` alongside a legacy reader; unversioned data is never reinterpreted.
2. Add the shared helpers for `rho`, `lambda`, the generalized ratios, and value/first/second jets to `riemannian_core.h`; add the positive stereographic forward and its domain checks.
3. Convert legacy Poincaré coordinates by `x_v2 = 2 x_v1`; transform tangent vectors and optimizer state by the differential of that map.
4. Recompute or migrate attention temperatures into squared-distance units.
5. Route qLLM fp32 geometry through the versioned core, or declare it a separate approximate backend with documented parity limits.
6. Change geodesic attention to squared distance; add intrinsic Fréchet aggregation as a separately gated change.
7. Deprecate creation under the legacy convention; keep read-only checkpoint import for one release window.

The migration is a dedicated PR after the in-flight geometry PRs merge; it is not folded into them.

## Required gates

- **Reference grid** (binary128 or MPFR, >= 113-bit): every finite `K` binade of both signs, all subnormal binades, `DBL_TRUE_MIN`, the largest finite `K`, exactly zero; fixed chart coordinates near zero plus radius-scaled points at fixed fractions of the negative ball boundary; positive points approaching the pole and cut locus; coincidence, one-ulp separations, exact and near antipodes, orthogonal subnormals, mixed `1e300 / 1e-300` scales, non-finite refusals; value, endpoint Jacobian, first and second `K` derivatives, and mixed endpoint/`K` directional derivatives.
- **Identity and AD:** native/VM/bridge bit identity for shared f64 outputs and refusal class; exp/log round trips where unique; transport isometry and destination tangency; left and right `K` jets equal at zero; reverse adjoints equal the derivative of the executed forward; forward-over-reverse agrees with binary128 jets; squared distance has zero endpoint gradient at coincidence; raw distance backward refuses at coincidence.
- **Attention:** finite logits and gradients across every `K` binade and zero; continuous weights and outputs through zero at fixed chart inputs; no curvature-singular temperature; identical causal mask and stabilized softmax in forward and reverse; universal-geometric-attention flat-reduction and Fréchet-adjoint goldens.
- **Serialization:** convention id and `alpha` mandatory; legacy-to-v2 coordinate, tangent, optimizer and temperature migration with round-trip fixtures; mismatched checkpoints refused.

## Documentation changes

`docs/reference/stdlib/geometry.md` (replace the discontinuity section with the chosen metric, domain, `K = 0` jets, cut-locus rules and migration note); `docs/api/backend/riemannian_core.md` (signed generalized functions and error bounds); `docs/api/bridge/qllm_bridge.md` and its public header (signed `K` versus legacy `c`, convention ids); qLLM model/checkpoint documentation (chart, `alpha`, signed `K`, factor type, temperature units); the Universal Geometric Attention note (bind its flat reduction to `alpha = 1`, remove the curvature-singular scale); release notes (changed numerical values and a deterministic migration command).

## References

Bachmann, Bécigneul, Ganea, *Constant Curvature Graph Convolutional Networks* (ICML 2020, arXiv:1911.05076); Skopek, Ganea, Bécigneul, *Mixed-curvature Variational Autoencoders* (ICLR 2020, arXiv:1911.08411); Ungar, *Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity* (2008); Nickel & Kiela, *Poincaré Embeddings* (NeurIPS 2017, arXiv:1705.08039) and *Learning Continuous Hierarchies in the Lorentz Model* (ICML 2018, arXiv:1806.03417); Gu, Sala, Gunel, Ré, *Learning Mixed-Curvature Representations in Product Spaces* (ICLR 2019); Chami et al., *Hyperbolic Graph Convolutional Neural Networks* (NeurIPS 2019, arXiv:1910.12933).

## Implementation (v1.4.0-connection)

- **Core.** `inc/eshkol/backend/riemannian_core.h` evaluates the chart for every
  sign with the signed parameter `B = -K alpha^2/4` (`ESHKOL_RM_ALPHA = 1`). The
  conformal denominator `1 + K|x|^2/4` is formed from `K` directly, and below
  `|K| = 2^-1020`, where `B` would drop bits, the ops use the chart's scale
  covariance `g_K(x) = s^2 g_{K s^2}(x/s)` at `s = 2`. The ratios `psi`,
  `tan_ratio` and `artan_ratio` switch to the degree-10 series at
  `|q| <= 2^-8`. The curvature jet (`eshkol_rm_distance_dK`) differentiates the
  series termwise inside the window, uses series-evaluated `psi'`/`psi''` up to
  `|w| <= 1/2`, and the `tanh`/`tan` form of the log-derivatives beyond, with
  `dB/dK` folded into the exponent-separated scaling.
- **Embedded sphere.** The ambient-coordinate ops (`great-circle-distance`,
  `slerp`, `spherical-*`, the space-form `SPHERICAL` factor) use the
  `eshkol_rm_sphere_*` functions; `eshkol_rm_sphere_to_chart` and
  `eshkol_rm_chart_to_sphere` are the isometry to the chart.
- **Consumers.** The VM geometry ops, the AD bridge (Poincare primitives,
  squared-distance attention with `ad_geodesic_attention_tempered`, the
  space-form squared distance with the new `STEREOGRAPHIC` factor, the Frechet
  mean with chart parameter `-K/4`) and the native `core.curvature` module
  (`lib/core/curvature_runtime.c`) call the same core.
- **Attention.** `eshkol_rm_attention_row` is the one stabilized row kernel:
  logits `-d^2/(2t)`, default `t = sqrt(head_dim)`, max-shift by the row's
  smallest squared distance. The reverse rule uses `-2 lambda^2 log`, exactly
  zero at coincidence.
- **Serialization.** `ESHKOL_GEOMETRY_CONVENTION_STEREOGRAPHIC_V2_ALPHA1` (2),
  `eshkol_rm_check_geometry_header` (refuses unversioned, legacy, unknown or
  `alpha != 1` records) and the migration helpers for points, tangents and
  attention temperatures (`t = d_ref / s`, slope-matched at a recorded
  reference distance).
- **Gates.** `tests/bridge/signed_curvature_reference_grid_test.cpp` (binary128,
  every finite binade of both signs; golden subset elsewhere),
  `tests/geometry/signed_curvature_test.esk` (JIT and AOT),
  `tests/vm/signed_curvature_surface.esk` (VM), producer
  `scripts/run_signed_curvature_gate.sh`.
- **Planned for v1.4.1.** Routing the qLLM fp32 geometry through the versioned
  core (Migration step 5), intrinsic Frechet aggregation in attention (step 6,
  separately gated), the positive-curvature Frechet mean, mixed endpoint/K
  directional derivatives in the reference grid, the chart for the bridge
  Poincare primitives' K > 0 reverse rules, and `core.curvature` on the
  bytecode VM.
